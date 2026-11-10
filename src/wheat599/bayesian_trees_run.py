"""Joint sparse Bayesian forests with a Gaussian polygenic background."""
import concurrent.futures
import json
import multiprocessing
import os
import shutil
import subprocess
import time
from pathlib import Path

import arviz as az
import numpy as np
from wheat599.data import load, sha256_of
from wheat599.distill_quantile import read, write
from wheat599.metrics import pcc, rmse, mae

SOURCE = Path(__file__).with_name('bayesian_trees')


def r_command():
    prefix = os.environ.get('WHEAT599_R_PREFIX')
    if prefix:
        executable = str(Path(prefix)/'bin/Rscript')
    else:
        executable = shutil.which('Rscript')
    if not executable:
        raise RuntimeError('Rscript with Rcpp/RcppArmadillo/jsonlite is required')
    return executable


def posterior(directory, name):
    meta = read(directory/'metadata.json')
    shape = meta['dimensions'][name]
    a = np.fromfile(directory/f'{name}.f64', dtype='<f8')
    if a.size != int(np.prod(shape)) or not np.isfinite(a).all():
        raise ValueError('Invalid posterior array: '+name)
    return a.reshape(shape, order='F')


def diagnostics(chains, target_sd):
    training = np.stack([np.moveaxis(posterior(c,'training'),-1,0) for c in chains])
    covariance = np.stack([np.moveaxis(posterior(c,'sigma'),-1,0) for c in chains])
    traces = np.stack([posterior(c,'trace') for c in chains])
    nchain, draws, n, env = training.shape
    tri = np.triu_indices(env)
    monitored = dict(training=(training/target_sd).reshape(nchain,draws,-1),
                     covariance=covariance[:,:,tri[0],tri[1]],
                     state=traces[:,:,[*range(8),*range(12,20)]])
    report = {}
    for name, values in monitored.items():
        varying = np.std(values.reshape(-1,values.shape[-1]),axis=0)>0
        if not varying.any():
            raise ValueError('Entire MCMC diagnostic block is constant: '+name)
        data = az.convert_to_dataset({'value':values[:,:,varying]})
        rh = az.rhat(data,method='rank')['value'].values
        ess = az.ess(data,method='bulk')['value'].values
        mcse = az.mcse(data,method='mean')['value'].values
        if not all(np.isfinite(a).all() for a in (rh,ess,mcse)):
            raise ValueError('Undefined MCMC diagnostics: '+name)
        report[name] = dict(max_rhat=float(rh.max()),min_bulk_ess=float(ess.min()),
                            max_mcse=float(mcse.max()),constant_columns=int((~varying).sum()))
    report.update(chains=nchain,draws_per_chain=draws)
    return report


def sample_chain(request, output):
    output.mkdir()
    request = dict(request,output=str(output.resolve()))
    write(output/'request.json',request)
    env = dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    if env.get('WHEAT599_R_PREFIX'):
        env['PATH']=str(Path(env['WHEAT599_R_PREFIX'])/'bin')+os.pathsep+env['PATH']
    start=time.monotonic()
    with (output/'sampler.log').open('w') as log:
        subprocess.run([r_command(),str(SOURCE/'run.R'),str(output/'request.json')],env=env,
                       stdout=log,stderr=subprocess.STDOUT,check=True)
    write(output/'execution.json',dict(seconds=time.monotonic()-start,
          sha256={p.name:sha256_of(p) for p in output.glob('*.f64')}))


def run_unit(request):
    x,y,query,fold,inner,cfg,output = request
    directory=Path(output)/f'unit_{fold}_{inner}';directory.mkdir()
    for name,a in dict(x=x,y=y,query=query).items():
        np.savetxt(directory/f'{name}.csv',a,delimiter=',',fmt='%.17g')
    predictions={};records=[]
    for c,case in enumerate(cfg['configurations']):
        case_dir=directory/f'configuration_{c}';case_dir.mkdir()
        attempt=case_dir/'posterior';attempt.mkdir()
        base=dict(source=str((SOURCE/'sampler.cpp').resolve()),directory=str(directory.resolve()),
            loader=read(Path(output)/'compile/loader.json')['loader'],
            **case,warmup=cfg['mcmc']['warmup'],draws=cfg['mcmc']['draws'])
        chains=[]
        for seed in cfg['seeds']:
            if (Path(output)/'STOP').exists():
                raise RuntimeError('Another unit failed; remaining chains stopped')
            chain=attempt/f'chain_{seed}';chains.append(chain)
            sample_chain(dict(base,seed=seed+100000*fold+1000*inner),chain)
        check=diagnostics(chains,y.std(0))
        write(attempt/'diagnostics.json',check)
        pred=np.mean([posterior(chain,'query').mean(axis=-1) for chain in chains],axis=0)
        if pred.shape!=(len(query),4) or not np.isfinite(pred).all():
            raise ValueError('Invalid posterior mean prediction')
        predictions[f'c{c}']=pred
        records.append(dict(configuration=c,diagnostics=check))
        write(directory/'sampling.json',records)
    np.savez_compressed(directory/'predictions.npz',**predictions)
    write(directory/'execution_status.json',dict(status='complete',configurations=len(records)))
    return dict(fold=fold,inner=inner,configurations=len(records))


def summarize(data,folds,cfg,out):
    source=Path(cfg['quantile_source']);selected=read(source/'summary.json')['folds'];rows=[]
    for fold in folds:
        f=fold['fold'];arrays=[np.load(out/f'unit_{f}_{j}/predictions.npz') for j in range(3)]
        scores=[float(np.mean([pcc(data.Y[s['val_idx'],e],arrays[j][f'c{c}'][:,e])
                for j,s in enumerate(fold['inner']) for e in range(4)])) for c in range(len(cfg['configurations']))]
        if not np.isfinite(scores).all(): raise ValueError('Undefined development PCC')
        choice=int(np.argmax(scores));methods={name:[] for name in ('candidate','original_quantile','huber','fixed_fusion')}
        qconfigs=next(v['selected'] for v in selected if v['fold']==f)
        for j,s in enumerate(fold['inner']):
            old=np.load(source/f'predictions_{f}_{j}.npz')
            q=np.column_stack([np.mean([old[f'e{e}_c{c}_s{seed}'] for seed in cfg['reference_seeds']],axis=0)
                               for e,c in enumerate(qconfigs)])
            hub=np.load(Path(cfg['huber_reference'])/f'references_{f}_{j}.npz')['huber']
            predictions=dict(candidate=arrays[j][f'c{choice}'],original_quantile=q,huber=hub,fixed_fusion=(q+hub)/2)
            for name,pred in predictions.items():
                methods[name].append([[fn(data.Y[s['val_idx'],e],pred[:,e]) for fn in (pcc,rmse,mae)] for e in range(4)])
        rows.append(dict(fold=f,selected_index=choice,selected=cfg['configurations'][choice],
                         pcc_by_configuration=scores,metrics={k:np.mean(v,axis=0).tolist() for k,v in methods.items()}))
    macro={name:dict(zip(('pcc','rmse','mae'),np.mean([r['metrics'][name] for r in rows],axis=(0,1)).tolist()))
           for name in rows[0]['metrics']}
    return dict(macro=macro,folds=rows)


def run(cfg,args,runner):
    if args.selection is not None: raise ValueError('Development only; no outer evaluation authorized')
    data=load(Path(cfg['data']),cfg['data_sha256']);folds=read(cfg['splits'])['folds']
    if len(folds)!=10 or len(cfg['seeds'])!=4 or len(cfg['configurations'])!=3:
        raise ValueError('Unexpected development budget')
    if read(Path(cfg['quantile_source'])/'failures.json'):raise ValueError('Reference experiment has failures')
    if read(Path(cfg['quantile_source'])/'provenance.json')['sha256'][cfg['splits']]!=sha256_of(Path(cfg['splits'])):
        raise ValueError('Reference split mismatch')
    args.out.mkdir(parents=True,exist_ok=False)
    files=[Path(args.config),Path(cfg['splits']),Path(__file__),Path(runner),*SOURCE.glob('*')]
    files.append(Path(cfg['quantile_source'])/'summary.json')
    for f in folds:
        for j in range(3):
            files.extend([Path(cfg['quantile_source'])/f"predictions_{f['fold']}_{j}.npz",
                          Path(cfg['huber_reference'])/f"references_{f['fold']}_{j}.npz"])
    write(args.out/'provenance.json',dict(config=cfg,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        sha256={str(p):sha256_of(p) for p in files if p.is_file()},rscript=r_command(),
        arviz=az.__version__,upstream='MateusMaiaDS/subart@deddf4ea1290513cd6bce4e7867349d6c32fc2fa'))
    requests=[]
    for f in folds:
        if len(f['inner'])!=3:raise ValueError('Expected three inner folds')
        for j,s in enumerate(f['inner']):
            tr,va=s['train_idx'],s['val_idx']
            if set(tr)&set(va) or set(tr)|set(va)!=set(f['dev_idx']) or (set(tr)|set(va))&set(f['test_idx']):
                raise ValueError('Invalid development split')
            requests.append((data.X[tr],data.Y[tr],data.X[va],f['fold'],j,cfg,str(args.out)))
    state=dict(status='running',completed_units=0,expected_units=30,configurations=3);start=time.monotonic();failures=[]
    write(args.out/'execution_status.json',state)
    try:
        # Compile once before starting independent R workers, avoiding cache writers racing.
        compile_dir=args.out/'compile';compile_dir.mkdir()
        write(compile_dir/'request.json',dict(source=str((SOURCE/'sampler.cpp').resolve()),compile_only=True,
                                            loader_manifest=str((compile_dir/'loader.json').resolve())))
        env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
        if env.get('WHEAT599_R_PREFIX'):env['PATH']=str(Path(env['WHEAT599_R_PREFIX'])/'bin')+os.pathsep+env['PATH']
        with (compile_dir/'build.log').open('w') as log:
            subprocess.run([r_command(),str(SOURCE/'run.R'),str(compile_dir/'request.json')],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        with concurrent.futures.ProcessPoolExecutor(max_workers=cfg['workers'],mp_context=multiprocessing.get_context('spawn')) as pool:
            futures=[pool.submit(run_unit,r) for r in requests]
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except BaseException:
                    (args.out/'STOP').touch()
                    for other in futures:other.cancel()
                    raise
                state['completed_units']+=1;write(args.out/'execution_status.json',state)
        write(args.out/'summary.json',summarize(data,folds,cfg,args.out))
        write(args.out/'sampling.json',{
            f'unit_{f["fold"]}_{j}':read(args.out/f'unit_{f["fold"]}_{j}/sampling.json')
            for f in folds for j in range(3)})
        write(args.out/'artifacts.json',{
            str(p.relative_to(args.out)):read(p) for p in args.out.glob('unit_*/configuration_*/posterior/chain_*/execution.json')})
        state['status']='complete'
    except BaseException as exc:
        state['status']='failed';failures.append(dict(error=repr(exc)));raise
    finally:
        write(args.out/'execution_status.json',state);write(args.out/'failures.json',failures)
        write(args.out/'timing.json',dict(seconds=time.monotonic()-start))
