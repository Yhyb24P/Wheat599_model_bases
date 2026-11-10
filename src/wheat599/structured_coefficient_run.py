"""Development evaluation of one structured robust multienvironment predictor."""
import concurrent.futures
import itertools
import multiprocessing
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
from scipy.linalg import solve_triangular
from wheat599.data import load,sha256_of
from wheat599.distill_quantile import read,write
from wheat599.metrics import pcc,rmse,mae
from wheat599.robust_icm import fit_predict
from wheat599.structured_coefficient import expansion,prepared_features,spectrum,posterior_from_spectrum,huber_certificate


def configurations(cfg):
    return [dict(zip(('bandwidth','epsilon','lambda','eta','delta'),v)) for v in itertools.product(cfg['bandwidths'],cfg['epsilons'],cfg['lambdas'],cfg['etas'],cfg['deltas'])]


def references(cfg):
    return [dict(zip(('bandwidth','epsilon','ridge'),v)) for v in itertools.product(cfg['bandwidths'],cfg['epsilons'],cfg['reference_ridges'])]


def run_unit(request):
    x,y,query,fold,inner,cfg,output=request;torch.set_num_threads(1)
    directory=Path(output)/f'unit_{fold}_{inner}';directory.mkdir()
    cases=configurations(cfg);bases=references(cfg);predictions={};weights={};records=[];diagnostics=[];max_replay=0.
    cindex={tuple(v.values()):i for i,v in enumerate(cases)};bindex={tuple(v.values()):i for i,v in enumerate(bases)}
    for bandwidth,epsilon in itertools.product(cfg['bandwidths'],cfg['epsilons']):
        d,dc,meta=expansion(x,query,bandwidth,epsilon)
        prepared=prepared_features(d,dc,y);ref_spec=spectrum(d,dc,center=False)
        for ridge in cfg['reference_ridges']:
            b=bindex[bandwidth,epsilon,ridge];pred,theta=posterior_from_spectrum(ref_spec,prepared['Y'].numpy(),ridge)
            coefficient=solve_triangular(meta['lower'].T,theta,lower=False)
            pred=pred*prepared['scale']+prepared['mean'];replay=(meta['query_rbf']@coefficient)*prepared['scale']+prepared['mean']
            difference=float(np.max(abs(pred-replay)));max_replay=max(max_replay,difference)
            if difference>1e-6:raise ValueError('Reference coefficient replay mismatch')
            predictions[f'b{b}']=pred;weights[f'b{b}_coefficient']=coefficient
        for lam,eta,delta in itertools.product(cfg['lambdas'],cfg['etas'],cfg['deltas']):
            c=cindex[bandwidth,epsilon,lam,eta,delta];ridge=lam*y.size
            pred,diag,state=fit_predict(prepared,ridge,eta,delta,cfg['optimizer'],return_state=True)
            certificate=huber_certificate(prepared,state,ridge,diag['thresholds'])
            if not (-1e-10<=certificate['relative_gap']<=cfg['relative_gap_tolerance']) or certificate['dual_zero_sum_residual']>1e-12 or certificate['dual_box_violation']>1e-12:
                raise ValueError(f'Uncertified structured Huber optimization: {certificate}')
            diag.update(certificate)
            theta=prepared['rotation']@state['weights']@state['factor'].T
            coefficient=solve_triangular(meta['lower'].T,theta,lower=False)
            bias=state['intercept']-prepared['feature_mean']@theta
            replay=(meta['query_rbf']@coefficient+bias)*prepared['scale']+prepared['mean']
            difference=float(np.max(abs(pred-replay)));max_replay=max(max_replay,difference)
            if difference>1e-6:raise ValueError('Structured coefficient replay mismatch')
            predictions[f'c{c}']=pred;weights[f'c{c}_coefficient']=coefficient;weights[f'c{c}_intercept']=bias
            diagnostics.append(dict(config_index=c,**diag))
        records.append(dict(bandwidth=bandwidth,epsilon=epsilon,min_precision_eigenvalue=meta['min_precision_eigenvalue'],effective_mean_diagonal=meta['effective_mean_diagonal']))
        write(directory/'execution_status.json',dict(status='running',candidate_fits=len(diagnostics),reference_fits=len([k for k in predictions if k.startswith('b')])))
    for pred in predictions.values():
        if pred.shape!=(len(query),4) or not np.isfinite(pred).all():raise ValueError('Invalid development predictions')
    weights.update(mean=y.mean(0),scale=y.std(0))
    np.savez_compressed(directory/'predictions.npz',**predictions);np.savez_compressed(directory/'coefficients.npz',**weights)
    write(directory/'optimization.json',diagnostics);write(directory/'kernels.json',records)
    write(directory/'verification.json',dict(coefficient_replay_max_difference=max_replay,sha256={n:sha256_of(directory/n) for n in ('predictions.npz','coefficients.npz','optimization.json','kernels.json')}))
    write(directory/'execution_status.json',dict(status='complete',candidate_fits=len(cases),reference_fits=len(bases)))
    return dict(fold=fold,inner=inner,candidate_fits=len(cases),reference_fits=len(bases))


def summarize(data,folds,cfg,out):
    cases=configurations(cfg);bases=references(cfg);all_folds=[]
    original=Path(cfg['quantile_source']);selection=read(original/'summary.json')['folds']
    for fold in folds:
        f=fold['fold'];arr=[np.load(out/f'unit_{f}_{j}/predictions.npz') for j in range(3)]
        cscores=[float(np.mean([pcc(data.Y[s['val_idx'],e],arr[j][f'c{c}'][:,e]) for j,s in enumerate(fold['inner']) for e in range(4)])) for c in range(len(cases))]
        if not np.isfinite(cscores).all():raise ValueError('Undefined candidate validation PCC')
        best=max(cscores);chosen=next(c for c,v in enumerate(cscores) if best-v<=cfg['tie_tolerance'])
        rchoices=[]
        for e in range(4):
            scores=[float(np.mean([pcc(data.Y[s['val_idx'],e],arr[j][f'b{b}'][:,e]) for j,s in enumerate(fold['inner'])])) for b in range(len(bases))]
            if not np.isfinite(scores).all():raise ValueError('Undefined reference validation PCC')
            rchoices.append(int(np.argmax(scores)))
        methods={k:[] for k in ('candidate','unweighted_posterior','original_quantile','huber','fixed_fusion')}
        qconfigs=next(v['selected'] for v in selection if v['fold']==f)
        for j,split in enumerate(fold['inner']):
            old=np.load(original/f'predictions_{f}_{j}.npz');hub=np.load(Path(cfg['huber_reference'])/f'references_{f}_{j}.npz')['huber']
            q=np.column_stack([np.mean([old[f'e{e}_c{c}_s{seed}'] for seed in cfg['seeds']],axis=0) for e,c in enumerate(qconfigs)])
            pred=dict(candidate=arr[j][f'c{chosen}'],unweighted_posterior=np.column_stack([arr[j][f'b{rchoices[e]}'][:,e] for e in range(4)]),original_quantile=q,huber=hub,fixed_fusion=.5*(q+hub))
            for method,value in pred.items():methods[method].append([[fn(data.Y[split['val_idx'],e],value[:,e]) for fn in (pcc,rmse,mae)] for e in range(4)])
        all_folds.append(dict(fold=f,selected_index=chosen,selected=cases[chosen],candidate_pcc_by_config=cscores,reference_selected_indices=rchoices,reference_selected=[bases[b] for b in rchoices],metrics={k:np.mean(v,axis=0).tolist() for k,v in methods.items()}))
    macro={name:dict(zip(('pcc','rmse','mae'),np.mean([r['metrics'][name] for r in all_folds],axis=(0,1)).tolist())) for name in all_folds[0]['metrics']}
    return dict(macro=macro,folds=all_folds,configurations=cases,reference_configurations=bases,reference_label='Unweighted GWKBR posterior formula with binary G+epsilon I, fit-label standardization and effective ridge grid; not full author-pipeline reproduction')


def run(cfg,args,runner):
    if args.selection is not None:raise ValueError('Development comparison only')
    data=load(Path(cfg['data']),cfg['data_sha256']);folds=read(cfg['splits'])['folds'];cases=configurations(cfg);bases=references(cfg)
    if len(folds)!=10 or len(cases)!=504 or len(bases)!=63:raise ValueError('Unexpected frozen development budget')
    for source in (Path(cfg['quantile_source']),):
        if read(source/'execution_status.json')!=dict(status='complete',completed=1920,expected=1920) or read(source/'failures.json'):raise ValueError('Reference experiment incomplete')
        if read(source/'provenance.json')['sha256'][cfg['splits']]!=sha256_of(Path(cfg['splits'])):raise ValueError('Reference split mismatch')
    args.out.mkdir(parents=True,exist_ok=False)
    sources=[Path(args.config),Path(cfg['splits']),Path(__file__),Path(__file__).with_name('structured_coefficient.py'),Path(__file__).with_name('robust_icm.py'),Path(__file__).with_name('reml_icm.py'),Path(__file__).with_name('kernel_baselines.py'),Path(runner),Path(cfg['upstream_verification']),Path(cfg['quantile_source'])/'summary.json']
    for f in folds:
        for j in range(3):sources.extend([Path(cfg['quantile_source'])/f"predictions_{f['fold']}_{j}.npz",Path(cfg['huber_reference'])/f"references_{f['fold']}_{j}.npz"])
    write(args.out/'provenance.json',dict(config=cfg,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),sha256={str(p):sha256_of(p) for p in sources}))
    start=time.perf_counter();state=dict(status='running',completed_units=0,expected_units=30,candidate_fits=0,expected_candidate_fits=15120,reference_four_target_fits=0,expected_reference_four_target_fits=1890);failures=[]
    write(args.out/'execution_status.json',state)
    try:
        requests=[]
        for fold in folds:
            if len(fold['inner'])!=3:raise ValueError('Expected three inner splits')
            for j,split in enumerate(fold['inner']):
                tr,va=split['train_idx'],split['val_idx']
                if set(tr)&set(va) or set(tr)|set(va)!=set(fold['dev_idx']) or (set(tr)|set(va))&set(fold['test_idx']):raise ValueError('Invalid development boundary')
                requests.append((data.X[tr],data.Y[tr],data.X[va],fold['fold'],j,cfg,str(args.out)))
        with concurrent.futures.ProcessPoolExecutor(max_workers=cfg['workers'],mp_context=multiprocessing.get_context('spawn')) as pool:
            futures=[pool.submit(run_unit,r) for r in requests]
            for future in concurrent.futures.as_completed(futures):
                try:r=future.result()
                except BaseException:
                    for other in futures:other.cancel()
                    raise
                state['completed_units']+=1;state['candidate_fits']+=r['candidate_fits'];state['reference_four_target_fits']+=r['reference_fits'];write(args.out/'execution_status.json',state)
        if state['candidate_fits']!=15120 or state['reference_four_target_fits']!=1890:raise ValueError('Incomplete experiment budget')
        write(args.out/'summary.json',summarize(data,folds,cfg,args.out));state['status']='complete'
    except BaseException as exc:
        state['status']='failed';failures.append(dict(error=repr(exc)));raise
    finally:
        write(args.out/'execution_status.json',state);write(args.out/'failures.json',failures);write(args.out/'timing.json',dict(seconds=time.perf_counter()-start))
