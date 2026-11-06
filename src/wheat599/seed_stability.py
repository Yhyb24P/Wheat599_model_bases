"""Fixed-selection seed stability for Wheat599 paired systems."""
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from wheat599.data import load, sha256_of
from wheat599.metrics import pcc, rmse, mae
from wheat599.quantile_tabm_final import fit_one, read_oof


def quantile_rule(rows):
    if len(rows)!=2 or len({r['config_index'] for r in rows})!=1 or {r['seed'] for r in rows}!={42,2026}:
        raise ValueError('Unexpected frozen quantile selection')
    epochs=[epoch for row in rows for epoch in row['inner_best_epochs']]
    if len(epochs)!=6: raise ValueError('Missing inner epochs')
    return rows[0]['config_index'],int(np.ceil(np.median(epochs)))


def run(cfg,args,runner):
    if args.selection is not None:raise ValueError('Fixed selections only')
    qc=yaml.safe_load(Path(cfg['quantile_config']).read_text());ac=yaml.safe_load(Path(cfg['architecture_config']).read_text())['configurations']
    mc=yaml.safe_load(Path(cfg['moegp_config']).read_text());source=Path(mc['source'])
    commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    if commit!=mc['source_commit'] or subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True).strip():raise ValueError('MoEGP source mismatch')
    sys.path.insert(0,str(source));sys.dont_write_bytecode=True
    from utils.moe import creat_mlp_layer
    from run_moegp_nested import fit_moegp
    torch.set_num_threads(cfg['cpu_threads']);device=torch.device(cfg['device'])
    if device.type=='cuda':torch.cuda.set_per_process_memory_fraction(cfg['gpu_memory_fraction'])
    data=load(Path(cfg['data']),cfg['data_sha256']);folds=json.loads(Path(cfg['splits']).read_text())['folds'];manifest=json.loads(Path(cfg['quantile_manifest']).read_text())
    with Path(cfg['moegp_selection']).open() as f: selection={(int(r['fold']),r['environment']):r for r in csv.DictReader(f)}
    provenance=json.loads((Path(cfg['moegp_selection']).parent/'provenance.json').read_text())
    if provenance['config']!=mc or provenance['independent_experts']:raise ValueError('MoEGP selection mismatch')
    for path in (cfg['splits'],cfg['quantile_config'],cfg['architecture_config'],'src/wheat599/quantile_tabm.py'):
        if manifest['sha256'][path]!=sha256_of(Path(path)):raise ValueError('Quantile provenance mismatch')
    huber=read_oof(cfg['huber_oof']);args.out.mkdir(parents=True,exist_ok=False);weights=args.out/'weights';weights.mkdir()
    paths=[Path(v) for v in (args.config,runner,__file__,cfg['splits'],cfg['quantile_manifest'],cfg['quantile_config'],cfg['architecture_config'],cfg['moegp_config'],cfg['moegp_selection'],cfg['huber_oof'],'scripts/run_moegp_nested.py','src/wheat599/quantile_tabm_final.py','src/wheat599/quantile_tabm.py')]
    write=lambda name,obj:(args.out/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
    write('provenance.json',{'config':cfg,'source_commit':commit,'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'sha256':{str(p):sha256_of(p) for p in paths}})
    status={'status':'running','completed':0,'expected':len(folds)*4*len(cfg['seeds'])*2};failures=[];records=[];predictions={};start=time.perf_counter()
    try:
        for fold in folds:
            f=fold['fold'];dev=np.array(fold['dev_idx']);test=np.array(fold['test_idx'])
            if set(dev)&set(test) or len(dev)+len(test)!=len(data.X):raise ValueError('Invalid outer split')
            xt=torch.as_tensor(data.X,dtype=torch.float32,device=device)
            for e,env in enumerate(data.environments):
                rows=[r for r in manifest['rows'] if r['fold']==f and r['environment_index']==e]
                ci,epochs=quantile_rule(rows)
                if any(r['fit_ids']!=fold['dev_idx'] or r['test_ids']!=fold['test_idx'] for r in rows):raise ValueError('Quantile split mismatch')
                sel=selection[(f,str(env))]
                if sel['failed'].lower() in ('true','1'):raise ValueError('Failed MoEGP selection')
                cand=next(c for c in mc['candidates'] if c['name']==sel['selected']); schedule=json.loads(sel['lr_schedule']);n=int(sel['refit_epochs'])
                key=[(f,str(env),int(i)) for i in test]
                if any(k not in huber for k in key):raise ValueError('Huber OOF mismatch')
                if not np.allclose(data.Y[test,e],[huber[k][0] for k in key],rtol=0,atol=1e-12):raise ValueError('Huber target mismatch')
                hp=np.array([huber[k][1] for k in key])
                for seed in cfg['seeds']:
                    row={'fold':f,'environment_index':e,'seed':seed,'config_index':ci,'epochs':epochs}
                    qp,det=fit_one(data.X[dev],data.Y[dev,e],data.X[test],row,ac[ci],qc,device,weights)
                    result=fit_moegp(xt,data.Y[:,e],dev,None,cand,mc['model'],mc['training'],n,device,seed,creat_mlp_layer,lr_schedule=schedule)
                    with torch.no_grad():mp=result['model'](xt[test])[0].detach().float().cpu().numpy().ravel()*result['sd']+result['mu']
                    if not np.isfinite(mp).all():raise ValueError('Nonfinite MoEGP output')
                    preds={'quantile':qp,'fusion':.5*(qp+hp),'moegp':mp}
                    predictions[f'f{f}_e{e}_s{seed}']=np.column_stack([qp,hp,mp])
                    for method,pred in preds.items():
                        values=[fun(data.Y[test,e],pred) for fun in (pcc,rmse,mae)]
                        if not np.isfinite(values).all():raise ValueError('Undefined seed metric')
                        records.append({'fold':f,'environment':str(env),'seed':seed,'method':method,**dict(zip(('pcc','rmse','mae'),map(float,values)))})
                    status['completed']+=2;del result
                write('status.json',status)
            np.savez_compressed(args.out/f'predictions_{f}.npz',**{k:v for k,v in predictions.items() if k.startswith(f'f{f}_')})
            predictions.clear();write('records.json',records)
        if status['completed']!=status['expected'] or len(records)!=len(folds)*4*len(cfg['seeds'])*3:raise ValueError('Incomplete seed results')
        by_seed={str(seed):{method:{metric:float(np.mean([r[metric] for r in records if r['seed']==seed and r['method']==method])) for metric in ('pcc','rmse','mae')} for method in ('quantile','fusion','moegp')} for seed in cfg['seeds']}
        macro={method:{metric:float(np.mean([by_seed[str(seed)][method][metric] for seed in cfg['seeds']])) for metric in ('pcc','rmse','mae')} for method in ('quantile','fusion','moegp')}
        write('summary.json',{'by_seed':by_seed,'mean_of_seed_metrics':macro});status['status']='complete'
    except BaseException as exc:
        status['status']='failed';failures.append({'error':repr(exc),'completed':status['completed']});raise
    finally:
        write('status.json',status);write('failures.json',failures);write('records.json',records);write('timing.json',{'seconds':time.perf_counter()-start})
