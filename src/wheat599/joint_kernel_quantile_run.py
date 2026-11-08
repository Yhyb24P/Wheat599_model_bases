"""Development-only paired comparison of separable quantile kernel states."""
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from wheat599.data import load, sha256_of
from wheat599.joint_kernel_quantile import predict, solve
from wheat599.metrics import mae, pcc, rmse
from wheat599.robust_icm import features


def candidates(cfg, levels):
    q=len(levels);smooth=np.exp(-cfg['gamma']*(levels[:,None]-levels[None,:])**2)
    states={'shared':np.ones((q,q)),'smooth':smooth,'independent':np.eye(q)}
    return [(name,float(eta),float(scale*np.linalg.eigvalsh(c)[-1]/q),c)
            for name,c in states.items() for eta in cfg['etas'] for scale in cfg['ridge_shared']]


def run(cfg,args,runner):
    if args.selection is not None:raise ValueError('Development only')
    data=load(Path(cfg['data']),cfg['data_sha256']);folds=json.loads(Path(cfg['splits']).read_text())['folds']
    bases=json.loads(Path(cfg['base_source'],'base_choices.json').read_text())
    subfolds=json.loads(Path(cfg['base_source'],'training_splits.json').read_text())
    baseline=Path(cfg['baseline_source']);reference=Path(cfg['reference_source'])
    split_hash=sha256_of(Path(cfg['splits']))
    if json.loads(Path(cfg['base_source'],'provenance.json').read_text())['splits_sha256']!=split_hash:raise ValueError('Base split hash mismatch')
    if any(json.loads((p/'provenance.json').read_text())['sha256'][cfg['splits']]!=split_hash for p in (baseline,reference)):raise ValueError('Reference split hash mismatch')
    chosen=json.loads((baseline/'summary.json').read_text())['folds'];audit=json.loads(Path(cfg['baseline_audit']).read_text())['sha256']
    for f in folds:
        for j in range(3):
            for path in (baseline/f"predictions_{f['fold']}_{j}.npz",reference/f"references_{f['fold']}_{j}.npz"):
                if sha256_of(path)!=audit[str(path)]:raise ValueError('Baseline prediction hash mismatch')
    levels=np.linspace(cfg['quantile_min'],cfg['quantile_max'],cfg['n_quantiles']);choices=candidates(cfg,levels)
    torch.set_num_threads(cfg['cpu_threads']);device=torch.device(cfg['device'])
    if device.type=='cuda':torch.cuda.set_per_process_memory_fraction(cfg['gpu_memory_fraction'])
    paths=[args.config,Path(runner),Path(__file__),Path(__file__).with_name('joint_kernel_quantile.py'),Path(cfg['splits']),Path(cfg['base_source'],'base_choices.json'),Path(cfg['base_source'],'training_splits.json'),Path(cfg['baseline_audit']),baseline/'summary.json']
    args.out.mkdir(parents=True,exist_ok=False)
    def write(name,value):(args.out/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    write('provenance.json',{'config':cfg,'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'sha256':{str(p):sha256_of(p) for p in paths},'rule':'development only; shared, smooth and independent each select eta/ridge by macro PCC'})
    status={'status':'running','completed':0,'expected':len(folds)*3*len(choices)};records=[];failures=[];start=time.perf_counter();per=[]
    try:
        for fold in folds:
            f=fold['fold'];scores={str(i):[] for i in range(len(choices))};scores.update({'baseline':[],'fusion':[],'huber':[]})
            selected=next(v['selected'] for v in chosen if v['fold']==f)
            for j,inner in enumerate(fold['inner']):
                tr=np.array(inner['train_idx']);val=np.array(inner['val_idx'])
                if set(tr)&set(val) or set(tr)|set(val)!=set(fold['dev_idx']) or (set(tr)|set(val))&set(fold['test_idx']):raise ValueError('Invalid material split')
                sub=next(v['splits'] for v in subfolds if v['fold']==f and v['inner_fold']==j)
                if len(sub)!=3 or any(set(x['train_idx'])&set(x['val_idx']) or set(x['train_idx'])|set(x['val_idx'])!=set(tr) for x in sub):raise ValueError('Kernel selection split mismatch')
                base=next(v['homogeneous'] for v in bases if v['fold']==f and v['inner_fold']==j)
                prepared=features(data.X,data.Y,tr,val,base);a=prepared['train_features'];k=a@a.T;cross=prepared['target_features']@a.T
                truth=data.Y[val];old=np.load(baseline/f'predictions_{f}_{j}.npz');ref=np.load(reference/f'references_{f}_{j}.npz')['huber']
                orig=np.column_stack([np.mean([old[f'e{e}_c{selected[e]}_s{seed}'] for seed in (42,2026)],axis=0) for e in range(4)])
                predictions={'baseline':orig,'fusion':.5*(orig+ref),'huber':ref};curves={}
                for i,(state,eta,ridge,c) in enumerate(choices):
                    b=(1-eta)*np.eye(4)+eta*prepared['C_Y'];begin=time.perf_counter()
                    result=solve(k,b,c,prepared['Y'].numpy(),levels,ridge,tolerance=cfg['tolerance'],max_iterations=cfg['max_iterations'],check_every=cfg['check_every'],device=device)
                    curve=predict(cross,b,c,result)*prepared['scale'][None,:,None]+prepared['mean'][None,:,None]
                    if not np.isfinite(curve).all():raise ValueError('Nonfinite predictions')
                    curves[str(i)]=curve;predictions[str(i)]=curve.mean(axis=2)
                    records.append({'fold':f,'inner':j,'candidate':i,'state':state,'eta':eta,'ridge':ridge,'seconds':time.perf_counter()-begin,**result['diagnostics']})
                    status['completed']+=1
                for name,pred in predictions.items():
                    vals=[[fn(truth[:,e],pred[:,e]) for fn in (pcc,rmse,mae)] for e in range(4)]
                    if not np.isfinite(vals).all():raise ValueError('Nonfinite metric')
                    scores[name].append(vals)
                np.savez_compressed(args.out/f'curves_{f}_{j}.npz',**curves)
                write('status.json',status);write('trajectories.json',records)
            per.append({'fold':f,'metrics':{k:np.mean(v,axis=0).tolist() for k,v in scores.items()}})
        if status['completed']!=status['expected']:raise ValueError('Incomplete development run')
        selection=[]
        for fold in per:
            entry={'fold':fold['fold']}
            for state in ('shared','smooth','independent'):
                ids=[i for i,c in enumerate(choices) if c[0]==state]
                i=max(ids,key=lambda v:float(np.mean(np.asarray(fold['metrics'][str(v)])[:,0])))
                entry[state]={'candidate':i,'eta':choices[i][1],'ridge':choices[i][2]}
            selection.append(entry)
        macro={}
        for name in ('shared','smooth','independent'):
            v=np.mean([fold['metrics'][str(selection[j][name]['candidate'])] for j,fold in enumerate(per)],axis=(0,1))
            macro[name]=dict(zip(('pcc','rmse','mae'),v.tolist()))
        for name in ('baseline','fusion','huber'):
            v=np.mean([fold['metrics'][name] for fold in per],axis=(0,1));macro[name]=dict(zip(('pcc','rmse','mae'),v.tolist()))
        descriptors=[{'id':i,'state':state,'eta':eta,'ridge':ridge} for i,(state,eta,ridge,_) in enumerate(choices)]
        write('selection.json',{'candidates':descriptors,'folds':selection});write('summary.json',{'macro':macro,'folds':per,'selection':selection});status['status']='complete'
    except BaseException as exc:
        status['status']='failed';failures.append({'completed':status['completed'],'error':repr(exc)});raise
    finally:
        write('status.json',status);write('failures.json',failures);write('trajectories.json',records);write('timing.json',{'seconds':time.perf_counter()-start})
