"""Paired material-permutation and whole-quantile relation controls."""
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from wheat599.data import load, sha256_of
from wheat599.metrics import pcc, rmse, mae
from wheat599.relation_quantile import train_joint
from wheat599.relation_regularizer import build_kernel, make_relation, make_relation_curve


def permute_kernel(kernel, seed):
    generator=torch.Generator(device='cpu').manual_seed(int(seed))
    permutation=torch.randperm(len(kernel),generator=generator).to(kernel.device)
    return kernel[permutation][:,permutation],permutation


def run(cfg,args,runner):
    if args.selection is not None:raise ValueError('Development only')
    base=yaml.safe_load(Path(cfg['original_config']).read_text());settings=yaml.safe_load(Path(base['quantile_config']).read_text())
    architectures=yaml.safe_load(Path(base['architecture_config']).read_text())['configurations'];original=Path(cfg['original_result'])
    if json.loads((original/'status.json').read_text())!={'status':'complete','completed':180,'expected':180} or json.loads((original/'failures.json').read_text()):raise ValueError('Original development incomplete')
    provenance=json.loads((original/'provenance.json').read_text());digest=sha256_of(Path(cfg['original_config']))
    if provenance['sha256'][cfg['original_config']]!=digest or cfg['strength'] not in base['lambdas']:raise ValueError('Original configuration mismatch')
    folds=json.loads(Path(settings['splits']).read_text())['folds'];baseline=Path(base['baseline_source']);ref=Path(base['reference_source'])
    chosen=json.loads((baseline/'summary.json').read_text())['folds'];bases=json.loads(Path(base['base_choices']).read_text());base_splits=json.loads(Path(base['base_splits']).read_text())
    audit=json.loads(Path(base['baseline_audit']).read_text())['sha256']
    for fold in folds:
        for j in range(3):
            for path in (baseline/f"predictions_{fold['fold']}_{j}.npz",ref/f"references_{fold['fold']}_{j}.npz"):
                if audit[str(path)]!=sha256_of(path):raise ValueError('Baseline prediction hash mismatch')
    data=load(Path(settings['data']),settings['data_sha256']);device=torch.device(settings['device']);torch.set_num_threads(base['cpu_threads'])
    if device.type=='cuda':torch.cuda.set_per_process_memory_fraction(base['gpu_memory_fraction'])
    paths=[Path(args.config),Path(cfg['original_config']),Path(__file__),Path(runner),Path(__file__).with_name('relation_quantile.py'),Path(__file__).with_name('relation_regularizer.py'),original/'provenance.json',original/'summary.json',original/'status.json',original/'trajectories.json']
    paths += [original/f"predictions_{f['fold']}_{j}.npz" for f in folds for j in range(3)]
    args.out.mkdir(parents=True,exist_ok=False)
    def write(name,value):(args.out/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    write('provenance.json',{'config':cfg,'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(), 'sha256':{str(p):sha256_of(p) for p in paths}})
    state={'status':'running','completed':0,'expected':len(folds)*3*len(settings['seeds'])*2};records=[];failures=[];permutations=[];start=time.perf_counter()
    try:
        for fold in folds:
            f=fold['fold'];indices=next(v['selected'] for v in chosen if v['fold']==f);configs=[architectures[i] for i in indices]
            for j,inner in enumerate(fold['inner']):
                tr=np.array(inner['train_idx']);val=np.array(inner['val_idx'])
                if set(tr)&set(val) or set(tr)|set(val)!=set(fold['dev_idx']) or (set(tr)|set(val))&set(fold['test_idx']):raise ValueError('Invalid material split')
                sub=next(v['splits'] for v in base_splits if v['fold']==f and v['inner_fold']==j)
                if len(sub)!=3 or any(set(q['train_idx'])&set(q['val_idx']) or set(q['train_idx'])|set(q['val_idx'])!=set(tr) for q in sub):raise ValueError('Base split mismatch')
                b=next(v['homogeneous'] for v in bases if v['fold']==f and v['inner_fold']==j)
                gram=build_kernel(data.X[tr],bandwidth=b['bandwidth'],fraction=b['fraction'],device=device)
                shuffled,perm=permute_kernel(gram,cfg['shuffle_seed']+3*f+j)
                permutations.append({'fold':f,'inner':j,'sha256':hashlib.sha256(perm.cpu().numpy().astype('<i8').tobytes()).hexdigest()})
                relations={'shuffled':make_relation(shuffled,data.Y[tr],eta=base['eta'],kappa=base['kappa'],device=device),
                           'curve':make_relation_curve(gram,data.Y[tr],eta=base['eta'],kappa=base['kappa'],device=device)}
                predictions={}
                for mode in ('shuffled','curve'):
                    for seed in settings['seeds']:
                        pred,detail=train_joint(data.X[tr],data.Y[tr],data.X[val],data.Y[val],configs,seed,indices,cfg['strength'],settings,relations[mode],device,curve=mode=='curve')
                        predictions[f'{mode}_s{seed}']=pred;records.append({'fold':f,'inner':j,'mode':mode,'seed':seed,**detail});state['completed']+=1
                np.savez_compressed(args.out/f'predictions_{f}_{j}.npz',**predictions)
                write('status.json',state);write('trajectories.json',records);write('permutations.json',permutations)
        if state['completed']!=state['expected']:raise ValueError('Incomplete controls')
        per=[]
        for fold in folds:
            f=fold['fold'];indices=next(v['selected'] for v in chosen if v['fold']==f);scores={k:[] for k in ('center','shuffled','curve','baseline','fusion')}
            for j,inner in enumerate(fold['inner']):
                new=np.load(args.out/f'predictions_{f}_{j}.npz');original_pred=np.load(original/f'predictions_{f}_{j}.npz');old=np.load(baseline/f'predictions_{f}_{j}.npz');h=np.load(ref/f'references_{f}_{j}.npz')['huber']
                orig=np.column_stack([np.mean([old[f'e{e}_c{indices[e]}_s{seed}'] for seed in settings['seeds']],axis=0) for e in range(4)])
                preds={'baseline':orig,'fusion':.5*(orig+h),'center':np.mean([original_pred[f'l{cfg["strength"]}_s{seed}'] for seed in settings['seeds']],axis=0)}
                for mode in ('shuffled','curve'):preds[mode]=np.mean([new[f'{mode}_s{seed}'] for seed in settings['seeds']],axis=0)
                for key,pred in preds.items():
                    vals=[[fun(data.Y[inner['val_idx'],e],pred[:,e]) for fun in (pcc,rmse,mae)] for e in range(4)]
                    if not np.isfinite(vals).all():raise ValueError('Nonfinite metric')
                    scores[key].append(vals)
            per.append({'fold':f,'metrics':{k:np.mean(v,axis=0).tolist() for k,v in scores.items()}})
        macro={k:dict(zip(('pcc','rmse','mae'),np.mean([r['metrics'][k] for r in per],axis=(0,1)).tolist())) for k in per[0]['metrics']}
        prior=json.loads((original/'summary.json').read_text())['macro'][str(cfg['strength'])]
        if any(abs(macro['center'][key]-prior[key])>1e-12 for key in ('pcc','rmse','mae')):raise ValueError('Original center metric mismatch')
        write('summary.json',{'macro':macro,'folds':per});state['status']='complete'
    except BaseException as exc:
        state['status']='failed';failures.append({'completed':state['completed'],'error':repr(exc)});raise
    finally:
        write('status.json',state);write('failures.json',failures);write('trajectories.json',records);write('timing.json',{'seconds':time.perf_counter()-start})
