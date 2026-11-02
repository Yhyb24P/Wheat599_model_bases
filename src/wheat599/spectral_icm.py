"""Spectrally selective multi-environment kernel ridge for CV1."""
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import KFold

from wheat599.data import load, sha256_of
from wheat599.kernel_baselines import kernels
from wheat599.metrics import pcc, rmse, mae
from wheat599.multienv_kernel import choose_base, fit_predict, task_correlation


def split_spectrum(k, cross, multiplier):
    """Training-only filter, applied consistently to the cross kernel."""
    if multiplier == 0:
        return k, torch.zeros_like(k), cross, torch.zeros_like(cross)
    eig, u = torch.linalg.eigh((k + k.T) / 2)
    if eig.min().item() < -1e-8:
        raise ValueError('Base kernel is not PSD')
    eig = eig.clamp_min(0)
    tau = multiplier * torch.diagonal(k).mean()
    if tau <= 0:
        raise ValueError('Nonpositive spectral scale')
    ratio = eig / (eig + tau)
    ks = (u * (eig * ratio)) @ u.T
    kp = (u * (eig * (1-ratio))) @ u.T
    projected = cross @ u
    return ks, kp, (projected * ratio) @ u.T, (projected * (1-ratio)) @ u.T


def predict_spectral(mats, y, param, eta, multiplier):
    a, ac, r, rc = mats
    _, ridge, frac = param
    k = frac*a + (1-frac)*r
    cross = frac*ac + (1-frac)*rc
    ks, kp, cs, cp = split_spectrum(k, cross, multiplier)
    q = y.shape[1]
    identity = np.eye(q)
    task = (1-eta)*identity + eta*task_correlation(y)
    return fit_predict([ks,kp], [cs,cp], y, [task,identity], np.full(q,ridge))


def write(path, fields, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)


def run(cfg, args, runner_path):
    data=load(Path(cfg['data']),cfg['data_sha256'])
    folds=json.loads(Path(cfg['splits']).read_text())['folds']
    split_hash=sha256_of(Path(cfg['splits']))
    frozen=json.loads(args.selection.read_text()) if args.selection else None
    if frozen:
        if frozen['config'] != cfg or frozen['splits_sha256'] != split_hash:
            raise ValueError('Frozen selection configuration or split mismatch')
        keys=[v['fold'] for v in frozen['selected']]
        if len(keys)!=len(folds) or set(keys)!={v['fold'] for v in folds}:
            raise ValueError('Missing or duplicate frozen selections')
    args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'provenance.json').write_text(json.dumps({'config':cfg,'splits_sha256':split_hash,'model_sha256':sha256_of(Path(__file__)),'runner_sha256':sha256_of(runner_path),'multienv_sha256':sha256_of(Path(__file__).parent/'multienv_kernel.py'),'baseline_sha256':sha256_of(Path(__file__).parent/'kernel_baselines.py'),'selection_sha256':sha256_of(args.selection) if args.selection else None,'protocol':'CV1; inner macro PCC selection, full development refit'},indent=2))
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type=='cuda': torch.cuda.set_per_process_memory_fraction(0.35)
    selected,validation,rows,oof,failures=[],[],[],[],[]
    # ICM tie is preferred before a nonzero spectral scale.
    choices=[(tau,eta) for tau in cfg['tau_multipliers'] for eta in cfg['sharing']]
    for fold in folds:
        k=fold['fold']; test=np.asarray(fold['test_idx']); dev=np.sort(np.r_[fold['train_idx'],fold['val_idx']])
        if len(set(dev)&set(test)) or len(set(dev)|set(test))!=data.n: raise ValueError('Invalid outer split')
        print(f'fold {k}',flush=True)
        try:
            if frozen is None:
                scores=np.empty((len(choices),cfg['inner_folds'],len(data.environments)))
                inner=KFold(n_splits=cfg['inner_folds'],shuffle=True,random_state=cfg['seed']+k)
                for j,(itr,iva) in enumerate(inner.split(dev)):
                    tr,va=dev[itr],dev[iva]
                    _,common=choose_base(data.X[tr],data.Y[tr],cfg,cfg['seed']+1000+k*10+j,device)
                    mats=kernels(data.X,tr,va,common[0],device)
                    for i,(tau,eta) in enumerate(choices):
                        pred=predict_spectral(mats,data.Y[tr],common,eta,tau)
                        for e,env in enumerate(data.environments):
                            score=pcc(data.Y[va,e],pred[:,e])
                            if not np.isfinite(score): raise ValueError('Nonfinite inner PCC')
                            scores[i,j,e]=score
                            validation.append({'fold':k,'inner_fold':j,'environment':env,'tau_multiplier':tau,'eta':eta,'pcc':float(score)})
                    print(f'  inner {j} complete',flush=True)
                means=scores.mean(axis=(1,2)); winner=int(np.argmax(means)); tau,eta=choices[winner]
                selected.append({'fold':k,'tau_multiplier':tau,'eta':eta,'inner_pcc':float(means[winner])})
                (args.out/'selection.json').write_text(json.dumps({'config':cfg,'splits_sha256':split_hash,'selected':selected},indent=2))
            else:
                entry=next(v for v in frozen['selected'] if v['fold']==k)
                tau,eta=entry['tau_multiplier'],entry['eta']
                if (tau,eta) not in choices: raise ValueError('Invalid frozen choice')
                _,common=choose_base(data.X[dev],data.Y[dev],cfg,cfg['seed']+k,device)
                mats=kernels(data.X,dev,test,common[0],device)
                pred=predict_spectral(mats,data.Y[dev],common,eta,tau)
                for e,env in enumerate(data.environments):
                    metrics={'pcc':pcc(data.Y[test,e],pred[:,e]),'rmse':rmse(data.Y[test,e],pred[:,e]),'mae':mae(data.Y[test,e],pred[:,e])}
                    if not all(np.isfinite(v) for v in metrics.values()): raise ValueError('Nonfinite test metric')
                    rows.append({'environment':env,'fold':k,**metrics,'tau_multiplier':tau,'eta':eta,'bandwidth':common[0],'ridge':common[1],'additive_fraction':common[2],'inner_pcc':entry['inner_pcc']})
                    oof.extend({'environment':env,'fold':k,'material_index':int(idx),'y_true':float(data.Y[idx,e]),'y_pred':float(v)} for idx,v in zip(test,pred[:,e]))
        except Exception as exc:
            failures.append({'fold':k,'reason':repr(exc)}); print(f'FAILED {k}: {exc}',flush=True)
            if isinstance(exc,torch.cuda.OutOfMemoryError): raise
        finally:
            write(args.out/'validation.csv',['fold','inner_fold','environment','tau_multiplier','eta','pcc'],validation)
            write(args.out/'per_fold.csv',['environment','fold','pcc','rmse','mae','tau_multiplier','eta','bandwidth','ridge','additive_fraction','inner_pcc'],rows)
            write(args.out/'oof.csv',['environment','fold','material_index','y_true','y_pred'],oof)
            write(args.out/'failures.csv',['fold','reason'],failures)
    if failures or (len(selected)!=10 if frozen is None else len(rows)!=40): raise RuntimeError('Incomplete run')
    if frozen is None:
        print('Development selection complete; no test predictions made.',flush=True); return
    per={e:{m:float(np.mean([r[m] for r in rows if r['environment']==e])) for m in ('pcc','rmse','mae')} for e in data.environments}
    summary={'n_runs':len(rows),'n_failures':0,'per_environment':per,'macro':{m:float(np.mean([per[e][m] for e in data.environments])) for m in ('pcc','rmse','mae')}}
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary),flush=True)
