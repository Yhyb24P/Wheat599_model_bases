"""Conditional-LOO calibrated dual-kernel LMC with GLS intercepts."""
import csv
import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import KFold

from wheat599.data import load, sha256_of
from wheat599.kernel_baselines import kernels
from wheat599.metrics import pcc, rmse, mae
from wheat599.multienv_kernel import choose_base, fit_predict, task_correlation


def conditional_loo(k, y, ridge):
    """Fixed-kernel GLS leave-one-out errors, refitting intercept per removal."""
    n = k.shape[0]
    yy = torch.as_tensor(y, dtype=k.dtype, device=k.device)
    chol = torch.linalg.cholesky(k + ridge * torch.eye(n, dtype=k.dtype, device=k.device))
    inv = torch.cholesky_inverse(chol)
    ones = torch.ones((n, 1), dtype=k.dtype, device=k.device)
    u = inv @ ones
    projection = inv - (u @ u.T) / (ones.T @ u)
    diag = torch.diagonal(projection)
    if torch.any(diag <= 0):
        raise ValueError('Nonpositive LOO projection diagonal')
    err = (projection @ yy) / diag[:, None]
    if not torch.isfinite(err).all():
        raise ValueError('Non-finite LOO residual')
    return err.cpu().numpy()


def predicted_corr(Y, P, floor):
    yc, pc = Y-Y.mean(0), P-P.mean(0)
    ys, ps = np.linalg.norm(yc,axis=0), np.linalg.norm(pc,axis=0)
    if np.any(ys <= 0): raise ValueError('Constant training phenotype')
    good = ps >= floor * ys
    z = pc / np.where(good, ps, 1.0)
    z[:, ~good] = 0
    corr = z.T @ z
    np.fill_diagonal(corr, 1.0)
    reliability = np.zeros(Y.shape[1])
    reliability[good] = np.clip(np.sum(yc[:,good]*pc[:,good],axis=0)/(ys[good]*ps[good]),0,1)
    return corr, reliability


def calibration(a, r, Y, base, cfg):
    _, lam, frac = base
    ea = conditional_loo(a, Y, lam)
    er = conditional_loo(r, Y, lam)
    e0 = conditional_loo(frac*a+(1-frac)*r, Y, lam)
    ca, ra = predicted_corr(Y, Y-ea, cfg['correlation_floor'])
    cr, rr = predicted_corr(Y, Y-er, cfg['correlation_floor'])
    sst = ((Y-Y.mean(0))**2).sum(0)
    if np.any(sst <= 0): raise ValueError('Constant training target')
    q = np.clip(1-(e0**2).sum(0)/sst,cfg['reliability_floor'],cfg['reliability_ceiling'])
    eps=cfg['weight_epsilon']
    pi=(ra**2+eps)/(ra**2+rr**2+2*eps)
    return {'q':q,'pi':pi,'C_A':ca,'C_R':cr,'C_Y':task_correlation(Y),'r_A':ra,'r_R':rr}


def task_parameters(Y, base, eta_a, eta_r, gamma, cfg, stats=None):
    """Use GLS-compatible calibration; gamma zero bypasses LOO entirely."""
    _, lam, frac=base
    cy=task_correlation(Y); identity=np.eye(Y.shape[1])
    if gamma==0:
        weights=np.full(Y.shape[1],frac); penalties=np.full(Y.shape[1],lam); ca=cr=cy
    else:
        if stats is None: raise ValueError('Missing training-only calibration')
        weights=(1-gamma)*frac+gamma*stats['pi']
        nu=(1-stats['q'])/stats['q']; penalties=lam*(nu/np.exp(np.log(nu).mean()))**gamma
        ca=(1-.5*gamma)*cy+.5*gamma*stats['C_A']
        cr=(1-.5*gamma)*cy+.5*gamma*stats['C_R']
    root_a=np.sqrt(weights); root_r=np.sqrt(1-weights)
    ba=root_a[:,None]*((1-eta_a)*identity+eta_a*ca)*root_a[None,:]
    br=root_r[:,None]*((1-eta_r)*identity+eta_r*cr)*root_r[None,:]
    return ba,br,penalties,weights,ca,cr


def select_best(choices, means, tol):
    if not np.isfinite(means).all(): raise ValueError('Non-finite candidate score')
    best=float(np.max(means))
    eligible=[i for i,score in enumerate(means) if best-score<=tol]
    return min(eligible,key=lambda i:(choices[i][2],abs(choices[i][0]-choices[i][1]),choices[i][0]+choices[i][1],choices[i]))


def write(path,fields,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n'); writer.writeheader(); writer.writerows(rows)


def calibration_row(phase,fold,inner,base,gamma,stats,weights,penalties,ca,cr):
    return {'phase':phase,'fold':fold,'inner_fold':inner,'bandwidth':base[0],'ridge':base[1],'base_fraction':base[2],'gamma':gamma,'q':json.dumps(stats['q'].tolist()),'pi':json.dumps(stats['pi'].tolist()),'w':json.dumps(weights.tolist()),'lambda':json.dumps(penalties.tolist()),'C_Y':json.dumps(stats['C_Y'].tolist()),'C_A':json.dumps(ca.tolist()),'C_R':json.dumps(cr.tolist()),'r_A':json.dumps(stats['r_A'].tolist()),'r_R':json.dumps(stats['r_R'].tolist())}


def run(cfg,args,runner_path):
    data=load(Path(cfg['data']),cfg['data_sha256'])
    folds=json.loads(Path(cfg['splits']).read_text())['folds']
    if len(folds)!=10: raise ValueError('Expected ten outer folds')
    split_hash=sha256_of(Path(cfg['splits']))
    frozen=json.loads(args.selection.read_text()) if args.selection else None
    if frozen:
        if frozen['config']!=cfg or frozen['splits_sha256']!=split_hash: raise ValueError('Frozen configuration mismatch')
        keys=[v['fold'] for v in frozen['selected']]
        if len(keys)!=len(folds) or set(keys)!={f['fold'] for f in folds}: raise ValueError('Incomplete frozen selection')
    args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'provenance.json').write_text(json.dumps({'config':cfg,'splits_sha256':split_hash,'runner_sha256':sha256_of(runner_path),'model_sha256':sha256_of(Path(__file__)),'multienv_sha256':sha256_of(Path(__file__).parent/'multienv_kernel.py'),'baseline_sha256':sha256_of(Path(__file__).parent/'kernel_baselines.py'),'selection_sha256':sha256_of(args.selection) if args.selection else None,'intercept':'GLS, training-standardized Y; conditional LOO uses GLS projection','protocol':'CV1; nested macro PCC; full development refit'},indent=2))
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type=='cuda': torch.cuda.set_per_process_memory_fraction(0.35)
    choices=list(itertools.product(cfg['etas'],cfg['etas'],cfg['gammas']))
    icm_ids=[i for i,(a,r,g) in enumerate(choices) if g==0 and a==r]
    selected,validation,calibrations,rows,oof,failures=[],[],[],[],[],[]
    start=time.perf_counter()
    for fold in folds:
        k=fold['fold']; test=np.asarray(fold['test_idx']); dev=np.sort(np.r_[fold['train_idx'],fold['val_idx']])
        if len(set(dev)&set(test)) or len(set(dev)|set(test))!=data.n: raise ValueError('Invalid outer split')
        print(f'fold {k}',flush=True); fold_start=time.perf_counter()
        try:
            if frozen is None:
                scores=np.empty((len(choices),cfg['inner_folds'],len(data.environments)))
                inner=KFold(n_splits=cfg['inner_folds'],shuffle=True,random_state=cfg['seed']+k)
                for j,(itr,iva) in enumerate(inner.split(dev)):
                    tr,va=dev[itr],dev[iva]; _,base=choose_base(data.X[tr],data.Y[tr],cfg,cfg['seed']+1000+k*10+j,device)
                    a,ac,r,rc=kernels(data.X,tr,va,base[0],device); Y=data.Y[tr]
                    stats=calibration(a,r,Y,base,cfg)
                    for gamma in cfg['gammas']:
                        _,_,pen,w,ca,cr=task_parameters(Y,base,0,0,gamma,cfg,stats)
                        calibrations.append(calibration_row('inner',k,j,base,gamma,stats,w,pen,ca,cr))
                    for i,(eta_a,eta_r,gamma) in enumerate(choices):
                        ba,br,pen,_,_,_=task_parameters(Y,base,eta_a,eta_r,gamma,cfg,stats if gamma else None)
                        pred=fit_predict([a,r],[ac,rc],Y,[ba,br],pen)
                        for e,env in enumerate(data.environments):
                            score=pcc(data.Y[va,e],pred[:,e])
                            if not np.isfinite(score): raise ValueError('Non-finite validation PCC')
                            scores[i,j,e]=score
                            validation.append({'fold':k,'inner_fold':j,'environment':env,'eta_a':eta_a,'eta_r':eta_r,'gamma':gamma,'pcc':float(score)})
                    print(f'  inner {j} complete',flush=True)
                means=scores.mean(axis=(1,2)); best=select_best(choices,means,cfg['tie_tolerance'])
                sub=select_best([choices[i] for i in icm_ids],means[icm_ids],cfg['tie_tolerance']); matched=icm_ids[sub]
                selected.append({'fold':k,'bh':list(choices[best]),'bh_inner_pcc':float(means[best]),'matched_icm':list(choices[matched]),'matched_icm_inner_pcc':float(means[matched])})
                (args.out/'selection.json').write_text(json.dumps({'config':cfg,'splits_sha256':split_hash,'selected':selected},indent=2))
            else:
                entry=next(v for v in frozen['selected'] if v['fold']==k)
                _,base=choose_base(data.X[dev],data.Y[dev],cfg,cfg['seed']+k,device)
                a,ac,r,rc=kernels(data.X,dev,test,base[0],device); Y=data.Y[dev]
                stats=calibration(a,r,Y,base,cfg)
                for model,key in [('bh_lmc','bh'),('matched_icm','matched_icm')]:
                    ea,er,gamma=entry[key]
                    if (ea,er,gamma) not in choices or (model=='matched_icm' and (gamma!=0 or ea!=er)): raise ValueError('Invalid frozen choice')
                    ba,br,pen,w,ca,cr=task_parameters(Y,base,ea,er,gamma,cfg,stats if gamma else None)
                    calibrations.append(calibration_row(model,k,-1,base,gamma,stats,w,pen,ca,cr))
                    pred=fit_predict([a,r],[ac,rc],Y,[ba,br],pen)
                    for e,env in enumerate(data.environments):
                        metrics={'pcc':pcc(data.Y[test,e],pred[:,e]),'rmse':rmse(data.Y[test,e],pred[:,e]),'mae':mae(data.Y[test,e],pred[:,e])}
                        if not all(np.isfinite(v) for v in metrics.values()): raise ValueError('Non-finite test metric')
                        rows.append({'model':model,'environment':env,'fold':k,**metrics,'bandwidth':base[0],'ridge':base[1],'base_fraction':base[2],'eta_a':ea,'eta_r':er,'gamma':gamma,'inner_pcc':entry[key+'_inner_pcc'],'seconds':time.perf_counter()-fold_start})
                        oof.extend({'model':model,'environment':env,'fold':k,'material_index':int(idx),'y_true':float(data.Y[idx,e]),'y_pred':float(v)} for idx,v in zip(test,pred[:,e]))
        except Exception as exc:
            failures.append({'fold':k,'reason':repr(exc)}); print(f'FAILED {k}: {exc}',flush=True)
            if isinstance(exc,torch.cuda.OutOfMemoryError): raise
        finally:
            write(args.out/'validation.csv',['fold','inner_fold','environment','eta_a','eta_r','gamma','pcc'],validation)
            write(args.out/'calibration.csv',['phase','fold','inner_fold','bandwidth','ridge','base_fraction','gamma','q','pi','w','lambda','C_Y','C_A','C_R','r_A','r_R'],calibrations)
            write(args.out/'per_fold.csv',['model','environment','fold','pcc','rmse','mae','bandwidth','ridge','base_fraction','eta_a','eta_r','gamma','inner_pcc','seconds'],rows)
            write(args.out/'oof.csv',['model','environment','fold','material_index','y_true','y_pred'],oof)
            write(args.out/'failures.csv',['fold','reason'],failures)
    if failures or (len(selected)!=10 if frozen is None else len(rows)!=80): raise RuntimeError('Incomplete run; inspect failures')
    if frozen is None:
        print('Development selection complete; no test predictions made.',flush=True); return
    summary={'n_runs':len(rows),'n_failures':0,'elapsed_seconds':time.perf_counter()-start,'models':{}}
    for model in ('bh_lmc','matched_icm'):
        per={e:{m:{'mean':float(np.mean([v[m] for v in rows if v['model']==model and v['environment']==e])),'sd':float(np.std([v[m] for v in rows if v['model']==model and v['environment']==e],ddof=1))} for m in ('pcc','rmse','mae')} for e in data.environments}
        summary['models'][model]={'per_environment':per,'macro':{m:float(np.mean([per[e][m]['mean'] for e in data.environments])) for m in ('pcc','rmse','mae')}}
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary),flush=True)
