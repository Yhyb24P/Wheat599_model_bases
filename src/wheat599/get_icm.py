"""Conditional GLS-LOO structured residual ICM, with exact spectral inference."""
import csv
import json
import time
from pathlib import Path
import numpy as np
import torch
from wheat599.data import load, sha256_of
from wheat599.metrics import pcc, rmse, mae
from wheat599.kernel_baselines import kernels
from wheat599.multienv_kernel import choose_base
from wheat599.bh_lmc import conditional_loo


def correlation(y):
    z=y-y.mean(0); norm=torch.linalg.vector_norm(z,dim=0)
    if torch.any(norm<=0) or not torch.isfinite(z).all(): raise ValueError('Undefined task correlation')
    z=z/norm; c=z.T@z; c=(c+c.T)/2
    if torch.linalg.eigvalsh(c).min() < -1e-10: raise ValueError('Non-PSD correlation')
    return c


def prepare(k,Y,lam,cfg,eigensystem=None):
    y=torch.as_tensor(Y,dtype=k.dtype,device=k.device); mean=y.mean(0); scale=y.std(0,unbiased=False)
    if torch.any(scale<=0): raise ValueError('Constant training phenotype')
    y=(y-mean)/scale
    eig,u=torch.linalg.eigh((k+k.T)/2) if eigensystem is None else eigensystem
    if eig.min() < -1e-8: raise ValueError('Non-PSD material kernel')
    eig=eig.clamp_min(0)
    one=torch.ones((len(y),1),dtype=k.dtype,device=k.device); uy=u.T@y; uone=u.T@one; inverse=1/(eig+lam)
    invy=u@(uy*inverse[:,None]); invone=u@(uone*inverse[:,None]); mu=invy.sum(0)/invone.sum()
    diagonal=u.square()@inverse-invone.ravel().square()/invone.sum()
    if torch.any(diagonal<=0): raise ValueError('Nonpositive LOO diagonal')
    err=(invy-invone*mu)/diagonal[:,None]
    denom=(y-mu).square().sum(0)
    if torch.any(denom<=0): raise ValueError('Zero GLS target energy')
    q=(err.square().sum(0)/denom).clamp(cfg['error_floor'],cfg['error_ceiling'])
    return {'Y':y,'mean':mean,'scale':scale,'C_Y':correlation(y),'C_G':correlation(y-err),'R_E':correlation(err),'q':q,'eig':eig,'U':u,'UTY':uy,'UT1':uone}


def matrices(stats,eta,beta):
    q=stats['q']; identity=torch.eye(len(q),dtype=q.dtype,device=q.device)
    d=(q/torch.exp(torch.log(q).mean()))**(beta/2)
    g=(1-eta)*identity+eta*((1-beta)*stats['C_Y']+beta*stats['C_G'])
    r=(1-.5*beta)*identity+.5*beta*stats['R_E']
    bg=d[:,None]*g*d[None,:]; sigma=d[:,None]*r*d[None,:]
    return bg,sigma,d


def spectral_predict(stats,cross,lam,eta,beta,cross_projection=None):
    bg,sigma,d=matrices(stats,eta,beta)
    ev,t=torch.linalg.eigh(sigma)
    if ev.min()<=0: raise ValueError('Nonpositive residual covariance')
    half=(t*torch.sqrt(ev))@t.T; invhalf=(t*torch.rsqrt(ev))@t.T
    s,Q=torch.linalg.eigh(invhalf@bg@invhalf)
    if s.min()<=0: raise ValueError('Nonpositive signal covariance')
    W=invhalf@Q; V=half@Q; u=stats['U']; eig=stats['eig']
    inverse=1/(eig[:,None]*s[None,:]+lam)
    uy=stats['UTY']@W; uone=stats['UT1']
    nu=(uone.T@(uy*inverse)).ravel()/(uone.T@(uone*inverse)).ravel()
    alpha=(uy-uone*nu)*inverse*s
    projected=cross@u if cross_projection is None else cross_projection
    normalized=nu@V.T+projected@alpha@V.T
    pred=normalized*stats['scale']+stats['mean']; intercept=(nu@V.T)*stats['scale']+stats['mean']
    return pred.cpu().numpy(),intercept.cpu().numpy()


def write(path,fields,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows)


def run(cfg,args,runner_path):
    data=load(Path(cfg['data']),cfg['data_sha256']); folds=json.loads(Path(cfg['splits']).read_text())['folds']; split_hash=sha256_of(Path(cfg['splits']))
    frozen=json.loads(args.selection.read_text()) if args.selection else None
    if frozen and (frozen['config']!=cfg or frozen['splits_sha256']!=split_hash): raise ValueError('Frozen protocol mismatch')
    if frozen and (len(frozen['selected'])!=10 or {x['fold'] for x in frozen['selected']}!={f['fold'] for f in folds}): raise ValueError('Incomplete selections')
    args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'provenance.json').write_text(json.dumps({'config':cfg,'git_commit':__import__('subprocess').check_output(['git','rev-parse','HEAD'],text=True).strip(),'splits_sha256':split_hash,'model_sha256':sha256_of(Path(__file__)),'runner_sha256':sha256_of(runner_path),'selection_sha256':sha256_of(args.selection) if args.selection else None,'protocol':'CV1; training-standardized Y; GLS intercept; conditional GLS-LOO'},indent=2))
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type=='cuda':torch.cuda.set_per_process_memory_fraction(.35)
    choices=[(eta,beta) for beta in cfg['betas'] for eta in cfg['etas']]; selected=[]; validation=[]; records=[]; oof=[]; calibration=[]; failures=[]; start=time.perf_counter()
    for f in folds:
        k=f['fold'];dev=np.array(f['dev_idx']);test=np.array(f['test_idx']);print(f'fold {k}',flush=True)
        try:
            if frozen is None:
                scores=np.empty((len(choices),3,4))
                for j,inner in enumerate(f['inner']):
                    tr=np.array(inner['train_idx']);va=np.array(inner['val_idx']);_,base=choose_base(data.X[tr],data.Y[tr],cfg,cfg['seed']+1000+k*10+j,device)
                    a,ac,r,rc=kernels(data.X,tr,va,base[0],device); K=base[2]*a+(1-base[2])*r; cross=base[2]*ac+(1-base[2])*rc; stats=prepare(K,data.Y[tr],base[1],cfg)
                    for i,(eta,beta) in enumerate(choices):
                        pred,_=spectral_predict(stats,cross,base[1],eta,beta)
                        for e,env in enumerate(data.environments):
                            value=pcc(data.Y[va,e],pred[:,e]);scores[i,j,e]=value
                            if not np.isfinite(value):raise ValueError('Nonfinite validation PCC')
                            validation.append({'fold':k,'inner_fold':j,'environment':env,'eta':eta,'beta':beta,'pcc':float(value),'bandwidth':base[0],'ridge':base[1],'fraction':base[2]})
                    print(f'  inner {j} complete',flush=True)
                means=scores.mean((1,2));mx=means.max();idx=next(i for i,v in enumerate(means) if mx-v<=cfg['tie_tolerance']);eta,beta=choices[idx]
                selected.append({'fold':k,'eta':eta,'beta':beta,'inner_pcc':float(means[idx]),'selection_reason':'max macro PCC; tolerance ties prefer beta then eta'})
                (args.out/'selection.json').write_text(json.dumps({'config':cfg,'splits_sha256':split_hash,'selected':selected},indent=2))
            else:
                chosen=next(x for x in frozen['selected'] if x['fold']==k);eta,beta=chosen['eta'],chosen['beta']
                if (eta,beta) not in choices:raise ValueError('Invalid frozen candidate')
                _,base=choose_base(data.X[dev],data.Y[dev],cfg,cfg['seed']+k,device)
                a,ac,r,rc=kernels(data.X,dev,test,base[0],device);K=base[2]*a+(1-base[2])*r;cross=base[2]*ac+(1-base[2])*rc;stats=prepare(K,data.Y[dev],base[1],cfg);pred,intercept=spectral_predict(stats,cross,base[1],eta,beta);bg,sigma,d=matrices(stats,eta,beta)
                calibration.append({'fold':k,'q':stats['q'].cpu().tolist(),'C_Y':stats['C_Y'].cpu().tolist(),'C_G':stats['C_G'].cpu().tolist(),'R_E':stats['R_E'].cpu().tolist(),'D':d.cpu().tolist(),'B_g':bg.cpu().tolist(),'Sigma':sigma.cpu().tolist(),'intercept':intercept.tolist(),'min_signal_eigenvalue':float(torch.linalg.eigvalsh(bg).min()),'min_residual_eigenvalue':float(torch.linalg.eigvalsh(sigma).min())})
                for e,env in enumerate(data.environments):
                    met={'pcc':pcc(data.Y[test,e],pred[:,e]),'rmse':rmse(data.Y[test,e],pred[:,e]),'mae':mae(data.Y[test,e],pred[:,e])}
                    if not all(np.isfinite(v) for v in met.values()):raise ValueError('Nonfinite test metric')
                    records.append({'environment':env,'fold':k,**met,'bandwidth':base[0],'fraction':base[2],'ridge':base[1],'eta':eta,'beta':beta})
                    oof.extend({'environment':env,'fold':k,'material_index':int(idx),'y_true':float(data.Y[idx,e]),'y_pred':float(v)}for idx,v in zip(test,pred[:,e]))
        except Exception as exc:
            failures.append({'fold':k,'reason':repr(exc)});print(f'FAILED {k}: {exc}',flush=True)
            if isinstance(exc,torch.cuda.OutOfMemoryError):raise
        finally:
            write(args.out/'validation.csv',['fold','inner_fold','environment','eta','beta','pcc','bandwidth','ridge','fraction'],validation)
            write(args.out/'per_fold.csv',['environment','fold','pcc','rmse','mae','bandwidth','fraction','ridge','eta','beta'],records)
            write(args.out/'oof.csv',['environment','fold','material_index','y_true','y_pred'],oof)
            write(args.out/'failures.csv',['fold','reason'],failures)
            (args.out/'calibration.json').write_text(json.dumps(calibration,indent=2))
    if failures or (len(selected)!=10 if frozen is None else len(records)!=40):raise RuntimeError('Incomplete run')
    summary={'n_runs':len(records),'failures':failures,'elapsed_seconds':time.perf_counter()-start}
    if frozen:
        per={e:{m:{'mean':float(np.mean([r[m]for r in records if r['environment']==e])),'sd':float(np.std([r[m]for r in records if r['environment']==e],ddof=1))}for m in ('pcc','rmse','mae')}for e in data.environments};summary.update(per_environment=per,macro={m:float(np.mean([per[e][m]['mean']for e in data.environments]))for m in ('pcc','rmse','mae')})
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
