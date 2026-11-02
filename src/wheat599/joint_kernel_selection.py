"""Joint development-CV tuning of input and task parameters for GET and ICM."""
import gzip
import csv
import itertools
import json
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
from wheat599.data import load,sha256_of
from wheat599.kernel_baselines import kernels
from wheat599.get_icm import prepare,spectral_predict,matrices,write
from wheat599.metrics import pcc,rmse,mae


def run(cfg,args,runner):
    data=load(Path(cfg['data']),cfg['data_sha256']);folds=json.loads(Path(cfg['splits']).read_text())['folds'];sh=sha256_of(Path(cfg['splits']))
    candidates=list(itertools.product(cfg['bandwidths'],cfg['additive_fractions'],cfg['ridge'],cfg['betas'],cfg['etas']))
    candidate_ids={c:i for i,c in enumerate(candidates)}
    frozen=json.loads(args.selection.read_text())if args.selection else None
    if frozen and (frozen['config']!=cfg or frozen['splits_sha256']!=sh or len(frozen['selected'])!=10 or {v['fold']for v in frozen['selected']}!={f['fold']for f in folds}):raise ValueError('Frozen protocol mismatch')
    args.out.mkdir(exist_ok=False,parents=True)
    (args.out/'provenance.json').write_text(json.dumps({'config':cfg,'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'splits_sha256':sh,'selection_sha256':sha256_of(args.selection)if args.selection else None,'code_sha256':sha256_of(Path(__file__)),'solver_sha256':sha256_of(Path(__file__).parent/'get_icm.py'),'runner_sha256':sha256_of(runner),'candidate_count':len(candidates),'tie_rule':'beta, eta, configured base order within 1e-12'},indent=2))
    devic=torch.device('cuda'if torch.cuda.is_available()else'cpu')
    if devic.type=='cuda':torch.cuda.set_per_process_memory_fraction(.35)
    rows=[];oof=[];sel=[];fail=[];cal=[];start=time.perf_counter()
    for f in folds:
        k=f['fold'];dev=np.array(f['dev_idx']);test=np.array(f['test_idx']);print(f'fold {k}',flush=True)
        try:
            if frozen is None:
                scores=np.empty((len(candidates),3,4))
                for j,inner in enumerate(f['inner']):
                    tr=np.array(inner['train_idx']);va=np.array(inner['val_idx'])
                    for h in cfg['bandwidths']:
                        a,ac,r,rc=kernels(data.X,tr,va,h,devic)
                        for frac in cfg['additive_fractions']:
                            K=frac*a+(1-frac)*r;cross=frac*ac+(1-frac)*rc;eig,u=torch.linalg.eigh((K+K.T)/2);cp=cross@u
                            for lam in cfg['ridge']:
                                stats=prepare(K,data.Y[tr],lam,cfg,(eig,u))
                                for beta in cfg['betas']:
                                    for eta in cfg['etas']:
                                        i=candidate_ids[(h,frac,lam,beta,eta)];pred,_=spectral_predict(stats,cross,lam,eta,beta,cp)
                                        scores[i,j]=[pcc(data.Y[va,e],pred[:,e])for e in range(4)]
                    print(f'  inner {j} complete',flush=True)
                if not np.isfinite(scores).all():raise ValueError('Nonfinite validation score')
                with gzip.open(args.out/f'validation_fold_{k}.csv.gz','wt',newline='')as fh:
                    writer=csv.writer(fh);writer.writerow(['candidate','bandwidth','fraction','ridge','beta','eta','inner_fold','environment','pcc'])
                    for i,c in enumerate(candidates):
                        for j in range(3):
                            for e,env in enumerate(data.environments):writer.writerow([i,*c,j,env,float(scores[i,j,e])])
                means=scores.mean((1,2));chosen={}
                for model in ('joint_get','joint_icm'):
                    ids=[i for i,c in enumerate(candidates)if model=='joint_get'or c[3]==0];best=max(means[ids]);eligible=[i for i in ids if best-means[i]<=cfg['tie_tolerance']];i=min(eligible,key=lambda i:(candidates[i][3],candidates[i][4],i));chosen[model]={'candidate':list(candidates[i]),'inner_pcc':float(means[i])}
                sel.append({'fold':k,**chosen});(args.out/'selection.json').write_text(json.dumps({'config':cfg,'splits_sha256':sh,'selected':sel},indent=2))
            else:
                chosen=next(v for v in frozen['selected']if v['fold']==k)
                for model in ('joint_get','joint_icm'):
                    c=tuple(chosen[model]['candidate'])
                    if c not in candidates or(model=='joint_icm'and c[3]!=0):raise ValueError('Invalid choice')
                    h,frac,lam,beta,eta=c;a,ac,r,rc=kernels(data.X,dev,test,h,devic);K=frac*a+(1-frac)*r;cross=frac*ac+(1-frac)*rc;stats=prepare(K,data.Y[dev],lam,cfg);pred,intercept=spectral_predict(stats,cross,lam,eta,beta);bg,sg,d=matrices(stats,eta,beta)
                    cal.append({'model':model,'fold':k,'q':stats['q'].cpu().tolist(),'B_g':bg.cpu().tolist(),'Sigma':sg.cpu().tolist(),'intercept':intercept.tolist()})
                    for e,env in enumerate(data.environments):
                        m={'pcc':pcc(data.Y[test,e],pred[:,e]),'rmse':rmse(data.Y[test,e],pred[:,e]),'mae':mae(data.Y[test,e],pred[:,e])}
                        if not all(np.isfinite(v)for v in m.values()):raise ValueError('Nonfinite test score')
                        rows.append({'model':model,'environment':env,'fold':k,**m,'bandwidth':h,'fraction':frac,'ridge':lam,'beta':beta,'eta':eta,'inner_pcc':chosen[model]['inner_pcc']})
                        oof.extend({'model':model,'environment':env,'fold':k,'material_index':int(ix),'y_true':float(data.Y[ix,e]),'y_pred':float(v)}for ix,v in zip(test,pred[:,e]))
        except Exception as exc:
            fail.append({'fold':k,'reason':repr(exc)});print(f'FAILED {k}: {exc}',flush=True)
            if isinstance(exc,torch.cuda.OutOfMemoryError):raise
        finally:
            write(args.out/'per_fold.csv',['model','environment','fold','pcc','rmse','mae','bandwidth','fraction','ridge','beta','eta','inner_pcc'],rows);write(args.out/'oof.csv',['model','environment','fold','material_index','y_true','y_pred'],oof);write(args.out/'failures.csv',['fold','reason'],fail);(args.out/'calibration.json').write_text(json.dumps(cal,indent=2))
    if fail or(len(sel)!=10 if frozen is None else len(rows)!=80):raise RuntimeError('Incomplete run')
    summary={'n_runs':len(rows),'n_failures':0,'elapsed_seconds':time.perf_counter()-start,'models':{}}
    if frozen:
        for model in ('joint_get','joint_icm'):
            per={e:{m:{'mean':float(np.mean([r[m]for r in rows if r['model']==model and r['environment']==e])),'sd':float(np.std([r[m]for r in rows if r['model']==model and r['environment']==e],ddof=1))}for m in ('pcc','rmse','mae')}for e in data.environments};summary['models'][model]={'per_environment':per,'macro':{m:float(np.mean([per[e][m]['mean']for e in data.environments]))for m in ('pcc','rmse','mae')}}
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
