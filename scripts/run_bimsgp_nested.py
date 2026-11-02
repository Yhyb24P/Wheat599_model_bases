#!/usr/bin/env python
"""BiMSGP: development-only inner PCC selection, then frozen outer refit."""
import argparse
import csv
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'src'))
from wheat599.data import load, sha256_of
from wheat599.metrics import pcc, rmse, mae
from run_background_ridge import write_csv


def train(X,y,train,valid,c,cfg,device,schedule=None):
    from bimsgp.model import BiMSGP
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    xmean=X[train].mean(0); xstd=X[train].std(0); xstd[xstd==0]=1
    ym=y[train].mean(); ys=y[train].std()
    if ys==0: raise ValueError('Constant training target')
    xt=torch.tensor((X[train]-xmean)/xstd,dtype=torch.float32,device=device).unsqueeze(-1)
    yt=torch.tensor((y[train]-ym)/ys,dtype=torch.float32,device=device)
    xv=None if valid is None else torch.tensor((X[valid]-xmean)/xstd,dtype=torch.float32,device=device).unsqueeze(-1)
    model=BiMSGP(n_markers=X.shape[1],d_model=c['d_model'],d_state=c['d_state'],dropout=c['dropout']).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=c['lr'],weight_decay=cfg['weight_decay'])
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',patience=cfg['patience'])
    amp=device.type=='cuda' and cfg['precision']=='float16'
    scaler=torch.amp.GradScaler('cuda',enabled=amp)
    curves=[]; lrs=[]
    for epoch in range(cfg['epochs'] if schedule is None else len(schedule)):
        if schedule is not None:
            for group in opt.param_groups: group['lr']=schedule[epoch]
        lrs.append(opt.param_groups[0]['lr']); model.train()
        for ids in torch.randperm(len(train),device=device).split(cfg['batch_size']):
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=amp):
                loss=torch.nn.functional.mse_loss(model(xt[ids]),yt[ids])
            if not torch.isfinite(loss): raise ValueError('Non-finite training loss')
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            scaler.step(opt); scaler.update()
        if valid is not None:
            pred=predict(model,xv,cfg,device)*ys+ym
            score=pcc(y[valid],pred)
            if not np.isfinite(score): raise ValueError('Non-finite validation PCC')
            curves.append(score); scheduler.step(rmse(y[valid],pred))
    return model,(xmean,xstd,ym,ys),curves,lrs


@torch.no_grad()
def predict(model,x,cfg,device):
    model.eval(); out=[]
    for batch in x.split(cfg['batch_size']):
        with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda' and cfg['precision']=='float16'):
            out.append(model(batch).float().cpu().numpy())
    return np.concatenate(out).ravel()


def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,default=ROOT/'configs/bimsgp_nested.yaml'); p.add_argument('--out',type=Path,required=True); p.add_argument('--selection',type=Path)
    args=p.parse_args(); cfg=yaml.safe_load(args.config.read_text()); source=Path(cfg['source'])
    commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    if commit!=cfg['source_commit'] or subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True).strip(): raise ValueError('BiMSGP source differs from pinned clean revision')
    sys.dont_write_bytecode=True; sys.path[:0]=[str(source),cfg['addons']]
    from bimsgp.model import BiMSGP
    folds=json.loads(Path(cfg['splits']).read_text())['folds']; data=load(Path(cfg['data']),cfg['data_sha256'])
    frozen=json.loads(args.selection.read_text()) if args.selection else None
    if frozen is not None and (frozen['config']!=cfg or frozen['splits_sha256']!=sha256_of(Path(cfg['splits']))): raise ValueError('Selection mismatch')
    if frozen is not None:
        keys=[(v['environment'],v['fold']) for v in frozen['selected']]
        expected={(e,f['fold']) for e in data.environments for f in folds}
        if len(keys)!=len(expected) or set(keys)!=expected: raise ValueError('Frozen selection incomplete or duplicated')
    args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'provenance.json').write_text(json.dumps({'config':cfg,'source_commit':commit,'splits_sha256':sha256_of(Path(cfg['splits'])),'runner_sha256':sha256_of(Path(__file__)),'selection_sha256':sha256_of(args.selection) if args.selection else None,'rule':'3-fold mean PCC selects candidate and epoch; refit uses per-epoch median inner LR; test after frozen selection','dependencies':'mamba-ssm 2.3.2.post1, causal-conv1d 1.6.2.post1'},indent=2))
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type!='cuda': raise RuntimeError('Official Mamba selective scan requires CUDA')
    torch.cuda.set_per_process_memory_fraction(0.25)
    selections=[]; histories=[]; rows=[]; oof=[]; failures=[]
    for fold in folds:
        test=np.array(fold['test_idx'],dtype=int); dev=np.sort(np.r_[fold['train_idx'],fold['val_idx']]); inner=[(dev[a],dev[b]) for a,b in KFold(n_splits=cfg['inner_folds'],shuffle=True,random_state=cfg['seed']+fold['fold']).split(dev)]
        if len(set(test)&set(dev)) or len(set(test)|set(dev))!=data.n: raise ValueError('Invalid outer split')
        for j,env in enumerate(data.environments):
            start=time.time(); y=data.Y[:,j]
            try:
                if frozen is None:
                    candidate_scores=[]; schedules=[]
                    for i,c in enumerate(cfg['candidates']):
                        all_curves=[]; all_lrs=[]
                        for k,(tr,va) in enumerate(inner):
                            model,_,curve,lrs=train(data.X,y,tr,va,c,cfg,device); del model
                            all_curves.append(curve); all_lrs.append(lrs)
                            histories.extend({'environment':env,'fold':fold['fold'],'candidate':i,'inner_fold':k,'epoch':ep+1,'pcc':v,'lr':lrs[ep]} for ep,v in enumerate(curve))
                        avg=np.mean(all_curves,axis=0); epoch=int(np.argmax(avg)); candidate_scores.append((float(avg[epoch]),epoch+1)); schedules.append(np.median(all_lrs,axis=0).tolist())
                    winner=max(range(len(candidate_scores)),key=lambda i:candidate_scores[i][0]); score,epochs=candidate_scores[winner]
                    selections.append({'environment':env,'fold':fold['fold'],'candidate':winner,'epochs':epochs,'inner_pcc':score,'lr_schedule':schedules[winner][:epochs]})
                    (args.out/'selection.json').write_text(json.dumps({'config':cfg,'splits_sha256':sha256_of(Path(cfg['splits'])),'selected':selections},indent=2))
                    print(f"fold={fold['fold']} env={env} selected in {time.time()-start:.1f}s",flush=True)
                else:
                    items=[v for v in frozen['selected'] if v['environment']==env and v['fold']==fold['fold']]
                    if len(items)!=1: raise ValueError('Missing frozen selection')
                    s=items[0]; model,scales,_,_=train(data.X,y,dev,None,cfg['candidates'][s['candidate']],cfg,device,schedule=s['lr_schedule']); xm,xs,ym,ys=scales
                    x=torch.tensor((data.X[test]-xm)/xs,dtype=torch.float32,device=device).unsqueeze(-1)
                    pred=predict(model,x,cfg,device)*ys+ym; del model
                    metrics={'pcc':pcc(y[test],pred),'rmse':rmse(y[test],pred),'mae':mae(y[test],pred)}
                    if not all(np.isfinite(v) for v in metrics.values()): raise ValueError('Non-finite test metric')
                    rows.append({'environment':env,'fold':fold['fold'],**metrics,'candidate':s['candidate'],'epochs':s['epochs'],'inner_pcc':s['inner_pcc'],'seconds':round(time.time()-start,2),'failed':False,'reason':''})
                    oof.extend({'environment':env,'fold':fold['fold'],'material_index':int(idx),'y_true':float(y[idx]),'y_pred':float(v)} for idx,v in zip(test,pred))
                    print(f"fold={fold['fold']} env={env} evaluated",flush=True)
            except Exception as exc:
                failures.append({'environment':env,'fold':fold['fold'],'reason':repr(exc)}); print(f'FAILED {env} {fold["fold"]}: {exc}',flush=True)
                if isinstance(exc,torch.cuda.OutOfMemoryError): raise
            finally:
                write_csv(args.out/'validation.csv',['environment','fold','candidate','inner_fold','epoch','pcc','lr'],histories)
                write_csv(args.out/'per_fold.csv',['environment','fold','pcc','rmse','mae','candidate','epochs','inner_pcc','seconds','failed','reason'],rows)
                write_csv(args.out/'oof.csv',['environment','fold','material_index','y_true','y_pred'],oof)
                write_csv(args.out/'failures.csv',['environment','fold','reason'],failures)
    if failures or (len(selections)!=40 if frozen is None else len(rows)!=40): raise RuntimeError('Incomplete run; inspect failures')
    if frozen is None: print('Development selection complete; no test predictions made.',flush=True); return
    summary={'n_runs':len(rows),'n_failures':0,'per_environment':{e:{m:{'mean':float(np.mean([r[m] for r in rows if r['environment']==e])),'sd':float(np.std([r[m] for r in rows if r['environment']==e],ddof=1))} for m in ('pcc','rmse','mae')} for e in data.environments}}
    summary['macro']={m:float(np.mean([summary['per_environment'][e][m]['mean'] for e in data.environments])) for m in ('pcc','rmse','mae')}
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary),flush=True)


if __name__=='__main__': main()
