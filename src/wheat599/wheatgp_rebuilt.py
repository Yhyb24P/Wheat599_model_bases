"""Paired WheatGP and fixed TabM–Huber under a rebuilt 10-fold protocol."""
import csv
import importlib.util
import itertools
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.model_selection import KFold

from wheat599.data import load, sha256_of
from wheat599.heterogeneous_icm import choose
from wheat599.metrics import pcc, rmse, mae
from wheat599.quantile_tabm import train as train_quantile
from wheat599.quantile_tabm_final import fit_one as refit_quantile
from wheat599.robust_icm import features, fit_predict
from wheat599.wheatgp_nested import fit as fit_wheatgp, prepare, validation_pcc


def splits(n, seed, validation_seed):
    result=[]
    for fold,(dev,test) in enumerate(KFold(10,shuffle=True,random_state=seed).split(np.arange(n))):
        # Mirrors the public train.py random_split generator and 90/10 sizes.
        permutation=torch.randperm(len(dev),generator=torch.Generator().manual_seed(validation_seed)).numpy()
        length=int(.9*len(dev));tr=dev[permutation[:length]];val=dev[permutation[length:]]
        if set(tr)&set(val) or set(tr)|set(val)!=set(dev) or set(dev)&set(test): raise ValueError('Invalid material split')
        result.append({'fold':fold,'dev_idx':dev.tolist(),'train_idx':tr.tolist(),'val_idx':val.tolist(),'test_idx':test.tolist()})
    if sorted(i for f in result for i in f['test_idx'])!=list(range(n)): raise ValueError('Incomplete outer coverage')
    return result


def huber_base(x,y,indices,cfg,seed,device):
    inner=list(KFold(3,shuffle=True,random_state=seed).split(indices))
    _,base,_=choose(x[indices],y[indices],inner,cfg,device)
    return base


def run(cfg,args,runner):
    w=yaml.safe_load(Path(cfg['wheatgp_config']).read_text()); q=yaml.safe_load(Path(cfg['quantile_config']).read_text())
    configs=yaml.safe_load(Path(cfg['architecture_config']).read_text())['configurations']; h=yaml.safe_load(Path(cfg['huber_config']).read_text())
    if sha256_of(Path(w['source']))!=w['source_sha256']: raise ValueError('WheatGP source mismatch')
    spec=importlib.util.spec_from_file_location('wheatgp_official',w['source']);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    torch.set_num_threads(cfg['cpu_threads']);device=torch.device(cfg['device'])
    if device.type=='cuda': torch.cuda.set_per_process_memory_fraction(cfg['gpu_memory_fraction'])
    data=load(Path(cfg['data']),cfg['data_sha256']); folds=splits(len(data.X),cfg['outer_seed'],cfg['validation_seed'])
    paths=[Path(x) for x in (args.config,cfg['wheatgp_config'],cfg['quantile_config'],cfg['architecture_config'],cfg['huber_config'],w['source'],__file__,runner)]
    hashes={str(p):sha256_of(p) for p in paths}
    selection=json.loads(args.selection.read_text()) if args.selection else None
    if selection is not None and (selection['hashes']!=hashes or selection['splits']!=folds or len(selection['folds'])!=10): raise ValueError('Frozen protocol mismatch')
    args.out.mkdir(parents=True,exist_ok=False)
    write=lambda name,obj:(args.out/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
    write('provenance.json',{'config':cfg,'hashes':hashes,'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()})
    write('splits.json',folds); start=time.perf_counter(); records=[]; failures=[]; selections=[]; predictions={}
    status={'status':'running','phase':'outer' if selection else 'development','folds_completed':0,'expected_folds':10}
    candidates=[(kind,lr) for kind in ('raw','pca') for lr in w['learning_rates']]
    hchoices=list(itertools.product(h['etas'],h['huber_deltas'],h['ridge_multipliers']))
    try:
        if selection is None:
            for fold in folds:
                f=fold['fold'];tr=np.array(fold['train_idx']);val=np.array(fold['val_idx'])
                base=huber_base(data.X,data.Y,tr,h,cfg['inner_base_seed']+f,device)
                prepared=features(data.X,data.Y,tr,val,base); hscores=[]
                for eta,delta,mult in hchoices:
                    pred,_=fit_predict(prepared,base['ridge'][0]*mult,eta,delta,h)
                    score=float(np.mean([pcc(data.Y[val,e],pred[:,e]) for e in range(4)]))
                    if not np.isfinite(score): raise ValueError('Invalid Huber validation')
                    hscores.append(score)
                hi=int(np.argmax(hscores)); hentry={'eta':hchoices[hi][0],'delta':hchoices[hi][1],'multiplier':hchoices[hi][2],'scores':hscores}
                prepared_w={k:prepare(data.X,tr,val,k,w) for k in ('raw','pca')};ws=[];qs=[]
                for e in range(4):
                    wc=[]
                    for c,(kind,lr) in enumerate(candidates):
                        xf,xv=prepared_w[kind];pred,best,epochs=fit_wheatgp(module.wheatGP_base,xf,data.Y[tr,e],xv,data.Y[val,e],lr,w['max_epochs'],w,device)
                        wc.append({'candidate':c,'kind':kind,'learning_rate':lr,'best_epoch':best,'epochs_run':epochs,'pcc':validation_pcc(data.Y[val,e],pred)})
                    ws.append(max(wc,key=lambda z:z['pcc'])); records.extend({'fold':f,'environment':e,'model':'wheatgp',**v} for v in wc)
                    qc=[]
                    for c,config in enumerate(configs):
                        preds=[]; info=[]
                        for seed in q['seeds']:
                            pred,detail=train_quantile(data.X[tr],data.Y[tr,e],data.X[val],data.Y[val,e],config,seed,c,q,device)
                            preds.append(pred);info.append(detail)
                        score=pcc(data.Y[val,e],np.mean(preds,axis=0))
                        if not np.isfinite(score): raise ValueError('Invalid TabM validation')
                        qc.append({'candidate':c,'pcc':float(score),'seeds':dict(zip(map(str,q['seeds']),info))})
                    qs.append(max(qc,key=lambda z:z['pcc']));records.extend({'fold':f,'environment':e,'model':'quantile',**v} for v in qc)
                selections.append({'fold':f,'huber':hentry,'wheatgp':ws,'quantile':qs})
                status['folds_completed']+=1;write('status.json',status);write('trajectories.json',records)
            write('selection.json',{'hashes':hashes,'splits':folds,'folds':selections})
        else:
            weights=args.out/'weights';weights.mkdir()
            for fold in folds:
                f=fold['fold']; dev=np.array(fold['dev_idx']);test=np.array(fold['test_idx']);chosen=selection['folds'][f]
                if chosen['fold']!=f: raise ValueError('Selection fold mismatch')
                base=huber_base(data.X,data.Y,dev,h,cfg['inner_base_seed']+f,device);prepared=features(data.X,data.Y,dev,test,base);hc=chosen['huber']
                hp,_=fit_predict(prepared,base['ridge'][0]*hc['multiplier'],hc['eta'],hc['delta'],h)
                modes=set(z['kind'] for z in chosen['wheatgp']);pw={k:prepare(data.X,dev,test,k,w) for k in modes}
                wg=[];qt=[]
                for e in range(4):
                    wc=chosen['wheatgp'][e];xf,xt=pw[wc['kind']]
                    wp,_,_=fit_wheatgp(module.wheatGP_base,xf,data.Y[dev,e],xt,None,wc['learning_rate'],wc['best_epoch'],w,device,select=False);wg.append(wp)
                    qc=chosen['quantile'][e];c=qc['candidate'];seedpred=[]
                    for seed in q['seeds']:
                        row={'fold':f,'environment_index':e,'seed':seed,'config_index':c,'epochs':qc['seeds'][str(seed)]['best_epoch']}
                        pred,detail=refit_quantile(data.X[dev],data.Y[dev,e],data.X[test],row,configs[c],q,device,weights);seedpred.append(pred)
                        records.append({'fold':f,'environment':e,'model':'quantile_refit',**detail})
                    qt.append(np.mean(seedpred,axis=0))
                wg=np.column_stack(wg);qt=np.column_stack(qt);fusion=.5*(qt+hp)
                np.savez_compressed(args.out/f'predictions_{f}.npz',wheatgp=wg,quantile=qt,huber=hp,fusion=fusion)
                predictions[f]=dict(wheatgp=wg,quantile=qt,huber=hp,fusion=fusion)
                status['folds_completed']+=1;write('status.json',status);write('trajectories.json',records)
            rows=[]; metrics=[]
            for fold in folds:
                f=fold['fold'];test=fold['test_idx'];truth=data.Y[test]
                for e,name in enumerate(data.environments):
                    methods={}
                    for method,pred in predictions[f].items():
                        pp=pred[:,e];ms=[fun(truth[:,e],pp) for fun in (pcc,rmse,mae)]
                        if not np.isfinite(ms).all():raise ValueError('Nonfinite outer metric')
                        methods[method]=dict(zip(('pcc','rmse','mae'),ms))
                    metrics.append({'fold':f,'environment':name,'methods':methods})
                    rows.extend({'fold':f,'environment':name,'material_index':int(i),'y_true':float(truth[j,e]),**{method:float(p[j,e]) for method,p in predictions[f].items()}} for j,i in enumerate(test))
            if len(rows)!=len(data.X)*4:raise ValueError('OOF coverage mismatch')
            with (args.out/'oof.csv').open('w',newline='') as out:
                writer=csv.DictWriter(out,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            write('summary.json',{'per_fold':metrics,'macro':{m:{k:float(np.mean([v['methods'][m][k] for v in metrics])) for k in ('pcc','rmse','mae')} for m in ('wheatgp','quantile','huber','fusion')}})
        if status['folds_completed']!=10:raise ValueError('Incomplete experiment')
        status['status']='complete'
    except BaseException as exc:
        status['status']='failed';failures.append({'error':repr(exc),'folds_completed':status['folds_completed']});raise
    finally:
        write('status.json',status);write('failures.json',failures);write('timing.json',{'seconds':time.perf_counter()-start})
