"""Frozen outer refits for fusion-selected original and mean-supervised TabM."""
import csv
import json
import time
from pathlib import Path
import numpy as np
import torch
import yaml
from wheat599.data import load, sha256_of
from wheat599.metrics import pcc,rmse,mae
from wheat599.quantile_tabm import build_model,quantile_loss,update_epoch
from wheat599.mean_quantile import mean_loss
from wheat599.quantile_tabm_final import read_oof,summarize


def fit(x,y,xt,row,c,s,device,weights):
    model=build_model(x,y,c,row['seed'],s,device)
    x=torch.as_tensor(x,dtype=torch.float32,device=device);xt=torch.as_tensor(xt,dtype=torch.float32,device=device)
    y=model.standardize_y(torch.as_tensor(y[:,None],dtype=torch.float32,device=device))
    q=torch.linspace(s['quantile_min'],s['quantile_max'],s['n_quantiles'],device=device)
    opt=torch.optim.AdamW(model.parameters(),lr=c['learning_rate'],weight_decay=c['weight_decay'])
    gen=torch.Generator().manual_seed(row['seed']*1000+row['config_index']);spec=s['training']
    for _ in range(row['epochs']):
        if row['arm']=='original':update_epoch(model,x,y,q,opt,gen,spec,device);continue
        model.train();order=torch.randperm(len(x),generator=gen).to(device)
        for batch in order.split(spec['batch_size']):
            opt.zero_grad(set_to_none=True);out=model(x[batch]);loss=quantile_loss(out,y[batch],q)+mean_loss(out,y[batch],s['eta'])
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),spec['gradient_clip']);opt.step()
    model.eval()
    with torch.no_grad():pred=model.inverse_y(model(xt).mean(dim=(1,2))[:,None])[:,0].cpu().numpy()
    if not np.isfinite(pred).all():raise ValueError('Nonfinite prediction')
    path=weights/f"{row['arm']}_{row['fold']}_{row['environment_index']}_{row['seed']}.pt"
    torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},path)
    return pred,{'arm':row['arm'],'fold':row['fold'],'environment_index':row['environment_index'],
                 'seed':row['seed'],'epochs':row['epochs'],'config_index':row['config_index'],
                 'weight_path':str(path),'weight_sha256':sha256_of(path)}


def run(cfg,args,runner):
    if args.selection is not None:raise ValueError('Frozen manifest only')
    path=Path(cfg['manifest'])
    if sha256_of(path)!=cfg['manifest_sha256']:raise ValueError('Manifest mismatch')
    manifest=json.loads(path.read_text())
    for p,h in manifest['source_sha256'].items():
        if sha256_of(Path(p))!=h:raise ValueError(f'Source mismatch: {p}')
    if manifest['fusion_weight']!=.5 or len(manifest['rows'])!=160:raise ValueError('Invalid manifest')
    folds=json.loads(Path(cfg['splits']).read_text())['folds'];byfold={f['fold']:f for f in folds}
    seen=set()
    for row in manifest['rows']:
        key=(row['arm'],row['fold'],row['environment_index'],row['seed']);f=byfold[row['fold']]
        if key in seen or row['arm'] not in ('original','mean') or row['seed'] not in (42,2026) or row['environment_index'] not in range(4):raise ValueError('Invalid row')
        if row['fit_ids']!=f['dev_idx'] or row['test_ids']!=f['test_idx'] or set(row['fit_ids'])&set(row['test_ids']):raise ValueError('Invalid split')
        if row['epochs']!=sorted(row['inner_best_epochs'])[1]:raise ValueError('Invalid epoch')
        seen.add(key)
    if len(seen)!=160:raise ValueError('Incomplete coverage')
    settings={a:yaml.safe_load(Path(p).read_text()) for a,p in [('original','configs/quantile_tabm.yaml'),('mean','configs/mean_quantile.yaml')]}
    configs=yaml.safe_load(Path('configs/environment_tabm.yaml').read_text())['configurations']
    torch.set_num_threads(cfg['cpu_threads']);device=torch.device(cfg['device'])
    if device.type=='cuda':torch.cuda.set_per_process_memory_fraction(cfg['gpu_memory_fraction'])
    data=load(Path(cfg['data']),cfg['data_sha256']);args.out.mkdir(parents=True,exist_ok=False);weights=args.out/'weights';weights.mkdir()
    def write(name,obj):(args.out/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
    state={'status':'running','completed':0,'expected':160};predictions={'original':{},'mean':{}};records=[];failures=[];started=time.perf_counter()
    try:
        for row in manifest['rows']:
            e=row['environment_index'];tr=row['fit_ids'];te=row['test_ids'];arm=row['arm']
            pred,info=fit(data.X[tr],data.Y[tr,e],data.X[te],row,configs[row['config_index']],settings[arm],device,weights)
            predictions[arm][f"f{row['fold']}_e{e}_s{row['seed']}"]=pred;records.append(info);state['completed']+=1
            write('execution_status.json',state);write('fits.json',records)
        if state['completed']!=160:raise ValueError('Incomplete fits')
        np.savez_compressed(args.out/'predictions.npz',**{f'{a}_{k}':v for a,d in predictions.items() for k,v in d.items()})
        huber=read_oof('results/robust_icm_final_20261103/oof.csv');mse=read_oof('results/independent_tabm_final_20261105/oof.csv')
        output={};oof={}
        for arm in ('original','mean'):
            output[arm],oof[arm]=summarize(data,folds,predictions[arm],[42,2026],huber,mse)
        baseline={}
        with Path('results/quantile_tabm_final_20261106/oof.csv').open() as handle:
            for r in csv.DictReader(handle):
                key=(int(r['fold']),str(r['environment']),int(r['material_index']));
                if key in baseline:raise ValueError('Duplicate baseline')
                baseline[key]=r
        metrics=[]
        for arm in ('original','mean'):
            per=[]
            for fold in folds:
                for e,name in enumerate(data.environments):
                    rr=[r for r in oof[arm] if r['fold']==fold['fold'] and r['environment']==name];truth=np.array([r['y_true'] for r in rr]);keys=[(r['fold'],r['environment'],r['material_index']) for r in rr]
                    if any(k not in baseline for k in keys) or not np.allclose(truth,[float(baseline[k]['y_true']) for k in keys],atol=1e-12,rtol=0):raise ValueError('Baseline mismatch')
                    old=np.array([float(baseline[k]['quantile_huber']) for k in keys]);new=np.array([r['quantile_huber'] for r in rr]);per.append({'fold':fold['fold'],'environment':name,'baseline_pcc':pcc(truth,old),'candidate_pcc':pcc(truth,new)})
            metrics.append({'arm':arm,'fold_environment':per})
        if len(baseline)!=2396:raise ValueError('Baseline incomplete')
        write('summary.json',{'systems':output,'paired':metrics,'baseline_summary':json.loads(Path('results/quantile_tabm_final_20261106/summary.json').read_text())['macro']})
        state['status']='complete'
    except BaseException as exc:
        state['status']='failed';failures.append({'completed':state['completed'],'error':repr(exc)});raise
    finally:
        write('execution_status.json',state);write('fits.json',records);write('failures.json',failures);write('timing.json',{'seconds':time.perf_counter()-started})
