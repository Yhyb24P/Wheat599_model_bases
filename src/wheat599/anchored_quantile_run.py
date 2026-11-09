"""Bounded development comparison with a common configuration/update selection."""
import concurrent.futures
import multiprocessing
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
import yaml
from wheat599 import anchored_quantile as joint
from wheat599.distill_quantile import read,write,fixed_fit,predict
from wheat599.robust_icm import features,fit_predict
from wheat599.data import load,sha256_of
from wheat599.metrics import pcc,rmse,mae


def unit(request):
    x,y,query,f,j,fit_ids,config_ids,epochs,cfg,configs,base,choice,thresholds,kcfg,output=request
    torch.set_num_threads(1);device=torch.device(cfg['device'])
    if device.type=='cuda':torch.cuda.set_per_process_memory_fraction(cfg['gpu_memory_fraction']/cfg['workers'])
    root=Path(output)/f'unit_{f}_{j}';root.mkdir()
    # Query labels are absent; robust features only index fit phenotypes.
    joined_x=np.concatenate((x,query));joined_y=np.concatenate((y,np.full((len(query),4),np.nan)))
    prepared=features(joined_x,joined_y,np.arange(len(x)),np.arange(len(x),len(joined_x)),base)
    hp,diag,hs=fit_predict(prepared,base['ridge'][0]*choice['multiplier'],choice['eta'],choice['delta'],kcfg,fixed_thresholds=thresholds,return_state=True)
    reference=np.load(Path(cfg['huber_reference'])/f'references_{f}_{j}.npz')['huber']
    huber_diff=float(np.max(abs(hp-reference)))
    if huber_diff>1e-8:raise ValueError(f'Original Huber function mismatch: {huber_diff}')
    original=np.load(Path(cfg['baseline_source'])/f'predictions_{f}_{j}.npz')
    settings=dict(cfg,network_configs=[configs[c] for c in config_ids])
    records=[];predictions={}
    for seed in cfg['seeds']:
        models=[];old=[]
        for e,c in enumerate(config_ids):
            m=fixed_fit(x,y[:,e],configs[c],seed,c,epochs[e,seed],cfg,device)
            value=predict(m,query,device);reference=original[f'e{e}_c{c}_s{seed}']
            if np.max(abs(value-reference))>1e-6:raise ValueError('Original neural initialization mismatch')
            models.append(m);old.append(value)
        predictions[f'original_s{seed}']=np.column_stack(old)
        initial=.5*(np.column_stack(old)+hp)
        for c,parameters in enumerate(cfg['regularization']):
            def save(step,point,state,certificate):
                difference=float(np.max(abs(point-initial))) if step==0 else None
                if step==0 and difference>1e-6:raise ValueError('Joint initialization does not preserve fusion')
                key=f'c{c}_s{seed}_t{step}';path=root/f'{key}.pt'
                state.update(fit_ids=fit_ids,base=base,config_ids=config_ids,seed=seed)
                torch.save(state,path)
                predictions[key]=point
                records.append(dict(fold=f,inner=j,config_index=c,seed=seed,step=step,
                    initialization_max_difference=difference,kernel_certificate=certificate,
                    weight_sha256=sha256_of(path)))
                write(root/'execution_status.json',dict(status='running',saved_states=len(records)))
            joint.train(models,x,y,query,prepared,hs,choice['eta'],parameters,settings,save)
    np.savez_compressed(root/'predictions.npz',**predictions)
    write(root/'states.json',records);write(root/'initialization.json',dict(huber_max_difference=huber_diff,huber_optimization=diag))
    write(root/'execution_status.json',dict(status='complete',saved_states=len(records)))
    return records


def summarize(data,folds,cfg,out):
    results=[]
    for fold in folds:
        f=fold['fold'];arrays=[np.load(out/f'unit_{f}_{j}/predictions.npz') for j in range(3)]
        scores=[]
        for c in range(3):
            for t in range(cfg['updates']+1):
                values=[]
                for j,inner in enumerate(fold['inner']):
                    pred=np.mean([arrays[j][f'c{c}_s{seed}_t{t}'] for seed in cfg['seeds']],axis=0)
                    values.extend([pcc(data.Y[inner['val_idx'],e],pred[:,e]) for e in range(4)])
                scores.append(dict(config_index=c,step=t,pcc=float(np.mean(values))))
        best=max(v['pcc'] for v in scores)
        chosen=next(v for v in scores if best-v['pcc']<=cfg['training']['tie_tolerance'])
        metrics={}
        for name,c,t in [('joint',chosen['config_index'],chosen['step']),('initial_fusion',0,0)]:
            values=[]
            for j,inner in enumerate(fold['inner']):
                pred=np.mean([arrays[j][f'c{c}_s{seed}_t{t}'] for seed in cfg['seeds']],axis=0)
                values.append([[fn(data.Y[inner['val_idx'],e],pred[:,e]) for fn in (pcc,rmse,mae)] for e in range(4)])
            metrics[name]=np.mean(values,axis=0).tolist()
        values=[]
        for j,inner in enumerate(fold['inner']):
            pred=np.mean([arrays[j][f'original_s{seed}'] for seed in cfg['seeds']],axis=0)
            values.append([[fn(data.Y[inner['val_idx'],e],pred[:,e]) for fn in (pcc,rmse,mae)] for e in range(4)])
        metrics['original_quantile']=np.mean(values,axis=0).tolist()
        results.append(dict(fold=f,selected=chosen,scores=scores,metrics=metrics))
    macro={name:dict(zip(('pcc','rmse','mae'),np.mean([r['metrics'][name] for r in results],axis=(0,1)).tolist())) for name in ('joint','initial_fusion','original_quantile')}
    return dict(folds=results,macro=macro,rule='One configuration and one update count shared across three inner folds, four environments and two seeds; no per-trajectory checkpoint selection')


def run(cfg,args,runner):
    if args.selection is not None:raise ValueError('Development only')
    if len(cfg['regularization'])!=3 or cfg['seeds']!=[42,2026]:raise ValueError('Unexpected candidate budget')
    data=load(Path(cfg['data']),cfg['data_sha256']);folds=read(cfg['splits'])['folds']
    if len(folds)!=10 or any(len(f['inner'])!=3 for f in folds):raise ValueError('Expected thirty development units')
    configs=yaml.safe_load(Path(cfg['architecture_config']).read_text())['configurations']
    old=Path(cfg['baseline_source']);old_rows=read(old/'trajectories.json');index={(r['fold'],r['inner'],r['environment'],r['config_index'],r['seed']):r['best_epoch'] for r in old_rows}
    selections=read(old/'summary.json')['folds'];hub=Path(cfg['huber_source']);selection=read(hub/'selection.json');diagnostics=read(hub/'optimization.json');bases=read(cfg['base_choices'])
    if read(old/'execution_status.json')['status']!='complete' or read(old/'failures.json'):raise ValueError('Original neural experiment incomplete')
    if read(hub/'provenance.json')['splits_sha256']!=sha256_of(Path(cfg['splits'])):raise ValueError('Historical Huber splits changed')
    args.out.mkdir(parents=True,exist_ok=False)
    paths=[Path(__file__),Path(joint.__file__),Path(__file__).with_name('anchored_quantile_kernel.py'),Path(__file__).with_name('distill_quantile.py'),Path(__file__).with_name('quantile_tabm.py'),Path(__file__).with_name('robust_icm.py'),Path(runner),Path(args.config),Path(cfg['splits']),Path(cfg['architecture_config']),Path(cfg['base_choices']),old/'trajectories.json',old/'summary.json',hub/'selection.json',hub/'optimization.json']
    for f in folds:
        for j in range(3):paths.extend([old/f"predictions_{f['fold']}_{j}.npz",Path(cfg['huber_reference'])/f"references_{f['fold']}_{j}.npz"])
    write(args.out/'provenance.json',dict(config=cfg,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),sha256={str(p):sha256_of(p) for p in paths}))
    start=time.perf_counter();records=[];failures=[];state=dict(status='running',completed=0,expected=180)
    write(args.out/'execution_status.json',state)
    try:
        requests=[]
        for fold in folds:
            f=fold['fold'];choice=next(v['huber'] for v in selection['selected'] if v['fold']==f);cs=next(v['selected'] for v in selections if v['fold']==f)
            for j,inner in enumerate(fold['inner']):
                tr,va=inner['train_idx'],inner['val_idx']
                if set(tr)&set(va) or set(tr)|set(va)!=set(fold['dev_idx']) or (set(tr)|set(va))&set(fold['test_idx']):raise ValueError('Invalid development information boundary')
                base=next(v['homogeneous'] for v in bases if v['fold']==f and v['inner_fold']==j)
                matched=[v for v in diagnostics if v['fold']==f and v['inner_fold']==j and v['model']=='huber' and all(v[k]==choice[k] for k in ('eta','delta','multiplier'))]
                if len(matched)!=1:raise ValueError('Missing initial Huber thresholds')
                epochs={(e,seed):index[f,j,e,c,seed] for e,c in enumerate(cs) for seed in cfg['seeds']}
                requests.append((data.X[tr],data.Y[tr],data.X[va],f,j,tr,cs,epochs,cfg,configs,base,choice,matched[0]['thresholds'],selection['config'],str(args.out)))
        with concurrent.futures.ProcessPoolExecutor(max_workers=cfg['workers'],mp_context=multiprocessing.get_context('spawn')) as pool:
            futures=[pool.submit(unit,r) for r in requests]
            for future in concurrent.futures.as_completed(futures):
                try:saved=future.result()
                except BaseException:
                    for other in futures:other.cancel()
                    raise
                records.extend(saved);state['completed']+=6
                write(args.out/'execution_status.json',state)
        if state['completed']!=180:raise ValueError('Incomplete fixed development budget')
        write(args.out/'summary.json',summarize(data,folds,cfg,args.out));state['status']='complete'
    except BaseException as exc:
        state['status']='failed';failures.append(dict(error=repr(exc)));raise
    finally:
        write(args.out/'execution_status.json',state);write(args.out/'states.json',records)
        write(args.out/'failures.json',failures);write(args.out/'timing.json',dict(seconds=time.perf_counter()-start))
