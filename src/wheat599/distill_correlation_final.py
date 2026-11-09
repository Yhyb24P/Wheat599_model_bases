"""Frozen outer evaluation of a single block-correlation distillation rule."""
import concurrent.futures
import csv
import multiprocessing
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
import yaml
from wheat599 import distill_quantile as dq
from wheat599.data import load, sha256_of
from wheat599.quantile_tabm_final import summarize, read_oof, read_original_oof


def manifest_rows(folds, selections, trajectories, original_fits, seeds):
    index={(r['fold'],r['inner'],r['environment'],r['seed'],r['strength']):r for r in trajectories if r['kind']=='student_finetune'}
    if len(index)!=960: raise ValueError('Incomplete development selection records')
    originals={(r['fold'],r['environment_index'],r['seed']):r for r in original_fits}
    rows=[]
    for f in folds:
        choice=next(r['selected_strengths'] for r in selections if r['fold']==f['fold'])
        if set(f['dev_idx']) & set(f['test_idx']): raise ValueError('Overlapping outer split')
        for e in range(4):
            for seed in seeds:
                old=originals[f['fold'],e,seed];strength=choice[e]
                selected=[index[f['fold'],j,e,seed,strength] for j in range(3)]
                if any(r['config_index']!=old['config_index'] for r in selected):raise ValueError('Original configuration changed')
                epochs=[r['best_epoch'] for r in selected]
                if any(n<0 or n>100 for n in epochs):raise ValueError('Invalid selected epoch')
                rows.append(dict(fold=f['fold'],environment=e,seed=seed,config_index=old['config_index'],
                    initialization_epochs=old['epochs'],strength=strength,inner_best_epochs=epochs,
                    finetuning_epochs=sorted(epochs)[1],fit_ids=f['dev_idx'],test_ids=f['test_idx']))
    if len(rows)!=80 or len({(r['fold'],r['environment'],r['seed']) for r in rows})!=80:raise ValueError('Incomplete outer manifest')
    return rows


def fixed_finetune(model, x_fit, y_fit, teacher, blocks, strength, config, seed, index, cfg, epochs, device):
    # No evaluation inputs or labels enter fixed-epoch training.
    torch.manual_seed(int(seed)+900000)
    x=torch.as_tensor(x_fit,dtype=torch.float32,device=device)
    y=model.standardize_y(torch.as_tensor(y_fit[:,None],dtype=torch.float32,device=device))
    t=model.standardize_y(torch.as_tensor(teacher[:,None],dtype=torch.float32,device=device))
    q=torch.linspace(cfg['quantile_min'],cfg['quantile_max'],cfg['n_quantiles'],device=device)
    opt=torch.optim.AdamW(model.parameters(),lr=config['learning_rate']*cfg['finetuning']['learning_rate_multiplier'],weight_decay=config['weight_decay'])
    generator=torch.Generator().manual_seed(int(seed)*1000+index+900000)
    history=[]
    for epoch in range(epochs):
        pin,kd=dq.student_update_epoch(model,x,y,t,q,strength,opt,generator,cfg,device,blocks)
        history.append(dict(epoch=epoch+1,pinball=pin,auxiliary_times_q=kd))
    return history


def outer_unit(request):
    x,y,query,fold,rows,cfg,configs,kernel,output=request
    torch.set_num_threads(cfg['execution']['cpu_threads_per_worker']);device=torch.device(cfg['device'])
    if device.type=='cuda':torch.cuda.set_per_process_memory_fraction(cfg['gpu_memory_fraction']/cfg['execution']['workers'])
    unit=Path(output)/f'fold_{fold}';unit.mkdir();records=[]
    def record(kind,detail):
        records.append(dict(kind=kind,fold=fold,**detail));dq.write(unit/'execution_status.json',dict(status='running',completed=len(records)))
    ids=np.asarray(rows[0]['fit_ids'])
    teacher=dq.crossfit_teacher(x,y,ids,configs,cfg,kernel,cfg['split_seed']+fold*10,device,record,unit)
    parts=dq.read(unit/'teacher_partitions.json')
    blocks=[np.flatnonzero(np.isin(ids,p['heldout_ids'])) for p in parts]
    result={};original=np.load(cfg['original_seed_predictions'])
    for r in rows:
        e,seed,c=r['environment'],r['seed'],r['config_index'];config=configs[c]
        model=dq.fixed_fit(x,y[:,e],config,seed,c,r['initialization_epochs'],cfg,device)
        warm=dq.predict(model,query,device);key=f'f{fold}_e{e}_s{seed}'
        difference=float(np.max(abs(warm-original[key])))
        if difference>cfg['warm_prediction_tolerance']:raise ValueError(f'Original outer prediction mismatch {difference}')
        record('student_initialization',dict(environment=e,seed=seed,max_difference=difference))
        history=fixed_finetune(model,x,y[:,e],teacher[:,e],blocks,r['strength'],config,seed,c,cfg,r['finetuning_epochs'],device)
        prediction=dq.predict(model,query,device)
        if not np.isfinite(prediction).all():raise ValueError('Invalid final prediction')
        path=unit/f'e{e}_s{seed}.pt';torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},path)
        result[key]=prediction
        dq.write(unit/f'e{e}_s{seed}_training.json',history)
        record('student_finetune',dict(environment=e,seed=seed,epochs=r['finetuning_epochs'],strength=r['strength'],weight_sha256=sha256_of(path)))
    np.savez_compressed(unit/'seed_predictions.npz',**result);dq.write(unit/'trajectories.json',records)
    dq.write(unit/'execution_status.json',dict(status='complete',completed=len(records)))
    return records,result


def run(cfg,args,runner):
    manifest=dq.read(cfg['manifest'])
    for name,digest in manifest['sha256'].items():
        if sha256_of(Path(name))!=digest:raise ValueError(f'Frozen source changed: {name}')
    settings=yaml.safe_load(Path(cfg['development_config']).read_text());settings.update({k:cfg[k] for k in ('execution','original_seed_predictions')})
    settings['execution']={k:v for k,v in settings['execution'].items() if k!='resume_source'}
    folds=dq.read(cfg['splits'])['folds'];data=load(Path(cfg['data']),cfg['data_sha256'])
    configs=yaml.safe_load(Path(settings['architecture_config']).read_text())['configurations'];kernel=yaml.safe_load(Path(settings['huber_config']).read_text())
    source=Path(cfg['development_source'])
    expected_rows=manifest_rows(folds,dq.read(source/'summary.json')['folds'],dq.read(source/'trajectories.json'),dq.read(cfg['original_fits']),settings['seeds'])
    if expected_rows!=manifest['rows']:raise ValueError('Frozen selection mismatch')
    args.out.mkdir(parents=True,exist_ok=False);start=time.perf_counter();records=[];predictions={};failures=[]
    expected=dict(teacher_selection=5760,teacher_refit=240,huber_selection=2160,huber_refit=30,student_initialization=80,student_finetune=80)
    state=dict(status='running',completed={k:0 for k in expected},expected=expected)
    dq.write(args.out/'provenance.json',dict(config=cfg,manifest_sha256=sha256_of(Path(cfg['manifest'])),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),torch_version=str(torch.__version__)))
    dq.write(args.out/'execution_status.json',state)
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=cfg['execution']['workers'],mp_context=multiprocessing.get_context('spawn')) as pool:
            pending=[]
            for f in folds:
                ids=f['dev_idx'];query=f['test_idx'];rows=[r for r in manifest['rows'] if r['fold']==f['fold']]
                # Slice all training labels before dispatch; query consists only of X.
                pending.append(pool.submit(outer_unit,(data.X[ids],data.Y[ids],data.X[query],f['fold'],rows,settings,configs,kernel,str(args.out))))
            for future in concurrent.futures.as_completed(pending):
                try:done,bank=future.result()
                except BaseException:
                    for other in pending:other.cancel()
                    raise
                records.extend(done);predictions.update(bank)
                for r in done:state['completed'][r['kind']]+=1
                dq.write(args.out/'trajectories.json',records);dq.write(args.out/'execution_status.json',state)
        if state['completed']!=expected:raise ValueError('Incomplete frozen budget')
        np.savez_compressed(args.out/'seed_predictions.npz',**predictions)
        # Labels from outer test are used only after all fitted predictions are fixed.
        result,oof=summarize(data,folds,predictions,settings['seeds'],read_oof(cfg['huber_oof']),read_oof(cfg['mse_oof']),read_original_oof(cfg['original_oof']),candidate_fusion=False)
        result['candidate']='quantile denotes standalone block-correlation distilled student; no inference fusion'
        dq.write(args.out/'summary.json',result)
        with (args.out/'oof.csv').open('w') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(oof[0]));writer.writeheader();writer.writerows(oof)
        state['status']='complete'
    except BaseException as exc:
        state['status']='failed';failures.append(dict(error=repr(exc)));raise
    finally:
        dq.write(args.out/'execution_status.json',state);dq.write(args.out/'failures.json',failures)
        dq.write(args.out/'trajectories.json',records);dq.write(args.out/'timing.json',dict(seconds=time.perf_counter()-start))
