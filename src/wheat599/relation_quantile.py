"""Full-fit relationship penalty on centers of independent quantile TabMs."""
import copy
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from wheat599.data import load, sha256_of
from wheat599.metrics import pcc, rmse, mae
from wheat599.quantile_tabm import build_model, quantile_loss


def train_joint(x_fit, y_fit, x_val, y_val, configs, seed, indices, strength, settings, relation, device):
    """Fit four independent models with one shared full-fit relation penalty."""
    models=[];optimizers=[];rng=[];generators=[]
    x=torch.as_tensor(x_fit,dtype=torch.float32,device=device)
    xv=torch.as_tensor(x_val,dtype=torch.float32,device=device)
    targets=[]
    for e,config in enumerate(configs):
        model=build_model(x_fit,y_fit[:,e],config,seed,settings,device)
        models.append(model)
        optimizers.append(torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay']))
        rng.append((torch.get_rng_state(),torch.cuda.get_rng_state(device) if device.type=='cuda' else None))
        generators.append(torch.Generator().manual_seed(int(seed)*1000+int(indices[e])))
        targets.append(model.standardize_y(torch.as_tensor(y_fit[:,e,None],dtype=torch.float32,device=device)))
    levels=torch.linspace(settings['quantile_min'],settings['quantile_max'],settings['n_quantiles'],device=device)
    spec=settings['training'];best=np.full(4,-np.inf);best_epoch=np.full(4,-1);best_prediction=[None]*4
    best_states=[None]*4;active=[True]*4;hist=[]
    for epoch in range(spec['max_epochs']+1):
        if epoch:
            orders=[torch.randperm(len(x),generator=generators[e]).to(device).split(spec['batch_size']) if active[e] else None for e in range(4)]
            for j in range((len(x)+spec['batch_size']-1)//spec['batch_size']):
                losses=[]
                for e in range(4):
                    if not active[e]:continue
                    optimizers[e].zero_grad(set_to_none=True);models[e].train()
                    torch.set_rng_state(rng[e][0])
                    if device.type=='cuda':torch.cuda.set_rng_state(rng[e][1],device)
                    batch=orders[e][j]
                    losses.append(quantile_loss(models[e](x[batch]),targets[e][batch],levels))
                    rng[e]=(torch.get_rng_state(),torch.cuda.get_rng_state(device) if device.type=='cuda' else None)
                for model in models:model.eval()
                centers=torch.stack([model(x).mean(dim=(1,2)) for model in models],dim=1)
                penalty=relation(centers)
                objective=torch.stack(losses).sum()+strength*penalty
                if not torch.isfinite(objective):raise FloatingPointError('Nonfinite relation objective')
                objective.backward()
                for e in range(4):
                    if active[e]:
                        torch.nn.utils.clip_grad_norm_(models[e].parameters(),spec['gradient_clip'])
                        optimizers[e].step()
        for e,model in enumerate(models):
            if not active[e]:continue
            model.eval()
            with torch.no_grad():
                pred=model.inverse_y(model(xv).mean(dim=(1,2))[:,None])[:,0].cpu().numpy()
            score=pcc(y_val[:,e],pred)
            if not np.isfinite(score):raise ValueError('Nonfinite validation PCC')
            if score>best[e]+spec['tie_tolerance']:
                best[e]=score;best_epoch[e]=epoch;best_prediction[e]=pred.copy()
                best_states[e]=copy.deepcopy(model.state_dict())
            elif epoch-best_epoch[e]>=spec['patience']:
                model.load_state_dict(best_states[e]);model.requires_grad_(False);active[e]=False
        if not any(active):break
    if any(p is None for p in best_prediction):raise ValueError('Missing checkpoint')
    return np.column_stack(best_prediction),{'best_pcc':best.tolist(),'best_epoch':best_epoch.tolist(),
                                               'epochs_run':epoch}


def run(cfg,args,runner):
    if args.selection is not None:raise ValueError('Development only')
    from wheat599.relation_regularizer import build_kernel, make_relation
    settings=yaml.safe_load(Path(cfg['quantile_config']).read_text())
    architectures=yaml.safe_load(Path(cfg['architecture_config']).read_text())['configurations']
    device=torch.device(settings['device']);torch.set_num_threads(cfg['cpu_threads'])
    if device.type=='cuda':torch.cuda.set_per_process_memory_fraction(cfg['gpu_memory_fraction'])
    data=load(Path(settings['data']),settings['data_sha256']);folds=json.loads(Path(settings['splits']).read_text())['folds']
    baseline=Path(cfg['baseline_source']);reference=Path(cfg['reference_source'])
    chosen=json.loads((baseline/'summary.json').read_text())['folds'];bases=json.loads(Path(cfg['base_choices']).read_text())
    base_splits=json.loads(Path(cfg['base_splits']).read_text())
    if len(folds)!=10 or any(len(f['inner'])!=3 for f in folds) or len(chosen)!=10:raise ValueError('Unexpected split coverage')
    if json.loads((baseline/'execution_status.json').read_text())!={'status':'complete','completed':1920,'expected':1920}:raise ValueError('Incomplete baseline')
    split_hash=sha256_of(Path(settings['splits']))
    base_provenance=Path(cfg['base_choices']).parent/'provenance.json'
    if json.loads(base_provenance.read_text())['splits_sha256']!=split_hash:raise ValueError('Base split mismatch')
    for source in (baseline,reference):
        if json.loads((source/'provenance.json').read_text())['sha256'][settings['splits']]!=split_hash:raise ValueError('Reference split mismatch')
    if not cfg['lambdas'] or any(not np.isfinite(v) or v<=0 for v in cfg['lambdas']):raise ValueError('Penalty strengths must be positive')
    paths=[Path(args.config),Path(runner),Path(__file__),Path(__file__).with_name('relation_regularizer.py'),Path(settings['splits']),
           Path(cfg['quantile_config']),Path(cfg['architecture_config']),Path(cfg['base_choices']),Path(cfg['base_splits']),base_provenance,baseline/'summary.json',baseline/'provenance.json',reference/'provenance.json']
    paths += [baseline/f"predictions_{f['fold']}_{j}.npz" for f in folds for j in range(3)]
    paths += [reference/f"references_{f['fold']}_{j}.npz" for f in folds for j in range(3)]
    audit=json.loads(Path(cfg['baseline_audit']).read_text())['sha256']
    for path in paths[-60:]:
        if audit[str(path)]!=sha256_of(path):raise ValueError('Reference prediction hash mismatch')
    paths.append(Path(cfg['baseline_audit']))
    args.out.mkdir(parents=True,exist_ok=False)
    def write(name,value):(args.out/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    write('provenance.json',{'config':cfg,'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                              'sha256':{str(p):sha256_of(p) for p in paths}})
    state={'status':'running','completed':0,'expected':len(folds)*3*len(settings['seeds'])*len(cfg['lambdas'])}
    records=[];failures=[];start=time.perf_counter()
    try:
        for fold in folds:
            f=fold['fold'];indices=next(v['selected'] for v in chosen if v['fold']==f)
            configs=[architectures[i] for i in indices]
            for j,inner in enumerate(fold['inner']):
                tr=np.array(inner['train_idx']);val=np.array(inner['val_idx'])
                if set(tr)&set(val) or set(tr)|set(val)!=set(fold['dev_idx']) or (set(tr)|set(val))&set(fold['test_idx']):raise ValueError('Invalid split')
                provenance_splits=next(v['splits'] for v in base_splits if v['fold']==f and v['inner_fold']==j)
                if len(provenance_splits)!=3 or any(set(q['train_idx'])&set(q['val_idx']) or set(q['train_idx'])|set(q['val_idx'])!=set(tr) for q in provenance_splits):raise ValueError('Base training split mismatch')
                base=next(v['homogeneous'] for v in bases if v['fold']==f and v['inner_fold']==j)
                gram=build_kernel(data.X[tr],bandwidth=base['bandwidth'],fraction=base['fraction'],device=device)
                relation=make_relation(gram,data.Y[tr],eta=cfg['eta'],kappa=cfg['kappa'],device=device)
                predictions={}
                for strength in cfg['lambdas']:
                    for seed in settings['seeds']:
                        pred,detail=train_joint(data.X[tr],data.Y[tr],data.X[val],data.Y[val],configs,seed,indices,strength,settings,relation,device)
                        predictions[f'l{strength}_s{seed}']=pred
                        records.append({'fold':f,'inner':j,'lambda':strength,'seed':seed,'config_indices':indices,**detail})
                        state['completed']+=1
                np.savez_compressed(args.out/f'predictions_{f}_{j}.npz',**predictions)
                write('status.json',state);write('trajectories.json',records)
        if state['completed']!=state['expected']:raise ValueError('Incomplete training')
        per=[]
        for fold in folds:
            f=fold['fold'];indices=next(v['selected'] for v in chosen if v['fold']==f)
            scores={str(v):[] for v in [0.0]+cfg['lambdas']};scores['fusion']=[]
            for j,inner in enumerate(fold['inner']):
                old=np.load(baseline/f'predictions_{f}_{j}.npz');new=np.load(args.out/f'predictions_{f}_{j}.npz');ref=np.load(reference/f'references_{f}_{j}.npz')
                orig=np.column_stack([np.mean([old[f'e{e}_c{indices[e]}_s{seed}'] for seed in settings['seeds']],axis=0) for e in range(4)])
                preds={'0.0':orig,'fusion':.5*(orig+ref['huber'])}
                for strength in cfg['lambdas']:preds[str(strength)]=np.mean([new[f'l{strength}_s{seed}'] for seed in settings['seeds']],axis=0)
                for key,pred in preds.items():
                    vals=[[fn(data.Y[inner['val_idx'],e],pred[:,e]) for fn in (pcc,rmse,mae)] for e in range(4)]
                    if not np.isfinite(vals).all():raise ValueError('Nonfinite metric')
                    scores[key].append(vals)
            per.append({'fold':f,'metrics':{k:np.mean(v,axis=0).tolist() for k,v in scores.items()}})
        names=list(per[0]['metrics']);macro={k:dict(zip(('pcc','rmse','mae'),np.mean([r['metrics'][k] for r in per],axis=(0,1)).tolist())) for k in names}
        choices=[str(v) for v in [0.0]+cfg['lambdas']];selection=[]
        for entry in per:
            scores=[float(np.mean(np.asarray(entry['metrics'][k])[:,0])) for k in choices]
            index=int(np.argmax(scores));selection.append({'fold':entry['fold'],'lambda':float(choices[index]),'scores':scores})
        selected=np.mean([entry['metrics'][str(selection[i]['lambda'])] for i,entry in enumerate(per)],axis=(0,1))
        write('selection.json',{'choices':choices,'folds':selection,'split_sha256':sha256_of(Path(settings['splits']))})
        write('summary.json',{'macro':macro,'selected_development':dict(zip(('pcc','rmse','mae'),selected.tolist())),
                              'folds':per});state['status']='complete'
    except BaseException as exc:
        state['status']='failed';failures.append({'completed':state['completed'],'error':repr(exc)});raise
    finally:
        write('status.json',state);write('failures.json',failures);write('trajectories.json',records);write('timing.json',{'seconds':time.perf_counter()-start})
