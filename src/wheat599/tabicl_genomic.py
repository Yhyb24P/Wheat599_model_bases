"""Matched high-dimensional TabICLv2 prior adaptation on development splits."""
import importlib.metadata
import json
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
from wheat599.data import load,sha256_of
from wheat599.genetic_prior import task
from wheat599.metrics import pcc,rmse,mae


def adapt(x_fit,arm,cfg,seed,checkpoint,device):
    from tabicl._model.tabicl import TabICL
    from tabicl.prior import PriorDataset
    from tabicl.prior.graph_lib._config import PriorConfig
    np.random.seed(seed);torch.manual_seed(seed);rng=np.random.default_rng(seed)
    model_cfg=checkpoint['config'].copy();model_cfg['recompute']=True
    model=TabICL(**model_cfg).to(device);model.load_state_dict(checkpoint['state_dict']);model.train()
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    n=cfg['context']+cfg['query']
    prior=None
    if arm=='generic':
        prior=PriorDataset(regression=True,batch_size=1,batch_size_per_gp=1,
            min_features=cfg['features']-1,max_features=cfg['features'],min_seq_len=None,max_seq_len=n,
            min_train_size=cfg['context'],max_train_size=cfg['context']+1,prior_type='graph_scm',
            config=PriorConfig(),n_jobs=1,device='cpu')
    losses=[]
    for step in range(cfg['steps']):
        if arm=='generic':
            xx,yy,_,seq,train_size=prior.get_batch()
            if int(seq[0])!=n or int(train_size[0])!=cfg['context'] or xx.shape[-1]<cfg['features']-1:
                raise ValueError('Generic prior shape mismatch')
            xx=xx.to(device);yy=yy.to(device)
        else:
            xx,yy,ids=task(x_fit,rng,context=cfg['context'],query=cfg['query'])
            if len(np.unique(ids))!=n:raise ValueError('Synthetic material overlap')
            xx=torch.as_tensor(xx[None],device=device);yy=torch.as_tensor(yy[None],device=device)
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=device.type=='cuda'):
            pred=model(xx,yy[:,:cfg['context']]);levels=torch.linspace(0,1,pred.shape[-1]+2,device=device)[1:-1]
            error=yy[:,cfg['context']:,None]-pred
            loss=torch.maximum(levels*error,(levels-1)*error).mean()
        if not torch.isfinite(loss):raise ValueError(f'Nonfinite prior loss at step {step}')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'])
        if not torch.isfinite(norm):raise ValueError(f'Nonfinite gradient at step {step}')
        opt.step();losses.append(float(loss.detach()))
    state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    del model,opt
    if device.type=='cuda':torch.cuda.empty_cache()
    return state,losses


def predict(x_fit,y_fit,x_val,cfg,state):
    from tabicl import TabICLRegressor
    model=TabICLRegressor(model_path=cfg['checkpoint'],allow_auto_download=False,
        n_estimators=cfg['n_estimators'],batch_size=1,device=cfg['device'],use_amp=True,
        random_state=cfg['seed'])
    model.fit(x_fit,y_fit)
    if state is not None:model.model_.load_state_dict(state)
    result=np.asarray(model.predict(x_val),dtype=float)
    if not np.isfinite(result).all():raise ValueError('Nonfinite real-data prediction')
    del model
    return result


def run(cfg,args,runner):
    if args.selection is not None:raise ValueError('Development only')
    if importlib.metadata.version('tabicl')!=cfg['tabicl_version']:
        raise ValueError('Unexpected TabICL version')
    if sha256_of(Path(cfg['checkpoint']))!=cfg['checkpoint_sha256']:
        raise ValueError('Checkpoint SHA mismatch')
    source=Path(cfg['reference_source']);folds=json.loads(Path(cfg['splits']).read_text())['folds']
    info=json.loads((source/'provenance.json').read_text())
    if info['sha256'][cfg['splits']]!=sha256_of(Path(cfg['splits'])):
        raise ValueError('Reference split mismatch')
    if len(folds)!=10 or any(len(f['inner'])!=3 for f in folds):raise ValueError('Unexpected split count')
    torch.set_num_threads(cfg['cpu_threads']);device=torch.device(cfg['device'])
    if device.type=='cuda':torch.cuda.set_per_process_memory_fraction(cfg['gpu_memory_fraction'])
    data=load(Path(cfg['data']),cfg['data_sha256']);checkpoint=torch.load(cfg['checkpoint'],map_location='cpu',weights_only=True)
    args.out.mkdir(parents=True,exist_ok=False);weights=args.out/'weights';weights.mkdir()
    def write(name,value):(args.out/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    paths=[Path(cfg['splits']),Path(cfg['checkpoint']),Path(__file__),Path(__file__).with_name('genetic_prior.py'),Path(args.config),Path(runner),source/'provenance.json']
    paths += [source/f"references_{f['fold']}_{j}.npz" for f in folds for j in range(3)]
    write('provenance.json',{'config':cfg,'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                             'sha256':{str(p):sha256_of(p) for p in paths}})
    status={'status':'running','completed_units':0,'expected_units':30,'completed_adaptations':0,'expected_adaptations':60};records=[];failures=[];start=time.perf_counter()
    try:
        for fold in folds:
            f=fold['fold']
            for j,split in enumerate(fold['inner']):
                tr,val=np.array(split['train_idx']),np.array(split['val_idx'])
                if set(tr)&set(val) or set(tr)|set(val)!=set(fold['dev_idx']) or (set(tr)|set(val))&set(fold['test_idx']):raise ValueError('Invalid split')
                states={'official':None}
                for arm in ('generic','genetic'):
                    seed=cfg['seed']+f*3+j
                    state,losses=adapt(data.X[tr],arm,cfg,seed,checkpoint,device)
                    path=weights/f'{arm}_{f}_{j}.pt';torch.save(state,path)
                    records.append({'fold':f,'inner':j,'arm':arm,'seed':seed,'steps':len(losses),
                                    'losses':losses,'weight_path':str(path),'weight_sha256':sha256_of(path)})
                    states[arm]=state;status['completed_adaptations']+=1;write('execution_status.json',status)
                pred={arm:np.column_stack([predict(data.X[tr],data.Y[tr,e],data.X[val],cfg,state)
                                           for e in range(4)]) for arm,state in states.items()}
                np.savez_compressed(args.out/f'predictions_{f}_{j}.npz',**pred)
                status['completed_units']+=1;write('execution_status.json',status);write('adaptations.json',records)
        if status['completed_units']!=30 or status['completed_adaptations']!=60:raise ValueError('Incomplete budget')
        folds_out=[]
        names=('official','generic','genetic','huber','independent_tabm','fixed_fusion')
        for fold in folds:
            vals={name:[] for name in names}
            for j,split in enumerate(fold['inner']):
                arr=np.load(args.out/f"predictions_{fold['fold']}_{j}.npz");ref=np.load(source/f"references_{fold['fold']}_{j}.npz");truth=data.Y[split['val_idx']]
                for name in names:
                    pred=arr[name] if name in ('official','generic','genetic') else ref[name]
                    v=[[fun(truth[:,e],pred[:,e]) for fun in (pcc,rmse,mae)] for e in range(4)]
                    if not np.isfinite(v).all():raise ValueError('Nonfinite metric')
                    vals[name].append(v)
            folds_out.append({'fold':fold['fold'],'metrics':{k:np.mean(v,axis=0).tolist() for k,v in vals.items()}})
        macro={name:dict(zip(('pcc','rmse','mae'),np.mean([r['metrics'][name] for r in folds_out],axis=(0,1)).tolist())) for name in names}
        write('summary.json',{'macro':macro,'folds':folds_out});status['status']='complete'
    except BaseException as exc:
        status['status']='failed';failures.append({'completed_units':status['completed_units'],'error':repr(exc)});raise
    finally:
        write('execution_status.json',status);write('adaptations.json',records);write('failures.json',failures)
        write('timing.json',{'seconds':time.perf_counter()-start})
