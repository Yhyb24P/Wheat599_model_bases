"""Matched 99-output MSE control for independent quantile TabM."""
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


def multihead_mse(output, target):
    """Same reduction as the quantile arm: sum heads, average materials/members."""
    return (output - target[:, None, :]).square().sum(-1).mean()


def train(x_fit, y_fit, x_val, y_val, config, seed, index, settings, device, objective):
    model = build_model(x_fit, y_fit, config, seed, settings, device)
    x = torch.as_tensor(x_fit, dtype=torch.float32, device=device)
    xv = torch.as_tensor(x_val, dtype=torch.float32, device=device)
    y = model.standardize_y(torch.as_tensor(y_fit[:, None], dtype=torch.float32, device=device))
    q = torch.linspace(settings['quantile_min'], settings['quantile_max'], settings['n_quantiles'], device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    generator = torch.Generator().manual_seed(int(seed)*1000 + index)
    spec = settings['training']
    clipped = torch.zeros((), device=device); norm_sum = torch.zeros((), device=device)
    steps = 0
    best = {'best_pcc': -np.inf, 'best_epoch': -1}
    for epoch in range(spec['max_epochs']+1):
        if epoch:
            model.train()
            order = torch.randperm(len(x), generator=generator).to(device)
            for batch in order.split(spec['batch_size']):
                opt.zero_grad(set_to_none=True)
                output = model(x[batch])
                loss = multihead_mse(output, y[batch]) if objective == 'mse' else quantile_loss(output, y[batch], q)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite loss')
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), spec['gradient_clip'])
                if not torch.isfinite(norm):
                    raise ValueError('Nonfinite gradient')
                clipped += (norm > spec['gradient_clip']); norm_sum += norm.detach(); steps += 1
                opt.step()
        model.eval()
        with torch.no_grad():
            pred = model.inverse_y(model(xv).mean(dim=(1,2))[:,None])[:,0].cpu().numpy()
        score = pcc(y_val, pred)
        if not np.isfinite(score):
            raise ValueError('Nonfinite validation PCC')
        if score > best['best_pcc'] + spec['tie_tolerance']:
            best = dict(best_pcc=score, best_epoch=epoch, prediction=pred.copy(),
                        best_steps=steps, best_clipped=int(clipped.item()),
                        best_gradient_norm_sum=float(norm_sum.item()))
        elif epoch - best['best_epoch'] >= spec['patience']:
            break
    best['epochs_run'] = epoch
    return best


def read(path):
    return json.loads(Path(path).read_text())


def run(cfg, args, runner):
    if args.selection is not None:
        raise ValueError('Development only')
    settings = yaml.safe_load(Path(cfg['development_config']).read_text())
    configs = yaml.safe_load(Path(settings['architecture_config']).read_text())['configurations']
    folds = read(cfg['splits'])['folds']
    if len(folds) != 10 or any(len(f['inner']) != 3 for f in folds) or len(configs) != 8 or settings['n_quantiles'] != 99:
        raise ValueError('Unexpected frozen development grid')
    source = Path(cfg['quantile_source']); reference = Path(cfg['reference_source'])
    if read(source/'execution_status.json') != {'status':'complete','completed':1920,'expected':1920} or read(source/'failures.json'):
        raise ValueError('Incomplete quantile source')
    if read(source/'provenance.json')['sha256'][cfg['splits']] != sha256_of(Path(cfg['splits'])):
        raise ValueError('Quantile split mismatch')
    chosen = {v['fold']:v['selected'] for v in read(source/'summary.json')['folds']}
    original = {(v['fold'],v['inner'],v['environment'],v['config_index'],v['seed']):v
                for v in read(source/'trajectories.json')}
    if len(original) != 1920 or len(chosen) != 10:
        raise ValueError('Incomplete source trajectories')
    torch.set_num_threads(cfg['cpu_threads']); device = torch.device(cfg['device'])
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(cfg['gpu_memory_fraction'])
    data = load(Path(cfg['data']), cfg['data_sha256'])
    args.out.mkdir(parents=True, exist_ok=False)
    def write(name, value):
        (args.out/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    paths = [Path(cfg['splits']),Path(cfg['development_config']),Path(settings['architecture_config']),
             Path(__file__),Path(__file__).with_name('quantile_tabm.py'),Path(runner),Path(args.config),
             source/'provenance.json',source/'summary.json',source/'trajectories.json',reference/'provenance.json']
    paths += [source/f"predictions_{f['fold']}_{j}.npz" for f in folds for j in range(3)]
    paths += [reference/f"references_{f['fold']}_{j}.npz" for f in folds for j in range(3)]
    write('provenance.json', {'config':cfg,'git_commit':subprocess.check_output(
        ['git','rev-parse','HEAD'],text=True).strip(),'sha256':{str(p):sha256_of(p) for p in paths},
        'mse_reduction':'sum of 99 squared errors per member, mean over materials and members',
        'quantile_reduction':'sum of 99 pinball losses per member, mean over materials and members'})
    state = dict(status='running',completed=0,expected=2160)
    records, failures, started = [], [], time.perf_counter()
    try:
        for fold in folds:
            f = fold['fold']
            for j,split in enumerate(fold['inner']):
                tr,val = np.array(split['train_idx']),np.array(split['val_idx'])
                if (set(tr)&set(val) or set(tr)|set(val)!=set(fold['dev_idx'])
                        or (set(tr)|set(val))&set(fold['test_idx'])):
                    raise ValueError('Invalid development split')
                saved = np.load(source/f'predictions_{f}_{j}.npz'); outputs = {}
                for e in range(4):
                    for c,config in enumerate(configs):
                        for seed in settings['seeds']:
                            result = train(data.X[tr],data.Y[tr,e],data.X[val],data.Y[val,e],config,
                                           seed,c,settings,device,'mse')
                            outputs[f'e{e}_c{c}_s{seed}'] = result.pop('prediction')
                            records.append(dict(fold=f,inner=j,environment=e,config_index=c,seed=seed,
                                                objective='mse',**result))
                            state['completed']+=1;write('execution_status.json',state)
                    c = chosen[f][e]
                    for seed in settings['seeds']:
                        result = train(data.X[tr],data.Y[tr,e],data.X[val],data.Y[val,e],configs[c],
                                       seed,c,settings,device,'quantile')
                        pred = result.pop('prediction');old = original[(f,j,e,c,seed)]
                        error = float(np.max(np.abs(pred-saved[f'e{e}_c{c}_s{seed}'])))
                        if error > 1e-5 or result['best_epoch'] != old['best_epoch']:
                            raise ValueError(f'Quantile replay mismatch: fold={f}, inner={j}, env={e}, seed={seed}, error={error}')
                        records.append(dict(fold=f,inner=j,environment=e,config_index=c,seed=seed,
                                            objective='quantile',replay_max_abs=error,**result))
                        state['completed']+=1;write('execution_status.json',state)
                np.savez_compressed(args.out/f'predictions_{f}_{j}.npz',**outputs)
                write('trajectories.json',records)
        if state['completed'] != state['expected']:
            raise ValueError('Incomplete trajectories')
        results=[]
        for fold in folds:
            f=fold['fold']; a=[np.load(args.out/f'predictions_{f}_{j}.npz') for j in range(3)]
            b=[np.load(source/f'predictions_{f}_{j}.npz') for j in range(3)]
            refs=[np.load(reference/f'references_{f}_{j}.npz') for j in range(3)]
            selected=[]
            for e in range(4):
                scores=[np.mean([pcc(data.Y[split['val_idx'],e],
                    np.mean([a[j][f'e{e}_c{c}_s{seed}'] for seed in settings['seeds']],axis=0))
                    for j,split in enumerate(fold['inner'])]) for c in range(len(configs))]
                selected.append(int(np.argmax(scores)))
            metrics={name:[] for name in ('mse99','mse99_quantile_configs','quantile','mse1','huber','mse1_huber')}
            for j,split in enumerate(fold['inner']):
                truth=data.Y[split['val_idx']]
                def bank(arr, indices):
                    return np.column_stack([np.mean([arr[f'e{e}_c{c}_s{seed}'] for seed in settings['seeds']],axis=0)
                                            for e,c in enumerate(indices)])
                preds={'mse99':bank(a[j],selected),'mse99_quantile_configs':bank(a[j],chosen[f]),
                       'quantile':bank(b[j],chosen[f]),'mse1':refs[j]['independent_tabm'],
                       'huber':refs[j]['huber'],'mse1_huber':refs[j]['fixed_fusion']}
                for name,pred in preds.items():
                    values=[[fn(truth[:,e],pred[:,e]) for fn in (pcc,rmse,mae)] for e in range(4)]
                    if not np.isfinite(values).all():
                        raise ValueError('Nonfinite metrics')
                    metrics[name].append(values)
            results.append(dict(fold=f,mse99_selected=selected,quantile_selected=chosen[f],
                                metrics={k:np.mean(v,axis=0).tolist() for k,v in metrics.items()}))
        names=('mse99','mse99_quantile_configs','quantile','mse1','huber','mse1_huber')
        macro={name:dict(zip(('pcc','rmse','mae'),np.mean([r['metrics'][name] for r in results],axis=(0,1)).tolist()))
               for name in names}
        def telemetry(objective,selected_by):
            chosen_rows=[v for v in records if v['objective']==objective and
                         v['config_index']==selected_by[v['fold']][v['environment']]]
            n=sum(v['best_steps'] for v in chosen_rows);clipped=sum(v['best_clipped'] for v in chosen_rows)
            return dict(trajectories=len(chosen_rows),steps=n,clipped_steps=clipped,
                        clipped_fraction=clipped/n if n else None,
                        mean_gradient_norm=sum(v['best_gradient_norm_sum'] for v in chosen_rows)/n if n else None)
        mse_chosen={r['fold']:r['mse99_selected'] for r in results}
        write('summary.json',dict(macro=macro,folds=results,telemetry={
            'quantile_selected':telemetry('quantile',chosen),
            'mse99_quantile_configs':telemetry('mse',chosen),
            'mse99_selected':telemetry('mse',mse_chosen)}))
        state['status']='complete'
    except BaseException as exc:
        state['status']='failed';failures.append({'completed':state['completed'],'error':repr(exc)});raise
    finally:
        write('execution_status.json',state);write('trajectories.json',records)
        write('failures.json',failures);write('timing.json',{'seconds':time.perf_counter()-started})
