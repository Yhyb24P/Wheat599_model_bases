"""WheatGP official network on nested Wheat599 material splits."""
import csv
import importlib.util
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA

from wheat599.data import load, sha256_of
from wheat599.metrics import pcc, rmse, mae


def prepare(x, train, query, kind, cfg):
    if set(train) & set(query):
        raise ValueError('Overlapping fit and query materials')
    if kind == 'raw':
        return x[train].astype('float32'), x[query].astype('float32')
    pca = PCA(n_components=cfg['pca_components'], svd_solver='randomized', random_state=cfg['seed'])
    return pca.fit_transform(x[train]).astype('float32'), pca.transform(x[query]).astype('float32')


def segments(x):
    width = x.shape[1] // 5
    if width < 5:
        raise ValueError('Insufficient feature count')
    return torch.split(x, [width] * 4 + [x.shape[1] - 4 * width], dim=1)


def fit(model_type, xf, yf, xq, yq, lr, epochs, cfg, device, select=True):
    torch.manual_seed(cfg['seed']); np.random.seed(cfg['seed'])
    model = model_type(8 * (xf.shape[1] - 20)).to(device)
    x = torch.as_tensor(xf, device=device); q = torch.as_tensor(xq, device=device)
    center, scale = float(yf.mean()), float(yf.std())
    if scale <= 0: raise ValueError('Constant training target')
    y = torch.as_tensor((yf - center)[:, None] / scale, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=90, gamma=0.1)
    generator = torch.Generator(device='cpu').manual_seed(cfg['seed'])
    best, best_epoch, best_pred = -np.inf, 0, None
    for epoch in range(1, epochs + 1):
        model.train()
        for idx in torch.randperm(len(x), generator=generator).split(cfg['batch_size']):
            idx = idx.to(device); optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(*segments(x[idx])), y[idx])
            if not torch.isfinite(loss): raise ValueError('Nonfinite training loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        scheduler.step()
        if select:
            model.eval()
            with torch.no_grad(): pred = model(*segments(q)).flatten().cpu().numpy() * scale + center
            score = pcc(yq, pred)
            if np.isfinite(score) and score > best:
                best, best_epoch, best_pred = float(score), epoch, pred.copy()
            elif epoch - best_epoch >= cfg['patience']: break
    if not select:
        model.eval()
        with torch.no_grad(): best_pred = model(*segments(q)).flatten().cpu().numpy() * scale + center
        best_epoch = epochs
    if best_pred is None or not np.isfinite(best_pred).all(): raise ValueError('Invalid prediction')
    return best_pred, best_epoch, epoch


def run(cfg, args, runner):
    if sha256_of(Path(cfg['source'])) != cfg['source_sha256']: raise ValueError('Official source mismatch')
    spec = importlib.util.spec_from_file_location('wheatgp_official', cfg['source'])
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    torch.set_num_threads(cfg['cpu_threads']); device = torch.device(cfg['device'])
    if device.type == 'cuda': torch.cuda.set_per_process_memory_fraction(cfg['gpu_memory_fraction'])
    data = load(Path(cfg['data']), cfg['data_sha256']); folds = json.loads(Path(cfg['splits']).read_text())['folds']
    if len(folds) != 10 or any(len(f['inner']) != 3 for f in folds): raise ValueError('Unexpected splits')
    candidates = [(kind, lr) for kind in ('raw', 'pca') for lr in cfg['learning_rates']]
    args.out.mkdir(parents=True, exist_ok=False)
    def write(name, x): (args.out/name).write_text(json.dumps(x, indent=2, allow_nan=False)+'\n')
    write('provenance.json', {'config': cfg, 'git_commit': subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
          'sha256': {str(p): sha256_of(Path(p)) for p in (cfg['source'], cfg['splits'], args.config, __file__, runner)}})
    start = time.perf_counter(); records = []; failures = []
    phase = 'development' if args.selection is None else 'outer'
    status = {'status':'running', 'phase':phase, 'completed':0, 'expected':10*3*4*len(candidates) if phase=='development' else 40}
    try:
        if phase == 'development':
            for fold in folds:
                f = fold['fold']; predictions = []
                for j, split in enumerate(fold['inner']):
                    tr, val = np.array(split['train_idx']), np.array(split['val_idx'])
                    if set(tr)|set(val) != set(fold['dev_idx']) or (set(tr)|set(val)) & set(fold['test_idx']): raise ValueError('Split leakage')
                    prepared = {kind:prepare(data.X, tr, val, kind, cfg) for kind in ('raw','pca')}
                    arr = {}
                    for e in range(4):
                        for c,(kind,lr) in enumerate(candidates):
                            pred, best, epochs = fit(module.wheatGP_base, *prepared[kind][:1], data.Y[tr,e], prepared[kind][1], data.Y[val,e], lr, cfg['max_epochs'], cfg, device)
                            arr[f'e{e}_c{c}'] = pred
                            records.append({'fold':f,'inner':j,'environment':e,'candidate':c,'best_epoch':best,'epochs_run':epochs,'pcc':pcc(data.Y[val,e],pred)})
                            status['completed'] += 1
                    np.savez_compressed(args.out/f'predictions_{f}_{j}.npz', **arr)
                    predictions.append(arr); write('status.json',status); write('trajectories.json',records)
                selected=[]
                for e in range(4):
                    scores=[float(np.mean([pcc(data.Y[sp['val_idx'],e], predictions[j][f'e{e}_c{c}']) for j,sp in enumerate(fold['inner'])])) for c in range(len(candidates))]
                    c=int(np.argmax(scores)); ep=sorted(r['best_epoch'] for r in records if r['fold']==f and r['environment']==e and r['candidate']==c)[1]
                    selected.append({'environment':e,'candidate':c,'kind':candidates[c][0],'learning_rate':candidates[c][1],'epochs':ep,'scores':scores})
                write(f'selection_{f}.json',selected)
            write('selection.json',{'folds':[{'fold':f['fold'],'selected':json.loads((args.out/f"selection_{f['fold']}.json").read_text())} for f in folds], 'source_sha256':cfg['source_sha256'],'splits_sha256':sha256_of(Path(cfg['splits'])), 'config_sha256':sha256_of(args.config)})
        else:
            selection=json.loads(args.selection.read_text())
            if selection['source_sha256']!=cfg['source_sha256'] or selection['splits_sha256']!=sha256_of(Path(cfg['splits'])) or selection['config_sha256']!=sha256_of(args.config): raise ValueError('Selection provenance mismatch')
            rows=[]
            for fold in folds:
                f=fold['fold']; tr=np.array(fold['dev_idx']); test=np.array(fold['test_idx'])
                selected=selection['folds'][f]['selected']; prepared={k:prepare(data.X,tr,test,k,cfg) for k in set(s['kind'] for s in selected)}
                for entry in selected:
                    e=entry['environment']; kind=entry['kind']; xf,xq=prepared[kind]
                    pred,_,_=fit(module.wheatGP_base,xf,data.Y[tr,e],xq,None,entry['learning_rate'],entry['epochs'],cfg,device,select=False)
                    rows.extend({'fold':f,'environment':data.environments[e],'material_index':int(i),'y_true':float(y),'y_pred':float(p)} for i,y,p in zip(test,data.Y[test,e],pred));status['completed']+=1
                write('status.json',status)
            with (args.out/'oof.csv').open('w',newline='') as h:
                w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
            with Path(cfg['quantile_oof']).open() as h:
                baseline = {(int(r['fold']),str(r['environment']),int(r['material_index'])):r for r in csv.DictReader(h)}
            if len(baseline) != len(rows): raise ValueError('Baseline OOF coverage mismatch')
            metrics=[]
            for fold in folds:
                for e in data.environments:
                    r=[v for v in rows if v['fold']==fold['fold'] and v['environment']==e]
                    b=[baseline[(v['fold'],str(e),v['material_index'])] for v in r]
                    y=np.array([v['y_true'] for v in r])
                    if not np.allclose(y,[float(v['y_true']) for v in b],atol=1e-12,rtol=0): raise ValueError('Baseline target mismatch')
                    predictions={'wheatgp':[v['y_pred'] for v in r], 'quantile_huber':[float(v['quantile_huber']) for v in b]}
                    metrics.append({'fold':fold['fold'],'environment':e, 'methods':{name:{'pcc':pcc(y,p),'rmse':rmse(y,p),'mae':mae(y,p)} for name,p in predictions.items()}})
            write('summary.json',{'per_fold':metrics,'macro':{name:{k:float(np.mean([v['methods'][name][k] for v in metrics])) for k in ('pcc','rmse','mae')} for name in ('wheatgp','quantile_huber')}})
        if status['completed'] != status['expected']: raise ValueError('Incomplete run')
        status['status']='complete'
    except BaseException as exc:
        status['status']='failed';failures.append(repr(exc));raise
    finally:
        write('status.json',status);write('failures.json',failures);write('timing.json',{'seconds':time.perf_counter()-start})
