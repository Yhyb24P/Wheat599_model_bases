"""Gaussian and additive/Gaussian kernel ridge under nested CV."""
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import KFold

from wheat599.data import load, sha256_of
from wheat599.metrics import pcc, rmse, mae


def kernels(X, train, target, bandwidth, device):
    x = torch.as_tensor(X[train], dtype=torch.float64, device=device)
    t = torch.as_tensor(X[target], dtype=torch.float64, device=device)
    mean = x.mean(0)
    z, zt = x - mean, t - mean
    a = z @ z.T / X.shape[1]
    ac = zt @ z.T / X.shape[1]
    scale = torch.diagonal(a).mean()
    if scale <= 0:
        raise ValueError('zero additive kernel scale')
    norms = x.square().sum(1)
    d = (norms[:, None] + norms[None, :] - 2*x @ x.T).clamp_min(0)
    dc = (t.square().sum(1)[:, None] + norms[None, :] - 2*t @ x.T).clamp_min(0)
    d0 = torch.quantile(d[torch.triu_indices(len(train), len(train), 1, device=device).unbind()], 0.5)
    if d0 <= 0:
        raise ValueError('zero distance median')
    return a/scale, ac/scale, torch.exp(-bandwidth*d/d0), torch.exp(-bandwidth*dc/d0)


def predict(a, ac, r, rc, y, fraction, ridge):
    """Ridge with an unpenalized intercept, fitted on training data only."""
    y = torch.as_tensor(y[:, None], dtype=a.dtype, device=a.device)
    k = fraction*a + (1-fraction)*r
    cross = fraction*ac + (1-fraction)*rc
    chol = torch.linalg.cholesky(k + ridge*torch.eye(len(y), dtype=y.dtype, device=y.device))
    both = torch.cholesky_solve(torch.cat([y, torch.ones_like(y)], dim=1), chol)
    intercept = both[:, 0].sum()/both[:, 1].sum()
    return (intercept + cross @ (both[:, :1] - intercept*both[:, 1:])).cpu().numpy().ravel()


def write(path, fields, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        w.writeheader(); w.writerows(rows)


def run(cfg, args, runner_path):
    data = load(Path(cfg['data']), cfg['data_sha256'])
    folds = json.loads(Path(cfg['splits']).read_text())['folds']
    split_hash = sha256_of(Path(cfg['splits']))
    frozen = json.loads(args.selection.read_text()) if args.selection else None
    if frozen and (frozen['config'] != cfg or frozen['splits_sha256'] != split_hash):
        raise ValueError('Frozen selection does not match config or splits')
    expected = {(m,e,f['fold']) for m in ('gaussian','additive_rbf') for e in data.environments for f in folds}
    if frozen:
        keys = [(v['model'],v['environment'],v['fold']) for v in frozen['selected']]
        if len(keys) != len(expected) or set(keys) != expected:
            raise ValueError('Incomplete frozen selection')
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out/'provenance.json').write_text(json.dumps({'config':cfg,'splits_sha256':split_hash,'runner_sha256':sha256_of(runner_path),'model_sha256':sha256_of(Path(__file__)),'selection_sha256':sha256_of(args.selection) if args.selection else None,'rule':'mean inner PCC; first candidate on tie; full dev refit'},indent=2))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda': torch.cuda.set_per_process_memory_fraction(0.35)
    selected, valrows, rows, oof, failures = [], [], [], [], []
    models = ('gaussian','additive_rbf')
    for fold in folds:
        k=fold['fold']; test=np.asarray(fold['test_idx']); dev=np.sort(np.r_[fold['train_idx'],fold['val_idx']])
        if len(set(dev)&set(test)) or len(set(dev)|set(test)) != data.n: raise ValueError('Invalid split')
        inner=[(dev[a],dev[b]) for a,b in KFold(n_splits=cfg['inner_folds'],shuffle=True,random_state=cfg['seed']+k).split(dev)]
        print(f'fold {k}',flush=True)
        try:
            if frozen is None:
                candidates={m:[(h,ridge,frac) for h in cfg['bandwidths'] for ridge in cfg['ridge'] for frac in ([0.0] if m=='gaussian' else cfg['additive_fractions'])] for m in models}
                scores={m:np.empty((len(data.environments),len(candidates[m]),len(inner))) for m in models}
                for j,(train,valid) in enumerate(inner):
                    for h in cfg['bandwidths']:
                        mats=kernels(data.X,train,valid,h,device)
                        for m in models:
                            for i,(hh,ridge,frac) in enumerate(candidates[m]):
                                if hh != h: continue
                                for e,env in enumerate(data.environments):
                                    pred=predict(*mats,data.Y[train,e],frac,ridge)
                                    score=pcc(data.Y[valid,e],pred)
                                    if not np.isfinite(score): raise ValueError('non-finite validation PCC')
                                    scores[m][e,i,j]=score
                                    valrows.append({'model':m,'environment':env,'fold':k,'inner_fold':j,'bandwidth':h,'ridge':ridge,'additive_fraction':frac,'pcc':float(score)})
                for m in models:
                    for e,env in enumerate(data.environments):
                        means=scores[m][e].mean(1); i=int(np.argmax(means)); h,ridge,frac=candidates[m][i]
                        selected.append({'model':m,'environment':env,'fold':k,'bandwidth':h,'ridge':ridge,'additive_fraction':frac,'inner_pcc':float(means[i])})
                (args.out/'selection.json').write_text(json.dumps({'config':cfg,'splits_sha256':split_hash,'selected':selected},indent=2))
            else:
                for m in models:
                    for e,env in enumerate(data.environments):
                        entry=next(v for v in frozen['selected'] if (v['model'],v['environment'],v['fold'])==(m,env,k))
                        h,ridge,frac=entry['bandwidth'],entry['ridge'],entry['additive_fraction']
                        if h not in cfg['bandwidths'] or ridge not in cfg['ridge'] or frac not in ([0.0] if m=='gaussian' else cfg['additive_fractions']): raise ValueError('invalid frozen choice')
                        pred=predict(*kernels(data.X,dev,test,h,device),data.Y[dev,e],frac,ridge)
                        met={'pcc':pcc(data.Y[test,e],pred),'rmse':rmse(data.Y[test,e],pred),'mae':mae(data.Y[test,e],pred)}
                        if not all(np.isfinite(v) for v in met.values()): raise ValueError('non-finite test metric')
                        rows.append({'model':m,'environment':env,'fold':k,**met,'bandwidth':h,'ridge':ridge,'additive_fraction':frac,'inner_pcc':entry['inner_pcc']})
                        oof.extend({'model':m,'environment':env,'fold':k,'material_index':int(idx),'y_true':float(data.Y[idx,e]),'y_pred':float(v)} for idx,v in zip(test,pred))
        except Exception as exc:
            failures.append({'fold':k,'reason':repr(exc)}); print(f'FAILED {k}: {exc}',flush=True)
            if isinstance(exc,torch.cuda.OutOfMemoryError): raise
        finally:
            write(args.out/'validation.csv',['model','environment','fold','inner_fold','bandwidth','ridge','additive_fraction','pcc'],valrows)
            write(args.out/'per_fold.csv',['model','environment','fold','pcc','rmse','mae','bandwidth','ridge','additive_fraction','inner_pcc'],rows)
            write(args.out/'oof.csv',['model','environment','fold','material_index','y_true','y_pred'],oof)
            write(args.out/'failures.csv',['fold','reason'],failures)
    if failures or (len(selected)!=80 if frozen is None else len(rows)!=80): raise RuntimeError('Incomplete run')
    if frozen is None:
        print('Development selection complete; no test predictions made.',flush=True); return
    summary={'n_runs':len(rows),'n_failures':0,'models':{}}
    for m in models:
        per={e:{metric:float(np.mean([v[metric] for v in rows if v['model']==m and v['environment']==e])) for metric in ('pcc','rmse','mae')} for e in data.environments}
        summary['models'][m]={'per_environment':per,'macro':{metric:float(np.mean([per[e][metric] for e in data.environments])) for metric in ('pcc','rmse','mae')}}
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary),flush=True)
