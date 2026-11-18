"""Gaussian and additive/Gaussian kernel ridge under nested CV."""
import csv
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import KFold

from genopred.data.wheat599 import load, sha256_of
from genopred.evaluation.metrics import pcc, rmse, mae


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
    pairs = d[torch.triu_indices(len(train), len(train), 1, device=device).unbind()]
    if pairs.numel() > 2**24:
        # PyTorch quantile limits input size. The extracted pairs can be partitioned in place.
        median = np.quantile(pairs.cpu().numpy(), 0.5, overwrite_input=True)
        d0 = torch.as_tensor(median, dtype=d.dtype, device=device)
    else:
        d0 = torch.quantile(pairs, 0.5)
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


def gcv(eigenvalues, vectors, y, ridges):
    """GCV including an unpenalized intercept, evaluated on the original grid."""
    eigenvalues, vectors, y = map(np.asarray, (eigenvalues, vectors, y))
    if eigenvalues.min() < -1e-8 or any(r <= 0 for r in ridges):
        raise ValueError('GCV requires a PSD kernel and positive ridge values')
    eigenvalues = np.maximum(eigenvalues, 0)
    transformed_y = vectors.T @ y
    one = vectors.sum(0)
    scores = []
    for ridge in ridges:
        inverse = 1 / (eigenvalues + ridge)
        denominator = np.sum(one**2 * inverse)
        intercept = np.sum(one * inverse * transformed_y) / denominator
        alpha = inverse * (transformed_y - intercept * one)
        trace_projection = inverse.sum() - np.sum(one**2 * inverse**2) / denominator
        # Residual=lambda*P*y, n-tr(H)=lambda*tr(P); cancel lambda stably.
        scores.append(float(len(y) * np.sum(alpha**2) / trace_projection**2))
    if not np.isfinite(scores).all():
        raise FloatingPointError('Nonfinite GCV')
    return scores


def run_gcv(cfg, args, runner_path):
    """Freeze GCV choices before one outer evaluation; geometry stays frozen."""
    if args.selection is not None:
        raise ValueError('GCV comparison uses only its configured frozen kernel selections')
    torch.set_num_threads(1)
    start = time.perf_counter()
    read = lambda path: json.loads(Path(path).read_text())
    data = load(Path(cfg['data']), cfg['data_sha256'])
    folds = read(cfg['splits'])['folds']
    frozen = read(cfg['selection'])
    old_cfg = frozen['config']
    if old_cfg['data_sha256'] != cfg['data_sha256']:
        raise ValueError('Frozen selection data mismatch')
    old_folds = read(old_cfg['splits'])['folds']
    if len(folds) != 10 or sorted(i for f in folds for i in f['test_idx']) != list(range(data.n)):
        raise ValueError('Invalid outer material coverage')
    for f in folds:
        old = next(row for row in old_folds if row['fold'] == f['fold'])
        old_dev = old['dev_idx'] if 'dev_idx' in old else old['train_idx'] + old['val_idx']
        if (set(f['dev_idx']) != set(old_dev) or set(f['test_idx']) != set(old['test_idx'])
                or set(f['dev_idx']) & set(f['test_idx'])
                or set(f['dev_idx']) | set(f['test_idx']) != set(range(data.n))):
            raise ValueError('Frozen kernel split differs from protocol')
    expected = {(m, e, f['fold']) for m in ('gaussian', 'additive_rbf')
                for e in data.environments for f in folds}
    index = {(r['model'], r['environment'], r['fold']): r for r in frozen['selected']}
    if set(index) != expected or len(index) != len(frozen['selected']):
        raise ValueError('Incomplete frozen kernel selection')
    baseline_path = Path(cfg['reference']) / 'oof.csv'
    baseline, reference_cfg = {}, read(Path(cfg['reference']) / 'provenance.json')['config']
    if reference_cfg != old_cfg:
        raise ValueError('Reference and selection configurations differ')
    with baseline_path.open() as handle:
        for row in csv.DictReader(handle):
            key = row['model'], row['environment'], int(row['fold']), int(row['material_index'])
            e = data.environments.index(row['environment'])
            if key in baseline or abs(float(row['y_true']) - data.Y[key[3], e]) > 1e-6:
                raise ValueError('Reference duplicate or truth mismatch')
            baseline[key] = float(row['y_pred'])
    expected_predictions = {(m, e, f['fold'], i) for m in ('gaussian', 'additive_rbf')
                            for e in data.environments for f in folds for i in f['test_idx']}
    if set(baseline) != expected_predictions or not np.isfinite(list(baseline.values())).all():
        raise ValueError('Reference OOF coverage or finite-value mismatch')
    args.out.mkdir(parents=True, exist_ok=False)

    def save(name, obj):
        (args.out / name).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')

    paths = [Path(__file__), Path(runner_path), Path(args.config), Path(cfg['splits']),
             Path(cfg['selection']), Path(old_cfg['splits']), baseline_path,
             Path(cfg['reference']) / 'provenance.json']
    save('provenance.json', dict(config=cfg, original_grid=old_cfg['ridge'],
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        rule='GCV on complete development set, original ridge grid; freeze all choices before test; kernel geometry fixed from original inner PCC selection',
        sha256={str(p): sha256_of(p) for p in paths}))
    state, failures, choices, rows, oof = dict(status='selecting', completed=0, expected=80), [], [], [], []
    save('execution_status.json', state)
    try:
        for f in folds:
            dev = np.array(f['dev_idx'])
            for model in ('gaussian', 'additive_rbf'):
                for e, env in enumerate(data.environments):
                    original = index[model, env, f['fold']]
                    a, _, r, _ = kernels(data.X, dev, np.array([], dtype=int), original['bandwidth'], 'cpu')
                    fraction = original['additive_fraction']
                    eigenvalues, vectors = np.linalg.eigh(fraction * a.numpy() + (1 - fraction) * r.numpy())
                    scores = gcv(eigenvalues, vectors, data.Y[dev, e], old_cfg['ridge'])
                    ridge = old_cfg['ridge'][int(np.argmin(scores))]
                    choices.append({**original, 'pcc_selected_ridge': original['ridge'],
                                    'ridge': ridge, 'gcv_scores': scores})
        save('selection.json', dict(config=cfg, selected=choices, rule='Frozen before any test scoring'))
        state['status'] = 'evaluating'
        save('execution_status.json', state)
        for choice in choices:
            f = next(row for row in folds if row['fold'] == choice['fold'])
            dev, test = np.array(f['dev_idx']), np.array(f['test_idx'])
            e, env, model = data.environments.index(choice['environment']), choice['environment'], choice['model']
            pred = predict(*kernels(data.X, dev, test, choice['bandwidth'], 'cpu'), data.Y[dev, e],
                           choice['additive_fraction'], choice['ridge'])
            reference = np.array([baseline[model, env, f['fold'], i] for i in test])
            if pred.shape != (len(test),) or not np.isfinite(pred).all():
                raise ValueError('Invalid test prediction')
            oof.extend(dict(model=model, environment=env, fold=f['fold'], material_index=int(i),
                            y_true=float(data.Y[i, e]), gcv=float(p), inner_pcc=float(old))
                       for i, p, old in zip(test, pred, reference))
            for rule, values in (('gcv', pred), ('inner_pcc', reference)):
                row = dict(model=model, environment=env, fold=f['fold'], rule=rule,
                           ridge=choice['ridge'] if rule == 'gcv' else choice['pcc_selected_ridge'])
                row.update({fn.__name__: float(fn(data.Y[test, e], values)) for fn in (pcc, rmse, mae)})
                rows.append(row)
                if not np.isfinite(list(row[k] for k in ('pcc', 'rmse', 'mae'))).all():
                    raise ValueError('Undefined test metric; predictions retained and failure recorded')
            state['completed'] += 1
        summary = gcv_summary(choices, rows, oof, data.environments)
        save('summary.json', summary)
        state['status'] = 'complete'
    except BaseException as error:
        state['status'] = 'failed'
        failures.append(dict(completed=state['completed'], error=repr(error)))
        raise
    finally:
        write(args.out / 'per_fold.csv', ['model', 'environment', 'fold', 'rule', 'ridge', 'pcc', 'rmse', 'mae'], rows)
        write(args.out / 'oof.csv', ['model', 'environment', 'fold', 'material_index', 'y_true', 'gcv', 'inner_pcc'], oof)
        save('execution_status.json', state)
        save('failures.json', failures)
        save('timing.json', dict(seconds=time.perf_counter() - start))


def gcv_summary(choices, rows, oof, environments):
    """Summarize complete predictions, also used to recover serialization failures."""
    summary = dict(n_units=len(choices), n_failures=0, comparison={})
    for model in ('gaussian', 'additive_rbf'):
        part = [c for c in choices if c['model'] == model]
        summary['comparison'][model] = dict(
            same_lambda=sum(c['ridge'] == c['pcc_selected_ridge'] for c in part),
            within_factor_ten=sum(bool(abs(np.log10(c['ridge'] / c['pcc_selected_ridge'])) <= 1) for c in part),
            increased_lambda=sum(c['ridge'] > c['pcc_selected_ridge'] for c in part),
            decreased_lambda=sum(c['ridge'] < c['pcc_selected_ridge'] for c in part),
            median_lambda_ratio=float(np.median([c['ridge'] / c['pcc_selected_ridge'] for c in part])))
        for rule in ('gcv', 'inner_pcc'):
            per = {env: {metric: float(np.mean([r[metric] for r in rows if r['model'] == model
                and r['rule'] == rule and r['environment'] == env])) for metric in ('pcc', 'rmse', 'mae')}
                for env in environments}
            pooled = {}
            for env in environments:
                values = [r for r in oof if r['model'] == model and r['environment'] == env]
                pooled[env] = {fn.__name__: fn([r['y_true'] for r in values], [r[rule] for r in values])
                               for fn in (pcc, rmse, mae)}
            summary['comparison'][model][rule] = dict(per_environment=per, pooled_oof=pooled,
                macro={metric: float(np.mean([per[e][metric] for e in environments])) for metric in ('pcc', 'rmse', 'mae')})
    return summary


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
