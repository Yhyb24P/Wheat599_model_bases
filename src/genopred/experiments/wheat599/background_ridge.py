"""Background-conditioned ridge / RR experiments (Wheat599).

Three protocols preserved verbatim from the former scripts/run_background_ridge.py
and scripts/run_background_nested.py:
- run_ridge: non-nested background ridge, minimum validation RMSE selection, one test.
- run_hybrid: background_rbf nested mean-inner-PCC bandwidth selection, full-dev refit.
- run_nested: conditioned ridge + RR-BLUP nested mean-inner-PCC selection, full-dev refit.

Math core lives in genopred.models.classical.background_ridge. runner_path is the
actual executing entry file passed by the caller (genopred.cli) for provenance.
"""
import argparse
import csv
import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

from genopred.data.wheat599 import load, sha256_of
from genopred.evaluation.metrics import pcc, rmse, mae
from genopred.models.classical import background_rbf, kernel_ridge
from genopred.models.classical import background_ridge as ridge_core
from genopred.models.classical.background_ridge import kernel, fit_candidate
from genopred.models.classical.kernel_ridge import gates


def write_csv(path, fields, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)


def design(cfg):
    keys = ('pca_components', 'gates', 'global_penalty', 'background_penalty', 'gate_steps', 'gate_lr')
    combinations = list(itertools.product(*(cfg[k] for k in keys)))
    rng = np.random.default_rng(cfg['search_seed'])
    chosen = rng.choice(len(combinations), size=cfg['budget'], replace=False)
    return [dict(zip(keys, combinations[i])) for i in chosen]


def features(X, train, target, n_pc, seed, device):
    centered = X - X[train].mean(axis=0)
    pca = PCA(n_components=n_pc, svd_solver='randomized', random_state=seed).fit(centered[train])
    scores = pca.transform(centered[np.r_[train, target]])
    mu, sd = scores[:len(train)].mean(0), scores[:len(train)].std(0)
    if np.any(sd == 0):
        raise ValueError('Constant PCA component')
    x = torch.tensor(np.column_stack([np.ones(len(scores)), (scores - mu) / sd]), dtype=torch.float64, device=device)
    ztr = torch.tensor(centered[train], dtype=torch.float64, device=device)
    ztarget = torch.tensor(centered[target], dtype=torch.float64, device=device)
    return x[:len(train)], x[len(train):], ztr @ ztr.T / X.shape[1], ztarget @ ztr.T / X.shape[1]


def model_predict(x, xt, gram, cross, ytrain, c, cfg):
    ym, ys = ytrain.mean(), ytrain.std()
    if ys == 0:
        raise ValueError('Constant target')
    y = torch.tensor(((ytrain - ym) / ys)[:, None], dtype=torch.float64, device=x.device)
    torch.manual_seed(cfg['seed'])
    local = {'gates': c['gates'], 'gate_steps': c['gate_steps'], 'gate_lr': c['gate_lr'], 'gate_init_std': cfg['gate_init_std']}
    w, alpha, intercept, _ = fit_candidate(x, gram, y, c['global_penalty'], c['background_penalty'], local)
    with torch.no_grad():
        pred = intercept + kernel(gates(xt, w), gates(x, w), cross, c['global_penalty'], c['background_penalty']) @ alpha
    return pred.cpu().numpy().ravel() * ys + ym


def rr_predict(X, y, train, target, alpha):
    z = X - X[train].mean(axis=0)
    mu = y[train].mean()
    gram = z[train] @ z[train].T
    weights = np.linalg.solve(gram + alpha * np.eye(len(train)), y[train] - mu)
    return mu + (z[target] @ z[train].T) @ weights


def _device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(0.35)
    return device


def run_hybrid(cfg, args, runner_path):
    data = load(Path(cfg['data']), cfg['data_sha256'])
    folds = json.loads(Path(cfg['splits']).read_text())['folds']
    if len(folds) != 10:
        raise ValueError('Expected ten outer folds')
    split_hash = sha256_of(Path(cfg['splits']))
    frozen = json.loads(args.selection.read_text()) if args.selection else None
    if frozen is not None:
        if frozen['config'] != cfg or frozen['splits_sha256'] != split_hash:
            raise ValueError('Frozen selections do not match configuration or splits')
        keys = [(v['environment'], v['fold']) for v in frozen['selected']]
        expected = {(e, f['fold']) for e in data.environments for f in folds}
        if len(keys) != len(expected) or set(keys) != expected:
            raise ValueError('Frozen selection incomplete or duplicated')
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'provenance.json').write_text(json.dumps({
        'config': cfg, 'splits_sha256': split_hash,
        'runner_sha256': sha256_of(Path(runner_path)),
        'model_sha256': sha256_of(Path(background_rbf.__file__)),
        'solver_sha256': sha256_of(Path(kernel_ridge.__file__)),
        'selection_sha256': sha256_of(args.selection) if args.selection else None,
        'selection': 'mean inner-fold PCC; ties use first bandwidth',
        'refit': 'entire outer development set',
    }, indent=2))
    device = _device()
    selections, validation, rows, oof, failures = [], [], [], [], []
    for fold in folds:
        test = np.asarray(fold['test_idx'], dtype=int)
        dev = np.sort(np.r_[fold['train_idx'], fold['val_idx']])
        if len(set(test) & set(dev)) or len(set(test) | set(dev)) != data.n:
            raise ValueError('Invalid outer split')
        inner = [(dev[a], dev[b]) for a, b in KFold(n_splits=cfg['inner_folds'], shuffle=True, random_state=cfg['seed'] + fold['fold']).split(dev)]
        print(f"outer fold {fold['fold']}: dev={len(dev)} test={len(test)}", flush=True)
        for j, env in enumerate(data.environments):
            start = time.time(); y = data.Y[:, j]
            try:
                if frozen is None:
                    scores = np.empty((len(cfg['bandwidths']), len(inner)))
                    for k, (train, valid) in enumerate(inner):
                        feat = features(data.X, train, valid, cfg['pca_components'], cfg['seed'], device)
                        for i, h in enumerate(cfg['bandwidths']):
                            prepared = background_rbf.inputs(data.X, train, valid, feat, h, cfg.get('redundancy_shrinkage'))
                            pred, weights, loo = background_rbf.fit_predict(prepared, y[train], cfg, cfg['seed'])
                            scores[i, k] = pcc(y[valid], pred)
                            if not np.isfinite(scores[i, k]):
                                raise ValueError('Non-finite inner PCC')
                            validation.append({'environment': env, 'fold': fold['fold'], 'inner_fold': k, 'bandwidth': h, 'pcc': float(scores[i, k]), 'loo_mse': loo, 'weights': json.dumps(weights)})
                    winner = int(np.argmax(scores.mean(axis=1)))
                    selections.append({'environment': env, 'fold': fold['fold'], 'bandwidth': cfg['bandwidths'][winner], 'inner_pcc': float(scores[winner].mean())})
                    (args.out / 'selection.json').write_text(json.dumps({'config': cfg, 'splits_sha256': split_hash, 'selected': selections}, indent=2))
                    print(f'  env={env} selected in {time.time() - start:.1f}s', flush=True)
                else:
                    entry = next(v for v in frozen['selected'] if v['environment'] == env and v['fold'] == fold['fold'])
                    if entry['bandwidth'] not in cfg['bandwidths']:
                        raise ValueError('Selected bandwidth is not configured')
                    feat = features(data.X, dev, test, cfg['pca_components'], cfg['seed'], device)
                    prepared = background_rbf.inputs(data.X, dev, test, feat, entry['bandwidth'], cfg.get('redundancy_shrinkage'))
                    pred, weights, loo = background_rbf.fit_predict(prepared, y[dev], cfg, cfg['seed'])
                    metrics = {'pcc': pcc(y[test], pred), 'rmse': rmse(y[test], pred), 'mae': mae(y[test], pred)}
                    if not all(np.isfinite(v) for v in metrics.values()):
                        raise ValueError('Non-finite test metric')
                    rows.append({'environment': env, 'fold': fold['fold'], **metrics, 'bandwidth': entry['bandwidth'], 'inner_pcc': entry['inner_pcc'], 'additive_weight': weights[0], 'background_weight': weights[1], 'rbf_weight': weights[2], 'loo_mse': loo, 'seconds': round(time.time() - start, 2)})
                    oof.extend({'environment': env, 'fold': fold['fold'], 'material_index': int(idx), 'y_true': float(y[idx]), 'y_pred': float(v)} for idx, v in zip(test, pred))
                    print(f'  env={env} evaluated in {time.time() - start:.1f}s', flush=True)
            except Exception as exc:
                failures.append({'environment': env, 'fold': fold['fold'], 'reason': repr(exc)})
                print(f'FAILED {env} {fold["fold"]}: {exc}', flush=True)
                if isinstance(exc, torch.cuda.OutOfMemoryError):
                    raise
            finally:
                write_csv(args.out / 'validation.csv', ['environment', 'fold', 'inner_fold', 'bandwidth', 'pcc', 'loo_mse', 'weights'], validation)
                write_csv(args.out / 'per_fold.csv', ['environment', 'fold', 'pcc', 'rmse', 'mae', 'bandwidth', 'inner_pcc', 'additive_weight', 'background_weight', 'rbf_weight', 'loo_mse', 'seconds'], rows)
                write_csv(args.out / 'oof.csv', ['environment', 'fold', 'material_index', 'y_true', 'y_pred'], oof)
                write_csv(args.out / 'failures.csv', ['environment', 'fold', 'reason'], failures)
    if failures or (len(selections) != 40 if frozen is None else len(rows) != 40):
        raise RuntimeError('Incomplete run; inspect failures')
    if frozen is None:
        print('Development selection complete; no test predictions made.', flush=True)
        return
    summary = {'n_runs': len(rows), 'n_failures': 0, 'per_environment': {e: {m: {'mean': float(np.mean([r[m] for r in rows if r['environment'] == e])), 'sd': float(np.std([r[m] for r in rows if r['environment'] == e], ddof=1))} for m in ('pcc', 'rmse', 'mae')} for e in data.environments}}
    summary['macro'] = {m: float(np.mean([summary['per_environment'][e][m]['mean'] for e in data.environments])) for m in ('pcc', 'rmse', 'mae')}
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


def run_nested(cfg, args, runner_path):
    data = load(Path(cfg['data']), cfg['data_sha256'])
    folds = json.loads(Path(cfg['splits']).read_text())['folds']
    if len(folds) != 10:
        raise ValueError('Expected ten outer folds')
    candidates = design(cfg)
    frozen = json.loads(args.selection.read_text()) if args.selection else None
    if frozen is not None and (frozen['config'] != cfg or frozen['candidates'] != candidates or frozen['splits_sha256'] != sha256_of(Path(cfg['splits']))):
        raise ValueError('Frozen selections do not match configuration or splits')
    if frozen is not None:
        keys = [(v['environment'], v['fold']) for v in frozen['selected']]
        expected = {(e, f['fold']) for e in data.environments for f in folds}
        if len(keys) != len(expected) or set(keys) != expected:
            raise ValueError('Frozen selection is incomplete or duplicated')
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'provenance.json').write_text(json.dumps({'config': cfg, 'splits_sha256': sha256_of(Path(cfg['splits'])), 'runner_sha256': sha256_of(Path(runner_path)), 'core_sha256': sha256_of(Path(ridge_core.__file__)), 'selection_sha256': sha256_of(args.selection) if args.selection else None, 'selection': 'mean inner-fold PCC, ties use first candidate; RR uses same PCC rule', 'refit': 'entire outer development set; no test feedback', 'candidates': candidates}, indent=2))
    device = _device()
    rows, oof, validation, failures, selected_rows = [], [], [], [], []
    field = ['model', 'environment', 'fold', 'pcc', 'rmse', 'mae', 'selected', 'inner_pcc', 'seconds', 'failed', 'reason']
    for fold in folds:
        test = np.asarray(fold['test_idx'], dtype=int)
        dev = np.sort(np.r_[fold['train_idx'], fold['val_idx']])
        if len(set(test) & set(dev)) or len(set(test) | set(dev)) != data.n:
            raise ValueError('Invalid outer split')
        inner = [(dev[a], dev[b]) for a, b in KFold(n_splits=cfg['inner_folds'], shuffle=True, random_state=cfg['seed'] + fold['fold']).split(dev)]
        print(f"outer fold {fold['fold']}: dev={len(dev)} test={len(test)}", flush=True)
        for j, env in enumerate(data.environments):
            start = time.time(); y = data.Y[:, j]
            try:
                if frozen is not None:
                    match = [v for v in frozen['selected'] if v['environment'] == env and v['fold'] == fold['fold']]
                    if len(match) != 1:
                        raise ValueError('Missing or duplicate frozen selection')
                    entry = match[0]; i = entry['candidate']; alpha = entry['rr_alpha']; chosen = candidates[i]
                    x, xt, gram, cross = features(data.X, dev, test, chosen['pca_components'], cfg['seed'], device)
                    predictions = {'background': model_predict(x, xt, gram, cross, y[dev], chosen, cfg), 'rrblup': rr_predict(data.X, y, dev, test, alpha)}
                    for model, pred in predictions.items():
                        metrics = {'pcc': pcc(y[test], pred), 'rmse': rmse(y[test], pred), 'mae': mae(y[test], pred)}
                        if not all(np.isfinite(v) for v in metrics.values()):
                            raise ValueError('Non-finite test metric')
                        rows.append({'model': model, 'environment': env, 'fold': fold['fold'], **metrics, 'selected': i if model == 'background' else alpha, 'inner_pcc': entry['background_pcc'] if model == 'background' else entry['rr_pcc'], 'seconds': round(time.time() - start, 2), 'failed': False, 'reason': ''})
                        oof.extend({'model': model, 'environment': env, 'fold': fold['fold'], 'material_index': int(idx), 'y_true': float(y[idx]), 'y_pred': float(v)} for idx, v in zip(test, pred))
                    print(f"  env={env} evaluated in {time.time() - start:.1f}s", flush=True)
                    continue
                scores = np.empty((len(candidates), len(inner)))
                rr_scores = np.empty((len(cfg['rr_alphas']), len(inner)))
                for k, (train, valid) in enumerate(inner):
                    cache = {d: features(data.X, train, valid, d, cfg['seed'], device) for d in cfg['pca_components']}
                    for i, c in enumerate(candidates):
                        pred = model_predict(*cache[c['pca_components']], y[train], c, cfg)
                        scores[i, k] = pcc(y[valid], pred)
                        if not np.isfinite(scores[i, k]):
                            raise ValueError('Non-finite inner PCC')
                    for i, alpha in enumerate(cfg['rr_alphas']):
                        rr_scores[i, k] = pcc(y[valid], rr_predict(data.X, y, train, valid, alpha))
                        if not np.isfinite(rr_scores[i, k]):
                            raise ValueError('Non-finite RR inner PCC')
                for model, vals in (('background', scores), ('rrblup', rr_scores)):
                    validation.extend({'model': model, 'environment': env, 'fold': fold['fold'], 'candidate': i, 'inner_fold': k, 'pcc': float(v)} for i, row in enumerate(vals) for k, v in enumerate(row))
                i = int(np.argmax(scores.mean(axis=1))); r = int(np.argmax(rr_scores.mean(axis=1)))
                selected_rows.append({'environment': env, 'fold': fold['fold'], 'candidate': i, 'rr_alpha': cfg['rr_alphas'][r], 'background_pcc': float(scores[i].mean()), 'rr_pcc': float(rr_scores[r].mean())})
                (args.out / 'selection.json').write_text(json.dumps({'config': cfg, 'candidates': candidates, 'splits_sha256': sha256_of(Path(cfg['splits'])), 'selected': selected_rows}, indent=2))
                print(f"  env={env} selected in {time.time() - start:.1f}s", flush=True)
            except Exception as exc:
                failures.append({'environment': env, 'fold': fold['fold'], 'reason': repr(exc)})
                print(f'FAILED {env} {fold["fold"]}: {exc}', flush=True)
                if isinstance(exc, torch.cuda.OutOfMemoryError):
                    raise
            finally:
                write_csv(args.out / 'per_fold.csv', field, rows)
                write_csv(args.out / 'oof.csv', ['model', 'environment', 'fold', 'material_index', 'y_true', 'y_pred'], oof)
                write_csv(args.out / 'validation.csv', ['model', 'environment', 'fold', 'candidate', 'inner_fold', 'pcc'], validation)
                write_csv(args.out / 'failures.csv', ['environment', 'fold', 'reason'], failures)
    if frozen is None:
        if len(selected_rows) != 40 or failures:
            raise RuntimeError('Development selection incomplete; no test evaluation permitted')
        print('Development selection complete; no test predictions made.', flush=True)
        return
    summary = {'n_runs': len(rows), 'failures': failures}
    if len(rows) == 80 and not failures:
        summary['models'] = {m: {e: {metric: {'mean': float(np.mean([r[metric] for r in rows if r['model'] == m and r['environment'] == e])), 'sd': float(np.std([r[metric] for r in rows if r['model'] == m and r['environment'] == e], ddof=1))} for metric in ('pcc', 'rmse', 'mae')} for e in data.environments} for m in ('background', 'rrblup')}
        for m in summary['models']:
            summary['models'][m]['macro'] = {metric: float(np.mean([summary['models'][m][e][metric]['mean'] for e in data.environments])) for metric in ('pcc', 'rmse', 'mae')}
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


def run_ridge(cfg, args, runner_path):
    data = load(Path(cfg['data']), cfg['data_sha256'])
    folds = json.loads(Path(cfg['splits']).read_text())['folds']
    if len(folds) != 10:
        raise ValueError('Expected ten fixed folds')
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'provenance.json').write_text(json.dumps({'config': cfg, 'splits_sha256': sha256_of(Path(cfg['splits'])), 'runner_sha256': sha256_of(Path(runner_path)), 'core_sha256': sha256_of(Path(ridge_core.__file__)), 'selection': 'minimum validation RMSE; ties use first configured pair', 'loo': 'analytic leave-one-out conditional on training PCA and gate values; gates are not refit per held-out sample', 'kernel': 'centered X X.T / p * (1/global_penalty + q q.T/background_penalty), unit noise, unpenalized intercept', 'test': 'once after validation selection'}, indent=2))
    device = _device()
    rows, oof, failures, choices = [], [], [], []
    fields = ['environment', 'fold', 'pcc', 'rmse', 'mae', 'global_penalty', 'background_penalty', 'val_rmse', 'loo_mse', 'seconds', 'failed', 'reason']
    for f in folds:
        tr, va, te = (np.asarray(f[k], dtype=int) for k in ('train_idx', 'val_idx', 'test_idx'))
        if len(set(tr) | set(va) | set(te)) != data.n or set(tr) & set(va) or set(tr) & set(te) or set(va) & set(te):
            raise ValueError('Invalid split')
        centered = data.X - data.X[tr].mean(axis=0)
        pca = PCA(n_components=cfg['pca_components'], svd_solver='randomized', random_state=cfg['seed']).fit(centered[tr])
        scores = pca.transform(centered)
        scale = scores[tr].std(axis=0)
        if np.any(scale == 0):
            raise ValueError('Zero PCA scale')
        scores = (scores - scores[tr].mean(axis=0)) / scale
        x = torch.tensor(np.column_stack([np.ones(data.n), scores]), dtype=torch.float64, device=device)
        z = torch.tensor(centered, dtype=torch.float64, device=device)
        gram = z @ z[tr].T / data.p
        for j, env in enumerate(data.environments):
            t0 = time.time()
            try:
                torch.manual_seed(cfg['seed'])
                y = data.Y[:, j]
                ym, ys = y[tr].mean(), y[tr].std()
                if ys == 0:
                    raise ValueError('Constant training target')
                yt = torch.tensor(((y[tr] - ym) / ys)[:, None], dtype=torch.float64, device=device)
                selected = None
                for gl, bl in itertools.product(cfg['global_penalty'], cfg['background_penalty']):
                    torch.manual_seed(cfg['seed'])
                    w, alpha, intercept, loo = fit_candidate(x[tr], gram[tr], yt, gl, bl, cfg)
                    with torch.no_grad():
                        predval = (intercept + kernel(gates(x[va], w), gates(x[tr], w), gram[va], gl, bl) @ alpha).cpu().numpy().ravel() * ys + ym
                    vr = rmse(y[va], predval)
                    if not np.isfinite(vr):
                        raise ValueError('Non-finite validation metric')
                    choices.append({'environment': env, 'fold': f['fold'], 'global_penalty': gl, 'background_penalty': bl, 'val_rmse': vr, 'loo_mse': loo})
                    if selected is None or vr < selected[0]:
                        selected = (vr, gl, bl, w, alpha, intercept, loo)
                vr, gl, bl, w, alpha, intercept, loo = selected
                with torch.no_grad():
                    pred = (intercept + kernel(gates(x[te], w), gates(x[tr], w), gram[te], gl, bl) @ alpha).cpu().numpy().ravel() * ys + ym
                metrics = {'pcc': pcc(y[te], pred), 'rmse': rmse(y[te], pred), 'mae': mae(y[te], pred)}
                if not all(np.isfinite(v) for v in metrics.values()):
                    raise ValueError('Non-finite test metric')
                rows.append({'environment': env, 'fold': f['fold'], **metrics, 'global_penalty': gl, 'background_penalty': bl, 'val_rmse': vr, 'loo_mse': loo, 'seconds': round(time.time() - t0, 2), 'failed': False, 'reason': ''})
                oof.extend({'environment': env, 'fold': f['fold'], 'material_index': int(idx), 'y_true': float(y[idx]), 'y_pred': float(p)} for idx, p in zip(te, pred))
                print(f"env={env} fold={f['fold']} PCC={metrics['pcc']:.4f} {time.time() - t0:.1f}s", flush=True)
            except Exception as exc:
                failure = {'environment': env, 'fold': f['fold'], 'reason': repr(exc)}
                failures.append(failure)
                rows.append({'environment': env, 'fold': f['fold'], 'failed': True, 'reason': repr(exc)})
                print(f'FAILED {failure}', flush=True)
                if isinstance(exc, torch.cuda.OutOfMemoryError):
                    raise
            finally:
                write_csv(args.out / 'per_fold.csv', fields, rows)
                write_csv(args.out / 'oof.csv', ['environment', 'fold', 'material_index', 'y_true', 'y_pred'], oof)
                write_csv(args.out / 'failures.csv', ['environment', 'fold', 'reason'], failures)
                write_csv(args.out / 'validation.csv', ['environment', 'fold', 'global_penalty', 'background_penalty', 'val_rmse', 'loo_mse'], choices)
    summary = {'n_runs': len(rows), 'n_failures': len(failures)}
    if len(rows) == 40 and not failures:
        summary['per_environment'] = {e: {m: {'mean': float(np.mean([r[m] for r in rows if r['environment'] == e])), 'sd': float(np.std([r[m] for r in rows if r['environment'] == e], ddof=1))} for m in ('pcc', 'rmse', 'mae')} for e in data.environments}
        summary['macro'] = {m: float(np.mean([summary['per_environment'][e][m]['mean'] for e in data.environments])) for m in ('pcc', 'rmse', 'mae')}
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Non-nested background-conditioned ridge.')
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--selection', type=Path, default=None)
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text())
    return run_ridge(cfg, args, Path(__file__))


if __name__ == '__main__':
    raise SystemExit(main())