"""RBF-SVR, Random Forest, XGBoost and LightGBM under the nested protocol.

Marker-only; every scaler and every choice is fitted inside the current
training set. Random models use the protocol seeds 42 and 2026: inner
selection scores the seed-averaged validation predictions, the outer test
prediction is the seed average, and per-seed predictions are kept aside.
Fits are spread over process workers (joblib loky); estimators themselves
run single-threaded so the configured job count is the true parallelism.
"""
import csv
import importlib.metadata
import itertools
import json
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from genopred.data.wheat599 import load, sha256_of
from genopred.evaluation.metrics import pcc, rmse, mae

RANDOM_MODELS = ('rf', 'xgb', 'lgbm')


def candidates(cfg, model):
    space = cfg['search'][model]
    keys = list(space)
    return [dict(zip(keys, values)) for values in itertools.product(*(space[k] for k in keys))]


def model_seeds(cfg, model):
    return tuple(cfg['seeds']) if model in RANDOM_MODELS else (cfg['seeds'][0],)


def fit_predict(model, spec, X, y, train, target, seed):
    if model == 'svr':
        scaler = StandardScaler().fit(X[train])
        est = SVR(kernel='rbf', C=spec['C'], gamma=spec['gamma'], epsilon=spec['epsilon'])
        return est.fit(scaler.transform(X[train]), y[train]).predict(scaler.transform(X[target]))
    if model == 'rf':
        est = RandomForestRegressor(
            n_estimators=spec['n_estimators'], max_features=spec['max_features'],
            min_samples_leaf=spec['min_samples_leaf'], random_state=seed, n_jobs=1)
    elif model == 'xgb':
        from xgboost import XGBRegressor
        est = XGBRegressor(
            n_estimators=spec['n_estimators'], max_depth=spec['max_depth'],
            learning_rate=spec['learning_rate'], subsample=spec['subsample'],
            colsample_bytree=spec['colsample_bytree'], random_state=seed,
            tree_method='hist', n_jobs=1, verbosity=0)
    elif model == 'lgbm':
        from lightgbm import LGBMRegressor
        est = LGBMRegressor(
            n_estimators=spec['n_estimators'], num_leaves=spec['num_leaves'],
            learning_rate=spec['learning_rate'], feature_fraction=spec['feature_fraction'],
            bagging_fraction=spec['bagging_fraction'], bagging_freq=1,
            random_state=seed, n_jobs=1, verbose=-1)
    else:
        raise ValueError(f'unknown model {model!r}')
    return est.fit(X[train], y[train]).predict(X[target])


def inner_score(job):
    model, spec, seeds, X, y, train, valid = job
    start = time.time()
    preds = [fit_predict(model, spec, X, y, train, valid, seed) for seed in seeds]
    return pcc(y[valid], np.mean(preds, axis=0)), time.time() - start


def outer_predict(job):
    model, spec, seeds, X, y, dev, test = job
    start = time.time()
    return {seed: fit_predict(model, spec, X, y, dev, test, seed) for seed in seeds}, time.time() - start


def write(path, fields, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        w.writeheader(); w.writerows(rows)


def run(cfg, args, runner_path):
    data = load(Path(cfg['data']), cfg['data_sha256'])
    folds = json.loads(Path(cfg['splits']).read_text())['folds']
    split_hash = sha256_of(Path(cfg['splits']))
    models = tuple(cfg['search'])
    space = {m: candidates(cfg, m) for m in models}
    frozen = json.loads(args.selection.read_text()) if args.selection else None
    if frozen and (frozen['config'] != cfg or frozen['splits_sha256'] != split_hash):
        raise ValueError('Frozen selection does not match config or splits')
    expected = {(m, e, f['fold']) for m in models for e in data.environments for f in folds}
    if frozen:
        keys = [(v['model'], v['environment'], v['fold']) for v in frozen['selected']]
        if len(keys) != len(expected) or set(keys) != expected:
            raise ValueError('Incomplete frozen selection')
    args.out.mkdir(parents=True, exist_ok=False)
    versions = {name: importlib.metadata.version(name) for name in ('numpy', 'scikit-learn', 'xgboost', 'lightgbm')}
    (args.out/'provenance.json').write_text(json.dumps({
        'config': cfg, 'splits_sha256': split_hash,
        'runner_sha256': sha256_of(runner_path), 'model_sha256': sha256_of(Path(__file__)),
        'selection_sha256': sha256_of(args.selection) if args.selection else None,
        'versions': versions, 'jobs': cfg['jobs'], 'parallel': 'joblib loky processes; estimators n_jobs=1',
        'rule': 'mean inner PCC of seed-averaged predictions; first candidate on tie; full dev refit',
    }, indent=2))
    selected, valrows, rows, seedrows, oof, seedoof, failures, timing = [], [], [], [], [], [], [], []
    for fold in folds:
        k = fold['fold']; test = np.asarray(fold['test_idx']); dev = np.asarray(fold['dev_idx'])
        if len(set(dev) & set(test)) or len(set(dev) | set(test)) != data.n:
            raise ValueError('Invalid outer split')
        inner = [(np.asarray(v['train_idx']), np.asarray(v['val_idx'])) for v in fold['inner']]
        print(f'fold {k}: dev={len(dev)} test={len(test)}', flush=True)
        start = time.time()
        try:
            if frozen is None:
                jobs, keys = [], []
                for model in models:
                    for env in data.environments:
                        e = data.environments.index(env); y = data.Y[:, e]
                        for j, (train, valid) in enumerate(inner):
                            for i, spec in enumerate(space[model]):
                                jobs.append((model, spec, model_seeds(cfg, model), data.X, y, train, valid))
                                keys.append((model, env, j, i))
                results = Parallel(n_jobs=cfg['jobs'], backend='loky')(
                    delayed(inner_score)(job) for job in jobs)
                scores = {}
                for (model, env, j, i), (score, seconds) in zip(keys, results):
                    if not np.isfinite(score):
                        raise ValueError(f'non-finite inner PCC {model} env={env} inner={j}')
                    scores.setdefault((model, env), np.zeros((len(space[model]), len(inner))))[i, j] = score
                    valrows.append({'model': model, 'environment': env, 'fold': k, 'inner_fold': j,
                                    'candidate': i, 'pcc': float(score), 'seconds': round(seconds, 3)})
                for model in models:
                    for env in data.environments:
                        i = int(np.argmax(scores[(model, env)].mean(1)))
                        selected.append({'model': model, 'environment': env, 'fold': k, 'candidate': i,
                                         'spec': space[model][i], 'inner_pcc': float(scores[(model, env)][i].mean())})
                (args.out/'selection.json').write_text(json.dumps(
                    {'config': cfg, 'splits_sha256': split_hash, 'selected': selected}, indent=2))
                timing.append({'fold': k, 'phase': 'selection', 'seconds': round(time.time() - start, 1),
                               'jobs': len(jobs)})
                print(f'  selection complete in {time.time()-start:.1f}s', flush=True)
            else:
                entries = {}
                for model in models:
                    for env in data.environments:
                        entry = next(v for v in frozen['selected']
                                     if (v['model'], v['environment'], v['fold']) == (model, env, k))
                        spec = space[model][entry['candidate']]
                        if spec != entry['spec']:
                            raise ValueError('Frozen spec does not match candidate order')
                        entries[(model, env)] = (spec, entry)
                jobs = [(model, spec, model_seeds(cfg, model), data.X, data.Y[:, data.environments.index(env)],
                         dev, test) for (model, env), (spec, _) in entries.items()]
                results = Parallel(n_jobs=cfg['jobs'], backend='loky')(
                    delayed(outer_predict)(job) for job in jobs)
                for ((model, env), (spec, entry)), (preds, seconds) in zip(entries.items(), results):
                    y = data.Y[:, data.environments.index(env)]
                    mean_pred = np.mean(list(preds.values()), axis=0)
                    met = {'pcc': pcc(y[test], mean_pred), 'rmse': rmse(y[test], mean_pred), 'mae': mae(y[test], mean_pred)}
                    if not all(np.isfinite(v) for v in met.values()):
                        raise ValueError(f'non-finite or undefined outer metric {model} env={env}')
                    rows.append({'model': model, 'environment': env, 'fold': k, **met, 'candidate': entry['candidate'],
                                 'inner_pcc': entry['inner_pcc'], 'fit_seconds': round(seconds, 2)})
                    for seed, pred in preds.items():
                        seedrows.append({'model': model, 'environment': env, 'fold': k, 'seed': seed,
                                         'pcc': pcc(y[test], pred), 'rmse': rmse(y[test], pred), 'mae': mae(y[test], pred)})
                        seedoof.extend({'model': model, 'environment': env, 'fold': k, 'seed': seed,
                                        'material_index': int(idx), 'y_true': float(y[idx]), 'y_pred': float(v)}
                                       for idx, v in zip(test, pred))
                    oof.extend({'model': model, 'environment': env, 'fold': k, 'material_index': int(idx),
                                'y_true': float(y[idx]), 'y_pred': float(v)} for idx, v in zip(test, mean_pred))
                timing.append({'fold': k, 'phase': 'outer', 'seconds': round(time.time() - start, 1), 'jobs': len(jobs)})
                print(f'  outer evaluation complete in {time.time()-start:.1f}s', flush=True)
        except Exception as exc:
            failures.append({'fold': k, 'reason': repr(exc)}); print(f'FAILED {k}: {exc}', flush=True)
        finally:
            write(args.out/'validation.csv',
                  ['model', 'environment', 'fold', 'inner_fold', 'candidate', 'pcc', 'seconds'], valrows)
            write(args.out/'per_fold.csv',
                  ['model', 'environment', 'fold', 'pcc', 'rmse', 'mae', 'candidate', 'inner_pcc', 'fit_seconds'], rows)
            write(args.out/'per_fold_seed.csv',
                  ['model', 'environment', 'fold', 'seed', 'pcc', 'rmse', 'mae'], seedrows)
            write(args.out/'oof.csv',
                  ['model', 'environment', 'fold', 'material_index', 'y_true', 'y_pred'], oof)
            write(args.out/'oof_seed.csv',
                  ['model', 'environment', 'fold', 'seed', 'material_index', 'y_true', 'y_pred'], seedoof)
            write(args.out/'failures.csv', ['fold', 'reason'], failures)
            (args.out/'timing.json').write_text(json.dumps({'jobs': cfg['jobs'], 'folds': timing}, indent=2))
    expected_runs = len(models) * len(data.environments) * len(folds)
    if failures or (len(selected) != expected_runs if frozen is None else len(rows) != expected_runs):
        raise RuntimeError('Incomplete run')
    if frozen is None:
        print('Development selection complete; no test predictions made.', flush=True)
        return
    summary = {'n_runs': len(rows), 'failures': [], 'models': {}}
    for m in models:
        summary['models'][m] = {
            'per_environment': {e: {metric: float(np.mean([r[metric] for r in rows if r['model'] == m and r['environment'] == e]))
                                    for metric in ('pcc', 'rmse', 'mae')} for e in data.environments}}
        summary['models'][m]['macro'] = {
            metric: float(np.mean([summary['models'][m]['per_environment'][e][metric] for e in data.environments]))
            for metric in ('pcc', 'rmse', 'mae')}
        summary['models'][m]['seeds'] = list(model_seeds(cfg, m))
    (args.out/'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)
