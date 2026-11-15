"""Wheat2000 classical baselines under the frozen nested protocol.

Estimator helpers are reused from classical_baselines; kernel ridge reuses
kernel_baselines.kernels with the same unpenalized-intercept solve batched
over the eight traits. GBLUP/RR-BLUP is the existing centered linear-kernel
run (results/wheat2000_gblup_final_20261105) and is not repeated here.
"""
import csv
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

import numpy as np
import torch
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

from wheat599.classical_baselines import (RANDOM_MODELS, candidates, inner_score,
                                          model_seeds, outer_predict)
from wheat599.data import sha256_of
from wheat599.kernel_baselines import kernels
from wheat599.metrics import pcc, rmse, mae
from wheat599.wheat2000_data import load_wheat2000

KERNEL_MODELS = ('gaussian', 'additive_rbf')
CPU = torch.device('cpu')


def kernel_space(cfg):
    return {'gaussian': [(h, r, 0.0) for h in cfg['bandwidths'] for r in cfg['ridge']],
            'additive_rbf': [(h, r, f) for h in cfg['bandwidths'] for r in cfg['ridge']
                             for f in cfg['additive_fractions']]}


def kernel_predict_multi(mats, y, fraction, ridge):
    """kernel_baselines.predict with all traits as right-hand sides."""
    a, ac, r, rc = mats
    y = torch.as_tensor(y, dtype=a.dtype, device=a.device)
    k = fraction * a + (1 - fraction) * r
    cross = fraction * ac + (1 - fraction) * rc
    chol = torch.linalg.cholesky(k + ridge * torch.eye(len(y), dtype=y.dtype, device=y.device))
    both = torch.cholesky_solve(torch.cat([y, torch.ones(len(y), 1, dtype=y.dtype, device=y.device)], 1), chol)
    intercept = both[:, :-1].sum(0) / both[:, -1].sum()
    return (intercept + cross @ (both[:, :-1] - intercept * both[:, -1:])).cpu().numpy()


def kernel_inner(job):
    torch.set_num_threads(1)
    X, Y, train, valid, h, specs = job
    start = time.time()
    mats = kernels(X, train, valid, h, CPU)
    scores = {(m, r, f): [pcc(Y[valid, e], pred[:, e]) for e in range(Y.shape[1])]
              for m, r, f in specs
              for pred in [kernel_predict_multi(mats, Y[train], f, r)]}
    return scores, time.time() - start


def kernel_outer(job):
    torch.set_num_threads(1)
    X, Y, dev, test, entries = job
    start = time.time()
    pred = np.empty((len(test), Y.shape[1]))
    for h, r, f in sorted({(e[2], e[3], e[4]) for e in entries}):
        p = kernel_predict_multi(kernels(X, dev, test, h, CPU), Y[dev], f, r)
        for e in entries:
            if (e[2], e[3], e[4]) == (h, r, f):
                pred[:, e[0]] = p[:, e[0]]
    return pred, time.time() - start


def write(path, fields, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def run(cfg, args, runner_path):
    torch.set_num_threads(1)
    data = load_wheat2000(cfg['data'], cfg['data_sha256'], cfg['cache'], cfg['cache_sha256'])
    split_path = Path(cfg['splits'])
    if sha256_of(split_path) != cfg['splits_sha256']:
        raise ValueError('Frozen Wheat2000 splits SHA differs')
    splits = json.loads(split_path.read_text())
    if list(data.trait_names) != cfg['traits'] or splits['trait_names'] != cfg['traits']:
        raise ValueError('All eight traits and their order must match the frozen protocol')
    folds, traits = splits['folds'], data.trait_names
    models = tuple(cfg['search']) + KERNEL_MODELS
    space = {m: candidates(cfg, m) for m in cfg['search']}
    kspace = kernel_space(cfg)
    config_sha = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    sources = {str(p): sha256_of(Path(p)) for p in
               (Path(__file__), Path(runner_path), Path(args.config),
                Path(__file__).with_name('classical_baselines.py'),
                Path(__file__).with_name('kernel_baselines.py'),
                Path(__file__).with_name('wheat2000_data.py'),
                Path(__file__).with_name('metrics.py'), Path(__file__).with_name('data.py'))}
    frozen = json.loads(args.selection.read_text()) if args.selection else None
    if frozen:
        if (frozen['config_sha256'] != config_sha or frozen['splits_sha256'] != cfg['splits_sha256']
                or frozen['source_sha256'] != sources):
            raise ValueError('Frozen selection/config/sources differ')
        expected = {(m, t, f['fold']) for m in models for t in traits for f in folds}
        if ({(v['model'], v['trait'], v['fold']) for v in frozen['selected']} != expected
                or len(frozen['selected']) != len(expected)):
            raise ValueError('Frozen selection incomplete or duplicated')
    args.out.mkdir(parents=True, exist_ok=False)
    versions = {n: importlib.metadata.version(n)
                for n in ('numpy', 'scikit-learn', 'xgboost', 'lightgbm', 'torch', 'joblib')}
    (args.out / 'provenance.json').write_text(json.dumps({
        'config': cfg, 'config_sha256': config_sha, 'splits_sha256': cfg['splits_sha256'],
        'data_sha256': cfg['data_sha256'], 'cache_sha256': cfg['cache_sha256'],
        'source_git_commit': cfg['source_git_commit'], 'source_sha256': sources,
        'selection_sha256': sha256_of(args.selection) if args.selection else None,
        'versions': versions, 'jobs': cfg['jobs'],
        'parallel': 'joblib loky processes; estimators and torch single-threaded',
        'rule': 'mean inner PCC of seed-averaged predictions; first candidate on tie; full dev refit; test scored once',
        'gblup_rrblup': 'reused from results/wheat2000_gblup_final_20261105 (same centered linear kernel)',
    }, indent=2))
    selected, valrows, rows, seedrows, oof, seedoof, failures, timing = [], [], [], [], [], [], [], []
    state = {'status': 'running', 'phase': 'final' if frozen else 'development', 'folds_complete': []}
    (args.out / 'execution_status.json').write_text(json.dumps(state, indent=2))
    try:
        with threadpool_limits(limits=1):
            for fold in folds:
                k = fold['fold']
                dev, test = np.asarray(fold['dev_idx']), np.asarray(fold['test_idx'])
                inner = [(np.asarray(v['train_idx']), np.asarray(v['val_idx'])) for v in fold['inner']]
                if len(set(dev) & set(test)) or len(set(dev) | set(test)) != len(data.X):
                    raise ValueError('Invalid outer split')
                print(f'fold {k}: dev={len(dev)} test={len(test)}', flush=True)
                start = time.time()
                if frozen is None:
                    jobs, keys = [], []
                    for m in cfg['search']:
                        for e, trait in enumerate(traits):
                            for j, (train, valid) in enumerate(inner):
                                for i, spec in enumerate(space[m]):
                                    jobs.append((m, spec, model_seeds(cfg, m), data.X, data.Y[:, e], train, valid))
                                    keys.append((m, e, j, i))
                    scores = {}
                    results = Parallel(n_jobs=cfg['jobs'], backend='loky')(
                        delayed(inner_score)(job) for job in jobs)
                    for (m, e, j, i), (score, seconds) in zip(keys, results):
                        if not np.isfinite(score):
                            raise ValueError(f'non-finite inner PCC {m} trait={traits[e]} inner={j}')
                        scores.setdefault((m, e), np.zeros((len(space[m]), len(inner))))[i, j] = score
                        valrows.append({'model': m, 'trait': traits[e], 'fold': k, 'inner_fold': j,
                                        'candidate': i, 'pcc': float(score), 'seconds': round(seconds, 3)})
                    kjobs = [(data.X, data.Y, train, valid, h,
                              [(m, r, f) for m in KERNEL_MODELS for (hh, r, f) in kspace[m] if hh == h])
                             for j, (train, valid) in enumerate(inner) for h in cfg['bandwidths']]
                    kresults = Parallel(n_jobs=cfg['jobs'], backend='loky')(
                        delayed(kernel_inner)(job) for job in kjobs)
                    for j, (train, valid) in enumerate(inner):
                        for h, (kscores, seconds) in zip(cfg['bandwidths'], kresults[j * len(cfg['bandwidths']):
                                                                             (j + 1) * len(cfg['bandwidths'])]):
                            for m in KERNEL_MODELS:
                                for i, (hh, r, f) in enumerate(kspace[m]):
                                    if hh != h:
                                        continue
                                    for e, trait in enumerate(traits):
                                        score = kscores[(m, r, f)][e]
                                        if not np.isfinite(score):
                                            raise ValueError(f'non-finite inner PCC {m} trait={trait} inner={j}')
                                        scores.setdefault((m, e), np.zeros((len(kspace[m]), len(inner))))[i, j] = score
                                        valrows.append({'model': m, 'trait': trait, 'fold': k, 'inner_fold': j,
                                                        'candidate': i, 'pcc': float(score), 'seconds': round(seconds, 3)})
                    for m in models:
                        for e, trait in enumerate(traits):
                            i = int(np.argmax(scores[(m, e)].mean(1)))
                            spec = space[m][i] if m in space else dict(zip(
                                ('bandwidth', 'ridge', 'additive_fraction'), kspace[m][i]))
                            selected.append({'model': m, 'trait': trait, 'fold': k, 'candidate': i,
                                             'spec': spec, 'inner_pcc': float(scores[(m, e)][i].mean())})
                    (args.out / 'selection.json').write_text(json.dumps(
                        {'config_sha256': config_sha, 'splits_sha256': cfg['splits_sha256'],
                         'source_sha256': sources, 'selected': selected}, indent=2))
                    timing.append({'fold': k, 'phase': 'selection', 'seconds': round(time.time() - start, 1)})
                    print(f'  selection complete in {time.time() - start:.1f}s', flush=True)
                else:
                    entries = {}
                    for m in models:
                        for e, trait in enumerate(traits):
                            entry = next(v for v in frozen['selected']
                                         if (v['model'], v['trait'], v['fold']) == (m, trait, k))
                            if m in space:
                                if space[m][entry['candidate']] != entry['spec']:
                                    raise ValueError('Frozen spec does not match candidate order')
                            else:
                                spec = kspace[m][entry['candidate']]
                                if dict(zip(('bandwidth', 'ridge', 'additive_fraction'), spec)) != entry['spec']:
                                    raise ValueError('Frozen kernel spec does not match candidate order')
                            entries[(m, e)] = entry
                    cjobs = [(m, entries[(m, e)]['spec'], model_seeds(cfg, m), data.X, data.Y[:, e], dev, test)
                             for m in cfg['search'] for e in range(len(traits))]
                    cresults = Parallel(n_jobs=cfg['jobs'], backend='loky')(
                        delayed(outer_predict)(job) for job in cjobs)
                    kentries = [(e, m, entries[(m, e)]['spec']['bandwidth'], entries[(m, e)]['spec']['ridge'],
                                 entries[(m, e)]['spec']['additive_fraction'])
                                for m in KERNEL_MODELS for e in range(len(traits))]
                    kpred, kseconds = kernel_outer((data.X, data.Y, dev, test, kentries))
                    fold_preds = {}
                    for (m, e), (preds, seconds) in zip(
                            [(m, e) for m in cfg['search'] for e in range(len(traits))], cresults):
                        fold_preds[(m, e)] = (preds, seconds)
                    for m in KERNEL_MODELS:
                        for e in range(len(traits)):
                            fold_preds[(m, e)] = ({None: kpred[:, e]}, kseconds)
                    for (m, e), (preds, seconds) in fold_preds.items():
                        trait = traits[e]
                        entry = entries[(m, e)]
                        mean_pred = np.mean(list(preds.values()), axis=0)
                        met = {'pcc': pcc(data.Y[test, e], mean_pred), 'rmse': rmse(data.Y[test, e], mean_pred),
                               'mae': mae(data.Y[test, e], mean_pred)}
                        if not all(np.isfinite(v) for v in met.values()):
                            raise ValueError(f'non-finite or undefined outer metric {m} trait={trait}')
                        rows.append({'model': m, 'trait': trait, 'fold': k, **met,
                                     'candidate': entry['candidate'], 'inner_pcc': entry['inner_pcc'],
                                     'fit_seconds': round(seconds, 2)})
                        for seed, pred in preds.items():
                            if seed is None:
                                continue
                            seedrows.append({'model': m, 'trait': trait, 'fold': k, 'seed': seed,
                                             'pcc': pcc(data.Y[test, e], pred), 'rmse': rmse(data.Y[test, e], pred),
                                             'mae': mae(data.Y[test, e], pred)})
                            seedoof.extend({'model': m, 'trait': trait, 'fold': k, 'seed': seed,
                                            'material_index': int(idx), 'y_true': float(data.Y[idx, e]),
                                            'y_pred': float(v)} for idx, v in zip(test, pred))
                        oof.extend({'model': m, 'trait': trait, 'fold': k, 'material_index': int(idx),
                                    'genotype_id': str(data.genotype_ids[idx]),
                                    'phenotype_id': str(data.phenotype_ids[idx]),
                                    'y_true': float(data.Y[idx, e]), 'y_pred': float(v)}
                                   for idx, v in zip(test, mean_pred))
                    timing.append({'fold': k, 'phase': 'outer', 'seconds': round(time.time() - start, 1)})
                    print(f'  outer evaluation complete in {time.time() - start:.1f}s', flush=True)
                state['folds_complete'].append(k)
                (args.out / 'execution_status.json').write_text(json.dumps(state, indent=2))
                write(args.out / 'validation.csv',
                      ['model', 'trait', 'fold', 'inner_fold', 'candidate', 'pcc', 'seconds'], valrows)
                write(args.out / 'per_fold.csv',
                      ['model', 'trait', 'fold', 'pcc', 'rmse', 'mae', 'candidate', 'inner_pcc', 'fit_seconds'], rows)
                write(args.out / 'per_fold_seed.csv', ['model', 'trait', 'fold', 'seed', 'pcc', 'rmse', 'mae'], seedrows)
                write(args.out / 'oof.csv',
                      ['model', 'trait', 'fold', 'material_index', 'genotype_id', 'phenotype_id', 'y_true', 'y_pred'], oof)
                write(args.out / 'oof_seed.csv',
                      ['model', 'trait', 'fold', 'seed', 'material_index', 'y_true', 'y_pred'], seedoof)
                write(args.out / 'failures.csv', ['fold', 'reason'], failures)
                (args.out / 'timing.json').write_text(json.dumps({'jobs': cfg['jobs'], 'folds': timing}, indent=2))
        expected_runs = len(models) * len(traits) * len(folds)
        if failures or (len(selected) != expected_runs if frozen is None else len(rows) != expected_runs):
            raise RuntimeError('Incomplete run')
        if frozen is None:
            (args.out / 'summary.json').write_text(json.dumps(
                {'n_trait_folds': len(selected),
                 'macro_development_pcc': float(np.mean([r['inner_pcc'] for r in selected])),
                 'scope': 'Conditional inner validation only; no test predictions'}, indent=2))
            state['status'] = 'complete'
            (args.out / 'execution_status.json').write_text(json.dumps(state, indent=2))
            print('Development selection complete; no test predictions made.', flush=True)
            return
        summary = {'n_runs': len(rows), 'failures': [], 'models': {}}
        for m in models:
            per_trait = {t: {metric: float(np.mean([r[metric] for r in rows if r['model'] == m and r['trait'] == t]))
                             for metric in ('pcc', 'rmse', 'mae')} for t in traits}
            summary['models'][m] = {
                'macro': {metric: float(np.mean([per_trait[t][metric] for t in traits])) for metric in ('pcc', 'rmse', 'mae')},
                'per_trait': per_trait, 'seeds': list(model_seeds(cfg, m)) if m in space else [cfg['seeds'][0]]}
        (args.out / 'summary.json').write_text(json.dumps(summary, indent=2))
        pooled = {}
        for m in models:
            per_trait = {}
            for t in traits:
                trows = [r for r in oof if r['model'] == m and r['trait'] == t]
                y = np.array([r['y_true'] for r in trows])
                pred = np.array([r['y_pred'] for r in trows])
                per_trait[t] = {'pcc': float(pcc(y, pred)), 'rmse': float(rmse(y, pred)), 'mae': float(mae(y, pred))}
            pooled[m] = {'per_trait': per_trait,
                         'macro': {metric: float(np.mean([per_trait[t][metric] for t in traits]))
                                   for metric in ('pcc', 'rmse', 'mae')}}
        (args.out / 'pooled_oof_metrics.json').write_text(json.dumps(pooled, indent=2))
        state['status'] = 'complete'
        (args.out / 'execution_status.json').write_text(json.dumps(state, indent=2))
        print(json.dumps(summary), flush=True)
    except Exception as error:
        failures.append({'fold': state['folds_complete'][-1] if state['folds_complete'] else None,
                         'reason': repr(error)})
        state['status'] = 'failed'
        (args.out / 'execution_status.json').write_text(json.dumps(state, indent=2))
        write(args.out / 'failures.csv', ['fold', 'reason'], failures)
        raise
