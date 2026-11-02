"""Evaluate fixed top-five ICM averaging on held-out development materials."""
import csv
import gzip
import itertools
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import KFold

from wheat599.data import load, sha256_of
from wheat599.get_icm import prepare, spectral_predict, write
from wheat599.kernel_baselines import kernels
from wheat599.metrics import pcc, rmse, mae


def candidates(cfg):
    return [c for c in itertools.product(cfg['bandwidths'], cfg['additive_fractions'], cfg['ridge'], cfg['etas'])
            if c[1] != 1 or c[0] == cfg['bandwidths'][0]]


def best_indices(scores, configurations, cfg):
    remaining = list(range(len(configurations)))
    chosen = []
    for _ in range(cfg['ensemble_size']):
        best = max(scores[i] for i in remaining)
        eligible = [i for i in remaining if best - scores[i] <= cfg['tie_tolerance']]
        index = min(eligible, key=lambda i: (configurations[i][3], i))
        chosen.append(index)
        remaining.remove(index)
    return chosen


def ranking(X, Y, splits, configurations, cfg, device):
    ids = {c: i for i, c in enumerate(configurations)}
    scores = np.empty((len(configurations), len(splits), Y.shape[1]))
    for j, (train, valid) in enumerate(splits):
        for h, fraction in dict.fromkeys((c[0], c[1]) for c in configurations):
            a, ac, r, rc = kernels(X, train, valid, h, device)
            gram, cross = fraction * a + (1 - fraction) * r, fraction * ac + (1 - fraction) * rc
            eig, vectors = torch.linalg.eigh(gram)
            projected = cross @ vectors
            for lam in cfg['ridge']:
                stats = prepare(gram, Y[train], lam, cfg, (eig, vectors))
                for eta in cfg['etas']:
                    pred, _ = spectral_predict(stats, cross, lam, eta, 0, projected)
                    scores[ids[(h, fraction, lam, eta)], j] = [pcc(Y[valid, e], pred[:, e]) for e in range(Y.shape[1])]
    if not np.isfinite(scores).all():
        raise ValueError('Incomplete or nonfinite ranking scores')
    return scores.mean((1, 2))


def predictions(X, Y, train, target, selected, cfg, device):
    result = []
    for h, fraction, lam, eta in selected:
        a, ac, r, rc = kernels(X, train, target, h, device)
        gram, cross = fraction * a + (1 - fraction) * r, fraction * ac + (1 - fraction) * rc
        stats = prepare(gram, Y[train], lam, cfg)
        pred, _ = spectral_predict(stats, cross, lam, eta, 0)
        result.append(pred)
    return np.stack(result)


def source_choices(cfg, folds, configurations):
    source = Path(cfg['validation_source'])
    provenance = json.loads((source / 'provenance.json').read_text())
    for key in ('data', 'data_sha256', 'splits', 'bandwidths', 'additive_fractions', 'ridge', 'etas'):
        if provenance['config'][key] != cfg[key]:
            raise ValueError(f'Source configuration mismatch: {key}')
    if provenance['splits_sha256'] != sha256_of(Path(cfg['splits'])) or provenance['solver_sha256'] != sha256_of(Path(__file__).parent / 'get_icm.py'):
        raise ValueError('Source split or solver mismatch')
    selected, hashes = [], {}
    ids = {c: i for i, c in enumerate(configurations)}
    for fold in folds:
        k = fold['fold']
        path = source / f'validation_fold_{k}.csv.gz'
        hashes[str(k)] = sha256_of(path)
        scores = np.full((len(configurations), 3, 4), np.nan)
        with gzip.open(path, 'rt') as file:
            for row in csv.DictReader(file):
                if float(row['beta']) != 0:
                    continue
                c = tuple(float(row[key]) for key in ('bandwidth', 'fraction', 'ridge', 'eta'))
                if c not in ids:
                    continue  # duplicate pure-additive bandwidths are excluded by design
                i, j, e = ids[c], int(row['inner_fold']), ['1', '2', '4', '5'].index(row['environment'])
                if not np.isnan(scores[i, j, e]):
                    raise ValueError('Duplicate source score')
                scores[i, j, e] = float(row['pcc'])
        if not np.isfinite(scores).all():
            raise ValueError('Incomplete source scores')
        means = scores.mean((1, 2))
        chosen = best_indices(means, configurations, cfg)
        selected.append({'fold': k, 'candidates': [configurations[i] for i in chosen],
                         'ranking_scores': [float(means[i]) for i in chosen]})
    return selected, hashes


def run(cfg, args, runner):
    data = load(Path(cfg['data']), cfg['data_sha256'])
    folds = json.loads(Path(cfg['splits']).read_text())['folds']
    configurations = candidates(cfg)
    selected, hashes = source_choices(cfg, folds, configurations)
    frozen = json.loads(args.selection.read_text()) if args.selection else None
    split_hash = sha256_of(Path(cfg['splits']))
    if frozen and (frozen['config'] != cfg or frozen['splits_sha256'] != split_hash
                   or frozen['source_validation_sha256'] != hashes or not frozen['development_gate']['pass']
                   or json.loads(json.dumps(selected)) != frozen['selected']):
        raise ValueError('Frozen selection mismatch or development gate failed')
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'provenance.json').write_text(json.dumps({
        'config': cfg, 'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'splits_sha256': split_hash, 'source_validation_sha256': hashes,
        'model_sha256': sha256_of(Path(__file__)), 'solver_sha256': sha256_of(Path(__file__).parent / 'get_icm.py'),
        'runner_sha256': sha256_of(runner),
        'selection_sha256': sha256_of(args.selection) if args.selection else None,
        'candidate_count': len(configurations), 'rule': 'fixed equal average of five training-CV-ranked ICMs'}, indent=2))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(.35)
    rows, oof, failures, inner_choices, training_splits, ranking_rows = [], [], [], [], [], []
    start = time.perf_counter()
    for fold in folds:
        k = fold['fold']
        dev, test = np.array(fold['dev_idx']), np.array(fold['test_idx'])
        if set(dev) & set(test) or len(set(dev) | set(test)) != data.n:
            raise ValueError('Invalid outer split')
        print(f'fold {k}', flush=True)
        try:
            if frozen is None:
                for j, inner in enumerate(fold['inner']):
                    train, valid = np.array(inner['train_idx']), np.array(inner['val_idx'])
                    if set(train) & set(valid) or set(train) | set(valid) != set(dev):
                        raise ValueError('Invalid development split')
                    splits = [(train[a], train[b]) for a, b in KFold(
                        n_splits=3, shuffle=True, random_state=cfg['seed'] + 1000 + k * 10 + j).split(train)]
                    training_splits.append({'fold': k, 'inner_fold': j,
                                            'splits': [{'train_idx': a.tolist(), 'val_idx': b.tolist()} for a, b in splits]})
                    means = ranking(data.X, data.Y, splits, configurations, cfg, device)
                    indices = best_indices(means, configurations, cfg)
                    chosen = [configurations[i] for i in indices]
                    inner_choices.append({'fold': k, 'inner_fold': j, 'candidates': chosen})
                    ranking_rows.extend({'fold': k, 'inner_fold': j, 'candidate': i,
                                         'bandwidth': c[0], 'fraction': c[1], 'ridge': c[2],
                                         'eta': c[3], 'pcc': float(means[i])} for i, c in enumerate(configurations))
                    pred = predictions(data.X, data.Y, train, valid, chosen, cfg, device)
                    for model, values in (('single', pred[0]), ('average5', pred.mean(0))):
                        for e, env in enumerate(data.environments):
                            metrics = {'pcc': pcc(data.Y[valid, e], values[:, e]),
                                       'rmse': rmse(data.Y[valid, e], values[:, e]),
                                       'mae': mae(data.Y[valid, e], values[:, e])}
                            if not all(np.isfinite(v) for v in metrics.values()):
                                raise ValueError('Nonfinite development metric')
                            rows.append({'model': model, 'environment': env, 'fold': k, 'inner_fold': j, **metrics})
                    print(f'  inner {j} complete', flush=True)
            else:
                choice = next(s for s in frozen['selected'] if s['fold'] == k)
                pred = predictions(data.X, data.Y, dev, test, choice['candidates'], cfg, device).mean(0)
                for e, env in enumerate(data.environments):
                    metrics = {'pcc': pcc(data.Y[test, e], pred[:, e]), 'rmse': rmse(data.Y[test, e], pred[:, e]),
                               'mae': mae(data.Y[test, e], pred[:, e])}
                    if not all(np.isfinite(v) for v in metrics.values()):
                        raise ValueError('Nonfinite test metric')
                    rows.append({'model': 'average5', 'environment': env, 'fold': k, 'inner_fold': '', **metrics})
                    oof.extend({'environment': env, 'fold': k, 'material_index': int(ix),
                                'y_true': float(data.Y[ix, e]), 'y_pred': float(value)} for ix, value in zip(test, pred[:, e]))
        except Exception as exc:
            failures.append({'fold': k, 'reason': repr(exc)})
            print(f'FAILED {k}: {exc}', flush=True)
            if isinstance(exc, torch.cuda.OutOfMemoryError):
                raise
        finally:
            write(args.out / ('per_fold.csv' if frozen else 'validation.csv'),
                  ['model', 'environment', 'fold', 'inner_fold', 'pcc', 'rmse', 'mae'], rows)
            write(args.out / 'failures.csv', ['fold', 'reason'], failures)
            if frozen:
                write(args.out / 'oof.csv', ['environment', 'fold', 'material_index', 'y_true', 'y_pred'], oof)
            else:
                (args.out / 'training_splits.json').write_text(json.dumps(training_splits))
                (args.out / 'inner_choices.json').write_text(json.dumps(inner_choices, indent=2))
                with gzip.open(args.out / 'training_rankings.csv.gz', 'wt', newline='') as file:
                    writer = csv.DictWriter(file, fieldnames=['fold', 'inner_fold', 'candidate', 'bandwidth', 'fraction', 'ridge', 'eta', 'pcc'])
                    writer.writeheader(); writer.writerows(ranking_rows)
    if failures or len(rows) != (40 if frozen else 240):
        raise RuntimeError('Incomplete averaging run')
    summary = {'n_runs': len(rows), 'n_failures': 0, 'elapsed_seconds': time.perf_counter() - start}
    if frozen:
        per = {env: {metric: float(np.mean([r[metric] for r in rows if r['environment'] == env]))
                     for metric in ('pcc', 'rmse', 'mae')} for env in data.environments}
        summary.update(per_environment=per, macro={metric: float(np.mean([per[env][metric] for env in data.environments]))
                                                   for metric in ('pcc', 'rmse', 'mae')})
    else:
        paired = [float(np.mean([r['pcc'] for r in rows if r['fold'] == k and r['model'] == 'average5'])
                        - np.mean([r['pcc'] for r in rows if r['fold'] == k and r['model'] == 'single'])) for k in range(10)]
        gate = {'pcc_gain': float(np.mean(paired)), 'folds_improved': sum(value > 0 for value in paired),
                'per_fold_delta': paired}
        gate['pass'] = gate['pcc_gain'] >= cfg['minimum_development_gain'] and gate['folds_improved'] >= cfg['minimum_development_folds']
        summary['development_gate'] = gate
        (args.out / 'selection.json').write_text(json.dumps({
            'config': cfg, 'splits_sha256': split_hash, 'source_validation_sha256': hashes,
            'development_gate': gate, 'selected': selected}, indent=2))
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)
