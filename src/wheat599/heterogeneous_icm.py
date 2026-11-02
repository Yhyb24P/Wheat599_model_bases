"""ICM with task-specific validation-selected shrinkage and shared material kernel."""
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
from wheat599.get_icm import write
from wheat599.kernel_baselines import kernels
from wheat599.metrics import pcc, rmse, mae
from wheat599.reml_icm import profile


def choose(X, Y, splits, cfg, device):
    """Tune shared kernel and individual penalties entirely inside the fit subset."""
    kernel_configs = list(itertools.product(cfg['bandwidths'], cfg['additive_fractions']))
    scores = np.empty((len(kernel_configs), len(cfg['ridge']), len(splits), Y.shape[1]))
    penalties = torch.tensor(cfg['ridge'], dtype=torch.float64, device=device)
    for j, (train, valid) in enumerate(splits):
        y = torch.tensor(Y[train], dtype=torch.float64, device=device)
        for h in cfg['bandwidths']:
            a, ac, r, rc = kernels(X, train, valid, h, device)
            for fraction in cfg['additive_fractions']:
                i = kernel_configs.index((h, fraction))
                gram, cross = fraction * a + (1 - fraction) * r, fraction * ac + (1 - fraction) * rc
                eig, vectors = torch.linalg.eigh(gram)
                if eig.min() < -1e-8:
                    raise ValueError('Non-PSD material kernel')
                inv = 1 / (eig.clamp_min(0)[None, :] + penalties[:, None])
                uy, uone, projected = vectors.T @ y, vectors.sum(0), cross @ vectors
                intercept = (inv * uone) @ uy / (inv * uone.square()).sum(1)[:, None]
                alpha = (uy[None, :, :] - uone[None, :, None] * intercept[:, None, :]) * inv[:, :, None]
                pred = (intercept[:, None, :] + torch.einsum('vn,lne->lve', projected, alpha)).cpu().numpy()
                for l in range(len(cfg['ridge'])):
                    scores[i, l, j] = [pcc(Y[valid, e], pred[l, :, e]) for e in range(Y.shape[1])]
    if not np.isfinite(scores).all():
        raise ValueError('Nonfinite base validation score')
    means = scores.mean(2)
    individual_lambdas = means.argmax(1)
    kernel_scores = np.take_along_axis(means, individual_lambdas[:, None, :], axis=1).mean((1, 2))
    i = int(np.argmax(kernel_scores))
    hetero = {'bandwidth': kernel_configs[i][0], 'fraction': kernel_configs[i][1],
              'ridge': [cfg['ridge'][l] for l in individual_lambdas[i]], 'base_pcc': float(kernel_scores[i])}
    # Preserve historical h, ridge, fraction ordering for the shared-penalty control.
    shared_configs = list(itertools.product(cfg['bandwidths'], cfg['ridge'], cfg['additive_fractions']))
    shared_scores = [float(means[kernel_configs.index((h, f)), cfg['ridge'].index(lam)].mean())
                     for h, lam, f in shared_configs]
    index = int(np.argmax(shared_scores))
    h, lam, fraction = shared_configs[index]
    homo = {'bandwidth': h, 'fraction': fraction, 'ridge': [lam] * Y.shape[1], 'base_pcc': shared_scores[index]}
    return hetero, homo, scores


def predict(X, Y, train, target, base, eta, device):
    a, ac, r, rc = kernels(X, train, target, base['bandwidth'], device)
    gram = base['fraction'] * a + (1 - base['fraction']) * r
    cross = base['fraction'] * ac + (1 - base['fraction']) * rc
    eig, vectors = torch.linalg.eigh(gram)
    if eig.min() < -1e-8:
        raise ValueError('Non-PSD material kernel')
    y = torch.tensor(Y[train], dtype=torch.float64, device=device)
    mean, scale = y.mean(0), y.std(0, unbiased=False)
    if torch.any(scale <= 0):
        raise ValueError('Constant phenotype')
    y = (y - mean) / scale
    z = y / torch.linalg.vector_norm(y, dim=0)
    corr = z.T @ z
    signal = (1 - eta) * torch.eye(Y.shape[1], dtype=y.dtype, device=device) + eta * corr
    residual = torch.diag(torch.tensor(base['ridge'], dtype=y.dtype, device=device))
    _, coefficients, intercept = profile(eig.clamp_min(0), vectors.T @ y, vectors.sum(0), signal, residual)
    return ((intercept + cross @ vectors @ coefficients @ signal) * scale + mean).cpu().numpy()


def run(cfg, args, runner):
    data = load(Path(cfg['data']), cfg['data_sha256'])
    folds = json.loads(Path(cfg['splits']).read_text())['folds']
    split_hash = sha256_of(Path(cfg['splits']))
    frozen = json.loads(args.selection.read_text()) if args.selection else None
    if frozen and (frozen['config'] != cfg or frozen['splits_sha256'] != split_hash
                   or not frozen['development_gate']['pass'] or len(frozen['selected']) != 10
                   or {s['fold'] for s in frozen['selected']} != {f['fold'] for f in folds}):
        raise ValueError('Frozen selection mismatch or development condition not met')
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'provenance.json').write_text(json.dumps({
        'config': cfg, 'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'splits_sha256': split_hash, 'model_sha256': sha256_of(Path(__file__)),
        'solver_sha256': sha256_of(Path(__file__).parent / 'reml_icm.py'),
        'runner_sha256': sha256_of(runner),
        'selection_sha256': sha256_of(args.selection) if args.selection else None,
        'rule': 'training-fold kernel and penalties; development-fold macro PCC for eta'}, indent=2))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(.35)
    rows, oof, failures, selected, base_records, split_records, base_scores = [], [], [], [], [], [], []
    start = time.perf_counter()
    for fold in folds:
        k = fold['fold']
        train_dev, test = np.array(fold['dev_idx']), np.array(fold['test_idx'])
        if set(train_dev) & set(test) or len(set(train_dev) | set(test)) != data.n:
            raise ValueError('Invalid outer split')
        print(f'fold {k}', flush=True)
        try:
            if frozen is None:
                for j, inner in enumerate(fold['inner']):
                    train, valid = np.array(inner['train_idx']), np.array(inner['val_idx'])
                    if set(train) & set(valid) or set(train) | set(valid) != set(train_dev):
                        raise ValueError('Invalid development split')
                    splits = [(train[a], train[b]) for a, b in KFold(
                        n_splits=3, shuffle=True, random_state=cfg['seed'] + 1000 + k * 10 + j).split(train)]
                    split_records.append({'fold': k, 'inner_fold': j,
                                          'splits': [{'train_idx': a.tolist(), 'val_idx': b.tolist()} for a, b in splits]})
                    hetero, homo, score = choose(data.X, data.Y, splits, cfg, device)
                    base_records.append({'fold': k, 'inner_fold': j, 'heterogeneous': hetero, 'homogeneous': homo})
                    kernel_configs = list(itertools.product(cfg['bandwidths'], cfg['additive_fractions']))
                    for i, (h, fraction) in enumerate(kernel_configs):
                        for l, lam in enumerate(cfg['ridge']):
                            for q in range(3):
                                for e, env in enumerate(data.environments):
                                    base_scores.append({'fold': k, 'inner_fold': j, 'base_fold': q, 'environment': env,
                                                        'bandwidth': h, 'fraction': fraction, 'ridge': lam, 'pcc': score[i, l, q, e]})
                    for model, base in (('heterogeneous', hetero), ('homogeneous', homo)):
                        for eta in cfg['etas']:
                            prediction = predict(data.X, data.Y, train, valid, base, eta, device)
                            for e, env in enumerate(data.environments):
                                metrics = {'pcc': pcc(data.Y[valid, e], prediction[:, e]),
                                           'rmse': rmse(data.Y[valid, e], prediction[:, e]),
                                           'mae': mae(data.Y[valid, e], prediction[:, e])}
                                if not all(np.isfinite(value) for value in metrics.values()):
                                    raise ValueError('Nonfinite validation metric')
                                rows.append({'model': model, 'fold': k, 'inner_fold': j, 'environment': env, 'eta': eta, **metrics})
                    print(f'  inner {j} complete', flush=True)
                choice = {'fold': k}
                for model in ('heterogeneous', 'homogeneous'):
                    means = [float(np.mean([r['pcc'] for r in rows if r['fold'] == k and r['model'] == model and r['eta'] == eta]))
                             for eta in cfg['etas']]
                    index = next(i for i, score in enumerate(means) if max(means) - score <= cfg['tie_tolerance'])
                    choice[model] = {'eta': cfg['etas'][index], 'inner_pcc': means[index]}
                selected.append(choice)
            else:
                choice = next(s for s in frozen['selected'] if s['fold'] == k)['heterogeneous']
                base, eta = choice['base'], choice['eta']
                if base['bandwidth'] not in cfg['bandwidths'] or base['fraction'] not in cfg['additive_fractions'] or eta not in cfg['etas'] or len(base['ridge']) != 4 or any(l not in cfg['ridge'] for l in base['ridge']):
                    raise ValueError('Invalid frozen parameters')
                prediction = predict(data.X, data.Y, train_dev, test, base, eta, device)
                for e, env in enumerate(data.environments):
                    metrics = {'pcc': pcc(data.Y[test, e], prediction[:, e]), 'rmse': rmse(data.Y[test, e], prediction[:, e]),
                               'mae': mae(data.Y[test, e], prediction[:, e])}
                    if not all(np.isfinite(value) for value in metrics.values()):
                        raise ValueError('Nonfinite test metric')
                    rows.append({'model': 'heterogeneous', 'fold': k, 'inner_fold': '', 'environment': env, 'eta': eta, **metrics})
                    oof.extend({'environment': env, 'fold': k, 'material_index': int(ix), 'y_true': float(data.Y[ix, e]), 'y_pred': float(value)}
                               for ix, value in zip(test, prediction[:, e]))
        except Exception as exc:
            failures.append({'fold': k, 'reason': repr(exc)})
            print(f'FAILED {k}: {exc}', flush=True)
            if isinstance(exc, torch.cuda.OutOfMemoryError):
                raise
        finally:
            write(args.out / ('per_fold.csv' if frozen else 'validation.csv'),
                  ['model', 'fold', 'inner_fold', 'environment', 'eta', 'pcc', 'rmse', 'mae'], rows)
            write(args.out / 'failures.csv', ['fold', 'reason'], failures)
            if frozen:
                write(args.out / 'oof.csv', ['environment', 'fold', 'material_index', 'y_true', 'y_pred'], oof)
            else:
                (args.out / 'training_splits.json').write_text(json.dumps(split_records))
                (args.out / 'base_choices.json').write_text(json.dumps(base_records, indent=2))
                with gzip.open(args.out / 'base_validation.csv.gz', 'wt', newline='') as file:
                    writer = csv.DictWriter(file, fieldnames=['fold', 'inner_fold', 'base_fold', 'environment', 'bandwidth', 'fraction', 'ridge', 'pcc'])
                    writer.writeheader(); writer.writerows(base_scores)
    if failures or len(rows) != (40 if frozen else 960):
        raise RuntimeError('Incomplete heterogeneous ICM run')
    summary = {'n_runs': len(rows), 'n_failures': 0, 'elapsed_seconds': time.perf_counter() - start}
    if frozen:
        per = {env: {metric: float(np.mean([r[metric] for r in rows if r['environment'] == env])) for metric in ('pcc', 'rmse', 'mae')}
               for env in data.environments}
        summary.update(per_environment=per, macro={metric: float(np.mean([per[env][metric] for env in data.environments]))
                                                   for metric in ('pcc', 'rmse', 'mae')})
    else:
        deltas = [s['heterogeneous']['inner_pcc'] - s['homogeneous']['inner_pcc'] for s in selected]
        gate = {'pcc_gain': float(np.mean(deltas)), 'folds_improved': sum(v > 0 for v in deltas), 'per_fold_delta': deltas}
        gate['pass'] = gate['pcc_gain'] >= cfg['minimum_development_gain'] and gate['folds_improved'] >= cfg['minimum_development_folds']
        summary['development_gate'] = gate
        if gate['pass']:
            for choice, fold in zip(selected, folds):
                splits = [(np.array(s['train_idx']), np.array(s['val_idx'])) for s in fold['inner']]
                hetero, homo, _ = choose(data.X, data.Y, splits, cfg, device)
                choice['heterogeneous']['base'], choice['homogeneous']['base'] = hetero, homo
        (args.out / 'selection.json').write_text(json.dumps({'config': cfg, 'splits_sha256': split_hash,
                                                            'development_gate': gate, 'selected': selected}, indent=2))
    summary['elapsed_seconds'] = time.perf_counter() - start
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)
