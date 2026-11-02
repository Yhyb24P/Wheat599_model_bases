"""Convex Huber regression with the existing material and ICM task kernels."""
import itertools
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize

from wheat599.data import load, sha256_of
from wheat599.get_icm import write
from wheat599.heterogeneous_icm import choose
from wheat599.kernel_baselines import kernels
from wheat599.metrics import pcc, rmse, mae
from wheat599.reml_icm import profile


def features(X, Y, train, target, base):
    a, ac, r, rc = kernels(X, train, target, base['bandwidth'], 'cpu')
    gram = base['fraction'] * a + (1 - base['fraction']) * r
    cross = base['fraction'] * ac + (1 - base['fraction']) * rc
    eig, vectors = torch.linalg.eigh(gram)
    if eig.min() < -1e-8:
        raise ValueError('Non-PSD material kernel')
    y = torch.tensor(Y[train], dtype=torch.float64)
    mean, scale = y.mean(0), y.std(0, unbiased=False)
    if torch.any(scale <= 0):
        raise ValueError('Constant phenotype')
    y = (y - mean) / scale
    z = y / torch.linalg.vector_norm(y, dim=0)
    keep = eig > 1e-12
    root = torch.sqrt(eig[keep])
    return {'eigenvalues': eig.clamp_min(0), 'vectors': vectors, 'Y': y,
            'mean': mean.numpy(), 'scale': scale.numpy(), 'C_Y': (z.T @ z).numpy(),
            'keep': keep, 'root': root.numpy(),
            'train_features': (vectors[:, keep] * root).numpy(),
            'target_features': ((cross @ vectors[:, keep]) / root).numpy()}


def huber_objective(values, design, factor, targets, ridge, delta, precondition=None, intercept_scale=1.):
    tasks = targets.shape[1]
    weights, intercept = values[:-tasks].reshape(design.shape[1], tasks), values[-tasks:]
    if precondition is not None:
        weights = weights / precondition
    intercept = intercept / intercept_scale
    residual = design @ weights @ factor.T + intercept - targets
    magnitude = abs(residual)
    score = np.clip(residual, -delta, delta)
    loss = np.where(magnitude <= delta, .5 * residual ** 2,
                    delta * (magnitude - .5 * delta)).sum() + .5 * ridge * np.sum(weights ** 2)
    weight_gradient = design.T @ score @ factor + ridge * weights
    if precondition is not None:
        weight_gradient = weight_gradient / precondition
    gradient = np.r_[weight_gradient.ravel(), score.sum(0) / intercept_scale]
    return loss / targets.size, gradient / targets.size


def fit_predict(prepared, ridge, eta, delta, cfg):
    y, vectors, eig = prepared['Y'], prepared['vectors'], prepared['eigenvalues']
    n, tasks = y.shape
    signal = (1 - eta) * np.eye(tasks) + eta * prepared['C_Y']
    task_eig, task_vectors = np.linalg.eigh(signal)
    factor = task_vectors * np.sqrt(task_eig)
    _, alpha, intercept = profile(eig, vectors.T @ y, vectors.sum(0),
                                  torch.as_tensor(signal), ridge * torch.eye(tasks, dtype=y.dtype))
    weights = prepared['root'][:, None] * alpha.numpy()[prepared['keep']] @ factor
    a, target = prepared['train_features'], prepared['target_features']
    normalized = y.numpy()
    precondition = np.sqrt(eig.numpy()[prepared['keep'], None] * task_eig[None, :] + ridge)
    intercept_scale = np.sqrt(n)
    initial = np.r_[(weights * precondition).ravel(), intercept.numpy() * intercept_scale]
    initial_residual = a @ weights @ factor.T + intercept.numpy() - normalized
    mad_scale = np.maximum(1.4826 * np.median(abs(initial_residual - np.median(initial_residual, axis=0)), axis=0), 1e-6)

    diagnostics = {'iterations': 0, 'optimizer_success': True, 'gradient_inf': 0., 'clipped_fraction': 0.}
    values = initial
    if delta is not None:
        thresholds = delta * mad_scale
        arguments = (a, factor, normalized, ridge, thresholds, precondition, intercept_scale)
        initial_loss = huber_objective(initial, *arguments)[0]
        result = minimize(huber_objective, initial, args=arguments, jac=True, method='L-BFGS-B',
                          options={'maxiter': cfg['max_iterations'], 'maxls': 40,
                                   'ftol': cfg['function_tolerance'], 'gtol': cfg['gradient_tolerance']})
        raw_gradient = np.r_[result.jac[:-tasks] * precondition.ravel(), result.jac[-tasks:] * intercept_scale]
        gradient = float(np.max(abs(raw_gradient)))
        if not np.isfinite(result.fun) or not np.isfinite(result.x).all() or gradient > cfg['accepted_gradient']:
            raise ValueError(f'Huber optimization failed to reach fixed accuracy: {result.message}; gradient={gradient}')
        values = result.x
        diagnostics.update(iterations=int(result.nit), optimizer_success=bool(result.success),
                           message=str(result.message), objective=float(result.fun),
                           initial_objective=float(initial_loss), gradient_inf=gradient,
                           residual_mad_scale=mad_scale.tolist(), thresholds=thresholds.tolist())
    weights, intercept = values[:-tasks].reshape(weights.shape) / precondition, values[-tasks:] / intercept_scale
    residual = a @ weights @ factor.T + intercept - normalized
    if delta is not None:
        diagnostics['clipped_fraction'] = float(np.mean(abs(residual) > thresholds))
    prediction = (intercept + target @ weights @ factor.T) * prepared['scale'] + prepared['mean']
    return prediction, diagnostics


def run(cfg, args, runner):
    data = load(Path(cfg['data']), cfg['data_sha256'])
    folds = json.loads(Path(cfg['splits']).read_text())['folds']
    split_hash = sha256_of(Path(cfg['splits']))
    source = Path(cfg['base_source'])
    source_info = json.loads((source / 'provenance.json').read_text())
    if source_info['splits_sha256'] != split_hash or any(source_info['config'][key] != cfg[key]
            for key in ('data', 'data_sha256', 'splits', 'seed', 'bandwidths', 'additive_fractions', 'ridge', 'etas')):
        raise ValueError('Training-only base source mismatch')
    source_hash = sha256_of(source / 'base_choices.json')
    base_records = json.loads((source / 'base_choices.json').read_text())
    frozen = json.loads(args.selection.read_text()) if args.selection else None
    if frozen and (frozen['config'] != cfg or frozen['splits_sha256'] != split_hash
                   or frozen['base_source_sha256'] != source_hash or not frozen['development_gate']['pass']
                   or len(frozen['selected']) != 10 or {s['fold'] for s in frozen['selected']} != {f['fold'] for f in folds}):
        raise ValueError('Frozen selection mismatch or development condition not met')
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'provenance.json').write_text(json.dumps({
        'config': cfg, 'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'splits_sha256': split_hash, 'base_source_sha256': source_hash,
        'model_sha256': sha256_of(Path(__file__)), 'solver_sha256': sha256_of(Path(__file__).parent / 'reml_icm.py'),
        'runner_sha256': sha256_of(runner),
        'selection_sha256': sha256_of(args.selection) if args.selection else None,
        'rule': 'convex Huber loss; train-only shared base; macro development PCC selects eta, delta, penalty multiplier'}, indent=2))
    candidates = list(itertools.product(cfg['etas'], cfg['huber_deltas'], cfg['ridge_multipliers']))
    rows, oof, failures, optimization, selected = [], [], [], [], []
    start = time.perf_counter()
    for fold in folds:
        k = fold['fold']
        print(f'fold {k}', flush=True)
        try:
            if frozen is None:
                for j, inner in enumerate(fold['inner']):
                    train, valid = np.array(inner['train_idx']), np.array(inner['val_idx'])
                    if set(train) & set(valid) or set(train) | set(valid) != set(fold['dev_idx']) or set(np.r_[train, valid]) & set(fold['test_idx']):
                        raise ValueError('Invalid development split')
                    base = next(b['homogeneous'] for b in base_records if b['fold'] == k and b['inner_fold'] == j)
                    prepared = features(data.X, data.Y, train, valid, base)
                    configurations = [('squared', eta, None, 1.) for eta in cfg['etas']]
                    configurations += [('huber', *candidate) for candidate in candidates]
                    for model, eta, delta, multiplier in configurations:
                        pred, diagnostics = fit_predict(prepared, base['ridge'][0] * multiplier, eta, delta, cfg)
                        optimization.append({'fold': k, 'inner_fold': j, 'model': model, 'eta': eta,
                                             'delta': delta, 'multiplier': multiplier, **diagnostics})
                        for e, env in enumerate(data.environments):
                            metrics = {'pcc': pcc(data.Y[valid, e], pred[:, e]),
                                       'rmse': rmse(data.Y[valid, e], pred[:, e]),
                                       'mae': mae(data.Y[valid, e], pred[:, e])}
                            if not all(np.isfinite(v) for v in metrics.values()):
                                raise ValueError('Nonfinite validation metric')
                            rows.append({'model': model, 'fold': k, 'inner_fold': j, 'environment': env,
                                         'eta': eta, 'delta': delta, 'multiplier': multiplier, **metrics})
                    print(f'  inner {j} complete', flush=True)
                entry = {'fold': k}
                for model in ('squared', 'huber'):
                    choices = [(eta, None, 1.) for eta in cfg['etas']] if model == 'squared' else candidates
                    means = [float(np.mean([r['pcc'] for r in rows if r['fold'] == k and r['model'] == model
                                            and (r['eta'], r['delta'], r['multiplier']) == c])) for c in choices]
                    index = next(i for i, score in enumerate(means) if max(means) - score <= cfg['tie_tolerance'])
                    eta, delta, multiplier = choices[index]
                    entry[model] = {'eta': eta, 'delta': delta, 'multiplier': multiplier, 'inner_pcc': means[index]}
                selected.append(entry)
            else:
                entry = next(s for s in frozen['selected'] if s['fold'] == k)
                choice, base = entry['huber'], entry['base']
                if (choice['eta'], choice['delta'], choice['multiplier']) not in candidates:
                    raise ValueError('Invalid frozen Huber choice')
                train, test = np.array(fold['dev_idx']), np.array(fold['test_idx'])
                if set(train) & set(test) or len(set(train) | set(test)) != data.n:
                    raise ValueError('Invalid outer split')
                prepared = features(data.X, data.Y, train, test, base)
                pred, diagnostics = fit_predict(prepared, base['ridge'][0] * choice['multiplier'], choice['eta'], choice['delta'], cfg)
                optimization.append({'fold': k, **diagnostics})
                for e, env in enumerate(data.environments):
                    metrics = {'pcc': pcc(data.Y[test, e], pred[:, e]), 'rmse': rmse(data.Y[test, e], pred[:, e]),
                               'mae': mae(data.Y[test, e], pred[:, e])}
                    if not all(np.isfinite(v) for v in metrics.values()):
                        raise ValueError('Nonfinite test metric')
                    rows.append({'model': 'huber', 'fold': k, 'inner_fold': '', 'environment': env,
                                 'eta': choice['eta'], 'delta': choice['delta'], 'multiplier': choice['multiplier'], **metrics})
                    oof.extend({'environment': env, 'fold': k, 'material_index': int(ix), 'y_true': float(data.Y[ix, e]),
                                'y_pred': float(value)} for ix, value in zip(test, pred[:, e]))
        except Exception as exc:
            failures.append({'fold': k, 'reason': repr(exc)})
            print(f'FAILED {k}: {exc}', flush=True)
        finally:
            write(args.out / ('per_fold.csv' if frozen else 'validation.csv'),
                  ['model', 'fold', 'inner_fold', 'environment', 'eta', 'delta', 'multiplier', 'pcc', 'rmse', 'mae'], rows)
            write(args.out / 'failures.csv', ['fold', 'reason'], failures)
            (args.out / 'optimization.json').write_text(json.dumps(optimization, indent=2))
            if frozen:
                write(args.out / 'oof.csv', ['environment', 'fold', 'material_index', 'y_true', 'y_pred'], oof)
    if failures or len(rows) != (40 if frozen else 3360):
        raise RuntimeError('Incomplete robust ICM run')
    summary = {'n_runs': len(rows), 'n_failures': 0}
    if frozen:
        per = {env: {metric: float(np.mean([r[metric] for r in rows if r['environment'] == env]))
                     for metric in ('pcc', 'rmse', 'mae')} for env in data.environments}
        summary.update(per_environment=per, macro={metric: float(np.mean([per[env][metric] for env in data.environments]))
                                                   for metric in ('pcc', 'rmse', 'mae')})
    else:
        deltas = [s['huber']['inner_pcc'] - s['squared']['inner_pcc'] for s in selected]
        gate = {'pcc_gain': float(np.mean(deltas)), 'folds_improved': sum(v > 0 for v in deltas), 'per_fold_delta': deltas}
        gate['pass'] = gate['pcc_gain'] >= cfg['minimum_development_gain'] and gate['folds_improved'] >= cfg['minimum_development_folds']
        summary['development_gate'] = gate
        if gate['pass']:
            for entry, fold in zip(selected, folds):
                splits = [(np.array(s['train_idx']), np.array(s['val_idx'])) for s in fold['inner']]
                _, entry['base'], _ = choose(data.X, data.Y, splits, cfg, 'cpu')
        (args.out / 'selection.json').write_text(json.dumps({'config': cfg, 'splits_sha256': split_hash,
            'base_source_sha256': source_hash, 'development_gate': gate, 'selected': selected}, indent=2))
    summary['elapsed_seconds'] = time.perf_counter() - start
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)
