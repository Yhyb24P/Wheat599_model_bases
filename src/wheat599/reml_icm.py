"""Multi-output kernel regression with training-only REML covariance fitting.

After material eigendecomposition, each independent spectral row has
covariance eig[i] * signal + residual. The four intercepts are profiled by
GLS; their information determinant is included in the restricted likelihood.
"""
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
from wheat599.kernel_baselines import kernels
from wheat599.metrics import pcc, rmse, mae


def covariance(parameters, size, structure, floor):
    if structure == 'diagonal':
        return torch.diag(torch.exp(2 * parameters) + floor)
    indices = torch.tril_indices(size, size)
    factor = parameters.new_zeros((size, size))
    factor[indices[0], indices[1]] = parameters
    diagonal = torch.diagonal(factor)
    factor = factor - torch.diag(diagonal) + torch.diag(torch.exp(diagonal))
    return factor @ factor.T + floor * torch.eye(size, dtype=factor.dtype)


def encode(matrix, structure):
    if structure == 'diagonal':
        return torch.log(torch.diagonal(matrix)) / 2
    factor = torch.linalg.cholesky(matrix)
    indices = torch.tril_indices(len(matrix), len(matrix))
    values = factor[indices[0], indices[1]].clone()
    diagonal = indices[0] == indices[1]
    values[diagonal] = torch.log(values[diagonal])
    return values


def profile(eigenvalues, transformed_y, transformed_one, signal, residual):
    """Return normalized negative REML, spectral dual coefficients, intercept."""
    covariances = eigenvalues[:, None, None] * signal + residual
    factors = torch.linalg.cholesky(covariances)
    inverse = torch.cholesky_inverse(factors)
    information = (transformed_one.square()[:, None, None] * inverse).sum(0)
    inverse_y = (inverse @ transformed_y[:, :, None]).squeeze(2)
    rhs = (transformed_one[:, None] * inverse_y).sum(0)
    intercept = torch.linalg.solve(information, rhs)
    errors = transformed_y - transformed_one[:, None] * intercept
    coefficients = (inverse @ errors[:, :, None]).squeeze(2)
    determinant = 2 * torch.log(torch.diagonal(factors, dim1=1, dim2=2)).sum()
    objective = (determinant + torch.linalg.slogdet(information)[1]
                 + (errors * coefficients).sum()) / (2 * transformed_y.numel())
    return objective, coefficients, intercept


def fit(eigensystem, y, structure, cfg):
    eigenvalues, vectors = eigensystem
    y = torch.as_tensor(y, dtype=torch.float64)
    mean, scale = y.mean(0), y.std(0, unbiased=False)
    if torch.any(scale <= 0):
        raise ValueError('Constant training phenotype')
    normalized = (y - mean) / scale
    transformed_y = vectors.T @ normalized
    transformed_one = vectors.sum(0)
    size = y.shape[1]
    n_signal = size * (size + 1) // 2
    empirical = normalized.T @ normalized / len(y)
    identity = torch.eye(size, dtype=torch.float64)
    trials = []
    for fraction in cfg['initial_signal_fractions']:
        signal = fraction * (.9 * empirical + .1 * identity)
        residual = (1 - fraction) * identity
        initial = torch.cat([encode(signal, 'full'), encode(residual, structure)]).numpy()

        def objective(values):
            parameters = torch.tensor(values, dtype=torch.float64, requires_grad=True)
            signal = covariance(parameters[:n_signal], size, 'full', cfg['covariance_floor'])
            residual = covariance(parameters[n_signal:], size, structure, cfg['covariance_floor'])
            loss, _, _ = profile(eigenvalues, transformed_y, transformed_one, signal, residual)
            gradient, = torch.autograd.grad(loss, parameters)
            if not torch.isfinite(loss) or not torch.isfinite(gradient).all():
                raise ValueError('Nonfinite REML objective or gradient')
            return float(loss.detach()), gradient.detach().numpy()

        result = minimize(objective, initial, jac=True, method='L-BFGS-B',
                          bounds=[(-10., 10.)] * len(initial),
                          options={'maxiter': cfg['max_iterations'],
                                   'ftol': cfg['function_tolerance'],
                                   'gtol': cfg['gradient_tolerance'], 'maxls': 40})
        if not np.isfinite(result.fun) or not np.isfinite(result.x).all():
            raise ValueError('Nonfinite covariance estimate')
        trials.append(result)
    best = min(trials, key=lambda result: result.fun)
    parameters = torch.as_tensor(best.x, dtype=torch.float64)
    signal = covariance(parameters[:n_signal], size, 'full', cfg['covariance_floor'])
    residual = covariance(parameters[n_signal:], size, structure, cfg['covariance_floor'])
    loss, coefficients, intercept = profile(eigenvalues, transformed_y, transformed_one, signal, residual)
    diagnostics = {
        'reml_per_observation': float(loss), 'signal_covariance': signal.tolist(),
        'residual_covariance': residual.tolist(), 'intercept_standardized': intercept.tolist(),
        'starts': [{'objective': float(r.fun), 'iterations': int(r.nit),
                    'success': bool(r.success), 'message': str(r.message),
                    'gradient_inf': float(np.max(abs(r.jac)))} for r in trials],
    }
    return (vectors @ coefficients @ signal, intercept, mean, scale), diagnostics


def predict(state, cross):
    dual, intercept, mean, scale = state
    return ((intercept + cross @ dual) * scale + mean).numpy()


def run(cfg, args, runner):
    data = load(Path(cfg['data']), cfg['data_sha256'])
    folds = json.loads(Path(cfg['splits']).read_text())['folds']
    split_hash = sha256_of(Path(cfg['splits']))
    candidates = [(h, fraction, residual)
                  for h, fraction, residual in itertools.product(
                      cfg['bandwidths'], cfg['additive_fractions'], cfg['residual_structures'])
                  if fraction != 1 or h == cfg['bandwidths'][0]]
    frozen = json.loads(args.selection.read_text()) if args.selection else None
    if frozen and (frozen['config'] != cfg or frozen['splits_sha256'] != split_hash
                   or len(frozen['selected']) != len(folds)
                   or {s['fold'] for s in frozen['selected']} != {f['fold'] for f in folds}):
        raise ValueError('Frozen protocol mismatch')
    args.out.mkdir(parents=True, exist_ok=False)
    provenance = {'config': cfg, 'git_commit': subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], text=True).strip(), 'splits_sha256': split_hash,
        'model_sha256': sha256_of(Path(__file__)), 'runner_sha256': sha256_of(runner),
        'selection_sha256': sha256_of(args.selection) if args.selection else None,
        'candidate_count': len(candidates), 'device': 'cpu',
        'rule': 'mean three-fold four-environment PCC; first candidate on ties'}
    (args.out / 'provenance.json').write_text(json.dumps(provenance, indent=2))
    validation, selected, rows, oof, calibration, failures = [], [], [], [], [], []
    start = time.perf_counter()
    for fold in folds:
        k = fold['fold']
        print(f'fold {k}', flush=True)
        try:
            if frozen is None:
                scores = np.empty((len(candidates), len(fold['inner']), len(data.environments)))
                for j, inner in enumerate(fold['inner']):
                    train, valid = np.array(inner['train_idx']), np.array(inner['val_idx'])
                    for h, fraction in dict.fromkeys((c[0], c[1]) for c in candidates):
                        a, ac, r, rc = kernels(data.X, train, valid, h, 'cpu')
                        gram, cross = fraction * a + (1 - fraction) * r, fraction * ac + (1 - fraction) * rc
                        eig, vectors = torch.linalg.eigh(gram)
                        if eig.min() < -1e-8:
                            raise ValueError('Non-PSD material kernel')
                        for structure in cfg['residual_structures']:
                            i = candidates.index((h, fraction, structure))
                            state, diag = fit((eig.clamp_min(0), vectors), data.Y[train], structure, cfg)
                            pred = predict(state, cross)
                            for e, env in enumerate(data.environments):
                                score = pcc(data.Y[valid, e], pred[:, e])
                                if not np.isfinite(score):
                                    raise ValueError('Nonfinite validation score')
                                scores[i, j, e] = score
                                validation.append({'fold': k, 'inner_fold': j, 'environment': env,
                                                   'bandwidth': h, 'fraction': fraction,
                                                   'residual_structure': structure, 'pcc': score})
                            calibration.append({'fold': k, 'inner_fold': j, 'bandwidth': h,
                                                'fraction': fraction, 'residual_structure': structure, **diag})
                    print(f'  inner {j} complete', flush=True)
                means = scores.mean((1, 2))
                best = next(i for i, score in enumerate(means)
                            if means.max() - score <= cfg['tie_tolerance'])
                selected.append({'fold': k, 'candidate': candidates[best], 'inner_pcc': float(means[best])})
                (args.out / 'selection.json').write_text(json.dumps(
                    {'config': cfg, 'splits_sha256': split_hash, 'selected': selected}, indent=2))
            else:
                choice = next(s for s in frozen['selected'] if s['fold'] == k)
                h, fraction, structure = choice['candidate']
                if (h, fraction, structure) not in candidates:
                    raise ValueError('Invalid frozen candidate')
                train, test = np.array(fold['dev_idx']), np.array(fold['test_idx'])
                a, ac, r, rc = kernels(data.X, train, test, h, 'cpu')
                gram, cross = fraction * a + (1 - fraction) * r, fraction * ac + (1 - fraction) * rc
                eig, vectors = torch.linalg.eigh(gram)
                if eig.min() < -1e-8:
                    raise ValueError('Non-PSD material kernel')
                state, diag = fit((eig.clamp_min(0), vectors), data.Y[train], structure, cfg)
                pred = predict(state, cross)
                calibration.append({'fold': k, 'bandwidth': h, 'fraction': fraction,
                                    'residual_structure': structure, **diag})
                for e, env in enumerate(data.environments):
                    metrics = {'pcc': pcc(data.Y[test, e], pred[:, e]),
                               'rmse': rmse(data.Y[test, e], pred[:, e]),
                               'mae': mae(data.Y[test, e], pred[:, e])}
                    if not all(np.isfinite(v) for v in metrics.values()):
                        raise ValueError('Nonfinite test metric')
                    rows.append({'environment': env, 'fold': k, **metrics, 'bandwidth': h,
                                 'fraction': fraction, 'residual_structure': structure,
                                 'inner_pcc': choice['inner_pcc']})
                    oof.extend({'environment': env, 'fold': k, 'material_index': int(idx),
                                'y_true': float(data.Y[idx, e]), 'y_pred': float(value)}
                               for idx, value in zip(test, pred[:, e]))
        except Exception as exc:
            failures.append({'fold': k, 'reason': repr(exc)})
            print(f'FAILED {k}: {exc}', flush=True)
        finally:
            write(args.out / 'validation.csv', ['fold', 'inner_fold', 'environment', 'bandwidth',
                  'fraction', 'residual_structure', 'pcc'], validation)
            write(args.out / 'per_fold.csv', ['environment', 'fold', 'pcc', 'rmse', 'mae',
                  'bandwidth', 'fraction', 'residual_structure', 'inner_pcc'], rows)
            write(args.out / 'oof.csv', ['environment', 'fold', 'material_index', 'y_true', 'y_pred'], oof)
            write(args.out / 'failures.csv', ['fold', 'reason'], failures)
            (args.out / 'calibration.json').write_text(json.dumps(calibration, indent=2))
    if failures or (len(selected) != 10 if frozen is None else len(rows) != 40):
        raise RuntimeError('Incomplete run')
    summary = {'n_runs': len(rows), 'n_failures': 0, 'elapsed_seconds': time.perf_counter() - start}
    if frozen:
        per = {env: {metric: {'mean': float(np.mean([r[metric] for r in rows if r['environment'] == env])),
                              'sd': float(np.std([r[metric] for r in rows if r['environment'] == env], ddof=1))}
                     for metric in ('pcc', 'rmse', 'mae')} for env in data.environments}
        summary.update(per_environment=per, macro={metric: float(np.mean(
            [per[env][metric]['mean'] for env in data.environments])) for metric in ('pcc', 'rmse', 'mae')})
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)
