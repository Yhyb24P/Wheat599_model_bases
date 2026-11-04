"""Fixed synthetic diagnostic: shared-support regression, support oracle and kernels.

The unknown-support solver receives only observed context and query genotypes.
Generating pairs enter a separate oracle fit and post-fit support diagnostics.
No function reads Wheat599 or retrains the frozen neural predictors.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
import warnings

import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import MultiTaskElasticNet
import torch

from wheat599.matrix_tasks import sample_task
from wheat599.matrix_training import task_panel
from wheat599.sparse_interaction import SparseInteraction


def error_record(prediction, target):
    difference = np.asarray(prediction) - np.asarray(target)
    if not np.isfinite(difference).all():
        raise FloatingPointError('Nonfinite diagnostic prediction or target')
    mse = float(np.mean(difference ** 2))
    if not math.isfinite(mse):
        raise FloatingPointError('Nonfinite diagnostic MSE')
    return {'mse': mse,
            'per_environment_mse': np.mean(difference ** 2, axis=0).tolist()}


def support_oracle(solver, pairs, query, ratios, maximum):
    """Same convex objective restricted to known pairs; no generating effects used."""
    context, cross = solver.phi(pairs, query=query)
    w, future = context.cpu().numpy(), cross.cpu().numpy()
    y = solver.y.cpu().numpy()
    a = solver.a.cpu().numpy()
    operator = solver.operator.cpu().numpy()
    half = solver._half.cpu().numpy()
    gram = a @ a.T + solver.lambda_a * np.eye(len(a))
    transformed = half.T @ w
    transformed_y = half.T @ y
    fitted = None
    records = []
    for ratio in ratios:
        strength = ratio * maximum
        if len(pairs) and maximum > 0:
            if fitted is None:
                fitted = MultiTaskElasticNet(fit_intercept=False, warm_start=True,
                    max_iter=200000, tol=1e-12, selection='cyclic')
            fitted.set_params(alpha=strength + solver.lambda_e,
                              l1_ratio=strength / (strength + solver.lambda_e))
            with warnings.catch_warnings():
                warnings.simplefilter('error', ConvergenceWarning)
                fitted.fit(transformed, transformed_y)
            coefficients = fitted.coef_.T.copy()
            iterations = int(fitted.n_iter_)
        else:
            coefficients = np.zeros((len(pairs), y.shape[1]))
            iterations = 0
        residual = y - w @ coefficients
        score = w.T @ operator @ residual / len(y)
        cnorm = np.linalg.norm(coefficients, axis=1)
        snorm = np.linalg.norm(score, axis=1)
        nonzero = cnorm > 0
        deviations = np.maximum(snorm - strength, 0)
        if nonzero.any():
            deviations[nonzero] = np.linalg.norm(score[nonzero]
                - solver.lambda_e * coefficients[nonzero]
                - strength * coefficients[nonzero] / cnorm[nonzero, None], axis=1)
        penalty = strength * cnorm.sum() + .5 * solver.lambda_e * (coefficients ** 2).sum()
        objective = float((residual * (operator @ residual)).sum() / (2 * len(y)) + penalty)
        gap = float(penalty + np.maximum(snorm - strength, 0).dot(
            np.maximum(snorm - strength, 0)) / (2 * solver.lambda_e)
            - (coefficients * score).sum())
        kkt = float(deviations.max()) if len(deviations) else 0.
        if (kkt > solver.kkt_tolerance * max(1., maximum)
                or gap > solver.dual_gap_tolerance * max(1., objective)
                or gap < -1e-10 * max(1., objective)):
            raise RuntimeError(f'Oracle solver unresolved: KKT={kkt}, gap={gap}')
        b = a.T @ np.linalg.solve(gram, residual)
        prediction = np.asarray(query) @ b / math.sqrt(solver.markers) + future @ coefficients
        records.append({'ratio': ratio, 'prediction': prediction, 'objective': objective,
                        'kkt_absolute': kkt, 'dual_gap': gap,
                        'kkt_scope': 'given_support_only', 'iterations': iterations,
                        'active_pairs': int(nonzero.sum())})
    return records


def dense_pair_predictions(solver, query, ridges):
    """L2 reference using the SAME standardized distinct-pair features as group fits."""
    n, m = solver.materials, len(query)
    kernel = torch.zeros((n, n), dtype=solver.dtype, device=solver.device)
    cross = torch.zeros((m, n), dtype=solver.dtype, device=solver.device)
    for start in range(0, len(solver.pairs), solver.block_size):
        pairs = solver.pairs[start:start + solver.block_size]
        context, future = solver.phi(pairs, query=query)
        kernel += context @ context.T
        cross += future @ context.T
    scale = torch.diagonal(kernel).mean()
    if scale > 0:
        kernel, cross = kernel / scale, cross / scale
    q = torch.as_tensor(np.asarray(query), device=solver.device, dtype=solver.dtype)
    additive = solver.a @ solver.a.T / solver.lambda_a
    additive_cross = (q / math.sqrt(solver.markers)) @ solver.a.T / solver.lambda_a
    identity = torch.eye(n, dtype=solver.dtype, device=solver.device)
    predictions = []
    for ridge in ridges:
        k = additive if ridge is None else additive + kernel / ridge
        c = additive_cross if ridge is None else additive_cross + cross / ridge
        prediction = c @ torch.linalg.solve(k + identity, solver.y)
        predictions.append(prediction.cpu().numpy())
    return predictions


def score_task(task, config, ratios, oracle_ratios, pair_ridges, *, design=None):
    data, target = task.prepare()
    target = target.double().numpy()
    begun = time.perf_counter()
    solver = SparseInteraction(data.x.numpy(), data.y.numpy(),
        lambda_a=config['lambda_a'], lambda_e=config['lambda_e'],
        device=config['device'], dtype=torch.float64, **config['solver'])
    candidates = solver.fit(ratios, query=data.query.numpy())
    if any(candidate.status != 'optimal' for candidate in candidates):
        raise RuntimeError(json.dumps([candidate.as_dict() for candidate in candidates]))
    # Truth is obtained only AFTER unknown-support optimization, as a separate return.
    if task.metadata['family'] == 'pairwise':
        duplicate, truth = sample_task(config['tasks'], task.metadata['seed'],
            family='pairwise', design=design, return_pair_support=True)
        if not (torch.equal(duplicate.context_y, task.context_y)
                and torch.equal(duplicate.query_signal, task.query_signal)):
            raise ValueError('Oracle reconstruction changed the frozen task')
        pairs = truth.numpy()
    else:
        pairs = np.empty((0, 2), dtype=np.int64)
    truth_set = {tuple(pair) for pair in pairs}
    unknown = []
    for candidate in candidates:
        found = {tuple(pair) for pair in candidate.active}
        unknown.append({**candidate.as_dict(), **error_record(candidate.prediction, target),
                        'true_positive_pairs': len(found & truth_set),
                        'generating_pair_count': len(truth_set)})
    oracle = []
    for result in support_oracle(solver, pairs, data.query.numpy(), oracle_ratios,
                                  solver.lambda_max_value):
        prediction = result.pop('prediction')
        oracle.append({**result, **error_record(prediction, target),
                       'generating_pair_count': len(pairs)})
    dense = [{**error_record(prediction, target), 'pair_ridge': ridge}
             for ridge, prediction in zip(pair_ridges,
                 dense_pair_predictions(solver, data.query.numpy(), pair_ridges))]
    if config['device'] == 'cuda':
        torch.cuda.synchronize()
    result = {'task': task.metadata, 'unknown': unknown, 'oracle': oracle, 'dense_pair': dense,
              'context_mean_mse': float(np.mean(target ** 2)),
              'seconds': time.perf_counter() - begun,
              'constant_pair_count': int(solver._constant.sum()),
              'lambda_max': solver.lambda_max_value,
              'peak_megabytes': (torch.cuda.max_memory_allocated() / 1024 ** 2
                                 if config['device'] == 'cuda' else None)}
    del solver
    if config['device'] == 'cuda':
        torch.cuda.empty_cache()
    return result


def select(records, family):
    scores = [sum(record[family][index]['mse'] for record in records) / len(records)
              for index in range(len(records[0][family]))]
    chosen = min(range(len(scores)), key=scores.__getitem__)
    return {'index': chosen, 'validation_mse': scores[chosen], 'candidate_means': scores}


def summarize(validation, history, chosen, neural):
    summary = {'validation_tasks': len(validation), 'historical_tasks': len(history),
               'failures': [], 'selected': chosen, 'metrics': {}}
    for name, records, field in (('validation', validation, 'validation_tasks'),
                                  ('historical', history, 'held_out_tasks')):
        metrics = {}
        for method in chosen:
            index = chosen[method]['index'] if name == 'validation' else 0
            values = [(r['task']['family'], r[method][index]['mse']) for r in records]
            metrics[method] = {'mean_mse': sum(v for _, v in values) / len(values),
                'per_family': {family: float(np.mean([v for f, v in values if f == family]))
                               for family in sorted({f for f, _ in values})}}
        ids = {r['task']['seed'] for r in records}
        for variant in ('matrix', 'direct'):
            prior = [r for r in neural if r['variant'] == variant]
            values = [row for r in prior for row in r[field] if row['seed'] in ids]
            if len(values) != len(ids) * len(prior):
                raise ValueError('Incomplete frozen neural comparison')
            metrics[variant] = {'mean_mse': float(np.mean([r['mse'] for r in values])),
                'per_family': {family: float(np.mean([r['mse'] for r in values if r['family'] == family]))
                               for family in sorted({r['family'] for r in values})},
                'meaning': 'Mean of frozen seed errors; not a prediction ensemble'}
        metrics['context_mean_mse'] = float(np.mean([r['context_mean_mse'] for r in records]))
        summary['metrics'][name] = metrics
    return summary


def run(config, args, runner):
    if args.selection is not None:
        raise ValueError('Synthetic diagnostic has no outer-test or selection-input path')
    torch.set_num_threads(config['cpu_threads'])
    if config['device'] == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('Requested CUDA is unavailable')
        torch.cuda.set_per_process_memory_fraction(config['gpu_memory_fraction'])
        torch.cuda.reset_peak_memory_stats()
    args.out.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), Path(__file__).with_name('sparse_interaction.py'),
               Path(__file__).with_name('matrix_tasks.py'),
               Path(__file__).with_name('matrix_training.py'), Path(runner), Path(args.config),
               Path(config['neural_reference'])]
    provenance = {'config': config, 'git_commit': subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        'torch_version': torch.__version__, 'sklearn_version': sklearn.__version__,
        'thread_environment': {k: os.environ.get(k) for k in
            ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')},
        'scope': 'Frozen synthetic panels only; no Wheat599, no new neural training',
        'selection_rule': 'Mean clean-query MSE over all 32 validation tasks; earliest tie',
        'oracle_information': 'Generating pair support only (empty outside pairwise family); '
                              'same objective, context lambda_max, grids and scoring',
        'historical_panel': 'Previously inspected admixture tasks; not fresh confirmation'}
    (args.out / 'provenance.json').write_text(json.dumps(provenance, indent=2))
    neural = json.loads(Path(config['neural_reference']).read_text())
    failures, validation, history = [], [], []
    try:
        training = config['training']
        panel = task_panel(config['tasks'], config['tasks']['families'],
                           training['validation_seed'], training['tasks_per_family'])
        for task in panel:
            if task.metadata['seed'] not in {row['seed'] for row in neural[0]['validation_tasks']}:
                raise ValueError('Validation task is not in the frozen neural reference')
            record = score_task(task, config, config['ratios'], config['ratios'],
                                config['pair_ridges'])
            validation.append(record)
            (args.out / 'validation.json').write_text(json.dumps(validation, indent=2))
            print(json.dumps({'stage': 'validation', 'task': task.metadata['seed'],
                              'seconds': record['seconds']}), flush=True)
        chosen = {family: select(validation, family)
                  for family in ('unknown', 'oracle', 'dense_pair')}
        (args.out / 'selection.json').write_text(json.dumps(chosen, indent=2))
        held = config['held_out']
        panel = task_panel(config['tasks'], held['families'], held['seed'],
                           held['tasks_per_family'], design='admixture')
        for task in panel:
            record = score_task(task, config,
                [config['ratios'][chosen['unknown']['index']]],
                [config['ratios'][chosen['oracle']['index']]],
                [config['pair_ridges'][chosen['dense_pair']['index']]], design='admixture')
            history.append(record)
            (args.out / 'historical.json').write_text(json.dumps(history, indent=2))
            print(json.dumps({'stage': 'historical', 'task': task.metadata['seed'],
                              'seconds': record['seconds']}), flush=True)
        (args.out / 'summary.json').write_text(json.dumps(
            summarize(validation, history, chosen, neural), indent=2))
    except Exception as exc:
        failures.append({'reason': repr(exc), 'validation_completed': len(validation),
                         'historical_completed': len(history)})
        raise
    finally:
        (args.out / 'failures.json').write_text(json.dumps(failures, indent=2))
