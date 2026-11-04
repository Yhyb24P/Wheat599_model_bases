"""Finite-kernel reference on the existing synthetic panels.

Two Gram kernels are built from context markers only:
    K_A = Z Z^T / p,        cross_A = Q Z^T / p
    K_Q = K_A elementwise^2, cross_Q = cross_A elementwise^2
Each is context-centered (both support and query features subtract the context feature
mean) and then divided by its own context mean diagonal, so both branches
and every mixture carry mean diagonal one. Prediction is ordinary kernel ridge,
alpha = (K + ridge I)^-1 Y with context-centered and context-standardized Y, and the
query prediction is cross alpha; no query phenotype, clean target, true heritability or
generating covariance enters it.

One common hyperparameter set is chosen from all candidates by mean clean-target MSE over
the fixed validation panel only, ties resolved by configuration order. The held-out
admixture panel is scored once, after that choice is frozen, and is therefore a historical
diagnostic rather than fresh evidence. No Wheat599 data is read.
"""
import dataclasses
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path

import torch

from wheat599.matrix_training import task_panel


def context_center(kernel, cross):
    """Center the support Gram and apply the matching shift to the query cross block."""
    centered = (kernel - kernel.mean(0, keepdim=True) - kernel.mean(1, keepdim=True)
                + kernel.mean())
    cross_centered = (cross - cross.mean(1, keepdim=True)
                      - kernel.mean(0, keepdim=True) + kernel.mean())
    return centered, cross_centered


def normalized_kernels(data):
    """Context-only kernels and cross blocks, each with mean context diagonal one."""
    markers = data.x.shape[1]
    additive = data.x @ data.x.T / markers
    additive_cross = data.query @ data.x.T / markers
    blocks = {'additive': (additive, additive_cross),
              'quadratic': (additive.square(), additive_cross.square())}
    kernels = {}
    for name, (kernel, cross) in blocks.items():
        kernel, cross = context_center(kernel, cross)
        scale = torch.diagonal(kernel).mean()
        if not torch.isfinite(scale) or scale <= 0:
            raise ValueError(f'Constant context kernel block: {name}')
        kernels[name] = kernel / scale
        kernels[f'{name}_cross'] = cross / scale
    return kernels


def candidate_grid(config):
    """Candidate order defines tie-breaking: mixture first, then ridge."""
    kernels = config['kernels']
    return [(fraction, ridge) for fraction in kernels['mixtures'] for ridge in kernels['ridge']]


def eigendecompose(kernels, fraction):
    """One factorization per mixture; the ridge sweep then costs only diagonal solves."""
    mixed = (1 - fraction) * kernels['additive'] + fraction * kernels['quadratic']
    cross = (1 - fraction) * kernels['additive_cross'] + fraction * kernels['quadratic_cross']
    values, vectors = torch.linalg.eigh(mixed)
    tolerance = 1e-10 * max(1., float(values.abs().max()))
    if float(values.min()) < -tolerance:
        raise ValueError('Mixed Gram is not positive semidefinite')
    return values, vectors, cross


def predict_from(response, ridge, decomposition):
    values, vectors, cross = decomposition
    rotated = vectors.T @ response
    return cross @ (vectors @ (rotated / (values.clamp_min(0) + ridge).unsqueeze(-1)))


def predict(kernels, response, fraction, ridge):
    """Kernel ridge on the mixed kernel; only the mixed Gram and context response enter."""
    return predict_from(response, ridge, eigendecompose(kernels, fraction))


def as_double(data):
    return dataclasses.replace(data, x=data.x.double(), y=data.y.double(),
                               query=data.query.double(), y_mean=data.y_mean.double(),
                               y_scale=data.y_scale.double())


def score_panel(panel, candidates):
    """Per-task clean-target MSE for every candidate; failures are never filtered."""
    per_task = []
    for task in panel:
        data, target = task.prepare()
        data, target = as_double(data), target.double()
        kernels = normalized_kernels(data)
        decompositions = {fraction: eigendecompose(kernels, fraction)
                          for fraction in {candidate[0] for candidate in candidates}}
        errors = []
        for fraction, ridge in candidates:
            prediction = predict_from(data.y, ridge, decompositions[fraction])
            loss = float((prediction - target).square().mean())
            if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
                raise FloatingPointError(f'Nonfinite kernel prediction at {fraction}, {ridge}')
            if not math.isfinite(loss):
                raise FloatingPointError(f'Nonfinite kernel MSE at {fraction}, {ridge}')
            errors.append(loss)
        per_task.append({'task': task.metadata, 'mse': errors})
    return per_task


def select(scores, candidates, subset=None):
    """Lowest mean validation MSE wins; the earliest candidate in grid order breaks ties."""
    means = [sum(row) / len(row) for row in zip(*[entry['mse'] for entry in scores])]
    order = range(len(candidates)) if subset is None else subset
    best, best_mean, best_index = None, float('inf'), None
    for index in order:
        if means[index] < best_mean:
            best, best_mean, best_index = candidates[index], means[index], index
    return {'fraction': best[0], 'ridge': best[1], 'validation_mse': best_mean,
            'candidate_index': best_index}


def panels(config):
    training = config['training']
    validation = task_panel(config['tasks'], config['tasks']['families'],
                            training['validation_seed'], training['tasks_per_family'])
    held = config['held_out']
    return validation, task_panel(config['tasks'], held['families'], held['seed'],
                                  held['tasks_per_family'], design='admixture')


def run(config, args, runner):
    if args.selection is not None:
        raise ValueError('The kernel reference has no selection-input or outer-test path')
    if config['device'] != 'cpu':
        raise ValueError('This reference is defined for CPU float64 only')
    torch.set_num_threads(config.get('cpu_threads', 2))
    args.out.mkdir(parents=True, exist_ok=False)
    here = Path(__file__).parent
    sources = [here / name for name in ('matrix_kernel_reference.py', 'matrix_tasks.py',
                                        'matrix_training.py')]
    sources += [Path(args.config), Path(runner)]
    timings, failures, provenance = {}, [], {}
    start = time.perf_counter()
    try:
        validation, held_out = panels(config)
        candidates = candidate_grid(config)
        drawn = time.perf_counter() - start
        timings['panel_draw_seconds'] = drawn
        begun = time.perf_counter()
        scores = score_panel(validation, candidates)
        timings['validation_seconds'] = time.perf_counter() - begun
        subset = lambda values: [index for index, candidate in enumerate(candidates)
                                 if candidate[0] in values]
        chosen = {'additive_endpoint': select(scores, candidates, subset({0.0})),
                  'quadratic_endpoint': select(scores, candidates, subset({1.0})),
                  'mixed_all_candidates': select(scores, candidates)}
        begun = time.perf_counter()
        unique = list(dict.fromkeys((entry['fraction'], entry['ridge']) for entry in chosen.values()))
        held_records = score_panel(held_out, unique)
        timings['held_out_seconds'] = time.perf_counter() - begun
        held_per_selection = {name: [entry['mse'][unique.index((chosen[name]['fraction'],
                                                                chosen[name]['ridge']))]
                                     for entry in held_records] for name in chosen}
        provenance.update({
            'config': config,
            'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            'source_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in sources},
            'torch_version': torch.__version__, 'dtype': 'float64', 'device': 'cpu',
            'scope': 'Synthetic panels only; no Wheat599 genotype or phenotype read',
            'selection_rule': 'One common (mixture, ridge) per reported family of candidates from '
                              'mean clean-target MSE over the fixed validation panel; ties by grid order',
            'held_out_status': 'Scored once after the frozen selection, and only for the selected '
                               'candidates; already seen, so it is a historical diagnostic and not '
                               'independent evidence',
            'candidates': [{'fraction': f, 'ridge': r} for f, r in candidates],
        })
        (args.out / 'provenance.json').write_text(json.dumps(provenance, indent=2))
        (args.out / 'validation.json').write_text(json.dumps(
            {'candidates': [{'fraction': f, 'ridge': r} for f, r in candidates],
             'per_task': scores, 'mean_per_candidate': [sum(row) / len(row)
                                                        for row in zip(*[e['mse'] for e in scores])]},
            indent=2))
        (args.out / 'selection.json').write_text(json.dumps(
            {'rule': provenance['selection_rule'], 'selected': chosen}, indent=2))
        (args.out / 'held_out.json').write_text(json.dumps(
            {'status': provenance['held_out_status'], 'candidates_evaluated': [
                {'fraction': f, 'ridge': r} for f, r in unique],
             'task_metadata': [entry['task'] for entry in held_records],
             'per_task_mse_by_candidate_order': [entry['mse'] for entry in held_records],
             'per_task_mse_by_selection': held_per_selection,
             'selected': {name: {**chosen[name],
                                 'held_out_mean_mse': sum(values) / len(values)}
                          for name, values in held_per_selection.items()}}, indent=2))
    except Exception as exc:
        failures.append({'reason': repr(exc)})
        raise
    finally:
        timings['total_seconds'] = time.perf_counter() - start
        (args.out / 'timings.json').write_text(json.dumps(timings, indent=2))
        (args.out / 'failures.json').write_text(json.dumps(failures, indent=2))
