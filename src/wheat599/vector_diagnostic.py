"""Pure readout diagnostic over the frozen joint checkpoints. Nothing here trains.

Fold-out rule. Every outer context hold re-selects lambda_A by a nested three-fold additive
cross-validation *inside that hold's fit rows* (``dataclasses.replace`` shrinks only the context;
the frozen joint modules are untouched). That fold lambda is shared by the network branch, the
oracle branch and the statistical reference for the same hold; the full-context lambda stays the
plain context cross-validation.

Candidates are the product of the configured shapes and scales. Families are nested --- scalar =
identity, diagonal = identity plus both diagonal sources, full = every configured shape --- so each
family's fold-best candidate is scored once on the query and the pre-registered group-mean
comparisons use those scores. The original-precision readout is recorded separately and is not
cross-validated.

Covariances come from cn's ``wheat599.vector_readout``: ``covariance_shapes`` returns an ordered
four-tuple and ``residual_shapes`` an ordered three-tuple, both built once per support. An empty
support falls back to the additive fit and requests no shapes. Shapes are diagnostic objects, never
true effect covariances.

The oracle branch takes the generating support (``return_support`` key ``pairs``) and runs the same
folded comparison with unit precisions, once per task and never per network; it never selects a
readout. Nothing here reads measured data.
"""
import json
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from wheat599.joint_evidence import Evidence
from wheat599.joint_training import (additive_cv, build_network, context, folds_of,
                                     inner_context, record_hash, solver_result)
from wheat599.mixed_tasks import sample_mixed_task

try:
    from wheat599 import vector_readout
except ImportError:  # cn owns that module; report instead of guessing
    vector_readout = None

def mse_pair(prediction, truth):
    """Synthetic MSE with both operands moved to CPU, so a device split cannot raise."""
    value = float((prediction.detach().cpu() - truth.detach().cpu()).square().mean())
    if value != value:
        raise FloatingPointError('Nonfinite synthetic MSE')
    return value


CV_FAMILIES = ('scalar', 'diagonal', 'full')
FAMILY_SHAPES = {'scalar': {'identity'},
                 'diagonal': {'identity', 'diagonal', 'residual_diagonal'},
                 'full': None}  # every configured shape


# -- panel and options -----------------------------------------------------
def build_panel(config, role='confirmation'):
    spec = config[role]
    tasks = dict(config['tasks'])
    for key in ('query_size', 'marker_counts', 'context_sizes'):
        if key in spec:
            tasks[key] = spec[key]
    entries, index = [], 0
    for background in config['tasks']['backgrounds']:
        for _ in range(int(spec['bases_per_background'])):
            seed = int(spec['role_base']) + index * 100_000_000
            index += 1
            for gamma in config['tasks']['gammas']:
                entries.append((background, float(gamma), seed))
    return tasks, entries


def shapes(config):
    return list(config['readout'].get('shape_order', ('identity', 'diagonal', 'half', 'full')))


def options(config):
    scales = [float(value) for value in config['readout'].get('scales', (0.1, 1.0, 10.0))]
    grid = [{'name': f'{shape}_{scale:g}', 'shape': shape, 'scale': scale}
            for shape in shapes(config) for scale in scales]
    if config['readout'].get('include_zero', True):
        grid.insert(0, {'name': 'zero', 'shape': None, 'scale': 1.0})
    return grid


def family_options(config, family):
    allowed = FAMILY_SHAPES[family] or set(shapes(config))
    names = [item['name'] for item in options(config) if item['shape'] in allowed]
    if config['readout'].get('include_zero', True):
        names.insert(0, 'zero')  # the empty model competes inside every family
    return names


# -- shapes and readouts ---------------------------------------------------
def shape_map(evidence, indices):
    """Shape name to tensor for one support; identity is represented by None."""
    if indices.numel() == 0:
        return {}
    if vector_readout is None:
        raise RuntimeError('vector_readout is unavailable; covariances cannot be built')
    mapping = {'identity': None}
    if hasattr(vector_readout, 'covariance_shapes'):
        built = vector_readout.covariance_shapes(evidence, indices)
        if not isinstance(built, tuple) or len(built) != 4:
            raise RuntimeError('covariance_shapes must return a four-tuple')
        for name, value in zip(('identity', 'diagonal', 'half', 'full'), built):
            if name != 'identity':  # identity stays represented by None
                mapping[name] = value
    if hasattr(vector_readout, 'residual_shapes'):
        residual = vector_readout.residual_shapes(evidence)
        if not isinstance(residual, tuple) or len(residual) != 3:
            raise RuntimeError('residual_shapes must return a three-tuple')
        for name, value in zip(('residual_diagonal', 'residual_half', 'residual_full'), residual):
            mapping[name] = value
    return mapping


def readout(evidence, indices, precision, item, mapping):
    """(additive, coefficients); identity stays local, an empty support is the additive limit."""
    if item['shape'] is None:
        return evidence.readout(torch.zeros(0, dtype=torch.long, device=evidence.device),
                                torch.zeros(0, device=evidence.device))
    if indices.numel() == 0 or item['shape'] == 'identity':
        return evidence.readout(indices, precision * float(item['scale']))
    if vector_readout is None or not hasattr(vector_readout, 'readout'):
        raise RuntimeError('vector_readout.readout is unavailable')
    return vector_readout.readout(evidence, indices, precision * float(item['scale']),
                                  mapping[item['shape']])


def active_for(item, indices):
    if item['shape'] is None:
        return torch.zeros(0, dtype=torch.long, device=indices.device)
    return indices


def option_mse(evidence, query, truth, indices, precision, config):
    mapping = shape_map(evidence, indices)
    if indices.numel():
        missing = [name for name in shapes(config) if name not in mapping]
        if missing:
            raise RuntimeError(f'vector_readout provides no {missing} shapes')
    values = {}
    for item in options(config):
        additive, coefficients = readout(evidence, indices, precision, item, mapping)
        values[item['name']] = mse_pair(evidence.predict(query, active_for(item, indices),
                                                         additive, coefficients), truth)
    return values


# -- fold machinery --------------------------------------------------------
def fold_lambdas(task, rows, config):
    """Nested context cross-validation inside each hold's fit rows only."""
    lambdas = {}
    for hold in folds_of(rows, int(config.get('context_cv_folds', 3))):
        keep = torch.ones(rows, dtype=torch.bool)
        keep[hold] = False
        inner_task = replace(task, context_x=task.context_x[keep], context_y=task.context_y[keep])
        lambdas[tuple(hold.tolist())] = additive_cv(inner_task, config)[0]
    return lambdas


def extracted(network, evidence, query):
    with torch.no_grad():
        output = network(evidence, query)
    return output['indices'], output['d']


def fold_totals(config, network, task, lambdas, device, oracle_pairs=None):
    data, _ = context(task)
    rows = len(data.x)
    totals = {item['name']: 0.0 for item in options(config)}
    for hold in folds_of(rows, int(config.get('context_cv_folds', 3))):
        lambda_a = lambdas[tuple(hold.tolist())]
        x, y, query, truth = inner_context(task, hold, rows)
        x, y, query, truth = (value.to(device) for value in (x, y, query, truth))
        evidence = Evidence(x, y, lambda_a=float(lambda_a))
        if oracle_pairs is None:
            indices, precision = extracted(network, evidence, query)
        else:
            indices = support_indices(evidence, oracle_pairs).to(device)
            precision = torch.ones(int(indices.numel()), dtype=torch.float64, device=device)
        for name, value in option_mse(evidence, query, truth, indices, precision, config).items():
            totals[name] += value * int(len(hold))
    return {name: value / rows for name, value in totals.items()}


def pick(totals, names):
    eligible = [name for name in names if name in totals]
    if not eligible:
        return None
    identity_one = f'identity_{1.0:g}'
    rank = {'zero': 0, identity_one: 1}
    return min(eligible, key=lambda name: (round(totals[name], 12), rank.get(name, 2),
                                           eligible.index(name)))


def support_indices(evidence, pairs):
    markers = int(evidence.markers)
    positions = [int(left) * (2 * markers - int(left) - 1) // 2 + (int(right) - int(left) - 1)
                 for left, right in np.asarray(pairs)]
    return torch.tensor(positions, dtype=torch.long)


# -- statistical reference with the fold lambdas ---------------------------
def reference_local(config, task, lambdas, full_lambda, device):
    """The solver takes numpy arrays, so this stage stays on CPU tensors; it moves to `device` itself."""
    data, target = context(task)
    x_full, y_full, query_full = data.x, data.y, data.query
    full = Evidence(x_full, y_full, lambda_a=float(full_lambda))
    rows = len(data.x)
    ratios = [float(value) for value in config['statistical']['ratios']]
    scores = []
    for order, ratio in enumerate(ratios):
        totals = {'original': 0.0, 'relaxed': 0.0}
        pairs = []
        for hold in folds_of(rows, int(config.get('context_cv_folds', 3))):
            lambda_a = lambdas[tuple(hold.tolist())]
            x, y, query, truth = inner_context(task, hold, rows)
            evidence = Evidence(x, y, lambda_a=float(lambda_a))
            outcome = solver_result(config, evidence, x, y, query, lambda_a, ratio, device)
            for kind in totals:
                totals[kind] += mse_pair(outcome[kind], truth) * int(len(hold))
            pairs.append(outcome['pairs'])
        for kind in ('original', 'relaxed'):
            scores.append({'ratio': ratio, 'kind': kind, 'fold_mse': totals[kind] / rows,
                           'order': order, 'fold_pairs': pairs})
    best = min(scores, key=lambda item: (round(item['fold_mse'], 12),
                                        0 if item['kind'] == 'original' else 1, item['order']))
    outcome = solver_result(config, full, x_full, y_full, query_full, full_lambda, best['ratio'],
                            device)
    return {'grid': scores, 'chosen': {key: best[key] for key in ('ratio', 'kind', 'fold_mse')},
            'full_context_pairs': outcome['pairs'], 'mse': mse_pair(outcome[best['kind']], target),
            'fold_lambdas': list(lambdas.values())}


# -- driver ----------------------------------------------------------------
def load_network(config, filename, device):
    path = Path(config['checkpoint_root']) / filename
    digest = record_hash(path)
    if config['checkpoints'].get(filename) and digest != config['checkpoints'][filename]:
        raise ValueError(f'Frozen checkpoint hash mismatch for {filename}')
    seed = int(filename.rsplit('_', 1)[1].split('.')[0])
    state = torch.load(path, map_location=device, weights_only=True)
    network = build_network(config, 'joint', seed).to(device)
    network.load_state_dict({key: value.to(device) for key, value in state.items()})
    network.eval()
    return network, digest


def family_report(config, totals, evidence, query, truth, indices, precision):
    """Family fold-best picks and their single query scores; the raw readout is not CV'd."""
    report = {'fold_totals': totals, 'picks': {}, 'family_query_mse': {}}
    mapping = shape_map(evidence, indices)
    for family in CV_FAMILIES:
        choice = pick(totals, family_options(config, family))
        report['picks'][family] = choice
        if choice is None:
            continue
        item = next(entry for entry in options(config) if entry['name'] == choice)
        additive, coefficients = readout(evidence, indices, precision, item, mapping)
        report['family_query_mse'][family] = mse_pair(
            evidence.predict(query, active_for(item, indices), additive, coefficients), truth)
    if indices.numel():
        additive, coefficients = readout(evidence, indices, precision,
                                         {'shape': 'identity', 'scale': 1.0}, mapping)
        score = mse_pair(evidence.predict(query, indices, additive, coefficients), truth)
    else:
        # With no candidates the original-precision readout is exactly the additive fit.
        score = mse_pair(evidence.ridge(query), truth)
    report['original_precision'] = {'name': f'identity_{1.0:g}', 'selected_by_cv': False,
                                    'query_mse': score}
    return report


def decision(records, config):
    """Group-mean ratios (not mean per-task ratios); the floor uses the full family."""
    rules = config.get('decision', {})
    out = {}
    for name in sorted({key for record in records for key in record['frozen']}):
        rows = [record for record in records if name in record['frozen']]
        strong = [row for row in rows if row['gamma'] >= 0.5]
        floor = [row for row in rows if row['gamma'] == 0.0
                 and row['background'] in ('dense', 'sparse')]

        def mean_mse(family, subset):
            values = [row['frozen'][name]['family_query_mse'][family] for row in subset
                      if family in row['frozen'][name]['family_query_mse']]
            return float(np.mean(values)) if values else None

        full, scalar, diagonal = (mean_mse('full', strong), mean_mse('scalar', strong),
                                 mean_mse('diagonal', strong))
        wins = [row['frozen'][name]['family_query_mse'].get('full', np.inf) <
                row['frozen'][name]['family_query_mse'].get('diagonal', np.inf) for row in strong]
        floors = [row['frozen'][name]['family_query_mse']['full'] for row in floor]
        bases = [row['additive_mse'] for row in floor]
        out[name] = {
            'tasks_gamma_ge_half': len(strong),
            'group_mean_mse': {'full': full, 'scalar': scalar, 'diagonal': diagonal},
            'relative_gain_vs_scalar': (1 - full / scalar) if full and scalar else None,
            'relative_gain_vs_diagonal': (1 - full / diagonal) if full and diagonal else None,
            'paired_win_fraction_vs_diagonal': float(np.mean(wins)) if wins else None,
            'floor_relative_degradation': (float(np.mean(floors)) / float(np.mean(bases)) - 1)
            if floors else None,
            'thresholds': rules,
        }
    return out


def run(config, args, runner):
    if args.selection is not None:
        raise ValueError('This diagnostic has no selection-input path')
    device = torch.device(config['device'])
    torch.set_num_threads(int(config.get('cpu_threads', 2)))
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(float(config['gpu_memory_fraction']))
    args.out.mkdir(parents=True, exist_ok=False)
    start, failures, records = time.perf_counter(), [], []
    networks = {name: load_network(config, name, device)[0] for name in config['checkpoints']}
    (args.out / 'provenance.json').write_text(json.dumps({
        'config': config,
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'checkpoint_sha256': {name: record_hash(Path(config['checkpoint_root']) / name)
                              for name in networks},
        'torch_version': torch.__version__, 'device': str(device),
        'scope': 'Synthetic confirmation panel; frozen weights; no training, no measured data',
        'note': 'Fold lambdas are nested inside each hold; family picks and support hits are '
                'diagnostic; no neural originality claim'}, indent=2))
    tasks, panel = build_panel(config)
    try:
        for background, gamma, seed in panel:
            task, support = sample_mixed_task(tasks, seed, background=background, gamma=gamma,
                                              return_support=True)
            data, target = context(task)
            rows = len(data.x)
            lambdas = fold_lambdas(task, rows, config)
            full_lambda, cv_losses = additive_cv(task, config)
            x, y, query, truth = (value.to(device) for value in (data.x, data.y, data.query, target))
            evidence = Evidence(x, y, lambda_a=float(full_lambda))
            pairs = support['pairs'] if gamma > 0 else np.zeros((0, 2), dtype=np.int64)
            known = support_indices(evidence, pairs).to(device)
            entry = {'background': background, 'gamma': gamma, 'seed': seed,
                     'metadata': task.metadata, 'full_lambda': full_lambda,
                     'fold_lambdas': list(lambdas.values()),
                     'additive_cv_losses': cv_losses,
                     'additive_mse': mse_pair(evidence.ridge(query), truth),
                     'frozen': {}, 'oracle_cv': {}}
            oracle = family_report(config, fold_totals(config, None, task, lambdas, device,
                                                       oracle_pairs=pairs),
                                   evidence, query, truth, known,
                                   torch.ones(int(known.numel()), dtype=torch.float64, device=device))
            oracle.update({'support_pairs': int(pairs.shape[0]), 'known_indices': known.tolist(),
                           'note': 'CV-only: unit precisions, no neural d; never selects a readout'})
            for name, network in networks.items():
                indices, precision = extracted(network, evidence, query)
                report = family_report(config, fold_totals(config, network, task, lambdas, device),
                                       evidence, query, truth, indices, precision)
                report.update({'full_context_indices': indices.tolist(),
                               'known_support_hits': int(len(set(indices.tolist())
                                                           & set(known.tolist()))),
                               'candidate_count': len(options(config))})
                entry['frozen'][name] = report
                entry['oracle_cv'][name] = oracle
            entry['reference'] = reference_local(config, task, lambdas, full_lambda, device)
            records.append(entry)
            (args.out / 'vector_dimension.json').write_text(json.dumps(records, indent=2, default=str))
            (args.out / 'decision.json').write_text(json.dumps(decision(records, config), indent=2))
    except Exception as exc:
        failures.append({'reason': repr(exc)})
        raise
    finally:
        (args.out / 'timings.json').write_text(json.dumps(
            {'total_seconds': time.perf_counter() - start}, indent=2))
        (args.out / 'failures.json').write_text(json.dumps(failures, indent=2))
