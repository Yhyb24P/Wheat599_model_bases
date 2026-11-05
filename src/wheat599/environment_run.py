"""Development runner for the environment-sharing TabM comparison.

Variants. ``shared`` is the frozen raw multi-output TabM arm; its 480 trajectories are reused
read-only after strict verification (same data/splits/configuration list/seeds, unchanged shared
model source, no ``selection_error`` in the source selection, and one trajectory rebuilt to the saved
prediction within 1e-10). ``conditioned`` trains new fits with per-member output and an environment
multiplicative scale; ``independent`` trains one model per environment, with checkpoint and
configuration chosen separately for each environment. The fixed equal-weight fusion of the old raw arm
and the old Huber ICM is reported as a reference only and is never re-tuned or replaced.

Outputs. Every trajectory stores its per-material member-mean validation predictions together with
fold, inner split, variant, environment, seed and configuration. Ensemble (seed-averaged) and
seed-mean readings are computed separately; per-environment metrics and per-environment improvement
against the shared arm are reported. Checkpoints are hashed and discarded (hash-only). Failures abort
the run and are kept as they occur. The outer test rows are never indexed.
"""
import csv
import hashlib
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from wheat599.data import load, sha256_of
from wheat599.genomic_tabm_run import model_source_sha, verify_versions
from wheat599.metrics import mae, pcc, rmse
from wheat599.ragp_run import state_sha256
from wheat599.retrieval_run import check_split

VARIANTS = ('shared', 'independent', 'conditioned')
METRICS = ('pcc', 'rmse', 'mae')


# -- adapters --------------------------------------------------------------
def shared_predictor_module():
    from wheat599 import retrieval_tabm
    return retrieval_tabm


def environment_module():
    from wheat599 import environment_tabm
    return environment_tabm


def genomic_tabm_sha():
    return sha256_of(Path(__file__).with_name('genomic_tabm.py'))


def shared_source_module_sha(config):
    source = Path(config['shared_source']) / 'provenance.json'
    provenance = json.loads(source.read_text())
    for path, value in provenance['source_sha256'].items():
        if Path(path).name == 'genomic_tabm.py':
            return value
    raise RuntimeError('The shared provenance records no genomic_tabm.py hash')


def build_environment_model(config, variant, seed, train_x, train_y, configuration, device):
    """Seeded construction through cn's model; the adapter is the only place that knows it.

    ``conditioned`` builds the environment-scaled module; ``independent`` (and the shared
    equivalence check) build one plain wrapper on the already single-environment target, which
    avoids constructing four models per trajectory.
    """
    module = environment_module()
    dtype = getattr(torch, config.get('dtype', 'float32'))
    settings = {key: configuration[key] for key in ('n_blocks', 'd_block', 'dropout', 'k')}
    torch.manual_seed(int(seed))
    if variant == 'conditioned':
        builder = getattr(module, 'EnvironmentTabM', None)
        if builder is None:
            raise RuntimeError('wheat599.environment_tabm.EnvironmentTabM is unavailable')
        expected_penalty = float(config.get('environment_scale_l2', 0.001))
        model = builder(train_x.to(dtype), train_y.to(dtype), dtype=dtype,
                        environment_penalty=expected_penalty, **settings)
        if float(getattr(model, 'environment_penalty_coefficient', expected_penalty)) \
                != expected_penalty:
            raise RuntimeError('The model environment penalty disagrees with the configuration')
    else:
        builder = getattr(module, 'SharedTabM', None)
        if builder is None:
            raise RuntimeError('wheat599.environment_tabm.SharedTabM is unavailable')
        model = builder(train_x.to(dtype), train_y.to(dtype), dtype=dtype, **settings)
    return model.to(device)


def optimizer_and_penalty(model, configuration):
    """Conditioned exposes its environment parameters so they can be excluded from decay."""
    groups = getattr(model, 'parameter_groups', None)
    if groups is not None:
        optimizer = torch.optim.AdamW(groups(float(configuration['weight_decay'])),
                                      lr=float(configuration['learning_rate']))
    else:
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=float(configuration['learning_rate']),
                                      weight_decay=float(configuration['weight_decay']))
    penalty = getattr(model, 'environment_penalty', None)
    return optimizer, penalty


def member_output(output):
    if output.ndim == 2:
        return output.unsqueeze(1)
    if output.ndim == 3:
        return output
    raise RuntimeError(f'Unexpected model output shape: {tuple(output.shape)}')


# -- metrics ---------------------------------------------------------------
def macro(values):
    return float(np.mean([np.mean(values[:, environment]) for environment in range(values.shape[1])]))


def unit_metrics(prediction, truth):
    """Macro-over-environment metrics that refuse a non-finite value outright."""
    values = {name: float(np.mean([function(truth[:, environment], prediction[:, environment])
                                   for environment in range(truth.shape[1])]))
              for name, function in zip(METRICS, (pcc, rmse, mae))}
    if not all(np.isfinite(value) for value in values.values()):
        raise FloatingPointError(f'Nonfinite validation metrics: {values}')
    return values


def per_environment_metrics(prediction, truth, environments):
    names = list(environments) if environments else list(range(truth.shape[1]))
    return {str(names[column]): {name: float(function(truth[:, column], prediction[:, column]))
                                 for name, function in zip(METRICS, (pcc, rmse, mae))}
            for column in range(truth.shape[1])}


# -- shared reuse ----------------------------------------------------------
def reuse_report(config):
    """Every check the shared arm must pass before its predictions are reused."""
    source = Path(config['shared_source'])
    provenance = json.loads((source / 'provenance.json').read_text())
    status = json.loads((source / 'execution_status.json').read_text())
    failures = json.loads((source / 'failures.json').read_text())
    selection = json.loads((source / 'selection.json').read_text())
    checks = {}
    checks['data_sha256'] = provenance['data_sha256'] == config['data_sha256']
    checks['splits_sha256'] = provenance['splits_sha256'] == sha256_of(Path(config['splits']))
    checks['seeds'] = provenance['config']['seeds'] == config['seeds']
    source_configs = [{key: value for key, value in entry.items() if key != 'context_size'}
                      for entry in provenance['config']['configurations']]
    checks['configurations'] = source_configs == [{key: value for key, value in entry.items()
                                                   if key != 'context_size'}
                                                  for entry in config['configurations']]
    checks['model_source_unchanged'] = shared_source_module_sha(config) == genomic_tabm_sha()
    checks['execution_complete'] = (status['status'] == 'complete'
                                    and status['completed_trajectories'] == status['expected']
                                    and len(status['folds_complete']) == 10)
    checks['no_failures'] = failures == []
    checks['no_selection_error'] = not any('selection_error' in (value if isinstance(value, dict)
                                                                 else {}) for value in selection.values())
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise RuntimeError(f'Shared arm cannot be reused; failed checks: {failed}')
    return provenance, selection


def load_shared(config):
    """Read the frozen raw predictions: per (fold, inner, seed, material) member-mean vectors."""
    source = Path(config['shared_source']) / 'predictions.csv'
    rows = {}
    with source.open() as handle:
        for row in csv.DictReader(handle):
            if row['variant'] != 'raw':
                continue
            key = (int(row['fold']), int(row['inner']), int(row['config_index']), int(row['seed']),
                   int(row['material_index']))
            rows[key] = np.array([float(row[f'y_pred_{e}']) for e in range(4)])
    return rows


def compare_saved(prediction, rows, materials, atol=1e-6, rtol=1e-6):
    """Compare a reloaded prediction with the saved CSV rows; float32 plus .10g serialisation."""
    if not np.allclose(prediction, rows, rtol=rtol, atol=atol):
        worst = float(np.max(np.abs(prediction - rows)))
        raise RuntimeError(f'Saved prediction mismatch: max |difference| {worst:.3e}')
    return {'max_absolute_difference': float(np.max(np.abs(prediction - rows))),
            'materials': int(len(materials))}


def verify_checkpoint_reuse(config, saved, device):
    """Reload one preserved raw checkpoint and require the saved prediction (no retraining)."""
    source = Path(config['shared_source'])
    provenance = json.loads((source / 'provenance.json').read_text())
    trajectories = json.loads((source / 'trajectories.json').read_text())
    match = next(row for row in trajectories
                 if row['trajectory']['variant'] == 'raw'
                 and row['trajectory']['fold'] == 0 and row['trajectory']['inner'] == 0
                 and row['trajectory']['seed'] == 42 and row['trajectory']['config_index'] == 0)
    path = source / match['checkpoint_path']
    if sha256_of(path) != match['checkpoint_sha256']:
        raise RuntimeError('Preserved raw checkpoint hash does not match its recorded hash')
    from wheat599 import retrieval_tabm
    source_config = dict(provenance['config'])
    data = load(Path(config['data']), config['data_sha256'])
    fold = json.loads(Path(config['splits']).read_text())['folds'][0]
    fit_index = np.asarray(fold['inner'][0]['train_idx'])
    val_index = np.asarray(fold['inner'][0]['val_idx'])
    dtype = getattr(torch, source_config.get('dtype', 'float32'))
    x = torch.as_tensor(data.X, dtype=torch.float64)
    y = torch.as_tensor(data.Y, dtype=torch.float64)
    torch.manual_seed(42)
    model = retrieval_tabm.RetrievalTabM(x[fit_index].to(dtype), y[fit_index].to(dtype),
                                         variant='raw', dtype=dtype,
                                         **{key: source_config['configurations'][0][key]
                                            for key in ('n_blocks', 'd_block', 'dropout', 'k')})
    model = model.to(device)
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        prediction = model.inverse_y(member_output(model(x[val_index].to(dtype).to(device))).mean(1))
    values = prediction.cpu().numpy().astype(np.float64)
    rows = np.stack([saved[(0, 0, 0, 42, int(material))] for material in val_index])
    return compare_saved(values, rows, val_index)


# -- training --------------------------------------------------------------
@dataclass
class Trajectory:
    fold: int
    inner: int
    seed: int
    variant: str
    config_index: int
    configuration: dict
    environment: int | None = None


def train_trajectory(data, fold, inner_index, trajectory, config, device):
    fit_index, val_index = check_split(fold, inner_index)
    training = config['training']
    dtype = getattr(torch, config.get('dtype', 'float32'))
    x = torch.as_tensor(data.X, dtype=torch.float64)
    y = torch.as_tensor(data.Y, dtype=torch.float64)
    fit_x, fit_y = x[fit_index].to(dtype).to(device), y[fit_index].to(dtype).to(device)
    val_x = x[val_index].to(dtype).to(device)
    val_y = y[val_index].to(device)
    if trajectory.environment is not None:
        fit_y = fit_y[:, trajectory.environment:trajectory.environment + 1]
        val_y = val_y[:, trajectory.environment:trajectory.environment + 1]
    started = time.perf_counter()
    model = build_environment_model(config, trajectory.variant, trajectory.seed, fit_x, fit_y,
                                    trajectory.configuration, device)
    preprocess_seconds = time.perf_counter() - started
    optimizer, penalty = optimizer_and_penalty(model, trajectory.configuration)
    target = model.standardize_y(fit_y) if hasattr(model, 'standardize_y') else fit_y
    generator = torch.Generator().manual_seed(int(trajectory.seed) * 1000 + trajectory.config_index)
    best = {'pcc': -np.inf, 'epoch': -1, 'state': None}
    curve, loss = [], None
    if device.type == 'cuda':
        baseline_memory = float(torch.cuda.memory_allocated(device))
        torch.cuda.reset_peak_memory_stats(device)
    else:
        baseline_memory = None
    if trajectory.environment is not None:
        fit_x = fit_x
    def evaluate():
        model.eval()
        with torch.no_grad():
            output = member_output(model(val_x)).mean(1)
            prediction = model.inverse_y(output) if hasattr(model, 'inverse_y') else output
        truth = val_y.cpu().numpy()
        return prediction.cpu().numpy().astype(np.float64), truth
    trained_start = time.perf_counter()
    for epoch in range(int(training['max_epochs']) + 1):
        if epoch:
            model.train()
            order = torch.randperm(len(fit_x), generator=generator).to(fit_x.device)
            for start in range(0, len(order), int(training['batch_size'])):
                batch = order[start:start + int(training['batch_size'])]
                optimizer.zero_grad(set_to_none=True)
                output = member_output(model(fit_x[batch]))
                loss = F.mse_loss(output, target[batch].unsqueeze(1).expand_as(output))
                if penalty is not None:
                    loss = loss + penalty()
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Nonfinite training loss at epoch {epoch}')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(training['gradient_clip']))
                optimizer.step()
        prediction, truth = evaluate()
        score = macro(np.stack([np.array([pcc(truth[:, e], prediction[:, e])
                                          for e in range(truth.shape[1])])]))
        curve.append({'epoch': epoch, 'validation_pcc': score,
                      'train_mse': None if loss is None else float(loss.detach())})
        if score > best['pcc'] + float(training['tie_tolerance']):
            best = {'pcc': score, 'epoch': epoch,
                    'state': {key: value.detach().cpu().clone()
                              for key, value in model.state_dict().items()}}
        elif epoch - best['epoch'] >= int(training['patience']):
            break
    trained_seconds = time.perf_counter() - trained_start
    peak = float(torch.cuda.max_memory_allocated(device)) if device.type == 'cuda' else None
    model.load_state_dict({key: value.to(device) for key, value in best['state'].items()})
    prediction, truth = evaluate()
    digest = state_sha256(best['state'])
    batch = val_x[:int(config['efficiency']['batch_size'])]
    for _ in range(int(config['efficiency']['warmup'])):
        with torch.no_grad():
            model(batch)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for _ in range(int(config['efficiency']['repeats'])):
        with torch.no_grad():
            model(batch)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    labels = [data.environments[trajectory.environment]] if trajectory.environment is not None \
        else data.environments
    return {'trajectory': {key: value for key, value in trajectory.__dict__.items()},
            'metrics': unit_metrics(prediction, truth),
            'per_environment': per_environment_metrics(prediction, truth, labels),
            'best_epoch': best['epoch'], 'epochs_run': len(curve), 'curve': curve,
            'prediction': prediction, 'material_index': val_index.tolist(),
            'preprocess_seconds': preprocess_seconds, 'train_seconds': trained_seconds,
            'inference_seconds': (time.perf_counter() - started) / int(config['efficiency']['repeats']),
            'memory': ({'baseline_megabytes': baseline_memory / 2 ** 20,
                        'total_allocated_peak_megabytes': peak / 2 ** 20,
                        'increment_megabytes': (peak - baseline_memory) / 2 ** 20}
                       if peak is not None else None),
            'parameters': model.parameter_count() if hasattr(model, 'parameter_count') else None,
            'checkpoint_sha256': digest, 'state': best['state']}


# -- selection -------------------------------------------------------------
def seed_ensemble(records, seeds, environment=None):
    by_seed = {int(record['trajectory']['seed']): record for record in records}
    if set(by_seed) != {int(seed) for seed in seeds}:
        raise ValueError('Missing seeds for an inner unit')
    return np.mean([by_seed[int(seed)]['prediction'] for seed in seeds], axis=0)


def select_common(records, config):
    """One configuration per fold; for independent, one per environment."""
    expected = 3 * len(config['seeds'])
    chosen = {}
    for fold in sorted({record['trajectory']['fold'] for record in records}):
        chosen[fold] = {}
        for variant in config['variants']:
            if variant == 'shared':
                continue  # reused from the frozen arm; its selection is not recomputed here
            environments = range(4) if variant == 'independent' else (None,)
            chosen[fold][variant] = {}
            for environment in environments:
                scored = []
                for index in range(len(config['configurations'])):
                    rows = [record for record in records
                            if record['trajectory']['fold'] == fold
                            and record['trajectory']['variant'] == variant
                            and record['trajectory']['config_index'] == index
                            and record['trajectory']['environment'] == environment]
                    if len(rows) != expected:
                        raise ValueError('Incomplete trajectory grid for one configuration')
                    ensembles, means = [], []
                    for inner in range(3):
                        unit = [row for row in rows if row['trajectory']['inner'] == inner]
                        prediction = seed_ensemble(unit, config['seeds'], environment)
                        full_truth = row_truth(unit[0])
                        truth = full_truth if environment is None else \
                            full_truth[:, environment:environment + 1]
                        ensembles.append(np.mean([pcc(truth[:, e], prediction[:, e])
                                                  for e in range(truth.shape[1])]))
                        means.append(float(np.mean([row['metrics']['pcc'] for row in unit])))
                    score = float(np.mean(ensembles))
                    if not np.isfinite(score):
                        raise FloatingPointError('Nonfinite selection score')
                    scored.append((score, index, float(np.mean(means))))
                best = max(scored, key=lambda item: (round(item[0], 12), -item[1]))
                chosen[fold][variant][environment] = {
                    'config_index': best[1], 'configuration': config['configurations'][best[1]],
                    'seed_ensemble_pcc': best[0], 'seed_mean_pcc': best[2]}
    return chosen


_truth_cache = {}


def row_truth(record):
    key = (record['trajectory']['fold'], record['trajectory']['inner'])
    return _truth_cache[key]


def write_predictions(path, records, environments):
    """Long format: one scalar prediction per (variant, unit, seed, config, material, environment).

    The column count never changes between rows, so every reading can be reconstructed from the
    file alone: a multi-output record emits one row per environment, and an independent record
    emits its single value under its own environment label.
    """
    labels = list(environments)
    with path.open('w', newline='') as handle:
        writer = csv.writer(handle, lineterminator='\n')
        writer.writerow(['variant', 'fold', 'inner', 'environment', 'seed', 'config_index',
                         'material_index', 'y_pred'])
        for record in records:
            trajectory = record['trajectory']
            for position, material in enumerate(record['material_index']):
                values = record['prediction'][position]
                if trajectory['environment'] is not None:
                    writer.writerow([trajectory['variant'], trajectory['fold'],
                                     trajectory['inner'], labels[trajectory['environment']],
                                     trajectory['seed'], trajectory['config_index'], material,
                                     f'{values[0]:.10g}'])
                else:
                    for environment, value in enumerate(values):
                        writer.writerow([trajectory['variant'], trajectory['fold'],
                                         trajectory['inner'], labels[environment],
                                         trajectory['seed'], trajectory['config_index'], material,
                                         f'{value:.10g}'])


# -- reference summaries ---------------------------------------------------
def fusion_unit(config, fold, inner, val_index):
    """The fixed equal-weight blend of the old raw arm and the old Huber ICM."""
    weight = float(config['fixed_fusion_weight'])
    payload = json.loads((Path(config['fusion_source']) / 'predictions.json').read_text())
    rows = {(int(row['fold']), int(row['inner']), int(row['material_index'])): row
            for row in payload}
    tabm = np.stack([np.array(rows[(fold, inner, int(m))]['tabm']) for m in val_index])
    icm = np.stack([np.array(rows[(fold, inner, int(m))]['icm']) for m in val_index])
    return weight * tabm + (1 - weight) * icm


def seeds_prediction(records, seeds):
    by_seed = {int(record['trajectory']['seed']): record for record in records}
    return np.mean([by_seed[int(seed)]['prediction'] for seed in seeds], axis=0)


def trained_unit(rows, seeds, truth, environments, chosen_indices):
    """Ensemble and seed-mean readings at the common configuration(s) chosen for this fold.

    No configuration is re-selected here: the summary uses exactly the indices that
    ``select_common`` fixed for this fold (and environment, for the independent variant).
    """
    predictions, means = [], []
    for position, environment in enumerate(environments):
        index = int(chosen_indices[position])
        unit_rows = [record for record in rows
                     if record['trajectory']['config_index'] == index
                     and record['trajectory']['environment'] == environment]
        if len(unit_rows) != len(seeds):
            raise ValueError('The common configuration has no complete seed set')
        predictions.append(seeds_prediction(unit_rows, seeds))
        means.append(float(np.mean([record['metrics']['pcc'] for record in unit_rows])))
    if environments == (None,):
        return {'ensemble': predictions[0], 'seed_mean': means[0],
                'config_index': int(chosen_indices[0])}
    assembled = np.zeros((len(truth), truth.shape[1]))
    for environment, prediction in enumerate(predictions):
        assembled[:, environment] = prediction[:, 0]
    return {'ensemble': assembled, 'seed_mean': float(np.mean(means)),
            'config_indices': [int(index) for index in chosen_indices]}


def _per_fold(entries):
    folds = {}
    for (k, inner, _), value in entries:
        metrics = unit_metrics(value['ensemble'], np.asarray(_truth_cache[(k, inner)]))
        folds.setdefault(k, []).append(metrics)
    return [{'fold': fold, **{metric: float(np.mean([entry[metric] for entry in values]))
                              for metric in METRICS}}
            for fold, values in sorted(folds.items())]


def summarise(config, records, saved, source_selection, selections, folds):
    """Readings for every variant: seed-ensemble and seed-mean, overall and per environment."""
    seeds = [int(seed) for seed in config['seeds']]
    units = {}
    for fold in folds:
        k = int(fold['fold'])
        for inner, split in enumerate(fold['inner']):
            val = np.asarray(split['val_idx'])
            truth = np.asarray(_truth_cache[(k, inner)])
            index = int(source_selection[str(k)]['raw']['config_index'])
            per_seed = [np.stack([saved[(k, inner, index, seed, int(m))] for m in val])
                        for seed in seeds]
            units[(k, inner, 'shared')] = {
                'ensemble': np.mean(per_seed, axis=0),
                'seed_mean': float(np.mean([unit_metrics(p, truth)['pcc'] for p in per_seed]))}
            units[(k, inner, 'fixed_fusion')] = {
                'ensemble': fusion_unit(config, k, inner, val), 'seed_mean': None}
            for variant in ('independent', 'conditioned'):
                rows = [record for record in records if record['trajectory']['variant'] == variant
                        and record['trajectory']['fold'] == k
                        and record['trajectory']['inner'] == inner]
                if not rows:
                    continue
                environments = tuple(range(truth.shape[1])) if variant == 'independent' \
                    else (None,)
                chosen = selections.get(str(k), {}).get(variant)
                if chosen is None:
                    raise ValueError('The summary requires the frozen common selection')
                if variant == 'independent':
                    indices = [chosen[environment]['config_index']
                               for environment in range(truth.shape[1])]
                else:
                    indices = [chosen[None]['config_index']]
                units[(k, inner, variant)] = trained_unit(rows, seeds, truth, environments, indices)
    report = {}
    for name in ('shared', 'fixed_fusion', 'independent', 'conditioned'):
        entries = [(key, value) for key, value in units.items() if key[2] == name]
        if not entries:
            continue
        unit_scores = []
        per_environment = {}
        for (k, inner, _), value in entries:
            truth = np.asarray(_truth_cache[(k, inner)])
            unit_scores.append(unit_metrics(value['ensemble'], truth))
            for environment, metrics in per_environment_metrics(value['ensemble'], truth,
                                                                None).items():
                per_environment.setdefault(environment, []).append(metrics)
        report[name] = {
            'units': len(unit_scores),
            'seed_ensemble': {metric: float(np.mean([entry[metric] for entry in unit_scores]))
                              for metric in METRICS},
            'per_environment_seed_ensemble': {
                environment: {metric: float(np.mean([entry[metric] for entry in values]))
                              for metric in METRICS}
                for environment, values in per_environment.items()},
            'seed_mean_pcc': float(np.mean([value['seed_mean'] for _, value in entries
                                            if value['seed_mean'] is not None])) if any(
                value['seed_mean'] is not None for _, value in entries) else None,
            'per_fold': _per_fold(entries),
            'wins_vs_shared_of_30': int(sum(
                unit_metrics(value['ensemble'], np.asarray(_truth_cache[(k, inner)]))['pcc'] >
                unit_metrics(units[(k, inner, 'shared')]['ensemble'],
                             np.asarray(_truth_cache[(k, inner)]))['pcc']
                for (k, inner, _), value in entries)),
        }
    for name in ('independent', 'conditioned'):
        if name not in report or 'shared' not in report:
            continue
        report[name]['per_environment_improvement_vs_shared'] = {
            environment: {metric: report[name]['per_environment_seed_ensemble'][environment][metric]
                          - report['shared']['per_environment_seed_ensemble'][environment][metric]
                          for metric in METRICS}
            for environment in report[name]['per_environment_seed_ensemble']}
    return report


# -- driver ----------------------------------------------------------------
def run(config, args, runner):
    if args.selection is not None:
        raise ValueError('This development run has no selection-input path')
    verified_versions = verify_versions(config)
    device = torch.device(config['device'])
    torch.set_num_threads(int(config.get('cpu_threads', 2)))
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(float(config['gpu_memory_fraction']))
    data = load(Path(config['data']), config['data_sha256'])
    manifest = json.loads(Path(config['splits']).read_text())
    folds = [fold for fold in manifest['folds'] if int(fold['fold']) in {int(value)
                                                                        for value in config['outer_folds']}]
    provenance, source_selection = reuse_report(config)
    saved = load_shared(config)
    spot = verify_checkpoint_reuse(config, saved, device)
    args.out.mkdir(parents=True, exist_ok=False)
    _truth_cache.clear()
    for fold in folds:
        for inner, split in enumerate(fold['inner']):
            _truth_cache[(int(fold['fold']), inner)] = np.asarray(data.Y[np.asarray(split['val_idx'])])
    here = Path(__file__).parent
    sources = [here / name for name in ('environment_run.py', 'environment_tabm.py',
                                       'genomic_tabm.py', 'retrieval_tabm.py', 'data.py',
                                       'metrics.py')] + [Path(args.config), Path(runner)]
    expected = len(folds) * 3 * len(config['seeds']) * (
        len(config['configurations']) * (1 if 'conditioned' in config['variants'] else 0)
        + len(config['configurations']) * (4 if 'independent' in config['variants'] else 0))
    state = {'status': 'running', 'completed_trajectories': 0, 'expected': expected,
             'folds_complete': []}
    (args.out / 'provenance.json').write_text(json.dumps({
        'config': config,
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'data_sha256': config['data_sha256'], 'splits_sha256': sha256_of(Path(config['splits'])),
        'source_sha256': {str(path): sha256_of(path) for path in sources if path.exists()},
        'shared_reuse': {'source': config['shared_source'],
                         'predictions_sha256': sha256_of(Path(config['shared_source'])
                                                         / 'predictions.csv'),
                         'checkpoint_reload_check': spot,
                         'checks': 'data/splits/configurations/seeds/model source/status/failures/'
                                   'selection_error all verified before reuse'},
        'torch_version': torch.__version__, 'numpy_version': np.__version__,
        'package_versions': verified_versions, 'model_source': model_source_sha(),
        'device': str(device), 'expected_excluding_shared': expected,
        'scope': 'Development only: inner fit/val; outer test rows never read; the fixed fusion '
                 'reference is the old raw + old ICM at weight 0.5 and is not re-tuned'}, indent=2))
    records, failures, selections = [], [], {}
    start = time.perf_counter()
    try:
        for fold in folds:
            for inner_index in range(3):
                for variant in config['variants']:
                    if variant == 'shared':
                        continue
                    environments = range(4) if variant == 'independent' else (None,)
                    for environment in environments:
                        for seed in config['seeds']:
                            for index, entry in enumerate(config['configurations']):
                                trajectory = Trajectory(int(fold['fold']), inner_index, int(seed),
                                                        variant, index, entry, environment)
                                try:
                                    record = train_trajectory(data, fold, inner_index, trajectory,
                                                              config, device)
                                    record.pop('state')
                                    records.append(record)
                                    state['completed_trajectories'] += 1
                                except Exception as exc:
                                    failures.append({'trajectory': trajectory.__dict__,
                                                     'reason': repr(exc)})
                                    raise
                                finally:
                                    (args.out / 'failures.json').write_text(
                                        json.dumps(failures, indent=2))
                                    (args.out / 'trajectories.json').write_text(json.dumps(
                                        [{key: value for key, value in record_.items()
                                          if key != 'prediction'} for record_ in records],
                                        indent=2, default=str))
            # A fold only counts as complete after its selection succeeded; an incomplete grid is
            # a failure that is recorded and raised, never swallowed.
            try:
                selections[str(fold['fold'])] = select_common(
                    [record for record in records
                     if record['trajectory']['fold'] == int(fold['fold'])], config)[int(fold['fold'])]
            except Exception as exc:
                failures.append({'stage': 'selection', 'fold': int(fold['fold']),
                                 'reason': repr(exc)})
                raise
            state['folds_complete'].append(int(fold['fold']))
            (args.out / 'selection.json').write_text(json.dumps(selections, indent=2))
            (args.out / 'execution_status.json').write_text(json.dumps(state, indent=2))
        state['status'] = 'complete'
    except Exception as exc:
        state['status'] = 'failed'
        raise
    finally:
        (args.out / 'failures.json').write_text(json.dumps(failures, indent=2))
        (args.out / 'selection.json').write_text(json.dumps(selections, indent=2))
        write_predictions(args.out / 'predictions.csv', records, data.environments)
        try:
            (args.out / 'summary.json').write_text(json.dumps(
                summarise(config, records, saved, source_selection, selections, folds),
                indent=2))
        except Exception as exc:
            # A summary failure fails the run: the training artifacts stay, completion is revoked.
            state['status'] = 'failed'
            failures.append({'stage': 'summary', 'reason': repr(exc)})
            (args.out / 'summary_error.json').write_text(json.dumps({'reason': repr(exc)}, indent=2))
        (args.out / 'timings.json').write_text(json.dumps(
            {'total_seconds': time.perf_counter() - start}, indent=2))
        (args.out / 'failures.json').write_text(json.dumps(failures, indent=2))
        (args.out / 'execution_status.json').write_text(json.dumps(state, indent=2))
