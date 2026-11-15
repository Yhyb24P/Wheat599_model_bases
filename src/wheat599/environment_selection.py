"""Re-select configurations from frozen predictions; no fitting or checkpoint changes."""
import csv
import json
import subprocess
from pathlib import Path

import numpy as np

from wheat599.data import load, sha256_of
from wheat599.metrics import pcc, rmse, mae


def select_indices(scores, tolerance=1e-12):
    """scores = inner x configuration x environment, all original checkpoints fixed."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 3 or not np.isfinite(scores).all():
        raise ValueError('Expected finite inner/configuration/environment scores')
    means = scores.mean(0)

    def first_max(values):
        return int(np.flatnonzero(values >= values.max() - tolerance)[0])

    common = first_max(means.mean(1))
    separate = [first_max(means[:, e]) for e in range(means.shape[1])]
    return [common] * means.shape[1], separate


def read_predictions(path, training, environment_names):
    """Canonical key: fold,inner,configuration,seed,material,environment-position."""
    predictions = {}
    with Path(path).open() as handle:
        for row in csv.DictReader(handle):
            variant = 'raw' if training == 'shared' else 'independent'
            if row['variant'] != variant:
                continue
            prefix = tuple(int(row[k]) for k in ('fold', 'inner', 'config_index', 'seed',
                                                'material_index'))
            values = ([(e, float(row[f'y_pred_{e}'])) for e in range(len(environment_names))]
                      if training == 'shared' else
                      [(environment_names.index(row['environment']), float(row['y_pred']))])
            for e, value in values:
                key = (*prefix, e)
                if key in predictions or not np.isfinite(value):
                    raise ValueError(f'Duplicate or nonfinite prediction: {key}')
                predictions[key] = value
    return predictions


def metric_matrix(prediction, truth):
    values = np.array([[fn(truth[:, e], prediction[:, e])
                        for fn in (pcc, rmse, mae)] for e in range(truth.shape[1])])
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite metrics')
    return values


def compute(data, folds, banks, seeds, configurations, tolerance=1e-12):
    per_fold, chosen = [], []
    expected_keys = {name: set() for name in banks}
    for fold in folds:
        k = int(fold['fold'])
        for training, bank in banks.items():
            cube, truth_by_inner = [], []
            for j, split in enumerate(fold['inner']):
                ids = list(map(int, split['val_idx']))
                fit = set(split['train_idx'])
                if fit & set(ids) or fit | set(ids) != set(fold['dev_idx']) \
                        or (fit | set(ids)) & set(fold['test_idx']):
                    raise ValueError('Invalid development boundary')
                truth_by_inner.append(np.asarray(data.Y[ids], dtype=np.float64))
                grid = []
                for c in range(configurations):
                    members = []
                    for seed in seeds:
                        keys = [[(k, j, c, int(seed), i, e)
                                 for e in range(data.Y.shape[1])] for i in ids]
                        expected_keys[training].update(key for row in keys for key in row)
                        try:
                            members.append([[bank[key] for key in row] for row in keys])
                        except KeyError as exc:
                            raise ValueError(f'Missing frozen prediction {exc}') from exc
                    grid.append(members)
                cube.append(np.asarray(grid, dtype=np.float64))
            # inner x configuration x seed x material x environment; inner sizes may differ.
            scores = np.array([[metric_matrix(grid.mean(0), truth_by_inner[j])[:, 0]
                                for grid in cube[j]] for j in range(len(cube))])
            common, separate = select_indices(scores, tolerance)
            for policy, indices in [('common', common), ('per_environment', separate)]:
                unit_metrics, unit_seed_metrics = [], []
                for j in range(len(cube)):
                    pred = np.column_stack([cube[j][c, :, :, e].mean(0)
                                            for e, c in enumerate(indices)])
                    unit_metrics.append(metric_matrix(pred, truth_by_inner[j]))
                    seed_values = []
                    for si in range(len(seeds)):
                        single = np.column_stack([cube[j][c, si, :, e]
                                                  for e, c in enumerate(indices)])
                        seed_values.append(metric_matrix(single, truth_by_inner[j]))
                    unit_seed_metrics.append(np.mean(seed_values, axis=0))
                metrics = np.mean(unit_metrics, axis=0)
                seed_metrics = np.mean(unit_seed_metrics, axis=0)
                per_fold.append({'fold': k, 'training': training, 'selection': policy,
                                 'config_indices': indices,
                                 'ensemble': dict(zip(('pcc', 'rmse', 'mae'), metrics.mean(0).tolist())),
                                 'seed_mean': dict(zip(('pcc', 'rmse', 'mae'), seed_metrics.mean(0).tolist())),
                                 'per_environment': metrics.tolist()})
                chosen.append({'fold': k, 'training': training, 'selection': policy,
                               'config_indices': indices, 'scores_config_environment': scores.mean(0).tolist()})
    for name, bank in banks.items():
        if set(bank) != expected_keys[name]:
            raise ValueError(f'Frozen prediction key set mismatch for {name}')
    summary = {}
    for training in banks:
        for policy in ('common', 'per_environment'):
            entries = [r for r in per_fold if r['training'] == training and r['selection'] == policy]
            summary[f'{training}_{policy}'] = {
                'ensemble': {m: float(np.mean([r['ensemble'][m] for r in entries]))
                             for m in ('pcc', 'rmse', 'mae')},
                'seed_mean': {m: float(np.mean([r['seed_mean'][m] for r in entries]))
                              for m in ('pcc', 'rmse', 'mae')},
                'per_environment': np.mean([r['per_environment'] for r in entries], axis=0).tolist()}
    return summary, per_fold, chosen


def run(config, args, runner):
    if args.selection is not None:
        raise ValueError('Re-summary only; no refit or checkpoint selection')
    data = load(Path(config['data']), config['data_sha256'])
    folds = json.loads(Path(config['splits']).read_text())['folds']
    sources, banks = {}, {}
    for name in ('shared', 'independent'):
        source = Path(config[f'{name}_source'])
        info = json.loads((source / 'provenance.json').read_text())
        status = json.loads((source / 'execution_status.json').read_text())
        if status['status'] != 'complete' or status['completed_trajectories'] != status['expected'] \
                or len(status['folds_complete']) != len(folds) \
                or json.loads((source / 'failures.json').read_text()) != []:
            raise ValueError('Incomplete or failed source run')
        if info['data_sha256'] != config['data_sha256'] \
                or info['splits_sha256'] != sha256_of(Path(config['splits'])):
            raise ValueError('Source data/splits mismatch')
        prediction_path = source / 'predictions.csv'
        prediction_sha = sha256_of(prediction_path)
        if prediction_sha != config[f'{name}_predictions_sha256']:
            raise ValueError('Frozen predictions SHA mismatch')
        sources[name] = {'path': str(source), 'predictions_sha256': prediction_sha,
                         'provenance_sha256': sha256_of(source / 'provenance.json'),
                         'selection_sha256': sha256_of(source / 'selection.json')}
        banks[name] = read_predictions(prediction_path, name, list(map(str, data.environments)))
        source_configs = [{k: v for k, v in c.items() if k != 'context_size'}
                          for c in info['config']['configurations']]
        if source_configs != config['configurations'] or info['config']['seeds'] != config['seeds']:
            raise ValueError('Source configuration/seed mismatch')
    summary, per_fold, selections = compute(data, folds, banks, config['seeds'],
                                            len(config['configurations']), config['tie_tolerance'])
    previous = json.loads((Path(config['independent_source']) / 'summary.json').read_text())
    for key, old in [('shared_common', 'shared'), ('independent_per_environment', 'independent')]:
        for m in ('pcc', 'rmse', 'mae'):
            if abs(summary[key]['ensemble'][m] - previous[old]['seed_ensemble'][m]) > 1e-8:
                raise ValueError(f'Known corner failed reproduction: {key}/{m}')
    summary['selection_gains'] = {
        t: summary[f'{t}_per_environment']['ensemble']['pcc'] - summary[f'{t}_common']['ensemble']['pcc']
        for t in banks}
    summary['training_differences'] = {
        p: summary[f'independent_{p}']['ensemble']['pcc'] - summary[f'shared_{p}']['ensemble']['pcc']
        for p in ('common', 'per_environment')}
    args.out.mkdir(parents=True, exist_ok=False)
    for name, payload in [('summary.json', summary), ('per_fold.json', per_fold),
                           ('selection.json', selections), ('failures.json', [])]:
        (args.out / name).write_text(json.dumps(payload, indent=2))
    (args.out / 'provenance.json').write_text(json.dumps({
        'config': config, 'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'sources': sources, 'data_sha256': config['data_sha256'],
        'splits_sha256': sha256_of(Path(config['splits'])),
        'code_sha256': sha256_of(Path(__file__)), 'runner_sha256': sha256_of(runner),
        'environment_order': list(map(str, data.environments)),
        'scope': 'Existing validation predictions only; checkpoints unchanged; no training/early stopping/test evaluation',
        'per_environment_metric_order': ['pcc', 'rmse', 'mae']}, indent=2))
    return summary
