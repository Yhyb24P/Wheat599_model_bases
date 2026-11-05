"""Freeze the 80-fit independent refit manifest from the existing development trajectories.

No model is built and no label is read: this module only extracts, for every fold,
environment and seed, the configuration chosen by the frozen per-environment selection
and the median of the three inner ``best_epoch`` values that the original independent
run recorded for that configuration.  The manifest keeps the source epoch semantics
(a source ``best_epoch`` of ``k`` means ``k`` update epochs; zero is allowed) and the
material IDs of the development and test splits, so the final runner can refit on the
development rows and evaluate the untouched test rows once.
"""

import csv
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from wheat599.data import load, sha256_of

SEEDS = (42, 2026)
POLICY = 'per_environment'
TRAINING = 'independent'


def ids_sha256(ids):
    return hashlib.sha256(','.join(str(int(i)) for i in ids).encode()).hexdigest()


def median_epoch(values):
    """Median of the three source best_epoch values; the source counts update epochs."""
    values = list(values)
    if len(values) != 3 or any(not isinstance(v, int) or v < 0 for v in values):
        raise ValueError('best_epoch values must be three non-negative integers')
    return sorted(values)[1]


def build_best_epochs(records):
    """Map (fold, inner, seed, config_index, environment) to source best_epoch."""
    out = {}
    for record in records:
        trajectory = record['trajectory']
        if trajectory['variant'] != TRAINING:
            continue
        inner = int(trajectory['inner'])
        if inner not in (0, 1, 2):
            raise ValueError(f'Unexpected inner split {inner}')
        key = (int(trajectory['fold']), inner, int(trajectory['seed']),
               int(trajectory['config_index']), int(trajectory['environment']))
        if key in out:
            raise ValueError(f'Duplicate source trajectory {key}')
        value = record['best_epoch']
        if not isinstance(value, int) or value < 0:
            raise ValueError(f'Invalid best_epoch for {key}: {value!r}')
        out[key] = value
    return out


def chosen_configs(selection, folds):
    """Per-environment configuration for every fold from the frozen re-summary."""
    chosen = {}
    for record in selection:
        if record['training'] == TRAINING and record['selection'] == POLICY:
            fold = int(record['fold'])
            if fold in chosen:
                raise ValueError(f'Duplicate selection for fold {fold}')
            indices = [int(i) for i in record['config_indices']]
            if len(indices) != 4:
                raise ValueError(f'Fold {fold} does not select one configuration per environment')
            chosen[fold] = indices
    if set(chosen) != set(int(fold['fold']) for fold in folds):
        raise ValueError('The frozen per-environment selection does not cover every fold')
    return chosen


def cross_check_selection(chosen, original, folds):
    """The new per-environment choice must equal the one the source run recorded."""
    for fold in folds:
        k = int(fold['fold'])
        expected = [int(original[str(k)]['independent'][str(e)]['config_index'])
                    for e in range(len(original[str(k)]['independent']))]
        if chosen[k] != expected:
            raise ValueError(f'Selection mismatch at fold {k}: {chosen[k]} vs {expected}')


def verify_splits(folds, historical, materials):
    """Development/test boundaries and exact reuse of the historical BiMSGP folds."""
    current = {}
    for fold in folds:
        k = int(fold['fold'])
        dev = [int(i) for i in fold['dev_idx']]
        test = [int(i) for i in fold['test_idx']]
        dev_set, test_set = set(dev), set(test)
        if len(dev) != len(dev_set) or len(test) != len(test_set):
            raise ValueError(f'Repeated material in fold {k}')
        if dev_set & test_set or dev_set | test_set != set(range(materials)):
            raise ValueError(f'Fold {k} does not partition the materials')
        for j, split in enumerate(fold['inner']):
            train = set(int(i) for i in split['train_idx'])
            val = set(int(i) for i in split['val_idx'])
            if train & val or train | val != dev_set:
                raise ValueError(f'Fold {k} inner {j} does not partition the development rows')
        current[k] = (dev, test)
    sizes = sorted((len(dev), len(test)) for dev, test in current.values())
    if sizes not in ([(539, 60)] * 10, [(539, 60)] * 9 + [(540, 59)]):
        raise ValueError(f'Unexpected fold sizes: {sizes}')
    if len(historical) != len(folds):
        raise ValueError('Historical fold count differs')
    for fold in historical:
        k = int(fold['fold'])
        test = [int(i) for i in fold['test_idx']]
        dev_historical = set(int(i) for i in fold['train_idx']) | set(int(i) for i in fold['val_idx'])
        if test != current[k][1]:
            raise ValueError(f'Fold {k} test IDs or order differ from the historical split')
        if dev_historical != set(current[k][0]):
            raise ValueError(f'Fold {k} historical development set differs')
    return {k: {'fit_count': len(dev), 'test_count': len(test)} for k, (dev, test) in current.items()}


def verify_reference(provenance, splits_sha256, data_sha256):
    if provenance.get('splits_sha256') != splits_sha256:
        raise ValueError('Reference result was produced on different splits')
    config = provenance.get('config', {})
    if config.get('data_sha256') != data_sha256:
        raise ValueError('Reference result was produced on different data')


def verify_source_config(source_info, configurations, seeds):
    source = [{k: v for k, v in c.items() if k != 'context_size'}
              for c in source_info['config']['configurations']]
    if source != configurations or [int(s) for s in source_info['config']['seeds']] != seeds:
        raise ValueError('Source configuration/seed mismatch')


def generate(environments, materials, folds, historical, best_epochs, chosen, seeds,
             configurations, data_sha256, splits_sha256, prediction_sha256,
             trajectory_sha256, selection_sha256, source_commit):
    boundaries = verify_splits(folds, historical, materials)
    if len(set(seeds)) != len(seeds) or not seeds:
        raise ValueError('Seeds must be distinct and non-empty')
    rows, seen = [], set()
    for fold in folds:
        k = int(fold['fold'])
        dev = [int(i) for i in fold['dev_idx']]
        test = [int(i) for i in fold['test_idx']]
        if len(chosen[k]) != len(environments):
            raise ValueError(f'Fold {k} does not select one configuration per environment')
        for e, label in enumerate(environments):
            index = chosen[k][e]
            if not 0 <= index < len(configurations):
                raise ValueError(f'Fold {k} environment {e} selects an unknown configuration')
            configuration = configurations[index]
            for seed in seeds:
                if (k, e, int(seed)) in seen:
                    raise ValueError(f'Duplicate row for fold {k} environment {e} seed {seed}')
                seen.add((k, e, int(seed)))
                values = []
                for j in range(3):
                    key = (k, j, int(seed), index, e)
                    if key not in best_epochs:
                        raise ValueError(f'Missing source best_epoch for {key}')
                    values.append(best_epochs[key])
                rows.append({
                    'fold': k, 'environment_index': e, 'environment': str(label), 'seed': int(seed),
                    'config_index': index, 'configuration': dict(configuration),
                    'inner_best_epochs': values, 'median_epochs': median_epoch(values),
                    'fit_ids': dev, 'test_ids': test,
                    'fit_count': len(dev), 'test_count': len(test),
                    'fit_ids_sha256': ids_sha256(dev), 'test_ids_sha256': ids_sha256(test),
                    'data_sha256': data_sha256, 'splits_sha256': splits_sha256,
                    'prediction_source_sha256': prediction_sha256,
                    'trajectory_source_sha256': trajectory_sha256,
                    'selection_source_sha256': selection_sha256,
                    'source_git_commit': source_commit,
                })
    expected = len(folds) * len(environments) * len(seeds)
    if len(rows) != expected:
        raise ValueError(f'Expected {expected} rows, built {len(rows)}')
    verification = {
        'fold_sizes': boundaries,
        'epoch_rule': 'median of the three inner best_epoch values of the selected config; '
                      'source best_epoch counts update epochs and epoch 0 is allowed',
        'selection': 'frozen independent per-environment selection, cross-checked against the '
                     'source environment run',
    }
    return rows, verification


def write_manifest(out, rows, verification, provenance):
    out.mkdir(parents=True, exist_ok=True)
    header = ['fold', 'environment_index', 'environment', 'seed', 'config_index', 'n_blocks',
              'd_block', 'dropout', 'k', 'learning_rate', 'weight_decay',
              'inner_best_epoch_0', 'inner_best_epoch_1', 'inner_best_epoch_2',
              'median_epochs', 'fit_count', 'test_count', 'fit_ids', 'test_ids',
              'fit_ids_sha256', 'test_ids_sha256', 'data_sha256', 'splits_sha256',
              'prediction_source_sha256', 'trajectory_source_sha256',
              'selection_source_sha256', 'source_git_commit']
    with (out / 'manifest.csv').open('w', newline='') as handle:
        writer = csv.writer(handle, lineterminator='\n')
        writer.writerow(header)
        for row in rows:
            settings = row['configuration']
            writer.writerow([
                row['fold'], row['environment_index'], row['environment'], row['seed'],
                row['config_index'], settings['n_blocks'], settings['d_block'],
                settings['dropout'], settings['k'], settings['learning_rate'],
                settings['weight_decay'], *row['inner_best_epochs'], row['median_epochs'],
                row['fit_count'], row['test_count'], ' '.join(map(str, row['fit_ids'])),
                ' '.join(map(str, row['test_ids'])), row['fit_ids_sha256'],
                row['test_ids_sha256'], row['data_sha256'], row['splits_sha256'],
                row['prediction_source_sha256'], row['trajectory_source_sha256'],
                row['selection_source_sha256'], row['source_git_commit']])
    digest = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    (out / 'manifest.json').write_text(json.dumps({
        'schema_version': 1, 'rows': rows, 'verification': verification,
        'rows_sha256': digest, 'row_count': len(rows)}, indent=2))
    (out / 'provenance.json').write_text(json.dumps({
        **provenance, 'rows_sha256': digest, 'row_count': len(rows),
        'scope': 'Frozen refit manifest only; no training, no test label read, test IDs only'},
        indent=2))
    (out / 'failures.json').write_text(json.dumps([], indent=2))
    return digest


def run(config, args, runner):
    if args.selection is not None:
        raise ValueError('This manifest has no selection-input path')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    failures = []
    try:
        data = load(Path(config['data']), config['data_sha256'])
        materials = int(data.Y.shape[0])
        environments = [str(name) for name in data.environments]
        folds = json.loads(Path(config['splits']).read_text())['folds']
        splits_sha256 = sha256_of(Path(config['splits']))
        source = Path(config['source_run'])
        selection_source = Path(config['selection_source'])
        trajectories_path = source / 'trajectories.json'
        prediction_path = source / 'predictions.csv'
        selection_path = selection_source / 'selection.json'
        if sha256_of(trajectories_path) != config['trajectories_sha256']:
            raise ValueError('Frozen trajectory SHA mismatch')
        if sha256_of(prediction_path) != config['predictions_sha256']:
            raise ValueError('Frozen prediction SHA mismatch')
        if sha256_of(selection_path) != config['selection_sha256']:
            raise ValueError('Frozen selection SHA mismatch')
        status = json.loads((source / 'execution_status.json').read_text())
        if status['status'] != 'complete' or status['completed_trajectories'] != status['expected'] \
                or json.loads((source / 'failures.json').read_text()) != []:
            raise ValueError('Source run is incomplete or has failures')
        source_info = json.loads((source / 'provenance.json').read_text())
        if source_info['data_sha256'] != config['data_sha256'] \
                or source_info['splits_sha256'] != splits_sha256:
            raise ValueError('Source run data/splits mismatch')
        verify_source_config(source_info, config['configurations'],
                             [int(seed) for seed in config['seeds']])
        historical_path = Path(config['historical_splits'])
        historical = json.loads(historical_path.read_text())
        splits_info = json.loads(Path(config['splits']).read_text())
        if sha256_of(historical_path) != splits_info['source_sha256']:
            raise ValueError('Historical split SHA mismatch')
        for name in ('huber_source', 'gls_source'):
            verify_reference(json.loads((Path(config[name]) / 'provenance.json').read_text()),
                             splits_sha256, config['data_sha256'])
        best_epochs = build_best_epochs(json.loads(trajectories_path.read_text()))
        selection = json.loads(selection_path.read_text())
        chosen = chosen_configs(selection, folds)
        original = json.loads((source / 'selection.json').read_text())
        cross_check_selection(chosen, original, folds)
        rows, verification = generate(
            environments, materials, folds, historical['folds'], best_epochs, chosen,
            [int(seed) for seed in config['seeds']], config['configurations'],
            config['data_sha256'], splits_sha256, config['predictions_sha256'],
            config['trajectories_sha256'], config['selection_sha256'], source_info['git_commit'])
        verification['historical_splits'] = {
            'path': str(historical_path), 'sha256': sha256_of(historical_path),
            'test_ids': 'identical and in the same order for all folds',
            'development': 'current dev equals historical train + val'}
        verification['references'] = 'Huber and GLS provenance split SHAs equal the current manifest'
        provenance = {
            'config': config,
            'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            'code_sha256': sha256_of(Path(__file__)),
            'source_run': {'path': str(source), 'git_commit': source_info['git_commit'],
                           'trajectories_sha256': config['trajectories_sha256'],
                           'predictions_sha256': config['predictions_sha256']},
            'selection_source': {'path': str(selection_source),
                                 'sha256': config['selection_sha256']},
            'data_sha256': config['data_sha256'], 'splits_sha256': splits_sha256,
            'environment_order': environments, 'materials': materials,
        }
        return write_manifest(out, rows, verification, provenance)
    except Exception as exc:
        failures.append({'reason': repr(exc)})
        raise
    finally:
        (out / 'failures.json').write_text(json.dumps(failures, indent=2))
