"""Re-evaluation of Mul-PheG2P on a frozen split manifest; splits are read, never regenerated.

Protocol discipline:
* every fit uses only train/val rows; the fitting phase never reads ``Y[test_idx]``;
* test labels are loaded for scoring only after all fits of the run have finished;
* model selection stays inside the model (validation early stopping); no test-side
  selection, no cross-arm choice made from test, and the two scalar-fusion arms are
  reported separately because they fuse different input spaces;
* ``full_dev_refit`` adds a second phase: the validation pass only freezes each arm's
  best epoch and each environment's source epoch, the scored run then refits those
  budgets on the whole development set without any validation split;
* ``execution_status.json`` reaches ``complete`` only after the summary and every table
  have been written, and ``artifact_manifest.json`` is written last so its recorded
  hashes match the files as they finally stand.
"""
from __future__ import annotations

import csv
import inspect
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from wheat599 import mulphe
from wheat599.data import load, sha256_of
from wheat599.metrics import mae, pcc, rmse

METRICS = ('pcc', 'rmse', 'mae')
METRIC_FUNCTIONS = (pcc, rmse, mae)


def git_value(*arguments):
    return subprocess.check_output(['git', *arguments], text=True).strip()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True))


def write_csv(path, fields, rows):
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


# -- frozen inputs ---------------------------------------------------------
def verify_manifest(path, expected_sha256, n_materials, n_environments):
    """Return the manifest, its folds and its verified SHA after checking schema and boundaries."""
    actual = sha256_of(Path(path))
    if actual != expected_sha256:
        raise ValueError(f'Split SHA mismatch: expected {expected_sha256}, got {actual}')
    manifest = json.loads(Path(path).read_text())
    folds = manifest.get('folds')
    if not isinstance(folds, list) or not folds:
        raise ValueError('Manifest has no folds')
    seen_folds, tested = set(), []
    for fold in folds:
        missing = {'fold', 'dev_idx', 'train_idx', 'val_idx', 'test_idx'} - set(fold)
        if missing:
            raise ValueError(f'Fold {fold.get("fold")} is missing {sorted(missing)}')
        number = int(fold['fold'])
        if number in seen_folds:
            raise ValueError(f'Duplicate fold {number}')
        seen_folds.add(number)
        parts = {name: np.asarray(fold[f'{name}_idx'], dtype=np.int64)
                 for name in ('train', 'val', 'dev', 'test')}
        for name, indices in parts.items():
            if indices.ndim != 1 or indices.size == 0:
                raise ValueError(f'Fold {number}: {name}_idx must be a non-empty 1-D list')
            if indices.min() < 0 or indices.max() >= n_materials:
                raise ValueError(f'Fold {number}: {name}_idx outside the material range')
            if len(set(indices.tolist())) != len(indices):
                raise ValueError(f'Fold {number}: duplicate indices in {name}_idx')
        if set(parts['train'].tolist()) & set(parts['val'].tolist()):
            raise ValueError(f'Fold {number}: train and val overlap')
        if set(parts['dev'].tolist()) != set(parts['train'].tolist()) | set(parts['val'].tolist()):
            raise ValueError(f'Fold {number}: dev_idx is not train plus val')
        if set(parts['test'].tolist()) & set(parts['dev'].tolist()):
            raise ValueError(f'Fold {number}: test overlaps the development set')
        tested.extend(parts['test'].tolist())
    if len(tested) != len(set(tested)):
        raise ValueError('A material is tested more than once across folds')
    if set(tested) != set(range(n_materials)):
        raise ValueError('The folds do not cover every material exactly once')
    if n_environments < 2:
        raise ValueError('At least two environments are required for the cross-environment arms')
    return manifest, folds, actual


def verify_references(references):
    """Check the files used for later same-material comparison before any fitting starts."""
    resolved = {}
    for name, entry in (references or {}).items():
        path = Path(entry['oof'])
        if not path.is_file():
            raise FileNotFoundError(f'Missing reference OOF: {path}')
        actual = sha256_of(path)
        if actual != entry['oof_sha256']:
            raise ValueError(f'Reference {name} SHA mismatch: expected {entry["oof_sha256"]}, got {actual}')
        resolved[name] = {'path': str(path), 'oof_sha256': actual}
    return resolved


def load_reference(path, environments):
    """Read a stored OOF table keyed by (fold, material) inside each environment."""
    known = set(environments)
    columns = {name: {} for name in environments}
    rows = 0
    with Path(path).open() as handle:
        for row in csv.DictReader(handle):
            environment = str(row['environment'])
            if environment not in known:
                raise ValueError(f'Reference environment {environment} is not in the data')
            key = (int(row['fold']), int(row['material_index']))
            if key in columns[environment]:
                raise ValueError(f'Duplicate reference row: {environment}/{key}')
            columns[environment][key] = (float(row['y_true']), float(row['y_pred']))
            rows += 1
    return {'columns': columns, 'rows': rows,
            'folds': sorted({key[0] for values in columns.values() for key in values})}


# -- json-safe projection --------------------------------------------------
def jsonable(value):
    if isinstance(value, np.ndarray):
        return {'shape': list(value.shape), 'dtype': str(value.dtype)}
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def diagnostics(entry):
    kept = {key: value for key, value in entry.items()
            if key not in ('prediction', 'val_prediction', 'source_scalars', 'history')}
    kept['epochs_recorded'] = len(entry.get('history', []))
    scalars = entry.get('source_scalars')
    if isinstance(scalars, dict):
        kept['source_scalars_shapes'] = {str(name): list(np.asarray(value).shape)
                                         for name, value in scalars.items()}
    return jsonable(kept)


# -- fitting ---------------------------------------------------------------
def job_sequence(folds, n_environments, seeds):
    return [(int(fold['fold']), target, int(seed))
            for fold in folds for target in range(n_environments) for seed in seeds]


def _arm_epochs(result, arms):
    """The epoch budget the validation pass froze for each trainable arm."""
    frozen = {}
    for arm in arms:
        if arm == 'linear_prediction':
            continue
        epoch = result['arms'][arm].get('best_epoch')
        if epoch is None:
            raise ValueError(f'The validation pass gave no best_epoch for {arm}')
        frozen[arm] = int(epoch)
    return frozen


def _source_epochs(result):
    return {str(entry['name']): int(entry['best_epoch'])
            for entry in (result.get('source_pretraining') or [])}


def _train_sources(config, x_rows, y_rows, environments, merged, seed, device, fixed_epochs=None,
                   x_val_rows=None, y_val_rows=None):
    """Shared source pretraining: validation-driven while freezing epochs, fixed-budget on dev."""
    if (fixed_epochs is None) != (x_val_rows is not None):
        raise ValueError('source pretraining needs a validation split exactly when epochs are not fixed')
    medians = mulphe._fit_impute(x_rows)
    return mulphe.train_source_encoders(
        mulphe._apply_impute(x_rows, medians), y_rows,
        None if x_val_rows is None else mulphe._apply_impute(x_val_rows, medians),
        None if y_val_rows is None else y_val_rows,
        environments, merged, int(seed) + 1000, device, fixed_epochs=fixed_epochs)


def _suite(config, data, arms, seed, fold_number, target, mode, train, val, test, device,
           rows, records, validation, source_models, shared_seconds,
           fixed_epochs=None, source_fixed_epochs=None):
    """One suite of arms for a single fold and target, in validation or frozen full-dev mode."""
    environments = [str(name) for name in data.environments]
    x_all = np.asarray(data.X, dtype=np.float64)
    truth = np.asarray(data.Y, dtype=np.float64)
    sources = [position for position in range(len(environments)) if position != target]
    fixed = mode == 'full_dev_fixed'
    result = mulphe.run_arms(
        x_all[train], truth[train, target],
        None if fixed else x_all[val], None if fixed else truth[val, target], x_all[test],
        target_name=environments[target],
        source_names=[environments[position] for position in sources],
        source_train=truth[np.ix_(train, sources)],
        source_val=None if fixed else truth[np.ix_(val, sources)],
        config=config.get('model_config'), seed=seed, device=device, arms=arms,
        source_models=source_models, fixed_epochs=fixed_epochs,
        source_fixed_epochs=source_fixed_epochs)
    expected = 'fixed_full_dev' if fixed else 'validation'
    if result['mode'] != expected:
        raise ValueError(f'Model reported mode {result["mode"]} for {mode}')
    if fixed and result['source_fixed_epochs'] is not None and source_models is not None:
        raise ValueError('Shared source models and source_fixed_epochs cannot both be given')
    for arm, entry in result['arms'].items():
        prediction = np.asarray(entry['prediction'], dtype=np.float64)
        if prediction.shape != (len(test),) or not np.isfinite(prediction).all():
            raise FloatingPointError(
                f'Non-finite or mis-shaped prediction: fold {fold_number} arm {arm} mode {mode}')
        rows.extend([{'arm': arm, 'fold': fold_number, 'target_environment': environments[target],
                      'seed': seed, 'material_index': int(material), 'y_pred': f'{value:.10g}'}
                     for material, value in zip(test, prediction)])
        records.append({'mode': mode, 'fold': fold_number, 'target_environment': environments[target],
                        'target_position': target, 'seed': seed, 'arm': arm, **diagnostics(entry)})
    validation.append({'mode': mode, 'fold': fold_number, 'target_environment': environments[target],
                       'seed': seed, 'model_mode': result['mode'],
                       'source_encoder_reuse': source_models is not None,
                       'source_pretraining_seconds': result['source_pretraining_seconds'],
                       'shared_source_pretraining_seconds': shared_seconds,
                       'fixed_epochs': result.get('fixed_epochs'),
                       'source_fixed_epochs': result.get('source_fixed_epochs'),
                       'best_epochs': {arm: result['arms'][arm].get('best_epoch') for arm in arms},
                       'source_pretraining': jsonable(result['source_pretraining'])})
    return result


def fit(config, data, folds, device, expected_fits, status_path):
    """Run the frozen grid. Full-dev mode first freezes epochs on validation, then refits on dev."""
    seeds = [int(seed) for seed in config['seeds']]
    arms = list(config['arms'])
    environments = [str(name) for name in data.environments]
    x_all = np.asarray(data.X, dtype=np.float64)
    truth = np.asarray(data.Y, dtype=np.float64)
    merged = {**mulphe.DEFAULT_CONFIG, **(config.get('model_config') or {})}
    reuse = bool(config.get('share_source_encoders', True))
    needs_sources = bool(set(arms) & (set(mulphe.ARMS)
                                      - {'direct', 'target_pretrain', 'linear_prediction'}))
    refit = bool((config.get('full_dev_refit') or {}).get('enabled'))
    rows, records, validation, frozen_epochs = [], [], [], []
    development = [] if refit else rows   # the freezing pass is kept out of the scored grid
    started = time.perf_counter()
    completed = 0
    for fold in folds:
        number = int(fold['fold'])
        train = np.asarray(fold['train_idx'], dtype=np.int64)
        val = np.asarray(fold['val_idx'], dtype=np.int64)
        test = np.asarray(fold['test_idx'], dtype=np.int64)
        dev = np.sort(np.concatenate([train, val]))
        if refit and len(train) + len(val) != len(dev):
            raise ValueError('Full-dev refit needs disjoint train and validation rows')
        for seed in seeds:
            shared = {}
            validation_results = {}
            for target in range(len(environments)):
                source_models, shared_seconds = None, None
                if reuse and needs_sources:
                    key = (number, seed)
                    if key not in shared:
                        began = time.perf_counter()
                        shared[key] = (_train_sources(config, x_all[train], truth[train],
                                                      environments, merged, seed, device,
                                                      x_val_rows=x_all[val], y_val_rows=truth[val]),
                                       time.perf_counter() - began)
                    source_models, shared_seconds = shared[key]
                validation_results[target] = _suite(
                    config, data, arms, seed, number, target, 'validation',
                    train, val, test, device, development, records, validation,
                    source_models, shared_seconds)
            if not refit:
                completed += len(environments)
                continue
            source_epochs = {}
            for result in validation_results.values():
                for name, epoch in _source_epochs(result).items():
                    if source_epochs.setdefault(name, epoch) != epoch:
                        raise ValueError(f'Shared source encoders froze different epochs for {name}')
            if needs_sources and set(source_epochs) != set(environments):
                raise ValueError('The validation pass did not freeze an epoch for every environment: '
                                 f'{sorted(set(environments) - set(source_epochs))}')
            dev_models = None
            if needs_sources:
                began = time.perf_counter()
                dev_models = _train_sources(config, x_all[dev], truth[dev], environments, merged,
                                            seed, device, fixed_epochs=source_epochs)
                dev_seconds = time.perf_counter() - began
            else:
                dev_seconds = None
            frozen_epochs.append({'fold': number, 'seed': seed, 'source_epochs': source_epochs,
                                  'source_encoder_rows': 'dev' if dev_models is not None else 'train',
                                  'arm_epochs': {}})
            current = frozen_epochs[-1]
            for target in range(len(environments)):
                result = validation_results[target]
                current['arm_epochs'][environments[target]] = _arm_epochs(result, arms)
                _suite(config, data, arms, seed, number, target, 'full_dev_fixed',
                       dev, None, test, device, rows, records, validation, dev_models, dev_seconds,
                       fixed_epochs=current['arm_epochs'][environments[target]],
                       source_fixed_epochs=None if dev_models else source_epochs)
                completed += 1
        write_json(status_path, {'status': 'running', 'phase': 'fitting',
                                 'completed_fits': completed, 'expected_fits': expected_fits,
                                 'full_dev_refit': refit,
                                 'seconds': round(time.perf_counter() - started, 3),
                                 'git_commit': git_value('rev-parse', 'HEAD')})
    return rows, records, validation, development, frozen_epochs, \
        round(time.perf_counter() - started, 3)


# -- scoring ---------------------------------------------------------------
def _macro(truth, prediction):
    values = {name: float(function(truth, prediction))
              for name, function in zip(METRICS, METRIC_FUNCTIONS)}
    if not all(np.isfinite(value) for value in values.values()):
        raise FloatingPointError(f'Non-finite score: {values}')
    return values


def _mean(entries):
    return {name: float(np.mean([entry[name] for entry in entries])) for name in METRICS}


def group_predictions(rows, arms, folds, environments, seeds):
    """Collect the per-material predictions into arm/fold/environment/seed vectors."""
    expected = {(arm, int(fold), environment, int(seed))
                for arm in arms for fold in folds for environment in environments for seed in seeds}
    grouped = {key: {} for key in expected}
    for row in rows:
        key = (row['arm'], int(row['fold']), str(row['target_environment']), int(row['seed']))
        if key not in grouped:
            raise ValueError(f'Unexpected prediction key: {key}')
        material = int(row['material_index'])
        if material in grouped[key]:
            raise ValueError(f'Duplicate prediction: {key}/{material}')
        grouped[key][material] = float(row['y_pred'])
    if any(not values for values in grouped.values()):
        raise ValueError('Empty prediction group')
    return grouped


def score(config, data, folds, rows, references):
    """Score stored predictions after every fit; fold means and pooled OOF are kept separate."""
    environments = [str(name) for name in data.environments]
    arms = list(config['arms'])
    seeds = [int(seed) for seed in config['seeds']]
    fold_numbers = [int(fold['fold']) for fold in folds]
    truth = np.asarray(data.Y, dtype=np.float64)
    grouped = group_predictions(rows, arms, fold_numbers, environments, seeds)

    def test_of(fold):
        return np.asarray(fold['test_idx'], dtype=np.int64)

    def vector(arm, number, environment, seed, test):
        values = grouped[(arm, number, environment, seed)]
        if set(values) != set(int(material) for material in test):
            raise ValueError('Prediction materials differ from the frozen test set')
        return np.array([values[int(material)] for material in test], dtype=np.float64)

    per_fold, pooled, oof_vectors = {}, {}, {}
    for arm in arms:
        oof = np.full((len(environments), len(seeds), data.n), np.nan)
        cells = {}
        for e_index, environment in enumerate(environments):
            for fold in folds:
                number, test = int(fold['fold']), test_of(fold)
                for s_index, seed in enumerate(seeds):
                    prediction = vector(arm, number, environment, seed, test)
                    cells[(number, environment, seed)] = (
                        _macro(truth[test, e_index], prediction), len(test))
                    oof[e_index, s_index, test] = prediction
        if not np.isfinite(oof).all():
            raise ValueError('Pooled OOF leaves materials uncovered')

        grid = [{'arm': arm, 'fold': number, 'environment': environment, 'seed': seed,
                 'materials': cells[(number, environment, seed)][1],
                 **cells[(number, environment, seed)][0]}
                for number in fold_numbers for environment in environments for seed in seeds]
        per_environment = {
            environment: _mean([cells[(number, environment, seed)][0]
                                for number in fold_numbers for seed in seeds])
            for environment in environments}
        per_fold[arm] = {
            'grid': grid,
            'per_environment': per_environment,
            'per_seed': {str(seed): {environment: _mean([cells[(number, environment, seed)][0]
                                                         for number in fold_numbers])
                                     for environment in environments} for seed in seeds},
            'macro': _mean([per_environment[environment] for environment in environments]),
            'scope': 'grid is one row per fold/environment/seed; per_environment and macro average '
                     'those per-fold readings'}

        pooled_environment = {
            environment: _mean([_macro(truth[:, e_index], oof[e_index, s_index])
                                for s_index in range(len(seeds))])
            for e_index, environment in enumerate(environments)}
        pooled[arm] = {
            'per_environment': pooled_environment,
            'macro': _mean([pooled_environment[environment] for environment in environments]),
            'per_seed_macro': {str(seed): _mean([_macro(truth[:, e_index], oof[e_index, index])
                                                 for e_index in range(len(environments))])
                               for index, seed in enumerate(seeds)},
            'seed_ensemble_macro': _mean([_macro(truth[:, e_index], oof[e_index].mean(axis=0))
                                          for e_index in range(len(environments))]),
            'scope': 'one pooled OOF vector per environment (every material predicted once); '
                     'not comparable with a fold mean'}
        oof_vectors[arm] = oof

    return per_fold, pooled, compare_reference(config, data, folds, grouped, oof_vectors, references)


# -- reference pairing -----------------------------------------------------
def compare_reference(config, data, folds, grouped, oof_vectors, references):
    """Pair with stored OOF on identical fold/environment/material triples when the folds agree."""
    environments = [str(name) for name in data.environments]
    arms, seeds = list(config['arms']), [int(seed) for seed in config['seeds']]
    truth = np.asarray(data.Y, dtype=np.float64)
    fold_test = {int(fold['fold']): [int(material) for material in fold['test_idx']] for fold in folds}
    comparison = {}
    for name, reference in (references or {}).items():
        columns = reference['columns']
        for e_index, environment in enumerate(environments):
            for (fold_number, material), (value_truth, _) in columns.get(environment, {}).items():
                if abs(float(truth[material, e_index]) - value_truth) > 1e-6:
                    raise ValueError(f'Reference {name} y_true disagrees with the data '
                                     f'for {environment}/{fold_number}/{material}')
        foreign = sorted({fold_number for values in columns.values()
                          for (fold_number, _) in values} - set(fold_test))
        misplaced = [(environment, fold_number, material)
                     for environment, values in columns.items()
                     for (fold_number, material) in values
                     if fold_number in fold_test and material not in set(fold_test[fold_number])]
        if misplaced:
            raise ValueError(f'Reference {name} places {misplaced[0][2]} outside fold '
                             f'{misplaced[0][1]} of the frozen manifest ({len(misplaced)} rows)')

        def pooled_shared(arm):
            """Same-material reading that stays available when the fold structures differ."""
            oof = oof_vectors[arm]
            result = {}
            for e_index, environment in enumerate(environments):
                predictions = {}
                for (_, material), (_, prediction) in columns.get(environment, {}).items():
                    if material in predictions and abs(predictions[material] - prediction) > 1e-12:
                        raise ValueError(f'Reference {name} predicts material {material} of '
                                         f'{environment} twice with different values')
                    predictions[material] = prediction
                materials = sorted(predictions)
                if not materials:
                    continue
                value_truth = truth[np.asarray(materials), e_index]
                scores = {
                    'reference': _macro(value_truth, np.array([predictions[m] for m in materials])),
                    'model': _macro(value_truth, np.array(
                        [float(oof[e_index, :, m].mean()) for m in materials]))}
                scores['model_minus_reference'] = {metric: scores['model'][metric] - scores['reference'][metric]
                                                   for metric in METRICS}
                result[environment] = {'materials': len(materials), **scores}
            return result

        entries = {}
        for arm in arms:
            paired, unpaired = [], []
            for fold_number, test in sorted(fold_test.items()):
                for e_index, environment in enumerate(environments):
                    stored = {material: values for (row_fold, material), values
                              in columns.get(environment, {}).items() if row_fold == fold_number}
                    materials = [material for material in test if material in stored]
                    record = {'fold': fold_number, 'environment': environment,
                              'reference_materials': len(stored), 'paired_materials': len(materials)}
                    if len(materials) != len(test):
                        unpaired.append(record)
                        continue
                    value_truth = truth[np.asarray(materials), e_index]
                    reference_pred = np.array([stored[material][1] for material in materials])
                    model_pred = np.array([np.mean([grouped[(arm, fold_number, environment, seed)][material]
                                                    for seed in seeds]) for material in materials])
                    scores = {'reference': _macro(value_truth, reference_pred),
                              'model': _macro(value_truth, model_pred)}
                    scores['model_minus_reference'] = {metric: scores['model'][metric] - scores['reference'][metric]
                                                       for metric in METRICS}
                    paired.append({**record, 'materials': len(materials), **scores})
            by_environment = {}
            for environment in environments:
                entries_for_environment = [entry for entry in paired if entry['environment'] == environment]
                if not entries_for_environment:
                    continue
                by_environment[environment] = {
                    'folds': [int(entry['fold']) for entry in entries_for_environment],
                    **{side: _mean([entry[side] for entry in entries_for_environment])
                       for side in ('reference', 'model')},
                    'fold_mean_delta': {metric: float(np.mean([
                        entry['model_minus_reference'][metric] for entry in entries_for_environment]))
                        for metric in METRICS},
                    'pcc_wins_of_folds': int(sum(entry['model_minus_reference']['pcc'] > 0
                                                 for entry in entries_for_environment))}
            macro = {}
            if by_environment:
                sides = [by_environment[environment] for environment in by_environment]
                macro = {**{side: _mean([entry[side] for entry in sides])
                             for side in ('reference', 'model')},
                         'delta': {metric: float(np.mean([entry['fold_mean_delta'][metric] for entry in sides]))
                                   for metric in METRICS}}
            entries[arm] = {
                'protocol': {'reference_folds': reference['folds'], 'run_folds': sorted(fold_test),
                             'reference_rows': reference['rows'],
                             'foreign_reference_folds': foreign,
                             'paired_units': len(paired), 'unpaired_units': len(unpaired),
                             'same_fold_structure': bool(not foreign and not unpaired)},
                'per_fold': paired, 'unpaired': unpaired,
                'fold_mean': {'per_environment': by_environment, 'macro': macro},
                'pooled_shared_materials': pooled_shared(arm),
                'note': 'fold-level pairing uses the same fold/environment/material rows on both sides; '
                        'when the reference folds differ, only the pooled same-material reading applies '
                        'and it is not a same-protocol comparison'}
        comparison[name] = entries
    return comparison


# -- driver ----------------------------------------------------------------
def run(config, args, runner):
    if args.selection is not None:
        raise ValueError('Mul-PheG2P selects on validation inside the model; there is no selection input')
    seeds = [int(seed) for seed in config['seeds']]
    arms = list(config['arms'])
    if not arms or len(set(arms)) != len(arms) or not set(arms) <= set(mulphe.ARMS):
        raise ValueError('Requested arms must be a non-empty distinct subset of the implemented arms')
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Seeds must be a non-empty distinct list')
    torch.set_num_threads(int(config.get('cpu_threads', 2)))
    device = torch.device(config['device'])
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(float(config['gpu_memory_fraction']))
    started = time.perf_counter()
    data = load(Path(config['data']), config['data_sha256'])
    manifest, folds, splits_sha256 = verify_manifest(
        Path(config['splits']), config['splits_sha256'], data.n, data.n_env)
    references = verify_references(config.get('references'))
    expected_fits = len(folds) * data.n_env * len(config['seeds'])
    args.out.mkdir(parents=True, exist_ok=False)

    write_json(args.out / 'manifest_validation.json', {
        'splits_sha256': splits_sha256, 'rule': manifest.get('rule'),
        'numpy': manifest.get('numpy'), 'sklearn': manifest.get('sklearn'),
        'description': manifest.get('description'),
        'folds': [{'fold': int(fold['fold']), 'train': len(fold['train_idx']),
                   'val': len(fold['val_idx']), 'dev': len(fold['dev_idx']),
                   'test': len(fold['test_idx'])} for fold in folds],
        'expected_fits': expected_fits, 'references': references,
        'arms': list(config['arms']), 'seeds': [int(seed) for seed in config['seeds']]})

    write_json(args.out / 'provenance.json', {
        'config': config, 'git_commit': git_value('rev-parse', 'HEAD'),
        'git_dirty': git_value('status', '--porcelain') != '',
        'data_sha256': config['data_sha256'], 'splits_sha256': splits_sha256,
        'code_sha256': {'mulphe.py': sha256_of(Path(mulphe.__file__)),
                        'mulphe_run.py': sha256_of(Path(__file__)),
                        'config.json': sha256_of(Path(args.config)),
                        'runner': sha256_of(Path(runner))},
        'environment_order': [str(name) for name in data.environments],
        'official': {'repo': mulphe.OFFICIAL_REPO, 'commit': mulphe.OFFICIAL_COMMIT},
        'model_selection': {
            'direct': 'validation Pearson', 'target_pretrain': 'validation MSE + 0.2 * Pearson',
            'source_pretraining': 'validation MSE + 0.2 * Pearson',
            'frozen_sources': 'validation MSE + 0.2 * Pearson',
            'scalar_prediction': 'validation MSE (official fine_tune_predict)',
            'embedding': 'validation Pearson with patience',
            'authority': 'mulphe.py implements these rules; the runner supplies no selection'},
        'source_encoder_sharing': bool(config.get('share_source_encoders', True))
                                   and bool(set(config['arms'])
                                            & (set(mulphe.ARMS) - {'direct', 'target_pretrain'})),
        'linear_source_reference': 'model arm linear_prediction (OLS on frozen source predictions)',
        'full_dev_refit': config.get('full_dev_refit') or {'enabled': False},
        'deviations': ['train-only scaling and imputation',
                       'no test phenotype enters any fit',
                       'source environments always exclude the target',
                       'no validation recalibration of the outputs'],
        'scope': 'frozen manifest read only; validation early stopping inside the model; '
                 'test labels accessed after every fit; no test-side selection or cross-arm choice'})

    rows, records, validation, development, frozen_epochs, fit_seconds = [], [], [], [], [], 0.0
    phase = 'fitting'
    status_path = args.out / 'execution_status.json'
    failures_path = args.out / 'failures.json'
    prediction_fields = ['arm', 'fold', 'target_environment', 'seed', 'material_index', 'y_pred']
    try:
        rows, records, validation, development, frozen_epochs, fit_seconds = fit(
            config, data, folds, device, expected_fits, status_path)
        phase = 'summary'
        write_csv(args.out / 'predictions.csv', prediction_fields,
                  sorted(rows, key=lambda row: (row['arm'], row['fold'], row['target_environment'],
                                                row['seed'], row['material_index'])))
        if development is not rows:
            write_csv(args.out / 'development_predictions.csv', prediction_fields,
                      sorted(development, key=lambda row: (row['arm'], row['fold'],
                                                           row['target_environment'],
                                                           row['seed'], row['material_index'])))
        write_json(args.out / 'frozen_epochs.json', frozen_epochs)
        write_json(args.out / 'fits.json', records)
        write_json(args.out / 'validation.json', validation)

        loaded = {name: load_reference(entry['path'], [str(e) for e in data.environments])
                  for name, entry in references.items()}
        per_fold, pooled, comparison = score(config, data, folds, rows, loaded)
        grid = [row for arm in arms for row in per_fold[arm]['grid']]
        write_csv(args.out / 'per_fold.csv',
                  ['arm', 'fold', 'environment', 'seed', 'materials', 'pcc', 'rmse', 'mae'],
                  [{key: (f'{row[key]:.12g}' if key in METRICS else row[key])
                    for key in ('arm', 'fold', 'environment', 'seed', 'materials',
                                'pcc', 'rmse', 'mae')} for row in grid])
        write_json(args.out / 'summary.json', {
            'per_fold': per_fold, 'pooled_oof': pooled, 'comparison_vs_reference': comparison,
            'metric_order': list(METRICS), 'arms': arms,
            'full_dev_refit': bool((config.get('full_dev_refit') or {}).get('enabled')),
            'scored_mode': 'full_dev_fixed' if (config.get('full_dev_refit') or {}).get('enabled')
                           else 'validation',
            'folds': [int(fold['fold']) for fold in folds],
            'seeds': [int(seed) for seed in config['seeds']],
            'completed_fits': expected_fits, 'training_seconds': fit_seconds,
            'note': 'per_fold.grid is one row per fold/environment/seed; pooled_oof evaluates one OOF '
                    'vector per environment where every material appears once — the two are different '
                    'readings and are reported separately'})

        write_json(args.out / 'timings.json', {
            'total_seconds': round(time.perf_counter() - started, 3), 'fit_seconds': fit_seconds,
            'device': str(device), 'amp': False, 'dtype': 'float32',
            'cuda': None if device.type != 'cuda' else {
                'name': torch.cuda.get_device_name(device),
                'peak_allocated_megabytes': float(torch.cuda.max_memory_allocated(device)) / 2 ** 20,
                'peak_reserved_megabytes': float(torch.cuda.max_memory_reserved(device)) / 2 ** 20}})

        write_json(failures_path, [])
        write_json(status_path, {'status': 'complete', 'phase': 'summary_written',
                                'completed_fits': expected_fits, 'expected_fits': expected_fits,
                                'folds_complete': sorted({row['fold'] for row in records}),
                                'seconds': round(time.perf_counter() - started, 3)})
    except Exception as error:
        failures_path.write_text(json.dumps(
            [{'type': type(error).__name__, 'message': str(error), 'phase': phase,
              'predictions_written': len(rows), 'at': time.strftime('%Y-%m-%dT%H:%M:%S%z')}], indent=2))
        write_json(status_path, {'status': 'failed', 'phase': phase,
                                'completed_fits': len({(row['fold'], row['target_environment'],
                                                        row['seed']) for row in records}),
                                'expected_fits': expected_fits,
                                'seconds': round(time.perf_counter() - started, 3)})
        raise
    finally:
        # the manifest is written last, after the terminal status and failures are settled
        artifacts = {path.name: {'bytes': path.stat().st_size, 'sha256': sha256_of(path)}
                     for path in sorted(args.out.iterdir())
                     if not path.is_dir() and path.name != 'artifact_manifest.json'}
        write_json(args.out / 'artifact_manifest.json', {
            'files': artifacts, 'weights_stored': False,
            'note': 'the model returns state hashes only, so no checkpoint files exist for this run'})
    return per_fold, pooled, comparison
