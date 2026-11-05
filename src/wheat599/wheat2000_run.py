"""Wheat2000 TabM development selection and frozen full-development refit.

Development trains only inner train/validation rows of every outer fold. Its
``selection.json`` freezes one configuration per (fold, trait) and one integer
epoch count per (fold, trait, seed). Final refits every (fold, trait, seed) on
the full outer development split for exactly those epochs, writes seed
predictions without reading outer test phenotypes, and scores the test set
once after every fit has finished.
"""
import csv
import json
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from wheat599.data import sha256_of
from wheat599.environment_run import Trajectory, train_trajectory
from wheat599.genomic_tabm_run import model_source_sha, verify_versions
from wheat599.independent_final import environment_provenance, fit_one, git_value
from wheat599.metrics import mae, pcc, rmse
from wheat599.wheat2000_data import load_wheat2000

METRICS = ('pcc', 'rmse', 'mae')
SOURCE_FILES = ('wheat2000_run.py', 'wheat2000_data.py', 'environment_run.py',
                'environment_tabm.py', 'genomic_tabm.py', 'independent_final.py',
                'metrics.py', 'retrieval_run.py', 'ragp_run.py', 'data.py')


# -- small utilities -------------------------------------------------------
def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str))


def write_csv(path, rows, fields):
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def metric_dict(truth, prediction):
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    result = {'pcc': float(pcc(truth, prediction)),
              'rmse': float(rmse(truth, prediction)),
              'mae': float(mae(truth, prediction))}
    if not all(np.isfinite(value) for value in result.values()):
        raise FloatingPointError('Nonfinite Wheat2000 metric')
    return result


def mean_metrics(entries):
    return {name: float(np.mean([entry[name] for entry in entries])) for name in METRICS}


def load_data(config):
    return load_wheat2000(Path(config['data']), config['data_sha256'],
                          Path(config['cache']), config['cache_sha256'])


def load_protocol(config):
    """Verify the frozen data/split facts before any fit or output directory."""
    data = load_data(config)
    splits_path = Path(config['splits'])
    expected_splits = config.get('splits_sha256')
    if expected_splits is not None and sha256_of(splits_path) != str(expected_splits):
        raise ValueError('Splits sha256 does not match the config')
    document = read_json(splits_path)
    folds = document['folds'] if isinstance(document, dict) else document
    if not isinstance(folds, list) or not folds:
        raise ValueError('Splits must be a non-empty list or a document with folds')
    traits = [str(value) for value in data.trait_names]
    if config.get('traits') is not None \
            and [str(value) for value in config['traits']] != traits:
        raise ValueError('Config traits do not match the cache trait order')
    if isinstance(document, dict):
        if str(document.get('data_sha256')) != str(config['data_sha256']) \
                or str(document.get('cache_sha256')) != str(config['cache_sha256']):
            raise ValueError('Splits manifest data/cache sha256 disagrees with the config')
        if document.get('trait_names') is not None \
                and [str(value) for value in document['trait_names']] != traits:
            raise ValueError('Splits manifest trait order disagrees with the cache')
        if int(document.get('outer_folds', len(folds))) != len(folds):
            raise ValueError('Splits manifest outer fold count disagrees')
        if int(document.get('inner_folds', 3)) != 3:
            raise ValueError('Splits manifest inner fold count must be three')
    expected_shape = config.get('expected_shape')
    if expected_shape is not None \
            and list(expected_shape) != [int(data.X.shape[0]), int(data.X.shape[1]),
                                         int(data.Y.shape[1])]:
        raise ValueError('Cache shape does not match expected_shape')
    universe = np.arange(len(data.sample_ids), dtype=np.int64)
    for fold in folds:
        dev = np.asarray(fold['dev_idx'], dtype=np.int64)
        test = np.asarray(fold['test_idx'], dtype=np.int64)
        if len(fold['inner']) != 3:
            raise ValueError('Every outer fold must contain three inner splits')
        if set(dev.tolist()) & set(test.tolist()) \
                or sorted(np.r_[dev, test].tolist()) != universe.tolist():
            raise ValueError('Outer fold does not partition every material')
        for inner in fold['inner']:
            fit_index = np.asarray(inner['train_idx'], dtype=np.int64)
            val_index = np.asarray(inner['val_idx'], dtype=np.int64)
            if set(fit_index.tolist()) & set(val_index.tolist()) \
                    or sorted(np.r_[fit_index, val_index].tolist()) != sorted(dev.tolist()):
                raise ValueError('Inner fold does not partition its outer dev split')
    return data, folds, traits


def prepare_device(config):
    device = torch.device(config.get('device', 'cuda'))
    torch.set_num_threads(int(config.get('cpu_threads', 2)))
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(float(config['gpu_memory_fraction']))
    return device


def source_hashes(config_path, runner):
    here = Path(__file__).parent
    paths = [here / name for name in SOURCE_FILES]
    if config_path is not None:
        paths.append(Path(config_path))
    if runner is not None:
        paths.append(Path(runner))
    return {str(path): sha256_of(path) for path in paths if Path(path).is_file()}


def safe_model_source():
    try:
        return model_source_sha()
    except Exception as error:  # pragma: no cover - depends on the installed packages
        return {'error': repr(error)}


def ensure_frozen_selection(path):
    repo = Path(git_value('rev-parse', '--show-toplevel'))
    try:
        relative = Path(path).resolve().relative_to(repo.resolve())
    except ValueError as error:
        raise ValueError('Selection must live inside the repository') from error
    subprocess.check_output(['git', 'ls-files', '--error-unmatch', str(relative)], text=True)
    dirty = git_value('status', '--porcelain', '--', str(relative))
    if dirty:
        raise ValueError(f'Selection is not committed cleanly: {dirty}')


def validate_config(config):
    required = ('data', 'data_sha256', 'cache', 'cache_sha256', 'splits',
                'seeds', 'configurations', 'training', 'efficiency')
    missing = [name for name in required if name not in config]
    if missing:
        raise ValueError(f'Missing config keys: {missing}')
    seeds = [int(value) for value in config['seeds']]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('seeds must be distinct and non-empty')
    if not config['configurations']:
        raise ValueError('configurations must be non-empty')
    training = config['training']
    for name in ('max_epochs', 'patience', 'batch_size', 'eval_batch_size', 'gradient_clip'):
        if name not in training:
            raise ValueError(f'Missing training key {name}')
    if int(training['max_epochs']) < 0:
        raise ValueError('max_epochs must be non-negative')
    if 'save_weights' in config and not bool(config['save_weights']):
        raise ValueError('Wheat2000 runs require save_weights for the final phase')


# -- development -----------------------------------------------------------
def _seed_ensemble(records, seeds):
    by_seed = {int(record['trajectory']['seed']): record for record in records}
    if set(by_seed) != {int(seed) for seed in seeds}:
        raise ValueError('Missing seeds in an inner unit')
    reference = np.asarray(by_seed[int(seeds[0])]['material_index'], dtype=np.int64)
    values = []
    for seed in seeds:
        record = by_seed[int(seed)]
        ids = np.asarray(record['material_index'], dtype=np.int64)
        if not np.array_equal(ids, reference):
            raise ValueError('Seed material order differs inside an inner unit')
        values.append(np.asarray(record['prediction'], dtype=np.float64))
    return reference, np.mean(values, axis=0)


def select_fold(records, fold_id, data, config):
    """One configuration per (fold, trait) and one median epoch per seed."""
    seeds = [int(value) for value in config['seeds']]
    configurations = config['configurations']
    fold_records = [record for record in records
                    if record['trajectory']['fold'] == int(fold_id)]
    config_rows, epoch_rows = [], []
    for trait_index, trait_name in enumerate(data.trait_names):
        scored = []
        for config_index in range(len(configurations)):
            unit_rows = [record for record in fold_records
                         if record['trajectory']['environment'] == trait_index
                         and record['trajectory']['config_index'] == config_index]
            if len(unit_rows) != 3 * len(seeds):
                raise ValueError('Incomplete trajectory grid for one trait/config')
            inner_scores, inner_seed_means = [], []
            for inner in range(3):
                unit = [record for record in unit_rows
                        if record['trajectory']['inner'] == inner]
                ids, ensemble = _seed_ensemble(unit, seeds)
                truth = np.asarray(data.Y[ids, trait_index], dtype=np.float64)
                inner_scores.append(float(pcc(truth, ensemble[:, 0])))
                inner_seed_means.append(float(np.mean(
                    [record['metrics']['pcc'] for record in unit])))
            score = float(np.mean(inner_scores))
            if not np.isfinite(score):
                raise FloatingPointError('Nonfinite configuration selection score')
            scored.append({'config_index': config_index,
                           'seed_ensemble_pcc': score,
                           'seed_mean_pcc': float(np.mean(inner_seed_means)),
                           'inner_seed_ensemble_pcc': inner_scores,
                           'inner_seed_mean_pcc': inner_seed_means})
        best = max(scored, key=lambda item: (round(item['seed_ensemble_pcc'], 12),
                                            -item['config_index']))
        config_rows.append({
            'fold': int(fold_id), 'trait': str(trait_name), 'trait_index': int(trait_index),
            'config_index': int(best['config_index']),
            'configuration': dict(configurations[best['config_index']]),
            'seed_ensemble_pcc': best['seed_ensemble_pcc'],
            'seed_mean_pcc': best['seed_mean_pcc'],
            'inner_seed_ensemble_pcc': best['inner_seed_ensemble_pcc'],
            'inner_seed_mean_pcc': best['inner_seed_mean_pcc'],
        })
        for seed in seeds:
            best_epochs = []
            for inner in range(3):
                matches = [record for record in fold_records
                           if record['trajectory']['environment'] == trait_index
                           and record['trajectory']['config_index'] == best['config_index']
                           and record['trajectory']['seed'] == seed
                           and record['trajectory']['inner'] == inner]
                if len(matches) != 1:
                    raise ValueError('Missing trajectory for a median epoch row')
                best_epochs.append(int(matches[0]['best_epoch']))
            epoch_rows.append({
                'fold': int(fold_id), 'trait': str(trait_name),
                'trait_index': int(trait_index), 'seed': int(seed),
                'config_index': int(best['config_index']),
                'per_inner_best_epochs': best_epochs,
                'median_epoch': int(sorted(best_epochs)[len(best_epochs) // 2]),
            })
    return config_rows, epoch_rows


def development_summary(records, selection_rows, epoch_rows, folds, data, config, cost):
    by_trait = {}
    for row in selection_rows:
        by_trait.setdefault(row['trait'], []).append(row)
    per_trait = [{
        'trait': trait, 'trait_index': int(rows[0]['trait_index']),
        'mean_seed_ensemble_pcc': float(np.mean(
            [row['seed_ensemble_pcc'] for row in rows])),
        'mean_seed_mean_pcc': float(np.mean([row['seed_mean_pcc'] for row in rows])),
        'folds': len(rows),
    } for trait, rows in sorted(by_trait.items(),
                                key=lambda item: item[1][0]['trait_index'])]
    return {
        'status': 'complete', 'phase': 'development',
        'scope': 'outer test rows never indexed; selection uses inner validation only',
        'traits': [str(value) for value in data.trait_names],
        'seeds': [int(value) for value in config['seeds']],
        'folds': [int(fold['fold']) for fold in folds],
        'configurations': len(config['configurations']),
        'counts': {
            'fits': len(records),
            'selection_rows': len(selection_rows),
            'median_epoch_rows': len(epoch_rows),
        },
        'primary_reading': 'seed-ensemble inner PCC, configuration selected per (fold, trait)',
        'mean_seed_ensemble_pcc': float(np.mean(
            [row['seed_ensemble_pcc'] for row in selection_rows])),
        'mean_seed_mean_pcc': float(np.mean(
            [row['seed_mean_pcc'] for row in selection_rows])),
        'per_trait': per_trait,
        'cost': cost,
    }


def write_dev_predictions(handle, records, trait_names):
    for record in records:
        trajectory = record['trajectory']
        trait = str(trait_names[trajectory['environment']])
        for material, value in zip(record['material_index'], record['prediction']):
            handle.write(f"{trajectory['fold']},{trait},{trajectory['environment']},"
                         f"{trajectory['inner']},{trajectory['seed']},"
                         f"{trajectory['config_index']},{material},{float(value[0]):.10g}\n")


def development(config, args, runner):
    validate_config(config)
    verified_versions = verify_versions(config)
    device = prepare_device(config)
    data, folds, trait_names = load_protocol(config)
    seeds = [int(value) for value in config['seeds']]
    expected = (len(folds) * len(trait_names) * len(config['configurations'])
                * len(seeds) * 3)
    if 'expected_development_fits' in config \
            and int(config['expected_development_fits']) != expected:
        raise ValueError('Development fit budget disagrees with expected_development_fits')
    model_data = SimpleNamespace(X=data.X, Y=data.Y, environments=trait_names)
    config_path = Path(args.config)
    hashes = source_hashes(config_path, runner)
    args.out.mkdir(parents=True, exist_ok=False)

    document = {
        'schema_version': 1, 'phase': 'development',
        'data_sha256': config['data_sha256'], 'cache_sha256': config['cache_sha256'],
        'splits_sha256': sha256_of(Path(config['splits'])),
        'config_sha256': sha256_of(config_path), 'config': config,
        'source_sha256': hashes, 'model_source': safe_model_source(),
        'traits': trait_names, 'seeds': seeds, 'configs': [], 'epochs': [],
    }
    state = {'status': 'running', 'phase': 'development', 'completed_fits': 0,
             'expected_fits': expected, 'folds_complete': []}
    write_json(args.out / 'execution_status.json', state)
    write_json(args.out / 'provenance.json', {
        **document, 'git_commit': git_value('rev-parse', 'HEAD'),
        'environment': environment_provenance(device),
        'package_versions': verified_versions,
        'scope': 'Development only: inner fit/validation; outer test rows never read',
    })

    records, failures = [], []
    started = time.perf_counter()
    predictions_path = args.out / 'predictions.csv'
    trajectories_path = args.out / 'trajectories.jsonl'
    predictions_path.write_text(
        'fold,trait,trait_index,inner,seed,config_index,material_index,y_pred\n')
    trajectories_path.write_text('')
    try:
        for fold in folds:
            fold_id = int(fold['fold'])
            for trait_index in range(len(trait_names)):
                for seed in seeds:
                    for config_index, configuration in enumerate(config['configurations']):
                        for inner in range(3):
                            trajectory = Trajectory(
                                fold_id, inner, seed, 'independent',
                                config_index, dict(configuration), trait_index)
                            try:
                                record = train_trajectory(model_data, fold, inner, trajectory,
                                                          config, device)
                            except Exception as error:
                                failures.append({'trajectory': trajectory.__dict__,
                                                 'reason': repr(error)})
                                write_json(args.out / 'failures.json', failures)
                                write_json(args.out / 'execution_status.json', state)
                                raise
                            record.pop('state')
                            records.append(record)
                            state['completed_fits'] += 1
                            with trajectories_path.open('a') as handle:
                                handle.write(json.dumps(
                                    {key: value for key, value in record.items()
                                     if key != 'prediction'}, default=str) + '\n')
                            write_json(args.out / 'failures.json', failures)
                            write_json(args.out / 'execution_status.json', state)
            config_rows, epoch_rows = select_fold(records, fold_id, data, config)
            document['configs'].extend(config_rows)
            document['epochs'].extend(epoch_rows)
            write_json(args.out / 'selection.json', document)
            state['folds_complete'].append(fold_id)
            write_json(args.out / 'execution_status.json', state)
            with predictions_path.open('a') as handle:
                write_dev_predictions(handle, [record for record in records
                                              if record['trajectory']['fold'] == fold_id],
                                      trait_names)
        state['status'] = 'complete'
    except Exception as error:
        if not failures or failures[-1].get('reason') != repr(error):
            failures.append({'stage': 'development', 'reason': repr(error)})
        state['status'] = 'failed'
        raise
    finally:
        write_json(args.out / 'failures.json', failures)
        write_json(args.out / 'execution_status.json', state)
        if document['configs']:
            write_json(args.out / 'selection.json', document)
        if records:
            write_json(args.out / 'trajectories.json',
                       [{key: value for key, value in record.items()
                         if key != 'prediction'} for record in records])
    if state['status'] != 'complete':
        return
    try:
        if len(document['configs']) != len(folds) * len(trait_names) \
                or len(document['epochs']) != len(folds) * len(trait_names) * len(seeds):
            raise RuntimeError('Incomplete development selection grid')
        cost = {
            'wall_seconds': time.perf_counter() - started,
            'fit_seconds': float(np.sum([record['train_seconds'] for record in records])),
            'preprocess_seconds': float(np.sum(
                [record['preprocess_seconds'] for record in records])),
            'inference_seconds': float(np.sum(
                [record['inference_seconds'] for record in records])),
            'parameters': sorted({int(record['parameters']) for record in records
                                  if record.get('parameters') is not None}),
            'max_peak_megabytes': max(
                [record['memory']['total_allocated_peak_megabytes']
                 for record in records if record.get('memory')], default=None),
        }
        summary = development_summary(records, document['configs'], document['epochs'],
                                      folds, data, config, cost)
        summary['selection_sha256'] = sha256_of(args.out / 'selection.json')
        summary['predictions_sha256'] = sha256_of(predictions_path)
        per_fold = [{'fold': row['fold'], 'trait': row['trait'],
                     'trait_index': row['trait_index'],
                     'config_index': row['config_index'],
                     'seed_ensemble_pcc': row['seed_ensemble_pcc'],
                     'seed_mean_pcc': row['seed_mean_pcc']}
                    for row in document['configs']]
        write_csv(args.out / 'per_fold.csv', per_fold,
                  ['fold', 'trait', 'trait_index', 'config_index',
                   'seed_ensemble_pcc', 'seed_mean_pcc'])
        write_csv(args.out / 'per_trait.csv', summary['per_trait'],
                  ['trait', 'trait_index', 'folds', 'mean_seed_ensemble_pcc',
                   'mean_seed_mean_pcc'])
        summary['per_fold_sha256'] = sha256_of(args.out / 'per_fold.csv')
        summary['per_trait_sha256'] = sha256_of(args.out / 'per_trait.csv')
        summary['trajectories_sha256'] = sha256_of(args.out / 'trajectories.json')
        write_json(args.out / 'summary.json', summary)
        write_json(args.out / 'timings.json',
                   {'total_seconds': time.perf_counter() - started})
    except Exception as error:
        state['status'] = 'failed'
        failures.append({'stage': 'summary', 'reason': repr(error)})
        write_json(args.out / 'failures.json', failures)
        write_json(args.out / 'execution_status.json', state)
        raise


# -- final -----------------------------------------------------------------
def validate_selection(config, selection, config_path, runner, folds, traits):
    if int(selection.get('schema_version', -1)) != 1:
        raise ValueError('Unsupported selection schema')
    if selection.get('phase') != 'development':
        raise ValueError('Selection is not a development output')
    if selection.get('config') != config:
        raise ValueError('Selection config does not match the current config')
    if str(selection.get('config_sha256')) != sha256_of(config_path):
        raise ValueError('Selection config sha256 does not match')
    if str(selection.get('data_sha256')) != str(config['data_sha256']):
        raise ValueError('Selection data sha256 does not match')
    if str(selection.get('cache_sha256')) != str(config['cache_sha256']):
        raise ValueError('Selection cache sha256 does not match')
    if str(selection.get('splits_sha256')) != sha256_of(Path(config['splits'])):
        raise ValueError('Selection splits sha256 does not match')
    if selection.get('source_sha256') != source_hashes(config_path, runner):
        raise ValueError('Selection source sha256 does not match')
    if selection.get('model_source') != safe_model_source():
        raise ValueError('Selection model source does not match')
    if [str(value) for value in selection.get('traits', [])] != [str(value) for value in traits]:
        raise ValueError('Selection traits do not match')
    if [int(value) for value in selection.get('seeds', [])] \
            != [int(value) for value in config['seeds']]:
        raise ValueError('Selection seeds do not match')

    fold_ids = [int(fold['fold']) for fold in folds]
    expected_rows = len(fold_ids) * len(traits)
    config_rows = selection.get('configs') or []
    epoch_rows = selection.get('epochs') or []
    if len(config_rows) != expected_rows:
        raise ValueError('Selection config row count is incomplete')
    if len(epoch_rows) != expected_rows * len(config['seeds']):
        raise ValueError('Selection median epoch row count is incomplete')
    chosen = {}
    for row in config_rows:
        key = (int(row['fold']), int(row['trait_index']))
        if key in chosen or key[0] not in fold_ids or not 0 <= key[1] < len(traits):
            raise ValueError('Invalid or duplicate selection config row')
        index = int(row['config_index'])
        if not 0 <= index < len(config['configurations']):
            raise ValueError('Selection config index out of range')
        if dict(row['configuration']) != dict(config['configurations'][index]):
            raise ValueError('Selection configuration does not match the config list')
        if not np.isfinite(float(row['seed_ensemble_pcc'])):
            raise ValueError('Nonfinite selection score')
        chosen[key] = row
    epochs = {}
    max_epochs = int(config['training']['max_epochs'])
    for row in epoch_rows:
        key = (int(row['fold']), int(row['trait_index']), int(row['seed']))
        if key in epochs or key[0] not in fold_ids or not 0 <= key[1] < len(traits) \
                or key[2] not in [int(value) for value in config['seeds']]:
            raise ValueError('Invalid or duplicate selection epoch row')
        config_row = chosen.get((key[0], key[1]))
        if config_row is None or int(row['config_index']) != int(config_row['config_index']):
            raise ValueError('Epoch row config disagrees with its config row')
        values = [int(value) for value in row['per_inner_best_epochs']]
        if len(values) != 3 or any(value < 0 or value > max_epochs for value in values):
            raise ValueError('Invalid per-inner best epochs')
        if int(row['median_epoch']) != sorted(values)[1]:
            raise ValueError('Median epoch is not the middle of the three inner epochs')
        epochs[key] = int(row['median_epoch'])
    if set(epochs) != {(fold, trait, int(seed)) for fold in fold_ids
                      for trait in range(len(traits)) for seed in config['seeds']}:
        raise ValueError('Selection epoch grid is incomplete')
    return chosen, epochs


def _local_fold(data, fold):
    dev = np.asarray(fold['dev_idx'], dtype=np.int64)
    test = np.asarray(fold['test_idx'], dtype=np.int64)
    local = SimpleNamespace(
        X=np.concatenate([np.asarray(data.X)[dev], np.asarray(data.X)[test]], axis=0),
        Y=np.asarray(data.Y[dev]),
        environments=[str(value) for value in data.trait_names])
    return dev, test, local


def score_final(data, folds, seeds, prediction_map, config_rows):
    universe = np.arange(len(data.sample_ids), dtype=np.int64)
    covered = np.concatenate([np.asarray(fold['test_idx'], dtype=np.int64)
                              for fold in folds])
    if not np.array_equal(np.sort(covered), universe):
        raise ValueError('Outer test folds do not cover every material exactly once')
    per_fold_rows, seed_metric_rows, oof_rows = [], [], []
    trait_pooled = {}
    for fold in folds:
        fold_id = int(fold['fold'])
        test_ids = np.asarray(fold['test_idx'], dtype=np.int64)
        for trait_index, trait_name in enumerate(data.trait_names):
            predictions = []
            for seed in seeds:
                prediction = np.asarray(prediction_map[(fold_id, trait_index, int(seed))],
                                        dtype=np.float64)
                if prediction.shape != (len(test_ids),) or not np.isfinite(prediction).all():
                    raise ValueError('Invalid final seed prediction')
                predictions.append(prediction)
            ensemble = np.mean(predictions, axis=0)
            truth = np.asarray(data.Y[test_ids, trait_index], dtype=np.float64)
            if not np.isfinite(truth).all():
                raise ValueError('Outer test truth is not finite')
            metrics = metric_dict(truth, ensemble)
            config_row = config_rows[(fold_id, trait_index)]
            per_fold_rows.append({
                'fold': fold_id, 'trait': str(trait_name),
                'trait_index': int(trait_index),
                'config_index': int(config_row['config_index']),
                **metrics,
                'seed_mean_pcc': float(np.mean(
                    [pcc(truth, prediction) for prediction in predictions])),
            })
            for seed, prediction in zip(seeds, predictions):
                seed_metric_rows.append({
                    'fold': fold_id, 'trait': str(trait_name),
                    'trait_index': int(trait_index), 'seed': int(seed),
                    **metric_dict(truth, prediction)})
            trait_pooled.setdefault(trait_index, {
                'truth': [], 'prediction': [], 'fold_metrics': [],
                'seed_truth': {int(seed): [] for seed in seeds},
                'seed_prediction': {int(seed): [] for seed in seeds}})
            trait_pooled[trait_index]['truth'].append(truth)
            trait_pooled[trait_index]['prediction'].append(ensemble)
            trait_pooled[trait_index]['fold_metrics'].append(metrics)
            for seed, prediction in zip(seeds, predictions):
                trait_pooled[trait_index]['seed_truth'][int(seed)].append(truth)
                trait_pooled[trait_index]['seed_prediction'][int(seed)].append(prediction)
            for material, y_true, y_pred in zip(test_ids, truth, ensemble):
                oof_rows.append({'fold': fold_id, 'trait': str(trait_name),
                                 'trait_index': int(trait_index),
                                 'material_index': int(material),
                                 'y_true': float(y_true), 'y_pred': float(y_pred)})
    per_trait_rows, seed_readings = [], {int(seed): [] for seed in seeds}
    for trait_index, trait_name in enumerate(data.trait_names):
        pooled = metric_dict(np.concatenate(trait_pooled[trait_index]['truth']),
                             np.concatenate(trait_pooled[trait_index]['prediction']))
        macro = mean_metrics(trait_pooled[trait_index]['fold_metrics'])
        per_seed = [row for row in seed_metric_rows if row['trait_index'] == trait_index]
        seed_macro = {int(seed): mean_metrics(
            [row for row in per_seed if row['seed'] == int(seed)]) for seed in seeds}
        seed_pooled = {}
        for seed in seeds:
            seed_pooled[int(seed)] = metric_dict(
                np.concatenate(trait_pooled[trait_index]['seed_truth'][int(seed)]),
                np.concatenate(trait_pooled[trait_index]['seed_prediction'][int(seed)]))
        per_trait_rows.append({
            'trait': str(trait_name), 'trait_index': int(trait_index),
            'macro_pcc': macro['pcc'], 'macro_rmse': macro['rmse'],
            'macro_mae': macro['mae'],
            'pooled_pcc': pooled['pcc'], 'pooled_rmse': pooled['rmse'],
            'pooled_mae': pooled['mae'],
            'seed_mean_macro_pcc': float(np.mean(
                [seed_macro[int(seed)]['pcc'] for seed in seeds])),
        })
        for seed in seeds:
            seed_readings[int(seed)].append({'trait': str(trait_name),
                                             'macro': seed_macro[int(seed)],
                                             'pooled': seed_pooled[int(seed)]})
    overall_macro = mean_metrics([
        {'pcc': row['macro_pcc'], 'rmse': row['macro_rmse'], 'mae': row['macro_mae']}
        for row in per_trait_rows])
    overall_pooled = metric_dict([row['y_true'] for row in oof_rows],
                                 [row['y_pred'] for row in oof_rows])
    return {
        'per_fold_trait': per_fold_rows,
        'per_trait': per_trait_rows,
        'seed_metrics': seed_metric_rows,
        'oof': oof_rows,
        'overall_macro': overall_macro,
        'overall_pooled': overall_pooled,
        'seed_readings': {str(seed): values for seed, values in seed_readings.items()},
    }


def final(config, args, runner):
    validate_config(config)
    verified_versions = verify_versions(config)
    device = prepare_device(config)
    data, folds, trait_names = load_protocol(config)
    seeds = [int(value) for value in config['seeds']]
    fit_config = dict(config)
    fit_config.setdefault('batch_size', int(config['training']['batch_size']))
    fit_config.setdefault('gradient_clip', float(config['training']['gradient_clip']))
    selection_path = Path(args.selection)
    selection = read_json(selection_path)
    config_path = Path(args.config)
    chosen, epochs = validate_selection(config, selection, config_path, runner,
                                        folds, trait_names)
    if bool(config.get('require_committed_selection', True)):
        ensure_frozen_selection(selection_path)
    args.out.mkdir(parents=True, exist_ok=False)
    selection_sha = sha256_of(selection_path)
    expected_fits = len(folds) * len(trait_names) * len(seeds)
    if 'expected_final_fits' in config \
            and int(config['expected_final_fits']) != expected_fits:
        raise ValueError('Final fit budget disagrees with expected_final_fits')
    expected_seed_rows = len(trait_names) * len(seeds) * sum(
        len(fold['test_idx']) for fold in folds)
    state = {'status': 'running', 'phase': 'final', 'completed_fits': 0,
             'expected_fits': expected_fits, 'folds_complete': [], 'fit_failures': 0}
    write_json(args.out / 'execution_status.json', state)
    write_json(args.out / 'provenance.json', {
        'config': config, 'config_sha256': sha256_of(config_path),
        'data_sha256': config['data_sha256'], 'cache_sha256': config['cache_sha256'],
        'splits_sha256': sha256_of(Path(config['splits'])),
        'selection': {'path': str(selection_path), 'sha256': selection_sha},
        'source_sha256': source_hashes(config_path, runner),
        'model_source': safe_model_source(),
        'git_commit': git_value('rev-parse', 'HEAD'),
        'environment': environment_provenance(device),
        'package_versions': verified_versions,
        'budget': expected_fits,
        'protocol': 'full-development refit at frozen median inner epoch; test scored once '
                    'after every fit; seed member-mean then seed mean',
    })

    prediction_map, failures, weight_rows, seed_rows = {}, [], [], []
    started = time.perf_counter()
    seed_fields = ['fold', 'trait', 'trait_index', 'seed', 'config_index',
                   'material_index', 'y_pred']
    seed_path = args.out / 'seed_predictions.csv'
    seed_path.write_text(','.join(seed_fields) + '\n')
    seed_handle = seed_path.open('a', newline='')
    seed_writer = csv.DictWriter(seed_handle, fieldnames=seed_fields, lineterminator='\n')
    try:
        for fold in folds:
            fold_id = int(fold['fold'])
            dev, test, local = _local_fold(data, fold)
            for trait_index, trait_name in enumerate(trait_names):
                for seed in seeds:
                    config_index = int(chosen[(fold_id, trait_index)]['config_index'])
                    row = {
                        'fold': fold_id, 'environment_index': trait_index,
                        'seed': int(seed), 'config_index': config_index,
                        'configuration': dict(config['configurations'][config_index]),
                        'epochs': int(epochs[(fold_id, trait_index, int(seed))]),
                        'fit_ids': list(range(len(dev))),
                        'test_ids': list(range(len(dev), len(dev) + len(test))),
                    }
                    try:
                        result = fit_one(row, local, fit_config, device,
                                         args.out / 'weights',
                                         save_weights=bool(config.get('save_weights', True)))
                    except Exception as error:
                        failures.append({'fit': {key: value for key, value in row.items()
                                                 if key != 'fit_ids'},
                                         'reason': repr(error)})
                        state['fit_failures'] += 1
                        raise
                    if [int(value) for value in result['test_ids']] != row['test_ids']:
                        raise ValueError('fit_one returned unexpected test rows')
                    prediction = np.asarray(result['prediction'], dtype=np.float64)
                    prediction_map[(fold_id, trait_index, int(seed))] = prediction
                    first_new_row = len(seed_rows)
                    for position, material in enumerate(test):
                        seed_rows.append({
                            'fold': fold_id, 'trait': str(trait_name),
                            'trait_index': int(trait_index), 'seed': int(seed),
                            'config_index': config_index,
                            'material_index': int(material),
                            'y_pred': float(prediction[position]),
                        })
                    weight_rows.append({
                        'fold': fold_id, 'trait': str(trait_name),
                        'trait_index': int(trait_index), 'seed': int(seed),
                        'config_index': config_index, 'epochs': int(row['epochs']),
                        'path': result['weight_path'],
                        'sha256': result['weight_sha256'],
                        'bytes': int(result['weight_bytes']),
                        'parameters': int(result['parameters']),
                        'fit_seconds': float(result['train_seconds']),
                        'predict_seconds': float(result['predict_seconds']),
                        'peak_megabytes': None if not result.get('memory') else
                        float(result['memory']['total_allocated_peak_megabytes']),
                    })
                    state['completed_fits'] += 1
                    seed_writer.writerows(seed_rows[first_new_row:])
                    seed_handle.flush()
                    write_json(args.out / 'failures.json', failures)
                    write_json(args.out / 'weights_manifest.json', weight_rows)
                    write_json(args.out / 'execution_status.json', state)
            state['folds_complete'].append(fold_id)
            write_json(args.out / 'execution_status.json', state)
        seed_handle.close()
        if len(prediction_map) != expected_fits or len(seed_rows) != expected_seed_rows \
                or state['completed_fits'] != expected_fits:
            raise RuntimeError('Final fit grid is incomplete')
        state['status'] = 'scoring'
        write_json(args.out / 'execution_status.json', state)
        scored = score_final(data, folds, seeds, prediction_map, chosen)
        write_csv(args.out / 'per_fold_trait.csv', scored['per_fold_trait'],
                  ['fold', 'trait', 'trait_index', 'config_index', *METRICS,
                   'seed_mean_pcc'])
        write_csv(args.out / 'per_trait.csv', scored['per_trait'],
                  ['trait', 'trait_index', 'macro_pcc', 'macro_rmse', 'macro_mae',
                   'pooled_pcc', 'pooled_rmse', 'pooled_mae', 'seed_mean_macro_pcc'])
        write_csv(args.out / 'seed_metrics.csv', scored['seed_metrics'],
                  ['fold', 'trait', 'trait_index', 'seed', *METRICS])
        write_csv(args.out / 'oof_predictions.csv', scored['oof'],
                  ['fold', 'trait', 'trait_index', 'material_index', 'y_true', 'y_pred'])
        state['status'] = 'complete'
        summary = {
            'status': 'complete', 'phase': 'final',
            'primary_reading': 'seed ensemble of member-mean predictions; '
                               'test scored once after all fits',
            'traits': trait_names, 'seeds': seeds,
            'folds': [int(fold['fold']) for fold in folds],
            'counts': {'fits': len(weight_rows),
                       'seed_prediction_rows': len(seed_rows),
                       'ensemble_oof_rows': len(scored['oof']),
                       'per_fold_trait_rows': len(scored['per_fold_trait'])},
            'overall_macro': scored['overall_macro'],
            'overall_pooled': scored['overall_pooled'],
            'per_trait': scored['per_trait'],
            'seed_readings': scored['seed_readings'],
            'selection': {'path': str(selection_path), 'sha256': selection_sha},
            'weights': {'count': len(weight_rows),
                        'bytes': int(sum(row['bytes'] for row in weight_rows)),
                        'manifest_sha256': sha256_of(args.out / 'weights_manifest.json')},
            'cost': {
                'wall_seconds': time.perf_counter() - started,
                'fit_seconds': float(sum(row['fit_seconds'] for row in weight_rows)),
                'predict_seconds': float(sum(row['predict_seconds'] for row in weight_rows)),
                'parameters': sorted({row['parameters'] for row in weight_rows}),
                'max_peak_megabytes': max(
                    [row['peak_megabytes'] for row in weight_rows
                     if row['peak_megabytes'] is not None], default=None),
            },
        }
        summary['per_fold_trait_sha256'] = sha256_of(args.out / 'per_fold_trait.csv')
        summary['per_trait_sha256'] = sha256_of(args.out / 'per_trait.csv')
        summary['oof_predictions_sha256'] = sha256_of(args.out / 'oof_predictions.csv')
        summary['seed_predictions_sha256'] = sha256_of(seed_path)
        write_json(args.out / 'summary.json', summary)
    except Exception as error:
        if not failures or failures[-1].get('reason') != repr(error):
            failures.append({'stage': 'final', 'reason': repr(error)})
        state['status'] = 'failed'
        raise
    finally:
        if not seed_handle.closed:
            seed_handle.close()
        write_json(args.out / 'failures.json', failures)
        write_json(args.out / 'weights_manifest.json', weight_rows)
        write_json(args.out / 'execution_status.json', state)
        write_json(args.out / 'timings.json',
                   {'total_seconds': time.perf_counter() - started})


# -- driver ----------------------------------------------------------------
def run(config, args, runner):
    if args.selection is None:
        return development(config, args, runner)
    return final(config, args, runner)
