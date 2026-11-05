"""Wheat2000 DeepGS/BiMSGP development selection and frozen full-dev refit.

Development trains inner train/validation rows only. For every (fold, trait) it
selects one configuration by the mean of three inner PCCs of the seed-averaged
best-validation prediction. For every seed it freezes the median of the three
inner best epochs and the per-epoch median learning-rate sequence. Final refits
every (fold, trait, seed) on the full outer development split for exactly those
epochs and learning rates, writes seed predictions without reading test
phenotypes, and scores the test set once after every fit has finished.
"""
import csv
import json
import subprocess
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from wheat599 import wheat2000_baselines as baselines
from wheat599.data import sha256_of
from wheat599.independent_final import ensure_frozen_manifest
from wheat599.metrics import mae, pcc, rmse
from wheat599.wheat2000_run import load_protocol, read_json, write_csv, write_json

MODELS = ('wheat2000_deepgs', 'wheat2000_bimsgp')
METRICS = ('pcc', 'rmse', 'mae')
TIE_TOLERANCE = 1e-12
SOURCE_FILES = ('wheat2000_neural_run.py', 'wheat2000_baselines.py', 'wheat2000_run.py',
                'wheat2000_data.py', 'independent_final.py', 'metrics.py', 'data.py')
DEFAULT_EVAL_BATCH = 256
DEFAULT_LR_FACTOR = 0.1


# -- utilities -------------------------------------------------------------
def metric_dict(truth, prediction):
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    values = {'pcc': float(pcc(truth, prediction)),
              'rmse': float(rmse(truth, prediction)),
              'mae': float(mae(truth, prediction))}
    if not np.isfinite(list(values.values())).all():
        raise FloatingPointError('Nonfinite evaluation metric')
    return values


def mean_metrics(entries):
    return {name: float(np.mean([entry[name] for entry in entries])) for name in METRICS}


def source_hashes(config_path, runner):
    here = Path(__file__).parent
    paths = [here / name for name in SOURCE_FILES]
    if config_path is not None:
        paths.append(Path(config_path))
    if runner is not None:
        paths.append(Path(runner))
    return {str(path): sha256_of(path) for path in paths if Path(path).is_file()}


def git_head():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()


def model_source_info(config, data):
    """Source identity of the selected network; verified separately in both phases."""
    if config['model'] == 'wheat2000_deepgs':
        return {'model': 'DeepGS',
                'module_sha256': sha256_of(Path(baselines.__file__))}
    if config['model'] == 'wheat2000_bimsgp':
        info = baselines.verify_bimsgp_source(config['source_path'], config['source_commit'])
        return {'model': 'BiMSGP', **info}
    raise ValueError('Unknown neural baseline model')


def prepare_device(config):
    device = torch.device(config.get('device', 'cuda'))
    torch.set_num_threads(int(config.get('cpu_threads', 2)))
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(float(config['gpu_memory_fraction']))
    return device


def validate_config(config):
    if config.get('model') not in MODELS:
        raise ValueError('model must be wheat2000_deepgs or wheat2000_bimsgp')
    required = ('data', 'data_sha256', 'cache', 'cache_sha256', 'splits',
                'seeds', 'configurations', 'training', 'traits')
    missing = [name for name in required if name not in config]
    if missing:
        raise ValueError(f'Missing config keys: {missing}')
    seeds = [int(value) for value in config['seeds']]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('seeds must be distinct and non-empty')
    if not config['configurations']:
        raise ValueError('configurations must be non-empty')
    for index, entry in enumerate(config['configurations']):
        if not isinstance(entry.get('architecture'), dict):
            raise ValueError(f'Configuration {index} lacks an architecture dict')
        optimizer = entry.get('optimizer') or {}
        if optimizer.get('name') not in ('sgd', 'adamw'):
            raise ValueError(f'Configuration {index} optimizer must be sgd or adamw')
        if float(optimizer.get('lr', 0)) <= 0:
            raise ValueError(f'Configuration {index} learning rate must be positive')
        if float(optimizer.get('weight_decay', 0)) < 0:
            raise ValueError(f'Configuration {index} weight decay must be non-negative')
        if optimizer['name'] == 'sgd' and float(optimizer.get('momentum', 0)) < 0:
            raise ValueError(f'Configuration {index} momentum must be non-negative')
    training = config['training']
    for name in ('max_epochs', 'early_stopping_patience', 'batch_size', 'gradient_clip',
                 'amp', 'genotype_preprocessing', 'scheduler', 'lr_patience',
                 'loss_scale', 'initialization', 'initialization_scale'):
        if name not in training:
            raise ValueError(f'Missing training key {name}')
    if int(training['max_epochs']) < 1:
        raise ValueError('max_epochs must be positive')
    if int(training['early_stopping_patience']) < 0:
        raise ValueError('early_stopping_patience must be non-negative')
    if int(training['batch_size']) < 1:
        raise ValueError('batch_size must be positive')
    if training['genotype_preprocessing'] not in ('raw', 'standardize'):
        raise ValueError('genotype_preprocessing must be raw or standardize')
    if training['scheduler'] not in ('none', 'plateau'):
        raise ValueError('scheduler must be none or plateau')
    if training['initialization'] not in ('default', 'uniform'):
        raise ValueError('initialization must be default or uniform')
    if float(training['loss_scale']) <= 0:
        raise ValueError('loss_scale must be positive')
    if training['initialization'] == 'uniform' \
            and float(training['initialization_scale']) <= 0:
        raise ValueError('initialization_scale must be positive')
    if float(training['gradient_clip']) < 0:
        raise ValueError('gradient_clip must be non-negative')
    if config['model'] == 'wheat2000_bimsgp' \
            and ('source_path' not in config or 'source_commit' not in config):
        raise ValueError('BiMSGP requires source_path and source_commit')


# -- preprocessing and model construction ---------------------------------
def fit_scaler(x_fit):
    """Marker mean/std from fit rows only; constant markers keep unit scale."""
    x = np.asarray(x_fit)
    mean = x.mean(axis=0, dtype=np.float64)
    std = x.std(axis=0, dtype=np.float64)
    return mean, np.where(std > 0, std, 1.0)


def transform_x(x, mean, std, dtype):
    x = np.asarray(x)
    np_dtype = np.dtype(str(dtype).split('.')[-1])
    if mean is None:
        return x.astype(np_dtype, copy=False)
    mean = mean.astype(x.dtype)
    std = std.astype(x.dtype)
    return ((x - mean) / std).astype(np_dtype, copy=False)


def prepare_tensors(data, fit_ids, query_ids, trait_index, training, dtype, device,
                    need_query_y, sequence_input=False):
    fit_ids = np.asarray(fit_ids, dtype=np.int64)
    query_ids = np.asarray(query_ids, dtype=np.int64)
    x_fit_raw = np.asarray(data.X)[fit_ids]
    x_query_raw = np.asarray(data.X)[query_ids]
    mode = training['genotype_preprocessing']
    mean, std = (None, None) if mode == 'raw' else fit_scaler(x_fit_raw)
    y_fit = np.asarray(data.Y[fit_ids, trait_index], dtype=np.float64)
    if not np.isfinite(y_fit).all():
        raise ValueError('Nonfinite fit phenotypes')
    y_mean = float(y_fit.mean())
    y_std = float(y_fit.std())
    if not np.isfinite(y_std) or y_std <= 0:
        raise ValueError('Fit phenotype standard deviation must be positive')
    target = torch.as_tensor((y_fit - y_mean) / y_std, dtype=dtype, device=device)
    y_query = None if not need_query_y else np.asarray(
        data.Y[query_ids, trait_index], dtype=np.float64)
    x_fit = torch.as_tensor(transform_x(x_fit_raw, mean, std, dtype), device=device)
    x_query = torch.as_tensor(transform_x(x_query_raw, mean, std, dtype), device=device)
    if sequence_input:
        x_fit, x_query = x_fit.unsqueeze(-1), x_query.unsqueeze(-1)
    return SimpleNamespace(
        x_fit=x_fit,
        target=target,
        x_query=x_query,
        y_query=y_query, y_mean=y_mean, y_std=y_std,
        scaler={'mode': mode, 'mean': mean, 'std': std})


def build_network(config, data, seed, architecture, device):
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    n_markers = int(data.X.shape[1])
    if config['model'] == 'wheat2000_deepgs':
        model = baselines.DeepGS(n_markers, **architecture)
    else:
        model = baselines.build_bimsgp(config['source_path'], config['source_commit'],
                                       n_markers, **architecture)
    training = config['training']
    if training['initialization'] == 'uniform':
        scale = float(training['initialization_scale'])
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module, (nn.Linear, nn.Conv1d)):
                    nn.init.uniform_(module.weight, -scale, scale)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
    return model.to(device)


def make_optimizer(model, settings):
    name = settings['name']
    common = {'lr': float(settings['lr']),
              'weight_decay': float(settings.get('weight_decay', 0.0))}
    if name == 'sgd':
        return torch.optim.SGD(model.parameters(), momentum=float(
            settings.get('momentum', 0.0)), **common)
    return torch.optim.AdamW(model.parameters(), foreach=False, **common)


def make_scheduler(optimizer, training):
    if training['scheduler'] == 'none':
        return None
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=float(training.get('lr_factor', DEFAULT_LR_FACTOR)),
        patience=int(training['lr_patience']), min_lr=float(training.get('min_lr', 0.0)))


def _amp_disabled(device):
    return device.type != 'cuda'


def _autocast(device, enabled):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=enabled) \
        if enabled else nullcontext()


def _train_epoch(model, optimizer, scaler, amp, x_fit, target, generator,
                 training, device):
    model.train()
    batch_size = int(training['batch_size'])
    clip = float(training['gradient_clip'])
    loss_scale = float(training['loss_scale'])
    order = torch.randperm(len(x_fit), generator=generator).to(device)
    total = 0.0
    for start in range(0, len(order), batch_size):
        batch = order[start:start + batch_size]
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, amp):
            output = model(x_fit[batch]).reshape(-1)
            loss = F.mse_loss(output.float(), target[batch]) * loss_scale
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite training loss')
        if amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
        total += float(loss.detach()) * len(batch)
    return total / len(order)


@torch.no_grad()
def predict(model, x, batch_size, *, device=None, amp=False):
    model.eval()
    chunks = []
    device = x.device if device is None else device
    for start in range(0, len(x), int(batch_size)):
        with _autocast(device, bool(amp) and device.type == 'cuda'):
            chunks.append(model(x[start:start + int(batch_size)]).reshape(-1))
    output = torch.cat(chunks).float().cpu().numpy().astype(np.float64)
    if not np.isfinite(output).all():
        raise FloatingPointError('Nonfinite neural prediction')
    return output


def _peak_memory(device):
    if device.type != 'cuda':
        return None
    return float(torch.cuda.max_memory_allocated(device)) / 2 ** 20


# -- development training --------------------------------------------------
def frozen_lr(sequences, epochs):
    """Center three inner learning-rate sequences; short runs pad with their last value."""
    epochs = int(epochs)
    if epochs < 1:
        raise ValueError('Frozen epoch count must be positive')
    if len(sequences) != 3:
        raise ValueError('Exactly three inner learning-rate sequences are required')
    padded = []
    for sequence in sequences:
        values = [float(value) for value in sequence]
        if not values:
            raise ValueError('Empty learning-rate sequence')
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError('Learning-rate sequence must be finite and positive')
        if len(values) < epochs:
            values = values + [values[-1]] * (epochs - len(values))
        padded.append(values[:epochs])
    return [float(np.median([values[position] for values in padded]))
            for position in range(epochs)]


def train_early_stopped(config, data, fold_id, trait_index, config_index, seed,
                        inner, fit_ids, val_ids, device):
    training = config['training']
    settings = config['configurations'][config_index]
    dtype = getattr(torch, config.get('dtype', 'float32'))
    started = time.perf_counter()
    tensors = prepare_tensors(data, fit_ids, val_ids, trait_index, training, dtype,
                              device, need_query_y=True,
                              sequence_input=config['model'] == 'wheat2000_bimsgp')
    model = build_network(config, data, seed, dict(settings['architecture']), device)
    preprocess_seconds = time.perf_counter() - started
    optimizer = make_optimizer(model, dict(settings['optimizer']))
    scheduler = make_scheduler(optimizer, training)
    amp = bool(training['amp']) and not _amp_disabled(device)
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    generator = torch.Generator().manual_seed(int(seed) * 1000 + int(config_index))
    if device.type == 'cuda':
        baseline_memory = float(torch.cuda.memory_allocated(device))
        torch.cuda.reset_peak_memory_stats(device)
    else:
        baseline_memory = None
    best = {'pcc': -np.inf, 'epoch': None, 'prediction': None, 'metrics': None}
    curve, lrs = [], []
    trained_start = time.perf_counter()
    for epoch in range(1, int(training['max_epochs']) + 1):
        lrs.append(float(optimizer.param_groups[0]['lr']))
        loss = _train_epoch(model, optimizer, scaler, amp, tensors.x_fit,
                            tensors.target, generator, training, device)
        prediction = predict(model, tensors.x_query,
                             int(training.get('eval_batch_size', DEFAULT_EVAL_BATCH)),
                             device=device, amp=amp)
        prediction = prediction * tensors.y_std + tensors.y_mean
        metrics = metric_dict(tensors.y_query, prediction)
        curve.append({'epoch': epoch, 'train_mse': float(loss),
                      'val_pcc': metrics['pcc'], 'val_rmse': metrics['rmse'],
                      'lr': lrs[-1]})
        if metrics['pcc'] > best['pcc'] + TIE_TOLERANCE:
            best = {'pcc': metrics['pcc'], 'epoch': epoch,
                    'prediction': prediction.copy(), 'metrics': metrics}
        if scheduler is not None:
            scheduler.step(metrics['rmse'])
        if epoch - best['epoch'] >= int(training['early_stopping_patience']):
            break
    trained_seconds = time.perf_counter() - trained_start
    peak = _peak_memory(device)
    return {
        'fold': int(fold_id), 'trait': str(data.trait_names[trait_index]),
        'trait_index': int(trait_index), 'inner': int(inner), 'seed': int(seed),
        'config_index': int(config_index), 'configuration': dict(settings),
        'best_epoch': int(best['epoch']), 'epochs_run': len(curve),
        'lr_sequence': lrs, 'curve': curve,
        'prediction': best['prediction'], 'material_index': np.asarray(val_ids).tolist(),
        'metrics': best['metrics'], 'parameters': int(sum(
            parameter.numel() for parameter in model.parameters())),
        'preprocess_seconds': preprocess_seconds, 'train_seconds': trained_seconds,
        'memory': (None if baseline_memory is None else {
            'baseline_megabytes': baseline_memory / 2 ** 20,
            'total_allocated_peak_megabytes': peak,
            'increment_megabytes': peak - baseline_memory / 2 ** 20}),
    }


def _seed_ensemble(records, seeds):
    by_seed = {int(record['seed']): record for record in records}
    if set(by_seed) != {int(seed) for seed in seeds}:
        raise ValueError('Missing seeds in an inner unit')
    reference = np.asarray(by_seed[int(seeds[0])]['material_index'], dtype=np.int64)
    values = []
    for seed in seeds:
        record = by_seed[int(seed)]
        if not np.array_equal(np.asarray(record['material_index'], dtype=np.int64),
                              reference):
            raise ValueError('Seed material order differs inside an inner unit')
        values.append(np.asarray(record['prediction'], dtype=np.float64))
    return reference, np.mean(values, axis=0)


def select_fold(records, fold_id, data, config):
    seeds = [int(value) for value in config['seeds']]
    fold_records = [record for record in records if record['fold'] == int(fold_id)]
    selected = []
    for trait_index, trait_name in enumerate(data.trait_names):
        scored = []
        for config_index in range(len(config['configurations'])):
            unit_rows = [record for record in fold_records
                         if record['trait_index'] == trait_index
                         and record['config_index'] == config_index]
            if len(unit_rows) != 3 * len(seeds):
                raise ValueError('Incomplete trajectory grid for one trait/config')
            inner_scores, inner_seed_means = [], []
            for inner in range(3):
                unit = [record for record in unit_rows if record['inner'] == inner]
                ids, ensemble = _seed_ensemble(unit, seeds)
                truth = np.asarray(data.Y[ids, trait_index], dtype=np.float64)
                inner_scores.append(float(pcc(truth, ensemble)))
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
        seed_blocks = {}
        for seed in seeds:
            best_epochs, lr_sequences = [], []
            for inner in range(3):
                matches = [record for record in fold_records
                           if record['trait_index'] == trait_index
                           and record['config_index'] == best['config_index']
                           and record['seed'] == seed and record['inner'] == inner]
                if len(matches) != 1:
                    raise ValueError('Missing trajectory for a frozen epoch row')
                best_epochs.append(int(matches[0]['best_epoch']))
                lr_sequences.append(list(matches[0]['lr_sequence']))
            median_epoch = int(sorted(best_epochs)[1])
            seed_blocks[str(int(seed))] = {
                'per_inner_best_epochs': best_epochs,
                'median_epoch': median_epoch,
                'per_inner_lr_sequences': lr_sequences,
                'lr': frozen_lr(lr_sequences, median_epoch),
            }
        selected.append({
            'fold': int(fold_id), 'trait': str(trait_name),
            'trait_index': int(trait_index),
            'config_index': int(best['config_index']),
            'configuration': dict(config['configurations'][best['config_index']]),
            'seed_ensemble_pcc': best['seed_ensemble_pcc'],
            'seed_mean_pcc': best['seed_mean_pcc'],
            'inner_seed_ensemble_pcc': best['inner_seed_ensemble_pcc'],
            'inner_seed_mean_pcc': best['inner_seed_mean_pcc'],
            'seeds': seed_blocks,
        })
    return selected


# -- development driver ----------------------------------------------------
def development(config, args, runner):
    validate_config(config)
    device = prepare_device(config)
    data, folds, trait_names = load_protocol(config)
    seeds = [int(value) for value in config['seeds']]
    expected = (len(folds) * len(trait_names) * len(config['configurations'])
                * len(seeds) * 3)
    if 'expected_development_fits' in config \
            and int(config['expected_development_fits']) != expected:
        raise ValueError('Development fit budget disagrees with expected_development_fits')
    config_path = Path(args.config)
    model_source = model_source_info(config, data)
    args.out.mkdir(parents=True, exist_ok=False)
    document = {
        'schema_version': 1, 'phase': 'development',
        'data_sha256': config['data_sha256'], 'cache_sha256': config['cache_sha256'],
        'splits_sha256': sha256_of(Path(config['splits'])),
        'config_sha256': sha256_of(config_path), 'config': config,
        'source_sha256': source_hashes(config_path, runner),
        'model_source': model_source,
        'traits': trait_names, 'seeds': seeds,
        'folds': [int(fold['fold']) for fold in folds], 'selected': [],
        'epoch_indexing': '1-based; best_epoch is the first epoch after one update',
    }
    state = {'status': 'running', 'phase': 'development', 'completed_fits': 0,
             'expected_fits': expected, 'folds_complete': []}
    write_json(args.out / 'execution_status.json', state)
    write_json(args.out / 'provenance.json', {
        **document, 'git_commit': git_head(), 'device': str(device),
        'scope': 'Development only: inner fit/validation; outer test rows never read',
    })
    predictions_path = args.out / 'predictions.csv'
    trajectories_path = args.out / 'trajectories.jsonl'
    predictions_path.write_text(
        'fold,trait,trait_index,inner,seed,config_index,material_index,y_pred\n')
    trajectories_path.write_text('')
    records, failures = [], []
    started = time.perf_counter()
    try:
        for fold in folds:
            fold_id = int(fold['fold'])
            for trait_index in range(len(trait_names)):
                for seed in seeds:
                    for config_index, settings in enumerate(config['configurations']):
                        for inner_index, inner in enumerate(fold['inner']):
                            try:
                                record = train_early_stopped(
                                    config, data, fold_id, trait_index, config_index,
                                    seed, inner_index, inner['train_idx'],
                                    inner['val_idx'], device)
                            except Exception as error:
                                failures.append({'fit': {
                                    'fold': fold_id, 'trait_index': trait_index,
                                    'seed': seed, 'config_index': config_index,
                                    'inner': inner_index}, 'reason': repr(error)})
                                write_json(args.out / 'failures.json', failures)
                                write_json(args.out / 'execution_status.json', state)
                                raise
                            finally:
                                if device.type == 'cuda':
                                    torch.cuda.empty_cache()
                            records.append(record)
                            state['completed_fits'] += 1
                            with trajectories_path.open('a') as handle:
                                handle.write(json.dumps(
                                    {key: value for key, value in record.items()
                                     if key != 'prediction'}, default=str) + '\n')
                            write_json(args.out / 'failures.json', failures)
                            write_json(args.out / 'execution_status.json', state)
            fold_selected = select_fold(records, fold_id, data, config)
            document['selected'].extend(fold_selected)
            state['folds_complete'].append(fold_id)
            write_json(args.out / 'selection.json', document)
            write_json(args.out / 'execution_status.json', state)
            with predictions_path.open('a') as handle:
                for record in records:
                    if record['fold'] != fold_id:
                        continue
                    for material, value in zip(record['material_index'],
                                               record['prediction']):
                        handle.write(
                            f"{record['fold']},{record['trait']},{record['trait_index']},"
                            f"{record['inner']},{record['seed']},{record['config_index']},"
                            f"{material},{float(value):.10g}\n")
        state['status'] = 'complete'
    except Exception as error:
        if not failures or failures[-1].get('reason') != repr(error):
            failures.append({'stage': 'selection', 'reason': repr(error)})
        state['status'] = 'failed'
        raise
    finally:
        write_json(args.out / 'failures.json', failures)
        write_json(args.out / 'execution_status.json', state)
        if document['selected']:
            write_json(args.out / 'selection.json', document)
        if records:
            write_json(args.out / 'trajectories.json',
                       [{key: value for key, value in record.items()
                         if key != 'prediction'} for record in records])
    if state['status'] != 'complete':
        return
    if len(document['selected']) != len(folds) * len(trait_names):
        raise RuntimeError('Incomplete development selection grid')
    try:
        cost = {
            'wall_seconds': time.perf_counter() - started,
            'train_seconds': float(np.sum([record['train_seconds'] for record in records])),
            'preprocess_seconds': float(np.sum(
                [record['preprocess_seconds'] for record in records])),
            'parameters': sorted({int(record['parameters']) for record in records}),
            'max_peak_megabytes': max(
                [record['memory']['total_allocated_peak_megabytes']
                 for record in records if record.get('memory')], default=None),
        }
        per_trait = []
        for trait_index, trait_name in enumerate(trait_names):
            rows = [row for row in document['selected']
                    if row['trait_index'] == trait_index]
            per_trait.append({
                'trait': str(trait_name), 'trait_index': int(trait_index),
                'folds': len(rows),
                'mean_seed_ensemble_pcc': float(np.mean(
                    [row['seed_ensemble_pcc'] for row in rows])),
                'mean_seed_mean_pcc': float(np.mean(
                    [row['seed_mean_pcc'] for row in rows])),
            })
        summary = {
            'status': 'complete', 'phase': 'development',
            'scope': 'outer test rows never indexed; selection uses inner validation only',
            'traits': [str(value) for value in trait_names], 'seeds': seeds,
            'folds': [int(fold['fold']) for fold in folds],
            'configurations': len(config['configurations']),
            'counts': {'fits': len(records),
                       'selected_rows': len(document['selected']),
                       'seed_epoch_rows': len(document['selected']) * len(seeds)},
            'primary_reading': 'seed-averaged best-validation PCC, '
                               'configuration selected per (fold, trait)',
            'mean_seed_ensemble_pcc': float(np.mean(
                [row['seed_ensemble_pcc'] for row in document['selected']])),
            'mean_seed_mean_pcc': float(np.mean(
                [row['seed_mean_pcc'] for row in document['selected']])),
            'per_trait': per_trait, 'cost': cost,
            'selection_sha256': sha256_of(args.out / 'selection.json'),
            'predictions_sha256': sha256_of(predictions_path),
            'trajectories_sha256': sha256_of(args.out / 'trajectories.json'),
        }
        write_json(args.out / 'summary.json', summary)
        write_csv(args.out / 'per_fold.csv', [
            {'fold': row['fold'], 'trait': row['trait'],
             'trait_index': row['trait_index'], 'config_index': row['config_index'],
             'seed_ensemble_pcc': row['seed_ensemble_pcc'],
             'seed_mean_pcc': row['seed_mean_pcc']}
            for row in document['selected']],
            ['fold', 'trait', 'trait_index', 'config_index',
             'seed_ensemble_pcc', 'seed_mean_pcc'])
        write_csv(args.out / 'per_trait.csv', per_trait,
                  ['trait', 'trait_index', 'folds', 'mean_seed_ensemble_pcc',
                   'mean_seed_mean_pcc'])
        write_json(args.out / 'timings.json',
                   {'total_seconds': time.perf_counter() - started})
    except Exception as error:
        state['status'] = 'failed'
        failures.append({'stage': 'summary', 'reason': repr(error)})
        write_json(args.out / 'failures.json', failures)
        write_json(args.out / 'execution_status.json', state)
        raise


# -- final -----------------------------------------------------------------
def validate_selection(config, selection, config_path, runner, folds, traits, data):
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
    if selection.get('model_source') != model_source_info(config, data):
        raise ValueError('Selection model source does not match')
    if [str(value) for value in selection.get('traits', [])] \
            != [str(value) for value in traits]:
        raise ValueError('Selection traits do not match')
    if [int(value) for value in selection.get('seeds', [])] \
            != [int(value) for value in config['seeds']]:
        raise ValueError('Selection seeds do not match')
    fold_ids = [int(fold['fold']) for fold in folds]
    rows = selection.get('selected') or []
    if len(rows) != len(fold_ids) * len(traits):
        raise ValueError('Selection row count is incomplete')
    chosen = {}
    for row in rows:
        key = (int(row['fold']), int(row['trait_index']))
        if key in chosen or key[0] not in fold_ids or not 0 <= key[1] < len(traits):
            raise ValueError('Invalid or duplicate selection row')
        if str(row.get('trait')) != str(traits[key[1]]):
            raise ValueError('Selection trait name disagrees with its index')
        index = int(row['config_index'])
        if not 0 <= index < len(config['configurations']):
            raise ValueError('Selection config index out of range')
        if dict(row['configuration']) != dict(config['configurations'][index]):
            raise ValueError('Selection configuration does not match the config list')
        if not np.isfinite(float(row['seed_ensemble_pcc'])):
            raise ValueError('Nonfinite selection score')
        seeds = row.get('seeds') or {}
        if set(seeds) != {str(int(seed)) for seed in config['seeds']}:
            raise ValueError('Selection seed epochs are incomplete')
        for seed, block in seeds.items():
            values = [int(value) for value in block['per_inner_best_epochs']]
            if len(values) != 3 or any(value < 1 or value > int(
                    config['training']['max_epochs']) for value in values):
                raise ValueError('Invalid per-inner best epochs')
            if int(block['median_epoch']) != sorted(values)[1]:
                raise ValueError('Median epoch is not the middle inner epoch')
            lr = [float(value) for value in block['lr']]
            if len(lr) != int(block['median_epoch']) or not np.isfinite(lr).all() \
                    or min(lr) <= 0:
                raise ValueError('Frozen learning-rate list is invalid')
            sequences = block.get('per_inner_lr_sequences')
            if not isinstance(sequences, list) or len(sequences) != 3:
                raise ValueError('Selection lacks three inner learning-rate sequences')
            if lr != frozen_lr(sequences, int(block['median_epoch'])):
                raise ValueError('Frozen learning-rate list disagrees with the median rule')
        chosen[key] = row
    return chosen


def _local_fold(data, fold):
    dev = np.asarray(fold['dev_idx'], dtype=np.int64)
    test = np.asarray(fold['test_idx'], dtype=np.int64)
    return dev, test


def train_fixed(config, data, fold_id, trait_index, config_index, seed,
                fit_ids, test_ids, epochs, lr_sequence, device, weights_dir,
                save_weights):
    training = config['training']
    settings = config['configurations'][config_index]
    dtype = getattr(torch, config.get('dtype', 'float32'))
    started = time.perf_counter()
    tensors = prepare_tensors(data, fit_ids, test_ids, trait_index, training, dtype,
                              device, need_query_y=False,
                              sequence_input=config['model'] == 'wheat2000_bimsgp')
    model = build_network(config, data, seed, dict(settings['architecture']), device)
    preprocess_seconds = time.perf_counter() - started
    if len(lr_sequence) != int(epochs):
        raise ValueError('Frozen learning-rate list length disagrees with the epoch count')
    optimizer = make_optimizer(model, dict(settings['optimizer']))
    amp = bool(training['amp']) and not _amp_disabled(device)
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    generator = torch.Generator().manual_seed(int(seed) * 1000 + int(config_index))
    if device.type == 'cuda':
        baseline_memory = float(torch.cuda.memory_allocated(device))
        torch.cuda.reset_peak_memory_stats(device)
    else:
        baseline_memory = None
    trained_lrs = []
    trained_start = time.perf_counter()
    for epoch in range(int(epochs)):
        for group in optimizer.param_groups:
            group['lr'] = float(lr_sequence[epoch])
        trained_lrs.append(float(optimizer.param_groups[0]['lr']))
        _train_epoch(model, optimizer, scaler, amp, tensors.x_fit, tensors.target,
                     generator, training, device)
    trained_seconds = time.perf_counter() - trained_start
    predict_started = time.perf_counter()
    prediction = predict(model, tensors.x_query,
                         int(training.get('eval_batch_size', DEFAULT_EVAL_BATCH)),
                         device=device, amp=amp)
    prediction = prediction * tensors.y_std + tensors.y_mean
    predict_seconds = time.perf_counter() - predict_started
    peak = _peak_memory(device)
    weight_path, weight_sha, weight_bytes = None, None, 0
    if save_weights:
        weights_dir.mkdir(parents=True, exist_ok=True)
        weight_path = weights_dir / (f"f{fold_id}_t{trait_index}_s{seed}_"
                                     f"c{config_index}.pt")
        torch.save(model.state_dict(), weight_path)
        weight_sha = sha256_of(weight_path)
        weight_bytes = weight_path.stat().st_size
    return {
        'prediction': prediction, 'test_ids': np.asarray(test_ids).tolist(),
        'epochs': int(epochs), 'trained_lrs': trained_lrs,
        'parameters': int(sum(parameter.numel() for parameter in model.parameters())),
        'preprocess_seconds': preprocess_seconds, 'train_seconds': trained_seconds,
        'predict_seconds': predict_seconds,
        'weights_saved': bool(save_weights), 'weight_path': None if weight_path is None
        else str(weight_path), 'weight_sha256': weight_sha,
        'weight_bytes': int(weight_bytes),
        'state_note': None if save_weights else 'state not persisted',
        'memory': (None if baseline_memory is None else {
            'baseline_megabytes': baseline_memory / 2 ** 20,
            'total_allocated_peak_megabytes': peak,
            'increment_megabytes': peak - baseline_memory / 2 ** 20}),
    }


def score_final(data, folds, seeds, prediction_map, chosen):
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
            config_row = chosen[(fold_id, trait_index)]
            per_fold_rows.append({
                'fold': fold_id, 'trait': str(trait_name),
                'trait_index': int(trait_index),
                'config_index': int(config_row['config_index']), **metrics,
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
        seed_pooled = {int(seed): metric_dict(
            np.concatenate(trait_pooled[trait_index]['seed_truth'][int(seed)]),
            np.concatenate(trait_pooled[trait_index]['seed_prediction'][int(seed)]))
            for seed in seeds}
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
    return {'per_fold_trait': per_fold_rows, 'per_trait': per_trait_rows,
            'seed_metrics': seed_metric_rows, 'oof': oof_rows,
            'overall_macro': overall_macro, 'overall_pooled': overall_pooled,
            'seed_readings': {str(seed): values for seed, values in seed_readings.items()}}


def final(config, args, runner):
    validate_config(config)
    device = prepare_device(config)
    data, folds, trait_names = load_protocol(config)
    seeds = [int(value) for value in config['seeds']]
    selection_path = Path(args.selection)
    selection = read_json(selection_path)
    config_path = Path(args.config)
    chosen = validate_selection(config, selection, config_path, runner, folds,
                                trait_names, data)
    if bool(config.get('require_committed_selection', True)):
        ensure_frozen_manifest(selection_path)
    save_weights = bool(config.get('save_weights', False))
    expected_fits = len(folds) * len(trait_names) * len(seeds)
    if 'expected_final_fits' in config and int(config['expected_final_fits']) != expected_fits:
        raise ValueError('Final fit budget disagrees with expected_final_fits')
    expected_seed_rows = len(trait_names) * len(seeds) * sum(
        len(fold['test_idx']) for fold in folds)
    args.out.mkdir(parents=True, exist_ok=False)
    selection_sha = sha256_of(selection_path)
    state = {'status': 'running', 'phase': 'final', 'completed_fits': 0,
             'expected_fits': expected_fits, 'folds_complete': [],
             'fit_failures': 0, 'weights_saved': save_weights}
    write_json(args.out / 'execution_status.json', state)
    write_json(args.out / 'provenance.json', {
        'config': config, 'config_sha256': sha256_of(config_path),
        'data_sha256': config['data_sha256'], 'cache_sha256': config['cache_sha256'],
        'splits_sha256': sha256_of(Path(config['splits'])),
        'selection': {'path': str(selection_path), 'sha256': selection_sha},
        'source_sha256': source_hashes(config_path, runner),
        'model_source': model_source_info(config, data), 'git_commit': git_head(),
        'device': str(device), 'budget': expected_fits,
        'protocol': 'full-development refit at frozen median inner epoch and LR list; '
                    'test scored once after every fit; seed member-mean then seed mean',
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
            dev, test = _local_fold(data, fold)
            for trait_index, trait_name in enumerate(trait_names):
                for seed in seeds:
                    row = chosen[(fold_id, trait_index)]
                    block = row['seeds'][str(int(seed))]
                    try:
                        result = train_fixed(
                            config, data, fold_id, trait_index,
                            int(row['config_index']), int(seed), dev.tolist(),
                            test.tolist(), int(block['median_epoch']),
                            [float(value) for value in block['lr']], device,
                            args.out / 'weights', save_weights)
                    except Exception as error:
                        failures.append({'fit': {
                            'fold': fold_id, 'trait_index': trait_index,
                            'seed': int(seed),
                            'config_index': int(row['config_index'])},
                            'reason': repr(error)})
                        state['fit_failures'] += 1
                        raise
                    finally:
                        if device.type == 'cuda':
                            torch.cuda.empty_cache()
                    if [int(value) for value in result['test_ids']] != test.tolist():
                        raise ValueError('Final training returned unexpected test rows')
                    prediction = np.asarray(result['prediction'], dtype=np.float64)
                    prediction_map[(fold_id, trait_index, int(seed))] = prediction
                    first_new_row = len(seed_rows)
                    for position, material in enumerate(test):
                        seed_rows.append({
                            'fold': fold_id, 'trait': str(trait_name),
                            'trait_index': int(trait_index), 'seed': int(seed),
                            'config_index': int(row['config_index']),
                            'material_index': int(material),
                            'y_pred': float(prediction[position]),
                        })
                    weight_rows.append({
                        'fold': fold_id, 'trait': str(trait_name),
                        'trait_index': int(trait_index), 'seed': int(seed),
                        'config_index': int(row['config_index']),
                        'epochs': int(result['epochs']),
                        'trained_lrs': result['trained_lrs'],
                        'weights_saved': result['weights_saved'],
                        'path': result['weight_path'],
                        'file_sha256': result['weight_sha256'],
                        'state_sha256': None,
                        'state_note': result['state_note'],
                        'bytes': result['weight_bytes'],
                        'parameters': result['parameters'],
                        'train_seconds': result['train_seconds'],
                        'predict_seconds': result['predict_seconds'],
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
        saved = [row for row in weight_rows if row['weights_saved']]
        summary = {
            'status': 'complete', 'phase': 'final',
            'primary_reading': 'seed ensemble of fit-only-standardized predictions; '
                               'test scored once after all fits',
            'traits': trait_names, 'seeds': seeds,
            'folds': [int(fold['fold']) for fold in folds],
            'counts': {'fits': len(weight_rows),
                       'seed_prediction_rows': len(seed_rows),
                       'ensemble_oof_rows': len(scored['oof']),
                       'per_fold_trait_rows': len(scored['per_fold_trait']),
                       'weights_saved': len(saved)},
            'overall_macro': scored['overall_macro'],
            'overall_pooled': scored['overall_pooled'],
            'per_trait': scored['per_trait'], 'seed_readings': scored['seed_readings'],
            'selection': {'path': str(selection_path), 'sha256': selection_sha},
            'weights': {'saved': save_weights, 'count': len(saved),
                        'bytes': int(sum(row['bytes'] for row in saved)),
                        'manifest_sha256': sha256_of(args.out / 'weights_manifest.json'),
                        'note': None if save_weights else 'weights not persisted'},
            'cost': {
                'wall_seconds': time.perf_counter() - started,
                'train_seconds': float(sum(row['train_seconds'] for row in weight_rows)),
                'predict_seconds': float(sum(row['predict_seconds']
                                             for row in weight_rows)),
                'parameters': sorted({row['parameters'] for row in weight_rows}),
                'max_peak_megabytes': max(
                    [row['peak_megabytes'] for row in weight_rows
                     if row['peak_megabytes'] is not None], default=None)},
        }
        summary['per_fold_trait_sha256'] = sha256_of(args.out / 'per_fold_trait.csv')
        summary['per_trait_sha256'] = sha256_of(args.out / 'per_trait.csv')
        summary['oof_predictions_sha256'] = sha256_of(args.out / 'oof_predictions.csv')
        summary['seed_predictions_sha256'] = sha256_of(seed_path)
        write_json(args.out / 'summary.json', summary)
        state['status'] = 'complete'
    except Exception as error:
        if not failures or failures[-1].get('reason') != repr(error):
            failures.append({'stage': state.get('status', 'final'), 'reason': repr(error)})
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


def run(config, args, runner):
    if args.selection is None:
        return development(config, args, runner)
    return final(config, args, runner)
