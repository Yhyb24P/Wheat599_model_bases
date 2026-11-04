"""Development-only runner for the RAGP Wheat599 adaptation.

One trajectory fits on one inner fit split, refreshes its retrieval library at the end of
every training epoch, and validates on that inner split. The first validation happens only
after the first epoch and its library update; the empty-library state is never a candidate.
The outer test rows are never indexed and the reference library holds fit materials and
their labels only.

Checkpoints are not persisted. The best state is kept in memory, hashed by a streaming
SHA-256 over sorted keys, loaded back through ``RAGPGenomic.load_state_dict`` (which
rebuilds the library), and compared with the recorded validation predictions. Failures
propagate immediately and are written as they occur.
"""
import csv
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from wheat599.data import load, sha256_of
from wheat599.metrics import mae, pcc, rmse
from wheat599.ragp import OFFICIAL_HIDDEN_DIMS, RAGPGenomic

METRICS = ('pcc', 'rmse', 'mae')
SELECTION_METRICS = ('seed_prediction_ensemble_macro_pcc',)
MODEL_KEYS = ('hidden_dims', 'n_blocks', 'dropout', 'pca_variance', 'neighbors',
              'fusion_alpha', 'label_scale', 'loss_alpha', 'triplet_anchors',
              'triplet_positives', 'triplet_negatives', 'triplet_margin')
TRAINING_KEYS = ('learning_rate', 'weight_decay', 'max_epochs', 'patience',
                 'scheduler_gamma', 'tie_tolerance', 'batch_size', 'eval_batch_size',
                 'gradient_clip')
CHECKPOINT_STORAGE = 'hash_only'


# -- configuration ---------------------------------------------------------
def model_settings(config, configuration):
    settings = dict(config.get('architecture', {}))
    settings.setdefault('hidden_dims', OFFICIAL_HIDDEN_DIMS)
    for key in MODEL_KEYS:
        if key in configuration:
            settings[key] = configuration[key]
    return settings


def training_settings(config, configuration):
    settings = dict(config.get('training', {}))
    for key in TRAINING_KEYS:
        if key in configuration:
            settings[key] = configuration[key]
    if 'learning_rate' not in settings or 'max_epochs' not in settings:
        raise ValueError('Each configuration needs learning_rate and max_epochs')
    settings.setdefault('weight_decay', 0.0)
    settings.setdefault('patience', 30)
    settings.setdefault('scheduler_gamma', 0.95)
    settings.setdefault('tie_tolerance', 1e-12)
    settings.setdefault('batch_size', 128)
    settings.setdefault('eval_batch_size', 256)
    if int(settings['max_epochs']) < 1:
        raise ValueError('max_epochs must be positive')
    if int(settings['batch_size']) < 2:
        raise ValueError('batch_size must be at least two for BatchNorm1d')
    if int(settings['eval_batch_size']) < 1:
        raise ValueError('eval_batch_size must be positive')
    return settings


def build_model(config, seed, train_x, train_y, configuration, device):
    """Seeded construction; PCA and the reference library use the fit rows only."""
    from wheat599 import ragp
    builder = getattr(ragp, 'RAGPGenomic', None)
    if builder is None:
        raise RuntimeError('wheat599.ragp.RAGPGenomic is unavailable')
    dtype = getattr(torch, config.get('dtype', 'float32'))
    settings = model_settings(config, configuration)
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    model = builder(train_x.to(dtype), train_y.to(dtype), dtype=dtype, **settings)
    return model.to(device)


# -- utilities -------------------------------------------------------------
def check_split(fold, inner_index):
    fit_index = np.asarray(fold['inner'][inner_index]['train_idx'])
    val_index = np.asarray(fold['inner'][inner_index]['val_idx'])
    dev = np.asarray(fold['dev_idx'])
    test = np.asarray(fold['test_idx'])
    if set(fit_index) & set(val_index) or set(fit_index) | set(val_index) != set(dev.tolist()):
        raise ValueError('Invalid inner split')
    if set(np.r_[fit_index, val_index]) & set(test.tolist()):
        raise ValueError('Outer test rows entered the development split')
    return fit_index, val_index


def make_batches(order, batch_size):
    """Split a permutation; merge a singleton tail into the previous batch for BatchNorm."""
    order = torch.as_tensor(order)
    batch_size = int(batch_size)
    if batch_size < 2:
        raise ValueError('batch_size must be at least two for BatchNorm1d')
    if order.ndim != 1 or order.numel() == 0:
        raise ValueError('order must be a non-empty vector')
    batches = [order[start:start + batch_size]
               for start in range(0, order.numel(), batch_size)]
    if len(batches) > 1 and batches[-1].numel() == 1:
        batches[-2] = torch.cat([batches[-2], batches[-1]])
        batches.pop()
    if batches[0].numel() < 2:
        raise ValueError('At least two fit materials are required')
    return batches


def state_sha256(state, chunk_bytes=8 * 2 ** 20):
    """Streaming SHA-256 of a state dict: sorted keys, dtype, shape and tensor bytes."""
    digest = hashlib.sha256()
    digest.update(b'ragp_state_v1\0')
    for key in sorted(state):
        tensor = state[key]
        if not torch.is_tensor(tensor):
            raise TypeError('state_dict values must be tensors')
        tensor = tensor.detach().cpu().contiguous()
        digest.update(key.encode('utf-8'))
        digest.update(b'\0')
        digest.update(str(tensor.dtype).encode('ascii'))
        digest.update(b'\0')
        digest.update(str(tuple(tensor.shape)).encode('ascii'))
        digest.update(b'\0')
        view = memoryview(tensor.numpy()).cast('B')
        for start in range(0, len(view), chunk_bytes):
            digest.update(view[start:start + chunk_bytes])
        digest.update(b'\0')
    return digest.hexdigest()


def ensemble_dim(model, output):
    if output.ndim != 3:
        raise RuntimeError(f'Expected (n, k, q) output, got {tuple(output.shape)}')
    return output


@torch.no_grad()
def predict(model, x, batch_size):
    """Validation queries are outside the fit library, so no query ids are passed."""
    model.eval()
    chunks = []
    for start in range(0, len(x), int(batch_size)):
        output = ensemble_dim(model, model(x[start:start + int(batch_size)]))
        chunks.append(output.mean(1))
    return torch.cat(chunks, dim=0)


def macro_pcc(prediction, truth):
    values = [pcc(truth[:, column], prediction[:, column]) for column in range(truth.shape[1])]
    if not all(np.isfinite(values)):
        raise FloatingPointError('Nonfinite validation PCC')
    return float(np.mean(values))


def macro_metrics(prediction, truth):
    return {name: float(np.mean([function(truth[:, column], prediction[:, column])
                                 for column in range(truth.shape[1])]))
            for name, function in zip(METRICS, (pcc, rmse, mae))}


def environment_metrics(prediction, truth, environments):
    names = list(environments) if environments is not None else list(range(truth.shape[1]))
    if len(names) != truth.shape[1]:
        raise ValueError('Environment labels disagree with the phenotype columns')
    return {str(names[column]): {name: float(function(truth[:, column], prediction[:, column]))
                                 for name, function in zip(METRICS, (pcc, rmse, mae))}
            for column in range(truth.shape[1])}


# -- one trajectory --------------------------------------------------------
@dataclass
class Trajectory:
    fold: int
    inner: int
    seed: int
    config_index: int
    configuration: dict


def train_trajectory(data, fold, inner_index, trajectory, config, device, environments):
    fit_index, val_index = check_split(fold, inner_index)
    training = training_settings(config, trajectory.configuration)
    dtype = getattr(torch, config.get('dtype', 'float32'))
    started = time.perf_counter()
    fit_x = np.asarray(data.X)[fit_index]
    fit_y = np.asarray(data.Y)[fit_index]
    val_x = np.asarray(data.X)[val_index]
    val_y = np.asarray(data.Y)[val_index]
    train_x = torch.as_tensor(fit_x, dtype=torch.float64).to(dtype)
    train_y = torch.as_tensor(fit_y, dtype=torch.float64).to(dtype)
    val_x_t = torch.as_tensor(val_x, dtype=torch.float64).to(dtype)
    val_y_np = np.asarray(val_y, dtype=np.float64)
    model = build_model(config, trajectory.seed, train_x, train_y,
                        trajectory.configuration, device)
    train_x = train_x.to(device)
    train_y = train_y.to(device)
    val_x_t = val_x_t.to(device)
    target = model.standardize_y(train_y)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=float(training['learning_rate']),
                                 weight_decay=float(training['weight_decay']))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=float(training['scheduler_gamma']))
    generator = torch.Generator().manual_seed(
        int(trajectory.seed) * 1000 + int(trajectory.config_index))
    model_info = {'parameters': model.parameter_count(),
                  'pca_components': int(model.pca_components.shape[0]),
                  'pca_explained': model.pca_explained_variance}
    preprocess_seconds = time.perf_counter() - started
    if device.type == 'cuda':
        baseline_memory = float(torch.cuda.memory_allocated(device))
        torch.cuda.reset_peak_memory_stats(device)
    else:
        baseline_memory = None

    best = {'pcc': -np.inf, 'epoch': -1, 'state': None, 'prediction': None}
    curve, update_seconds = [], 0.0
    stale, epochs_run = 0, 0
    trained_start = time.perf_counter()
    for epoch in range(int(training['max_epochs'])):
        model.train()
        order = torch.randperm(int(train_x.shape[0]), generator=generator)
        batches = make_batches(order, training['batch_size'])
        epoch_start = time.perf_counter()
        running_loss = 0.0
        for batch in batches:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction, embeddings = model(train_x[batch], query_ids=batch,
                                           return_embeddings=True)
            loss = model.loss(prediction, target[batch], embeddings)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite training loss at epoch {epoch}')
            loss.backward()
            clip = training.get('gradient_clip')
            if clip is not None and float(clip) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip))
            optimizer.step()
            running_loss += float(loss.detach())
        update_start = time.perf_counter()
        model.update_library()
        update_seconds += time.perf_counter() - update_start
        scheduler.step()
        standardised = predict(model, val_x_t, training['eval_batch_size'])
        values = model.inverse_y(standardised).cpu().numpy().astype(np.float64)
        score = macro_pcc(values, val_y_np)
        metrics = macro_metrics(values, val_y_np)
        epochs_run = epoch + 1
        curve.append({'epoch': epoch, 'train_loss': running_loss / len(batches),
                      'validation_pcc': score, 'validation_metrics': metrics,
                      'seconds': time.perf_counter() - epoch_start})
        if score > best['pcc'] + float(training['tie_tolerance']):
            best = {'pcc': score, 'epoch': epoch,
                    'state': {key: value.detach().cpu().clone()
                              for key, value in model.state_dict().items()},
                    'prediction': values.copy()}
            stale = 0
        else:
            stale += 1
            if int(training['patience']) and stale >= int(training['patience']):
                break
    trained_seconds = time.perf_counter() - trained_start
    if best['state'] is None:
        raise FloatingPointError('No finite validation prediction')
    peak_allocated = float(torch.cuda.max_memory_allocated(device)) \
        if device.type == 'cuda' else None
    peak_reserved = float(torch.cuda.max_memory_reserved(device)) \
        if device.type == 'cuda' else None

    hash_start = time.perf_counter()
    checkpoint_sha = state_sha256(best['state'])
    hash_seconds = time.perf_counter() - hash_start
    model.load_state_dict(best['state'])
    model.eval()
    with torch.no_grad():
        standardised = predict(model, val_x_t, training['eval_batch_size'])
        values = model.inverse_y(standardised).cpu().numpy().astype(np.float64)
    reload_difference = float(np.max(np.abs(values - best['prediction']))) \
        if values.size else 0.0
    if reload_difference > 1e-10:
        raise FloatingPointError(
            f'Best state reload changed validation predictions by {reload_difference}')
    metrics = macro_metrics(values, val_y_np)
    per_environment = environment_metrics(values, val_y_np, environments)

    efficiency = config.get('efficiency', {})
    efficiency_batch = int(efficiency.get('batch_size', training['eval_batch_size']))
    warmup = int(efficiency.get('warmup', 1))
    repeats = int(efficiency.get('repeats', 2))
    inference_batch = val_x_t[:min(efficiency_batch, len(val_x_t))]
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(inference_batch)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        inference_start = time.perf_counter()
        for _ in range(max(1, repeats)):
            model(inference_batch)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        inference_seconds = (time.perf_counter() - inference_start) / max(1, repeats)

    return {
        'trajectory': {'fold': int(trajectory.fold), 'inner': int(trajectory.inner),
                       'seed': int(trajectory.seed),
                       'config_index': int(trajectory.config_index),
                       'configuration': trajectory.configuration},
        'metrics': metrics,
        'per_environment': per_environment,
        'best_epoch': int(best['epoch']),
        'epochs_run': int(epochs_run),
        'curve': curve,
        'prediction': values,
        'material_index': val_index.tolist(),
        'reload_max_abs_difference': reload_difference,
        'checkpoint_sha256': checkpoint_sha,
        'checkpoint_storage': CHECKPOINT_STORAGE,
        'model': model_info,
        'preprocess_seconds': preprocess_seconds,
        'train_seconds': trained_seconds,
        'update_library_seconds': update_seconds,
        'hash_seconds': hash_seconds,
        'inference_seconds': inference_seconds,
        'memory': ({'baseline_megabytes': baseline_memory / 2 ** 20,
                    'total_allocated_peak_megabytes': peak_allocated / 2 ** 20,
                    'increment_megabytes': (peak_allocated - baseline_memory) / 2 ** 20,
                    'total_reserved_peak_megabytes': peak_reserved / 2 ** 20}
                   if peak_allocated is not None else None),
    }


# -- selection -------------------------------------------------------------
def unit_scores(records, truth):
    by_inner = {}
    for record in records:
        by_inner.setdefault(record['trajectory']['inner'], []).append(record)
    ensemble, per_seed = [], []
    for inner, rows in sorted(by_inner.items()):
        ordered = sorted(rows, key=lambda record: record['trajectory']['seed'])
        if len(ordered) != len({record['trajectory']['seed'] for record in ordered}):
            raise ValueError('Duplicate seeds inside one inner split')
        stacked = np.stack([record['prediction'] for record in ordered], axis=0)
        ensemble.append(macro_pcc(stacked.mean(0), truth[inner]))
        per_seed.append(float(np.mean([record['metrics']['pcc'] for record in ordered])))
    return float(np.mean(ensemble)), float(np.mean(per_seed))


def select_common(records, config, truth):
    metric = config.get('selection_metric', SELECTION_METRICS[0])
    if metric not in SELECTION_METRICS:
        raise ValueError(f'Unsupported selection metric: {metric}')
    chosen = {}
    for fold in sorted({record['trajectory']['fold'] for record in records}):
        scored = []
        for index in range(len(config['configurations'])):
            rows = [record for record in records
                    if record['trajectory']['fold'] == fold
                    and record['trajectory']['config_index'] == index]
            if len(rows) != 3 * len(config['seeds']):
                raise ValueError('Incomplete trajectory grid for one configuration')
            ensemble, per_seed = unit_scores(rows, truth)
            scored.append((ensemble, index, per_seed,
                           {name: float(np.mean([row['metrics'][name] for row in rows]))
                            for name in METRICS}))
        best = max(scored, key=lambda item: (round(item[0], 12), -item[1]))
        chosen[fold] = {'config_index': best[1],
                        'configuration': config['configurations'][best[1]],
                        'seed_ensemble_pcc': best[0],
                        'seed_mean_pcc': best[2],
                        'seed_mean_metrics': best[3]}
    return chosen


def write_predictions(path, records):
    with path.open('w', newline='') as handle:
        writer = csv.writer(handle, lineterminator='\n')
        header = ['fold', 'inner', 'seed', 'config_index', 'material_index'] + \
            [f'y_pred_{name}' for name in range(records[0]['prediction'].shape[1])]
        writer.writerow(header)
        for record in records:
            trajectory = record['trajectory']
            for position, material in enumerate(record['material_index']):
                writer.writerow([trajectory['fold'], trajectory['inner'],
                                 trajectory['seed'], trajectory['config_index'], material,
                                 *[f'{value:.10g}'
                                   for value in record['prediction'][position]]])


# -- versions --------------------------------------------------------------
def package_version(name):
    try:
        import importlib.metadata as metadata
        return metadata.version(name)
    except Exception:
        try:
            return getattr(__import__(name), '__version__', None)
        except ImportError:
            return None


def verify_versions(config):
    wanted = config.get('versions') or {}
    actual, mismatches = {}, {}
    for name, expected in wanted.items():
        found = package_version(name)
        actual[name] = found
        if found != str(expected):
            mismatches[name] = {'expected': str(expected), 'found': found}
    if mismatches:
        raise RuntimeError(f'Pinned package versions do not match: {mismatches}')
    return actual


def environment_provenance(device):
    device_name = None
    if device.type == 'cuda':
        index = device.index if device.index is not None else torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(index)
    return {
        'python_executable': sys.executable,
        'python_version': sys.version,
        'torch_version': torch.__version__,
        'torch_cuda': torch.version.cuda,
        'numpy_version': np.__version__,
        'scikit_learn_version': package_version('scikit-learn'),
        'device': str(device),
        'device_name': device_name,
    }


# -- driver ----------------------------------------------------------------
def run(config, args, runner):
    if args.selection is not None:
        raise ValueError('This development run has no selection-input path')
    if config.get('checkpoint_storage', CHECKPOINT_STORAGE) != CHECKPOINT_STORAGE:
        raise ValueError('Only hash-only checkpoint storage is supported')
    device = torch.device(config['device'])
    torch.set_num_threads(int(config.get('cpu_threads', 2)))
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(float(config['gpu_memory_fraction']))
    verified_versions = verify_versions(config)
    data = load(Path(config['data']), config['data_sha256'])
    manifest = json.loads(Path(config['splits']).read_text())
    wanted = {int(value) for value in config['outer_folds']}
    folds = [fold for fold in manifest['folds'] if int(fold['fold']) in wanted]
    if len(folds) != len(wanted):
        raise ValueError('Configured outer folds are missing from the split manifest')
    args.out.mkdir(parents=True, exist_ok=False)
    here = Path(__file__).parent
    sources = [here / name for name in ('ragp_run.py', 'ragp.py', 'data.py', 'metrics.py')] \
        + [Path(args.config), Path(runner)]
    configurations = [dict(entry) for entry in config['configurations']]
    config = {**config, 'configurations': configurations}
    expected = len(folds) * 3 * len(config['seeds']) * len(configurations)
    (args.out / 'provenance.json').write_text(json.dumps({
        'config': config,
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                              text=True).strip(),
        'data_sha256': config['data_sha256'],
        'splits_sha256': sha256_of(Path(config['splits'])),
        'source_sha256': {str(path): sha256_of(path) for path in sources if path.exists()},
        'package_versions': {
            'torch': package_version('torch'),
            'numpy': package_version('numpy'),
            'scikit-learn': package_version('scikit-learn'),
            **verified_versions,
        },
        'environment': environment_provenance(device),
        'device': str(device),
        'budget': expected,
        'selection_metric': config.get('selection_metric', SELECTION_METRICS[0]),
        'checkpoint_policy': 'in-memory best state only; streaming SHA-256; no weight files',
        'upstream': {
            'url': 'https://github.com/l00907l/RAGP',
            'commit': '85675b1e8054fe3e91fdb3e974150f1dde81eee3',
            'license': 'no LICENSE file at the pinned commit; independent implementation',
        },
        'scope': 'Development only: inner fit/val; outer test rows are never indexed; '
                 'PCA, scalers and the retrieval library use fit rows only',
    }, indent=2))
    state = {'status': 'running', 'completed_trajectories': 0, 'expected': expected,
             'folds_complete': []}
    records, failures, selections = [], [], {}
    truth = {}
    for fold in folds:
        for inner_index in range(3):
            _, val_index = check_split(fold, inner_index)
            truth[(int(fold['fold']), inner_index)] = \
                np.asarray(data.Y[val_index], dtype=np.float64)

    def write_state():
        (args.out / 'execution_status.json').write_text(json.dumps(state, indent=2))

    def fold_truth(fold_id):
        return {inner: truth[(fold_id, inner)] for inner in range(3)}

    write_state()
    start = time.perf_counter()
    try:
        for fold in folds:
            for inner_index in range(3):
                for seed in config['seeds']:
                    for index, entry in enumerate(configurations):
                        trajectory = Trajectory(int(fold['fold']), inner_index, int(seed),
                                                index, entry)
                        try:
                            record = train_trajectory(data, fold, inner_index, trajectory,
                                                      config, device,
                                                      getattr(data, 'environments', None))
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
                                [{key: value for key, value in record.items()
                                  if key != 'prediction'} for record in records],
                                indent=2, default=str))
                            if records:
                                write_predictions(args.out / 'predictions.csv', records)
            state['folds_complete'].append(int(fold['fold']))
            selections.update(select_common(
                [record for record in records
                 if record['trajectory']['fold'] == int(fold['fold'])],
                config, fold_truth(int(fold['fold']))))
            (args.out / 'selection.json').write_text(json.dumps(selections, indent=2))
            write_state()
            print(f"fold {fold['fold']} complete", flush=True)
        state['status'] = 'complete'
    except Exception as exc:
        state['status'] = 'failed'
        if not failures or failures[-1].get('reason') != repr(exc):
            failures.append({'stage': 'run', 'reason': repr(exc)})
        raise
    finally:
        write_state()
        (args.out / 'selection.json').write_text(json.dumps(selections, indent=2))
        (args.out / 'timings.json').write_text(json.dumps({
            'total_seconds': time.perf_counter() - start,
            'trajectories': len(records),
            'epochs': int(sum(record['epochs_run'] for record in records)),
            'train_seconds': sum(record['train_seconds'] for record in records),
            'preprocess_seconds': sum(record['preprocess_seconds'] for record in records),
            'update_library_seconds': sum(record['update_library_seconds']
                                          for record in records),
            'hash_seconds': sum(record['hash_seconds'] for record in records),
            'max_increment_megabytes': max(
                (record['memory']['increment_megabytes']
                 for record in records if record['memory'] is not None), default=None),
            'max_reserved_peak_megabytes': max(
                (record['memory']['total_reserved_peak_megabytes']
                 for record in records if record['memory'] is not None), default=None),
        }, indent=2))
