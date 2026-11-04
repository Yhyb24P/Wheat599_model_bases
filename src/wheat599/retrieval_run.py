"""Development-only runner for the retrieval-augmented TabM arms.

Three arms share one budget schedule: ``raw`` (the plain TabM backbone), ``retrieval_multi`` and
``single_retrieval`` (k = 1). Every variant returns ``(n, k, q)``; the runner averages members only where
the protocol says so.

Data contract. Each trajectory fits on one inner fit split and early-stops/selects on that inner
validation split; the outer test rows are never indexed. The retrieval library is built by the model
from the fit rows and their four environment labels only; inner validation and outer test rows never
enter it. Nothing here reads measured phenotypes beyond the fit/validation rows of the development
split.

Outputs. Per trajectory the runner stores the member-mean validation predictions keyed by fold,
inner split, variant, seed and configuration together with the material ids, so seed-mean scores and
seed-ensembled predictions can both be computed without reloading weights. The main configuration
choice is fixed to the pre-registered metric (mean validation PCC over the inner splits and seeds);
the seed-ensembled score is reported next to it and never used to pick a configuration after the
fact. Reference-encoding time is measured separately when the model exposes it, because re-encoding
the retrieval library every step is a real cost. Failures and non-finite metrics abort the run and
are kept as they occur.
"""
import csv
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from wheat599.data import load, sha256_of
from wheat599.genomic_tabm_run import model_source_sha
from wheat599.metrics import mae, pcc, rmse

VARIANTS = ('raw', 'retrieval', 'single_retrieval')
ARCHITECTURE_KEYS = ('n_blocks', 'd_block', 'dropout', 'k')
SELECTION_METRICS = ('seed_prediction_ensemble_macro_pcc',)
METRICS = ('pcc', 'rmse', 'mae')


# -- model adapter ---------------------------------------------------------
def architecture_for(configuration, variant):
    """Per-configuration architecture; ``single_retrieval`` pins k to one member."""
    settings = {key: configuration[key] for key in ARCHITECTURE_KEYS if key in configuration}
    if variant == 'single_retrieval':
        settings['k'] = 1
    return settings


def build_model(config, variant, seed, train_x, train_y, configuration, device):
    """Seeded construction; the model owns the retrieval library built from fit rows."""
    from wheat599 import retrieval_tabm
    builder = getattr(retrieval_tabm, 'RetrievalTabM', None)
    if builder is None:
        raise RuntimeError('wheat599.retrieval_tabm.RetrievalTabM is unavailable')
    dtype = getattr(torch, config.get('dtype', 'float32'))
    settings = architecture_for(configuration, variant)
    if variant != 'raw' and 'context_size' in configuration:
        settings['context_size'] = int(configuration['context_size'])
    torch.manual_seed(int(seed))
    model = builder(train_x.to(dtype), train_y.to(dtype), variant=variant, dtype=dtype,
                    **settings)
    return model.to(device)


def ensemble_dim(model, output):
    if output.ndim != 3:
        raise RuntimeError(f'Expected (n, k, q) output, got {tuple(output.shape)}')
    return output


def encode_reference(model):
    """Time the library encoding when the model exposes it as a separate step."""
    hook = getattr(model, 'encode_reference', None)
    if hook is None:
        return None
    started = time.perf_counter()
    hook()
    return time.perf_counter() - started


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
    return {str(names[column]): {name: float(function(truth[:, column], prediction[:, column]))
                                 for name, function in zip(METRICS, (pcc, rmse, mae))}
            for column in range(truth.shape[1])}


# -- one trajectory --------------------------------------------------------
@dataclass
class Trajectory:
    fold: int
    inner: int
    seed: int
    variant: str
    config_index: int
    configuration: dict


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


@torch.no_grad()
def predict(model, x, batch_size):
    """Validation queries are never in the retrieval library, so no query ids are passed."""
    model.eval()
    chunks = []
    for start in range(0, len(x), int(batch_size)):
        output = ensemble_dim(model, model(x[start:start + int(batch_size)]))
        chunks.append(output.mean(1))  # member mean
    return torch.cat(chunks, dim=0)


def train_trajectory(data, fold, inner_index, trajectory, config, device, environments):
    fit_index, val_index = check_split(fold, inner_index)
    training = config['training']
    dtype = getattr(torch, config.get('dtype', 'float32'))
    started = time.perf_counter()
    x = torch.as_tensor(data.X, dtype=torch.float64)[fit_index].to(dtype).to(device)
    y = torch.as_tensor(data.Y, dtype=torch.float64)[fit_index].to(dtype).to(device)
    val_x = torch.as_tensor(data.X, dtype=torch.float64)[val_index].to(dtype).to(device)
    val_y = torch.as_tensor(data.Y, dtype=torch.float64)[val_index].to(device)
    model = build_model(config, trajectory.variant, trajectory.seed, x, y,
                        trajectory.configuration, device)
    reference_seconds = encode_reference(model)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=float(trajectory.configuration['learning_rate']),
                                  weight_decay=float(trajectory.configuration['weight_decay']))
    target = model.standardize_y(y) if hasattr(model, 'standardize_y') else y
    generator = torch.Generator(device='cpu').manual_seed(
        int(trajectory.seed) * 1000 + trajectory.config_index)
    preprocess_seconds = time.perf_counter() - started
    best = {'pcc': -np.inf, 'epoch': -1, 'state': None}
    curve, loss = [], None
    if device.type == 'cuda':
        baseline_memory = float(torch.cuda.memory_allocated(device))
        torch.cuda.reset_peak_memory_stats(device)
    else:
        baseline_memory = None
    trained_start = time.perf_counter()
    for epoch in range(int(training['max_epochs']) + 1):
        if epoch:
            model.train()
            order = torch.randperm(len(x), generator=generator).to(x.device)
            for start in range(0, len(order), int(training['batch_size'])):
                batch = order[start:start + int(training['batch_size'])]
                optimizer.zero_grad(set_to_none=True)
                # Training rows are the library itself: pass local ids so the model can exclude
                # each query from its own retrieval set.
                output = ensemble_dim(model, model(x[batch], query_ids=batch))
                loss = F.mse_loss(output, target[batch].unsqueeze(1).expand_as(output))
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Nonfinite training loss at epoch {epoch}')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               float(training['gradient_clip']))
                optimizer.step()
        prediction = model.inverse_y(predict(model, val_x, training['eval_batch_size'])) \
            if hasattr(model, 'inverse_y') else predict(model, val_x, training['eval_batch_size'])
        score = macro_pcc(prediction.cpu().numpy().astype(np.float64), val_y.cpu().numpy())
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
    prediction = model.inverse_y(predict(model, val_x, training['eval_batch_size'])) \
        if hasattr(model, 'inverse_y') else predict(model, val_x, training['eval_batch_size'])
    values = prediction.cpu().numpy().astype(np.float64)
    truth = val_y.cpu().numpy()
    started = time.perf_counter()
    for _ in range(int(config['efficiency']['warmup'])):
        with torch.no_grad():
            model(val_x[:int(config['efficiency']['batch_size'])])
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    for _ in range(int(config['efficiency']['repeats'])):
        with torch.no_grad():
            model(val_x[:int(config['efficiency']['batch_size'])])
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    inference_seconds = (time.perf_counter() - started) / int(config['efficiency']['repeats'])
    return {'trajectory': {'fold': trajectory.fold, 'inner': trajectory.inner,
                           'seed': trajectory.seed, 'variant': trajectory.variant,
                           'config_index': trajectory.config_index,
                           'configuration': trajectory.configuration},
            'metrics': macro_metrics(values, truth),
            'per_environment': environment_metrics(values, truth, environments),
            'best_epoch': best['epoch'], 'epochs_run': len(curve), 'curve': curve,
            'prediction': values, 'material_index': val_index.tolist(),
            'preprocess_seconds': preprocess_seconds, 'train_seconds': trained_seconds,
            'reference_encode_seconds': reference_seconds,
            'inference_seconds': inference_seconds,
            'memory': ({'baseline_megabytes': baseline_memory / 2 ** 20,
                        'total_allocated_peak_megabytes': peak / 2 ** 20,
                        'increment_megabytes': (peak - baseline_memory) / 2 ** 20}
                       if peak is not None else None),
            'state': best['state']}


# -- selection -------------------------------------------------------------
def unit_key(record):
    trajectory = record['trajectory']
    return (trajectory['fold'], trajectory['inner'], trajectory['variant'])


def unit_scores(records, truth):
    """Both pre-registered readings for one configuration inside one fold and variant.

    ``seed_ensemble_pcc`` is the macro PCC of the seed-averaged prediction (the main selection
    metric); ``seed_mean_pcc`` is the mean of the per-seed macro PCCs and is reported next to it.
    """
    by_inner = {}
    for record in records:
        by_inner.setdefault(record['trajectory']['inner'], []).append(record)
    ensemble, per_seed = [], []
    for inner, rows in sorted(by_inner.items()):
        ordered = sorted(rows, key=lambda record: record['trajectory']['seed'])
        if len(ordered) != len({record['trajectory']['seed'] for record in ordered}):
            raise ValueError('Duplicate seeds inside one inner split')
        truth_rows = truth[inner]
        stacked = np.stack([record['prediction'] for record in ordered], axis=0)
        ensemble.append(macro_pcc(stacked.mean(0), truth_rows))
        per_seed.append(float(np.mean([record['metrics']['pcc'] for record in ordered])))
    return float(np.mean(ensemble)), float(np.mean(per_seed))


def select_arm(records, config, truth):
    """One common configuration per fold and variant, chosen by the fixed selection metric."""
    metric = config.get('selection_metric', SELECTION_METRICS[0])
    if metric not in SELECTION_METRICS:
        raise ValueError(f'Unsupported selection metric: {metric}')
    chosen = {}
    for fold in sorted({record['trajectory']['fold'] for record in records}):
        chosen[fold] = {}
        for variant in config['variants']:
            scored = []
            for index in range(len(config['configurations'])):
                rows = [record for record in records if record['trajectory']['fold'] == fold
                        and record['trajectory']['variant'] == variant
                        and record['trajectory']['config_index'] == index]
                if len(rows) != 3 * len(config['seeds']):
                    raise ValueError('Incomplete trajectory grid for one configuration')
                ensemble, per_seed = unit_scores(rows, truth)
                scored.append((ensemble, index, per_seed,
                               {name: float(np.mean([row['metrics'][name] for row in rows]))
                                for name in METRICS}))
            best = max(scored, key=lambda item: (round(item[0], 12), -item[1]))
            chosen[fold][variant] = {'config_index': best[1],
                                 'configuration': config['configurations'][best[1]],
                                 'seed_ensemble_pcc': best[0],
                                 'seed_mean_pcc': best[2],
                                 'seed_mean_metrics': best[3]}
    return chosen


def write_predictions(path, records):
    """Long-format member-mean predictions keyed by unit, seed and configuration."""
    with path.open('w', newline='') as handle:
        writer = csv.writer(handle, lineterminator='\n')
        header = ['fold', 'inner', 'variant', 'seed', 'config_index', 'material_index'] + \
            [f'y_pred_{name}' for name in range(records[0]['prediction'].shape[1])]
        writer.writerow(header)
        for record in records:
            trajectory = record['trajectory']
            for position, material in enumerate(record['material_index']):
                writer.writerow([trajectory['fold'], trajectory['inner'], trajectory['variant'],
                                 trajectory['seed'], trajectory['config_index'], material,
                                 *[f'{value:.10g}' for value in record['prediction'][position]]])


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
    """Pinned package versions; a mismatch aborts before anything is written."""
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


# -- driver ----------------------------------------------------------------
def run(config, args, runner):
    if args.selection is not None:
        raise ValueError('This development run has no selection-input path')
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
    sources = [here / name for name in ('retrieval_run.py', 'retrieval_tabm.py',
                                       'genomic_tabm.py', 'data.py', 'metrics.py')] \
        + [Path(args.config), Path(runner)]
    configurations = [dict(entry) for entry in config['configurations']]
    config = {**config, 'configurations': configurations}
    expected = len(folds) * 3 * len(config['seeds']) * len(config['variants']) \
        * len(configurations)
    (args.out / 'provenance.json').write_text(json.dumps({
        'config': config,
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'data_sha256': config['data_sha256'], 'splits_sha256': sha256_of(Path(config['splits'])),
        'source_sha256': {str(path): sha256_of(path) for path in sources if path.exists()},
        'torch_version': torch.__version__, 'numpy_version': np.__version__,
        'package_versions': verified_versions, 'model_source': model_source_sha(),
        'device': str(device), 'budget': expected,
        'selection_metric': config.get('selection_metric', SELECTION_METRICS[0]),
        'reference': ({'path': config['reference'],
                       'sha256': sha256_of(Path(config['reference']))}
                      if Path(config.get('reference', '')).is_file() else None),
        'scope': 'Development only: inner fit/val; outer test rows are never read; the retrieval '
                 'library holds fit rows and their environment labels only'}, indent=2))
    state = {'status': 'running', 'completed_trajectories': 0, 'expected': expected,
             'folds_complete': []}
    records, failures, selections = [], [], {}
    truth = {}
    for fold in folds:
        for inner_index in range(3):
            _, val_index = check_split(fold, inner_index)
            truth[(int(fold['fold']), inner_index)] = \
                torch.as_tensor(data.Y[val_index], dtype=torch.float64).numpy()

    def write_state():
        (args.out / 'execution_status.json').write_text(json.dumps(state, indent=2))

    def fold_truth(fold_id):
        return {inner: truth[(fold_id, inner)] for inner in range(3)}

    write_state()
    start = time.perf_counter()
    try:
        for fold in folds:
            for inner_index in range(3):
                for variant in config['variants']:
                    for seed in config['seeds']:
                        for index, entry in enumerate(configurations):
                            trajectory = Trajectory(int(fold['fold']), inner_index, int(seed),
                                                    variant, index, entry)
                            try:
                                record = train_trajectory(data, fold, inner_index, trajectory,
                                                          config, device,
                                                          getattr(data, 'environments', None))
                                path = args.out / (f"f{trajectory.fold}_i{inner_index}_{variant}_"
                                                   f"s{seed}_c{index}.pt")
                                torch.save(record.pop('state'), path)
                                record['checkpoint_sha256'] = sha256_of(path)
                                record['checkpoint_path'] = path.name
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
            try:
                selections.update(select_arm([record for record in records
                                              if record['trajectory']['fold'] == int(fold['fold'])],
                                             config, fold_truth(int(fold['fold']))))
            except Exception as exc:
                selections[str(fold['fold'])] = {'selection_error': repr(exc)}
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
        (args.out / 'timings.json').write_text(json.dumps(
            {'total_seconds': time.perf_counter() - start}, indent=2))
