"""Development-only runner for the genomic TabM adaptation. Outer test labels never enter fitting or selection.

Each trajectory fits on one inner fit split, early-stops and selects on that inner validation split,
and never indexes the outer test rows. The model owns its preprocessing (fit-row buffers), so the
runner passes raw markers and raw phenotypes; validation PCC is computed on ``inverse_y`` predictions
(scale-invariant), and RMSE/MAE are reported on the original scale.

Budget: every configured outer fold, its three inner splits, both variants, both seeds and the
learning-rate/weight-decay products. The checkpoint rule is the highest validation macro PCC with ties
to the earlier epoch. One common configuration per fold and variant is chosen by the mean recorded
validation PCC over its inner splits and seeds; RMSE never selects. The grouped-additive ridge lambda
is chosen inside the fit rows with a seeded re-standardising fold split, cached per (fold, inner), and
its solution initialises the model's ``beta``/``bias``; the model's ridge penalty uses the same lambda
divided by ``n * q`` and ``beta`` is excluded from weight decay.

Every trajectory writes its curve, timings, parameter count, group metadata, peak GPU memory relative
to the allocation already present (another tenant may share the card), the measured inference pass on
the efficiency batch including preprocessing, and the checkpoint SHA. Failures are written as they
occur; nothing is averaged with nanmean.
"""
import inspect
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from wheat599.data import load, sha256_of
from wheat599.metrics import mae, pcc, rmse

VARIANTS = ('raw', 'grouped_additive')
METRICS = ('pcc', 'rmse', 'mae')
LOSS_DESCRIPTION = ('mean over batch, ensemble members and environments of squared error '
                    '(mean_bke), plus the ridge penalty')


# -- model adapter ---------------------------------------------------------
def build_model(config, variant, seed, train_x, train_y, beta=None, bias=None, device='cpu'):
    """Seeded construction on the raw fit rows; the model owns its preprocessing."""
    from wheat599 import genomic_tabm
    builder = getattr(genomic_tabm, 'GenomicTabM', None)
    if builder is None:
        raise RuntimeError('wheat599.genomic_tabm.GenomicTabM is unavailable')
    penalty = 0.0 if beta is None else float(beta['ridge_lambda']) / (
        float(len(train_x)) * float(train_y.shape[1]))
    dtype = getattr(torch, config.get('dtype', 'float32'))
    torch.manual_seed(int(seed))
    model = builder(train_x.to(dtype), train_y.to(dtype), variant=variant,
                    beta=None if beta is None else beta['beta'].to(dtype),
                    bias=None if bias is None else bias.to(dtype),
                    beta_ridge=penalty, dtype=dtype, **dict(config['architecture']))
    return model.to(device)


def ensemble(output):
    """The official TabM returns (n, members, q); the member axis is the second one."""
    if output.ndim != 3:
        raise RuntimeError(f'Expected (n, members, q) output, got {tuple(output.shape)}')
    return output


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
    """Per-environment PCC/RMSE/MAE, so completions never need the weights reloaded."""
    names = list(environments) if environments is not None else list(range(truth.shape[1]))
    if len(names) != truth.shape[1]:
        raise ValueError('Environment labels disagree with the phenotype columns')
    return {str(names[column]): {name: float(function(truth[:, column], prediction[:, column]))
                                 for name, function in zip(METRICS, (pcc, rmse, mae))}
            for column in range(truth.shape[1])}


# -- grouped-additive ridge ------------------------------------------------
def standardise(x_fit, x_all):
    """Fit-row mean/std; constant columns become zero, matching the model's own convention."""
    mean, spread = x_fit.mean(dim=0), x_fit.std(dim=0, unbiased=False)
    constant = spread <= 0
    safe = torch.where(constant, torch.ones_like(spread), spread)
    block = (x_all - mean) / safe
    return torch.where(constant.unsqueeze(0), torch.zeros_like(block), block)


def ridge_solution(x, y, ridge):
    """Slope and intercept of the unpenalised-intercept ridge, via the n-by-n dual system.

    Centring the columns removes the intercept, so ``beta = Xc^T (Xc Xc^T + lambda I)^-1 Yc`` and
    ``bias = ybar - xbar beta``; this stays cheap when markers exceed materials.
    """
    x_mean, y_mean = x.mean(0, keepdim=True), y.mean(0, keepdim=True)
    centered_x, centered_y = x - x_mean, y - y_mean
    system = centered_x @ centered_x.T + ridge * torch.eye(len(x), dtype=x.dtype, device=x.device)
    beta = centered_x.T @ torch.linalg.solve(system, centered_y)
    bias = (y_mean - x_mean @ beta).reshape(-1)
    return beta, bias


def ridge_folds(rows, count, seed):
    order = torch.randperm(rows, generator=torch.Generator().manual_seed(int(seed)))
    return torch.tensor_split(order, int(count))


def choose_ridge(x_fit, y_fit, lambdas, folds):
    """Fit-internal cross-validation; every fold carries its own marker and phenotype scalers."""
    rows = len(x_fit)
    totals = {float(value): 0.0 for value in lambdas}
    for hold in folds:
        keep = torch.ones(rows, dtype=torch.bool)
        keep[hold] = False
        z = standardise(x_fit[keep], x_fit)
        y_mean = y_fit[keep].mean(dim=0)
        y_std = y_fit[keep].std(dim=0, unbiased=False).clamp_min(1e-12)
        scaled = (y_fit - y_mean) / y_std
        for value in lambdas:
            beta, bias = ridge_solution(z[keep], scaled[keep], float(value))
            prediction = z[hold] @ beta + bias
            totals[float(value)] += float((prediction - scaled[hold]).square().mean()) * int(hold.numel())
    best = min(totals, key=lambda value: (round(totals[value], 12), list(lambdas).index(value)))
    return best, totals


def ridge_init(x_fit, y_fit, value):
    """beta/bias of the model's linear branch, expressed in its standardised-phenotype space."""
    mean, spread = y_fit.mean(dim=0), y_fit.std(dim=0, unbiased=False).clamp_min(1e-12)
    scaled = (y_fit - mean) / spread
    z = standardise(x_fit, x_fit)
    return ridge_solution(z, scaled, float(value))


# -- one trajectory --------------------------------------------------------
@dataclass
class Trajectory:
    fold: int
    inner: int
    seed: int
    variant: str
    config_index: int
    learning_rate: float
    weight_decay: float


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


def evaluate(model, val_x, val_y, config, device, environments=None):
    training = config['training']
    chunks = []
    with torch.no_grad():
        for start in range(0, len(val_x), int(training['eval_batch_size'])):
            batch = val_x[start:start + int(training['eval_batch_size'])]
            chunks.append(model.inverse_y(ensemble(model(batch)).mean(1)))
        prediction = torch.cat(chunks, dim=0)
    truth = val_y.cpu().numpy()
    values = prediction.cpu().numpy().astype(np.float64)
    metrics = macro_metrics(values, truth)
    by_environment = environment_metrics(values, truth, environments)
    efficiency = config['efficiency']
    batch = val_x[:int(efficiency['batch_size'])]
    for _ in range(int(efficiency['warmup'])):
        with torch.no_grad():
            model(batch)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for _ in range(int(efficiency['repeats'])):
        with torch.no_grad():
            model(batch)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    seconds = (time.perf_counter() - started) / int(efficiency['repeats'])
    return metrics, by_environment, {'batch': int(len(batch)),
                                     'repeats': int(efficiency['repeats']),
                                     'seconds_per_repeat': seconds,
                                     'includes_preprocessing': bool(efficiency['include_preprocessing'])}


def train_trajectory(data, fold, inner_index, trajectory, config, device, ridge_cache,
                     environments=None):
    fit_index, val_index = check_split(fold, inner_index)
    training = config['training']
    dtype = getattr(torch, config.get('dtype', 'float32'))
    started = time.perf_counter()
    x = torch.as_tensor(data.X, dtype=torch.float64)
    y = torch.as_tensor(data.Y, dtype=torch.float64)
    fit_x, fit_y = x[fit_index].to(dtype).to(device), y[fit_index].to(dtype).to(device)
    val_x, val_y = x[val_index].to(dtype).to(device), y[val_index].to(dtype).to(device)
    weight, bias, value, ridge_seconds = None, None, None, 0.0
    cache_hit = False
    if trajectory.variant == 'grouped_additive':
        key = (int(trajectory.fold), int(inner_index))
        if key not in ridge_cache:
            ridge_started = time.perf_counter()
            chosen, _ = choose_ridge(x[fit_index], y[fit_index], training['ridge_grid'],
                                     ridge_folds(len(fit_index), training['ridge_folds'],
                                                 training['ridge_seed']))
            weight, bias = ridge_init(x[fit_index], y[fit_index], chosen)
            ridge_cache[key] = {'lambda': chosen, 'beta': weight, 'bias': bias,
                                'seconds': time.perf_counter() - ridge_started}
        else:
            cache_hit = True
        cached = ridge_cache[key]
        value, weight, bias = cached['lambda'], cached['beta'], cached['bias']
        ridge_seconds = cached['seconds']
    model = build_model(config, trajectory.variant, trajectory.seed, fit_x, fit_y,
                        beta={'beta': weight, 'ridge_lambda': value} if weight is not None else None,
                        bias=bias, device=device)
    decay = [parameter for name, parameter in model.named_parameters()
             if trajectory.variant != 'grouped_additive' or name != 'beta']
    groups = [{'params': decay, 'weight_decay': trajectory.weight_decay}]
    if trajectory.variant == 'grouped_additive':
        groups.append({'params': [model.beta], 'weight_decay': 0.0})
    if sum(len(group['params']) for group in groups) != len(list(model.parameters())):
        raise RuntimeError('Every parameter must belong to exactly one optimiser group')
    optimizer = torch.optim.AdamW(groups, lr=trajectory.learning_rate)
    target = model.standardize_y(fit_y)
    generator = torch.Generator(device='cpu').manual_seed(
        int(trajectory.seed) * 1000 + trajectory.config_index)
    preprocess_seconds = time.perf_counter() - started
    best = {'pcc': -np.inf, 'epoch': -1, 'state': None}
    curve = []
    if device.type == 'cuda':
        baseline_memory = float(torch.cuda.memory_allocated(device))
        torch.cuda.reset_peak_memory_stats(device)
    else:
        baseline_memory = None
    trained_start = time.perf_counter()
    loss = None
    for epoch in range(int(training['max_epochs']) + 1):
        mse_total, objective_total = 0.0, 0.0
        if epoch:
            model.train()
            order = torch.randperm(len(fit_x), generator=generator).to(fit_x.device)
            for start in range(0, len(order), int(training['batch_size'])):
                batch = order[start:start + int(training['batch_size'])]
                optimizer.zero_grad(set_to_none=True)
                output = ensemble(model(fit_x[batch]))
                batch_mse = F.mse_loss(output, target[batch].unsqueeze(1).expand_as(output))
                loss = batch_mse + model.ridge_penalty()
                mse_total += float(batch_mse.detach()) * len(batch)
                objective_total += float(loss.detach()) * len(batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Nonfinite training loss at epoch {epoch}')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               float(training['gradient_clip']))
                optimizer.step()
        # Epoch 0 is the initialisation itself and is eligible as the best checkpoint.
        model.eval()
        with torch.no_grad():
            prediction = model.inverse_y(ensemble(model(val_x)).mean(1))
        score = macro_pcc(prediction.cpu().numpy().astype(np.float64), val_y.cpu().numpy())
        curve.append({'epoch': epoch, 'validation_pcc': score,
                      'train_mse': mse_total / len(fit_x) if epoch else None,
                      'train_objective': objective_total / len(fit_x) if epoch else None})
        if score > best['pcc'] + float(training['tie_tolerance']):
            best = {'pcc': score, 'epoch': epoch,
                    'state': {key: value.detach().cpu().clone()
                              for key, value in model.state_dict().items()}}
        elif epoch - best['epoch'] >= int(training['patience']):
            break
    trained_seconds = time.perf_counter() - trained_start
    peak = float(torch.cuda.max_memory_allocated(device)) if device.type == 'cuda' else None

    model.load_state_dict({key: value.to(device) for key, value in best['state'].items()})
    model.eval()
    metrics, by_environment, inference = evaluate(model, val_x, val_y, config, device,
                                                  environments)
    return {'trajectory': trajectory.__dict__, 'metrics': metrics,
            'per_environment': by_environment,
            'best_epoch': best['epoch'], 'epochs_run': len(curve) - 1, 'validation_checks': len(curve), 'curve': curve,
            'parameters': model.parameter_count(), 'groups': model.group_metadata(),
            'preprocess_seconds': preprocess_seconds, 'train_seconds': trained_seconds,
            'inference': inference,
            'memory': ({'baseline_megabytes': baseline_memory / 2 ** 20,
                        'total_allocated_peak_megabytes': peak / 2 ** 20,
                        'increment_megabytes': (peak - baseline_memory) / 2 ** 20}
                       if peak is not None else None),
            'ridge_lambda': value, 'ridge_cv_seconds': ridge_seconds,
            'ridge_cache_hit': cache_hit, 'state': best['state']}


# -- versions and model source --------------------------------------------
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
    """Pinned package versions from the config; a mismatch aborts before anything is written."""
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


def model_source_sha():
    """SHA of the imported model packages, so the published source is pinned as well."""
    out = {}
    for name in ('tabm', 'rtdl_num_embeddings'):
        try:
            path = inspect.getsourcefile(__import__(name))
        except Exception:
            path = None
        out[name] = {'path': path, 'sha256': sha256_of(Path(path)) if path else None}
    return out


# -- driver ----------------------------------------------------------------
def select_common(records, config):
    """One configuration per fold and variant: mean validation PCC over inner splits and seeds."""
    expected = 3 * len(config['seeds'])
    out = {}
    for fold in sorted({record['trajectory']['fold'] for record in records}):
        chosen = {}
        for variant in config['variants']:
            scored = []
            for index in range(len(config['training']['learning_rates'])
                               * len(config['training']['weight_decays'])):
                rows = [record for record in records if record['trajectory']['fold'] == fold
                        and record['trajectory']['variant'] == variant
                        and record['trajectory']['config_index'] == index]
                if len(rows) != expected:
                    raise ValueError('Incomplete trajectory grid for one configuration')
                scored.append((float(np.mean([row['metrics']['pcc'] for row in rows])), index,
                               {name: float(np.mean([row['metrics'][name] for row in rows]))
                                for name in METRICS}))
            best = max(scored, key=lambda item: (round(item[0], 12), -item[1]))
            chosen[variant] = {'config_index': best[1], 'validation_pcc': best[0],
                               'validation_metrics': best[2]}
        out[fold] = chosen
    return out


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
    sources = [here / name for name in ('genomic_tabm_run.py', 'genomic_tabm.py', 'data.py',
                                       'metrics.py')] + [Path(args.config), Path(runner)]
    configurations = [{'learning_rate': float(lr), 'weight_decay': float(wd)}
                      for lr in config['training']['learning_rates']
                      for wd in config['training']['weight_decays']]
    reference = ({'path': config['reference'],
                  'sha256': sha256_of(Path(config['reference']))}
                 if Path(config['reference']).is_file() else None)
    (args.out / 'provenance.json').write_text(json.dumps({
        'config': config,
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'data_sha256': config['data_sha256'], 'splits_sha256': sha256_of(Path(config['splits'])),
        'source_sha256': {str(path): sha256_of(path) for path in sources if path.exists()},
        'torch_version': torch.__version__, 'numpy_version': np.__version__,
        'package_versions': verified_versions, 'model_source': model_source_sha(),
        'loss': LOSS_DESCRIPTION, 'device': str(device),
        'budget': len(folds) * 3 * len(config['seeds']) * len(config['variants'])
                  * len(configurations),
        'scope': 'Development only: inner fit/val of the configured outer folds; '
                 'outer test labels never enter fitting or selection',
        'reference': reference}, indent=2))
    records, failures, ridge_cache = [], [], {}
    expected = len(folds) * 3 * len(config['seeds']) * len(config['variants']) \
        * len(configurations)
    state = {'status': 'running', 'completed_trajectories': 0, 'expected': expected,
             'folds_complete': []}
    selections = {}

    def write_state():
        (args.out / 'execution_status.json').write_text(json.dumps(state, indent=2))

    write_state()
    start = time.perf_counter()
    try:
        for fold in folds:
            for inner_index in range(3):
                for variant in config['variants']:
                    for seed in config['seeds']:
                        for index, entry in enumerate(configurations):
                            trajectory = Trajectory(int(fold['fold']), inner_index, int(seed),
                                                    variant, index, entry['learning_rate'],
                                                    entry['weight_decay'])
                            try:
                                record = train_trajectory(
                                    data, fold, inner_index, trajectory, config, device,
                                    ridge_cache, getattr(data, 'environments', None))
                                state['completed_trajectories'] += 1
                                path = args.out / (f"f{trajectory.fold}_i{inner_index}_"
                                                   f"{variant}_s{seed}_c{index}.pt")
                                torch.save(record.pop('state'), path)
                                record['checkpoint_sha256'] = sha256_of(path)
                                record['checkpoint_path'] = path.name
                                records.append(record)
                            except Exception as exc:
                                failures.append({'trajectory': trajectory.__dict__,
                                                 'reason': repr(exc)})
                                raise
                            finally:
                                (args.out / 'trajectories.json').write_text(
                                    json.dumps(records, indent=2, default=str))
                                (args.out / 'failures.json').write_text(
                                    json.dumps(failures, indent=2))
            state['folds_complete'].append(int(fold['fold']))
            try:
                selections.update(select_common(
                    [record for record in records
                     if record['trajectory']['fold'] == int(fold['fold'])], config))
            except Exception as exc:  # an incomplete fold never masks the trajectory failure
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
