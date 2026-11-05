"""Final CV1 evaluation of independent single-environment TabM specialists.

The 80 fits are read from the frozen refit manifest: ten outer folds, the four
Wheat599 environments and seeds 42/2026. Each fit uses the manifest's full
development rows, its selected configuration and its fixed median epoch count;
there is no validation, checkpoint selection, scheduling or test-based rule
choice. Test genotypes are used only to form predictions after each fit, and
test phenotypes are read once after all 80 fits for the final metrics.

The manifest is validated before any fit: row completeness, fold/environment/
seed uniqueness, fit/test ID order and hashes, configuration and selection
cross-checks, data/splits hashes, source artifact hashes and the frozen rows
digest. Baselines are compared by per-fold test IDs and truth, not by requiring
a full split-manifest SHA match. The GLS reference is derived from the frozen
Huber per-fold metrics and the Huber-minus-GLS paired differences, with both
source SHAs recorded.
"""
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from wheat599.data import load, sha256_of
from wheat599.environment_tabm import SharedTabM
from wheat599.metrics import mae, pcc, rmse

METRICS = ('pcc', 'rmse', 'mae')
DEFAULT_MANIFEST = Path('results/independent_tabm_refit_manifest_20261105/manifest.json')
DEFAULT_SOURCE_RUN = Path('results/environment_tabm_development_20261105')
DEFAULT_SELECTION_SOURCE = Path('results/environment_selection_development_20261105')
DEFAULT_HUBER_SOURCE = Path('results/robust_icm_final_20261103')
DEFAULT_BIMSGP_SOURCE = Path('results/bimsgp_nested_final_20261102')
DEFAULT_HISTORICAL_SPLITS = Path('results/bimsgp_ftt_fixed_20261029/splits_manifest.json')


# -- basic helpers ---------------------------------------------------------
def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str))


def ids_sha256(ids):
    return hashlib.sha256(','.join(str(int(value)) for value in ids).encode()).hexdigest()


def rows_sha256(rows):
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def metric_dict(truth, prediction):
    truth = np.asarray(truth, dtype=np.float64).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if truth.size != prediction.size:
        raise ValueError('truth and prediction must have equal size')
    values = {'pcc': float(pcc(truth, prediction)),
              'rmse': float(rmse(truth, prediction)),
              'mae': float(mae(truth, prediction))}
    if not all(np.isfinite(value) for value in values.values()):
        raise FloatingPointError('Nonfinite metric')
    return values


def mean_metrics(rows):
    return {name: float(np.mean([row[name] for row in rows])) for name in METRICS}


def group_metrics(rows, keys):
    groups = {}
    for row in rows:
        key = tuple(row[name] for name in keys)
        groups.setdefault(key, {'truth': [], 'prediction': []})
        groups[key]['truth'].append(float(row['y_true']))
        groups[key]['prediction'].append(float(row['y_pred']))
    return {key: metric_dict(value['truth'], value['prediction'])
            for key, value in groups.items()}


def package_version(name):
    try:
        import importlib.metadata as metadata
        return metadata.version(name)
    except Exception:
        try:
            return getattr(__import__(name), '__version__', None)
        except ImportError:
            return None


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


def git_value(*arguments):
    return subprocess.check_output(['git', *arguments], text=True).strip()


def ensure_frozen_manifest(path):
    repo = Path(git_value('rev-parse', '--show-toplevel'))
    try:
        relative = Path(path).resolve().relative_to(repo.resolve())
    except ValueError as error:
        raise ValueError('Manifest must live inside the repository') from error
    subprocess.check_output(['git', 'ls-files', '--error-unmatch', str(relative)],
                            text=True)
    dirty = git_value('status', '--porcelain', '--', str(relative))
    if dirty:
        raise ValueError(f'Manifest is not committed cleanly: {dirty}')


# -- manifest validation ---------------------------------------------------
def load_manifest(path):
    document = read_json(path)
    if int(document.get('schema_version', -1)) != 1:
        raise ValueError('Unsupported manifest schema')
    rows = document.get('rows')
    if not isinstance(rows, list):
        raise ValueError('Manifest rows must be a list')
    if int(document.get('row_count', -1)) != len(rows):
        raise ValueError('Manifest row_count disagrees with rows')
    if document.get('rows_sha256') != rows_sha256(rows):
        raise ValueError('Manifest rows_sha256 mismatch')
    return document, rows


def freeze_document(config):
    path = Path(config.get('freeze_config', ''))
    if not path.is_file():
        return None
    try:
        import yaml
    except ImportError as error:
        raise ImportError('PyYAML is required for freeze_config') from error
    return yaml.safe_load(path.read_text())


def source_configuration(config):
    freeze = freeze_document(config)
    if freeze and freeze.get('configurations'):
        return [dict(entry) for entry in freeze['configurations']]
    if config.get('configurations'):
        return [dict(entry) for entry in config['configurations']]
    path = Path(config.get('source_config', 'configs/environment_tabm.yaml'))
    if path.is_file():
        try:
            import yaml
        except ImportError as error:
            raise ImportError('PyYAML is required for source_config') from error
        document = yaml.safe_load(path.read_text())
        if document.get('configurations'):
            return [dict(entry) for entry in document['configurations']]
    return None


def selection_maps(config):
    selection_source = Path(config.get('selection_source', DEFAULT_SELECTION_SOURCE))
    source_run = Path(config.get('source_run', DEFAULT_SOURCE_RUN))
    selection_path = Path(config.get('selection_file', selection_source / 'selection.json'))
    environment_path = Path(config.get('environment_selection_file',
                                      source_run / 'selection.json'))
    if not selection_path.is_file() or not environment_path.is_file():
        raise FileNotFoundError('Selection cross-check sources are missing')
    selection = read_json(selection_path)
    list_map = {}
    for entry in selection:
        if entry.get('training') == 'independent' and entry.get('selection') == 'per_environment':
            list_map[int(entry['fold'])] = [int(value) for value in entry['config_indices']]
    environment = read_json(environment_path)
    dict_map = {}
    for fold_key, fold_data in environment.items():
        independent = fold_data.get('independent')
        if independent:
            dict_map[int(fold_key)] = {int(env): int(value['config_index'])
                                       for env, value in independent.items()}
    return selection_path, environment_path, list_map, dict_map


def source_artifact_paths(config):
    source_run = Path(config.get('source_run', DEFAULT_SOURCE_RUN))
    selection_source = Path(config.get('selection_source', DEFAULT_SELECTION_SOURCE))
    return {
        'predictions': Path(config.get('prediction_source', source_run / 'predictions.csv')),
        'trajectories': Path(config.get('trajectory_source', source_run / 'trajectories.json')),
        'selection': Path(config.get('selection_file', selection_source / 'selection.json')),
        'environment_selection': Path(config.get('environment_selection_file',
                                                   source_run / 'selection.json')),
        'artifact_manifest': Path(config.get('artifact_manifest',
                                             source_run / 'artifact_manifest.json')),
    }


def validate_manifest(config, data, splits, document, rows, manifest_path):
    fold_map = {int(fold['fold']): fold for fold in splits['folds']}
    configured = config.get('outer_folds')
    folds = [int(value) for value in configured] if configured is not None \
        else sorted(fold_map)
    if any(fold not in fold_map for fold in folds):
        raise ValueError('Configured fold is absent from splits')
    seeds = [int(value) for value in config.get('seeds', [42, 2026])]
    environments = list(data.environments)
    expected_rows = len(folds) * len(environments) * len(seeds)
    if len(rows) != expected_rows:
        raise ValueError(f'Expected {expected_rows} manifest rows, found {len(rows)}')

    configurations = source_configuration(config)
    freeze = freeze_document(config)
    if freeze:
        for key in ('source_run', 'selection_source', 'historical_splits',
                    'huber_source', 'gls_source'):
            if config.get(key) is not None and freeze.get(key) is not None \
                    and str(config[key]) != str(freeze[key]):
                raise ValueError(f'freeze_config disagrees on {key}')
        if freeze.get('seeds') is not None \
                and [int(value) for value in freeze['seeds']] != seeds:
            raise ValueError('freeze_config seeds disagree')
        if freeze.get('configurations') is not None \
                and [dict(entry) for entry in freeze['configurations']] != configurations:
            raise ValueError('freeze_config configurations disagree')
    selection_path, environment_path, list_selection, dict_selection = selection_maps(config)
    for fold in folds:
        if fold not in list_selection or fold not in dict_selection:
            raise ValueError(f'Selection missing for fold {fold}')
        if list_selection[fold] != [dict_selection[fold].get(index)
                                     for index in range(len(environments))]:
            raise ValueError(f'Selection sources disagree for fold {fold}')

    artifacts = source_artifact_paths(config)
    for name in ('predictions', 'trajectories', 'selection'):
        if not artifacts[name].is_file():
            raise FileNotFoundError(f'Missing source artifact: {artifacts[name]}')
    actual_source = {
        'predictions': sha256_of(artifacts['predictions']),
        'trajectories': sha256_of(artifacts['trajectories']),
        'selection': sha256_of(artifacts['selection']),
    }
    expected_hashes = {
        'predictions': config.get('predictions_sha256')
        or (freeze or {}).get('predictions_sha256'),
        'trajectories': config.get('trajectories_sha256')
        or (freeze or {}).get('trajectories_sha256'),
        'selection': config.get('selection_sha256')
        or (freeze or {}).get('selection_sha256'),
    }
    for name, expected in expected_hashes.items():
        if expected and str(expected) != actual_source[name]:
            raise ValueError(f'Config {name} source sha256 does not match the artifact')
    artifact_commit = None
    if artifacts['artifact_manifest'].is_file():
        artifact_commit = read_json(artifacts['artifact_manifest']).get('git_commit')

    seen = set()
    normalized = []
    test_by_fold = {}
    for row in rows:
        fold = int(row['fold'])
        environment_index = int(row['environment_index'])
        seed = int(row['seed'])
        if fold not in folds or not 0 <= environment_index < len(environments):
            raise ValueError('Manifest fold/environment out of range')
        if seed not in seeds:
            raise ValueError('Manifest seed out of range')
        key = (fold, environment_index, seed)
        if key in seen:
            raise ValueError(f'Duplicate manifest row {key}')
        seen.add(key)
        if str(row['environment']) != str(environments[environment_index]):
            raise ValueError('Manifest environment label disagrees with data')

        configuration = dict(row.get('configuration') or {
            name: row[name] for name in ('n_blocks', 'd_block', 'dropout', 'k',
                                         'learning_rate', 'weight_decay')})
        config_index = int(row['config_index'])
        if configurations is not None:
            if not 0 <= config_index < len(configurations):
                raise ValueError('Manifest config_index out of range')
            if configuration != configurations[config_index]:
                raise ValueError('Manifest configuration disagrees with source config')
        for name in ('n_blocks', 'd_block', 'dropout', 'k',
                     'learning_rate', 'weight_decay'):
            if name in row and row[name] != configuration[name]:
                raise ValueError('Flattened configuration disagrees with configuration')
        if config_index != list_selection[fold][environment_index]:
            raise ValueError('Manifest config disagrees with selection')

        inner = [int(value) for value in row['inner_best_epochs']]
        if len(inner) != 3 or any(value < 0 for value in inner):
            raise ValueError('inner_best_epochs must be three non-negative integers')
        epochs = int(row['median_epochs'])
        if epochs != sorted(inner)[1]:
            raise ValueError('median_epochs is not the median of inner_best_epochs')
        if epochs < 0:
            raise ValueError('median_epochs must be non-negative')

        split = fold_map[fold]
        fit_ids = [int(value) for value in row['fit_ids']]
        test_ids = [int(value) for value in row['test_ids']]
        if fit_ids != [int(value) for value in split['dev_idx']]:
            raise ValueError('Manifest fit_ids order disagrees with splits')
        if test_ids != [int(value) for value in split['test_idx']]:
            raise ValueError('Manifest test_ids order disagrees with splits')
        if int(row['fit_count']) != len(fit_ids) or int(row['test_count']) != len(test_ids):
            raise ValueError('Manifest fit/test counts disagree')
        if row['fit_ids_sha256'] != ids_sha256(fit_ids) \
                or row['test_ids_sha256'] != ids_sha256(test_ids):
            raise ValueError('Manifest fit/test ids sha256 mismatch')
        if set(fit_ids) & set(test_ids):
            raise ValueError('fit_ids and test_ids overlap')
        if set(fit_ids) | set(test_ids) != set(range(len(data.X))):
            raise ValueError('fit_ids and test_ids do not partition materials')
        if fold in test_by_fold:
            if test_by_fold[fold] != test_ids:
                raise ValueError('A fold has inconsistent test order across rows')
        else:
            test_by_fold[fold] = test_ids

        if str(row['data_sha256']) != str(config['data_sha256']):
            raise ValueError('Manifest data_sha256 mismatch')
        if str(row['splits_sha256']) != sha256_of(Path(config['splits'])):
            raise ValueError('Manifest splits_sha256 mismatch')
        if row['prediction_source_sha256'] != actual_source['predictions']:
            raise ValueError('Manifest prediction source sha256 mismatch')
        if row['trajectory_source_sha256'] != actual_source['trajectories']:
            raise ValueError('Manifest trajectory source sha256 mismatch')
        if row['selection_source_sha256'] != actual_source['selection']:
            raise ValueError('Manifest selection source sha256 mismatch')
        if artifact_commit is not None and row['source_git_commit'] != artifact_commit:
            raise ValueError('Manifest source_git_commit disagrees with artifact manifest')

        normalized.append({
            'fold': fold, 'environment_index': environment_index,
            'environment': str(environments[environment_index]), 'seed': seed,
            'config_index': config_index, 'configuration': configuration,
            'inner_best_epochs': inner, 'epochs': epochs,
            'fit_ids': fit_ids, 'test_ids': test_ids,
            'source_git_commit': row['source_git_commit'],
        })

    if sorted(value for values in test_by_fold.values() for value in values) \
            != list(range(len(data.X))):
        raise ValueError('Per-fold test IDs do not cover materials exactly once')
    for environment_index in range(len(environments)):
        for seed in seeds:
            covered = [material for row in normalized
                       if row['environment_index'] == environment_index
                       and row['seed'] == seed for material in row['test_ids']]
            if sorted(covered) != list(range(len(data.X))):
                raise ValueError('Per-environment/seed OOF test IDs are not a partition')
    for fold in folds:
        for environment_index in range(len(environments)):
            for seed in seeds:
                if (fold, environment_index, seed) not in seen:
                    raise ValueError('Manifest is missing a fold/environment/seed row')
    historical = None
    historical_path = Path(config.get('historical_splits', DEFAULT_HISTORICAL_SPLITS))
    if historical_path.is_file():
        historical_document = read_json(historical_path)
        for fold in folds:
            if [int(value) for value in historical_document['folds'][fold]['test_idx']] \
                    != [int(value) for value in fold_map[fold]['test_idx']]:
                raise ValueError('Historical test ID order disagrees with current splits')
        historical = {'path': str(historical_path), 'sha256': sha256_of(historical_path)}

    return {
        'folds': folds, 'seeds': seeds, 'environments': environments,
        'rows': sorted(normalized, key=lambda item: (item['fold'],
                                                     item['environment_index'],
                                                     item['seed'])),
        'manifest_sha256': sha256_of(Path(manifest_path)),
        'rows_sha256': document.get('rows_sha256'),
        'row_count': len(rows),
        'verification': document.get('verification'),
        'source_artifacts': {name: {'path': str(path),
                                    'sha256': actual_source.get(name)}
                             for name, path in artifacts.items()},
        'selection_sources': {'selection': str(selection_path),
                              'environment_selection': str(environment_path)},
        'historical_splits': historical,
        'source_git_commit': rows[0]['source_git_commit'],
    }


# -- fitting ---------------------------------------------------------------
def fit_one(row, data, config, device, weights_dir, save_weights=True):
    dtype = getattr(torch, config.get('dtype', 'float32'))
    fit_ids, test_ids = row['fit_ids'], row['test_ids']
    environment_index = row['environment_index']
    configuration = row['configuration']
    batch_size = int(config.get('batch_size', 128))
    clip = config.get('gradient_clip')
    started = time.perf_counter()
    x_fit = torch.as_tensor(np.asarray(data.X)[fit_ids], dtype=torch.float64).to(dtype)
    y_fit = torch.as_tensor(np.asarray(data.Y)[fit_ids, environment_index:environment_index + 1],
                            dtype=torch.float64).to(dtype)
    torch.manual_seed(int(row['seed']))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(row['seed']))
    model = SharedTabM(
        x_fit, y_fit, n_blocks=int(configuration['n_blocks']),
        d_block=int(configuration['d_block']), dropout=float(configuration['dropout']),
        k=int(configuration['k']), dtype=dtype)
    model = model.to(device)
    x_fit = x_fit.to(device)
    y_fit = y_fit.to(device)
    target = model.standardize_y(y_fit)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=float(configuration['learning_rate']),
                                  weight_decay=float(configuration['weight_decay']))
    generator = torch.Generator().manual_seed(
        int(row['seed']) * 1000 + int(row['config_index']))
    preprocess_seconds = time.perf_counter() - started
    if device.type == 'cuda':
        baseline_memory = float(torch.cuda.memory_allocated(device))
        torch.cuda.reset_peak_memory_stats(device)
    else:
        baseline_memory = None

    train_started = time.perf_counter()
    for epoch in range(1, int(row['epochs']) + 1):
        model.train()
        order = torch.randperm(len(fit_ids), generator=generator)
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size].to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(x_fit[batch])
            loss = F.mse_loss(output, target[batch].unsqueeze(1).expand_as(output))
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite training loss at epoch {epoch}')
            loss.backward()
            if clip is not None and float(clip) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip))
            optimizer.step()
    train_seconds = time.perf_counter() - train_started

    test_started = time.perf_counter()
    x_test = torch.as_tensor(np.asarray(data.X)[test_ids], dtype=torch.float64).to(dtype)
    x_test = x_test.to(device)
    model.eval()
    with torch.no_grad():
        output = model(x_test)
        member_mean = output.mean(1)
        prediction = model.inverse_y(member_mean)[:, 0].cpu().numpy().astype(np.float64)
    predict_seconds = time.perf_counter() - test_started
    peak_allocated = float(torch.cuda.max_memory_allocated(device)) \
        if device.type == 'cuda' else None
    peak_reserved = float(torch.cuda.max_memory_reserved(device)) \
        if device.type == 'cuda' else None

    weight_path, weight_sha256, weight_bytes = None, None, 0
    if save_weights:
        weights_dir.mkdir(parents=True, exist_ok=True)
        weight_path = weights_dir / (f"f{row['fold']}_e{environment_index}_"
                                     f"s{row['seed']}_c{row['config_index']}.pt")
        state = {key: value.detach().cpu().clone()
                 for key, value in model.state_dict().items()}
        torch.save(state, weight_path)
        weight_sha256 = sha256_of(weight_path)
        weight_bytes = weight_path.stat().st_size

    return {
        'prediction': prediction,
        'test_ids': test_ids,
        'weight_path': None if weight_path is None else str(weight_path),
        'weight_sha256': weight_sha256,
        'weight_bytes': weight_bytes,
        'parameters': model.parameter_count(),
        'epochs': int(row['epochs']),
        'preprocess_seconds': preprocess_seconds,
        'train_seconds': train_seconds,
        'predict_seconds': predict_seconds,
        'memory': ({'baseline_megabytes': baseline_memory / 2 ** 20,
                    'total_allocated_peak_megabytes': peak_allocated / 2 ** 20,
                    'increment_megabytes': (peak_allocated - baseline_memory) / 2 ** 20,
                    'total_reserved_peak_megabytes': peak_reserved / 2 ** 20}
                   if peak_allocated is not None else None),
    }


# -- summaries and baselines ----------------------------------------------
def baseline_specs(config):
    huber = Path(config.get('huber_source', DEFAULT_HUBER_SOURCE))
    gls = Path(config.get('gls_source', DEFAULT_HUBER_SOURCE))
    bimsgp = Path(config.get('bimsgp_source', DEFAULT_BIMSGP_SOURCE))
    return {
        'huber': {'kind': 'oof', 'source': huber, 'oof': huber / 'oof.csv',
                  'per_fold': huber / 'per_fold.csv'},
        'gls': {'kind': 'derived', 'source': gls, 'per_fold': gls / 'per_fold.csv',
                'differences': gls / 'paired_differences.csv', 'base_model': 'huber',
                'base_oof': huber / 'oof.csv'},
        'bimsgp': {'kind': 'oof', 'source': bimsgp, 'oof': bimsgp / 'oof.csv',
                   'per_fold': bimsgp / 'per_fold.csv'},
    }


def read_oof(path):
    groups = {}
    with Path(path).open(newline='') as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = (str(row['environment']), int(row['fold']))
            groups.setdefault(key, []).append(row)
    return groups


def verify_baseline_groups(groups, candidate_groups):
    for key, candidate in candidate_groups.items():
        if key not in groups:
            raise ValueError(f'Baseline missing group {key}')
        baseline = groups[key]
        if [int(row['material_index']) for row in baseline] != candidate['ids']:
            raise ValueError(f'Baseline test ID order mismatch for {key}')
        truth = np.asarray([float(row['y_true']) for row in baseline], dtype=np.float64)
        if not np.allclose(truth, candidate['truth'], atol=1e-12, rtol=0):
            raise ValueError(f'Baseline truth mismatch for {key}')


def oof_metrics(groups):
    per_fold = {}
    per_environment = {}
    pooled_truth, pooled_prediction = [], []
    for key, rows in groups.items():
        environment, fold = key
        truth = np.asarray([float(row['y_true']) for row in rows], dtype=np.float64)
        prediction = np.asarray([float(row['y_pred']) for row in rows], dtype=np.float64)
        per_fold[key] = metric_dict(truth, prediction)
        per_environment.setdefault(environment, {'truth': [], 'prediction': []})
        per_environment[environment]['truth'].extend(truth)
        per_environment[environment]['prediction'].extend(prediction)
        pooled_truth.extend(truth)
        pooled_prediction.extend(prediction)
    environment_metrics = {environment: metric_dict(value['truth'], value['prediction'])
                           for environment, value in per_environment.items()}
    pooled = metric_dict(pooled_truth, pooled_prediction)
    return {'per_fold': per_fold, 'per_environment': environment_metrics, 'pooled': pooled}


def derived_metrics(spec, candidate_groups):
    per_fold_path = spec['per_fold']
    differences_path = spec['differences']
    rows = {}
    with Path(per_fold_path).open(newline='') as handle:
        for row in csv.DictReader(handle):
            if 'model' in row and row['model'] != spec.get('base_model', 'huber'):
                continue
            rows[(str(row['environment']), int(row['fold']))] = {
                name: float(row[name]) for name in METRICS}
    differences = {}
    with Path(differences_path).open(newline='') as handle:
        for row in csv.DictReader(handle):
            differences[(str(row['environment']), int(row['fold']))] = {
                name: float(row['delta_' + name]) for name in METRICS}
    if set(rows) != set(differences):
        raise ValueError('Derived baseline sources have different keys')
    per_fold = {}
    for key in rows:
        per_fold[key] = {name: rows[key][name] - differences[key][name]
                         for name in METRICS}
    if spec.get('base_oof') and Path(spec['base_oof']).is_file():
        verify_baseline_groups(read_oof(spec['base_oof']), candidate_groups)
    return {'per_fold': per_fold, 'per_environment': None, 'pooled': None,
            'derived': 'huber_minus_paired_difference',
            'sources': {
                'per_fold': {'path': str(per_fold_path),
                             'sha256': sha256_of(Path(per_fold_path))},
                'differences': {'path': str(differences_path),
                                'sha256': sha256_of(Path(differences_path))}}}


def baseline_result(spec, candidate_groups):
    if spec['kind'] == 'derived':
        return derived_metrics(spec, candidate_groups)
    if not Path(spec['oof']).is_file():
        raise FileNotFoundError(f'Missing baseline OOF: {spec["oof"]}')
    groups = read_oof(spec['oof'])
    verify_baseline_groups(groups, candidate_groups)
    result = oof_metrics(groups)
    result['derived'] = None
    result['sources'] = {'oof': {'path': str(spec['oof']),
                                 'sha256': sha256_of(Path(spec['oof']))}}
    if Path(spec['per_fold']).is_file():
        result['sources']['per_fold'] = {'path': str(spec['per_fold']),
                                         'sha256': sha256_of(Path(spec['per_fold']))}
    return result


def summary_block(per_fold, per_environment, pooled):
    if not per_fold:
        return None
    ordered = [per_fold[key] for key in sorted(per_fold, key=lambda value: (value[1], value[0]))]
    folds = sorted({key[1] for key in per_fold})
    fold_mean = []
    for fold in folds:
        fold_rows = [per_fold[key] for key in per_fold if key[1] == fold]
        fold_mean.append(mean_metrics(fold_rows))
    return {
        'primary_40_envfold': mean_metrics(ordered),
        'fold_mean': mean_metrics(fold_mean),
        'per_environment_fold_mean': {str(environment): mean_metrics(
            [per_fold[key] for key in per_fold if key[0] == str(environment)])
            for environment in sorted({key[0] for key in per_fold})},
        'per_environment': per_environment,
        'pooled': pooled,
        'per_fold': [{'environment': key[0], 'fold': key[1], **per_fold[key]}
                     for key in sorted(per_fold, key=lambda value: (value[1], value[0]))],
    }


def fold_macro(per_fold):
    folds = sorted({key[1] for key in per_fold})
    return {fold: mean_metrics([per_fold[key] for key in per_fold if key[1] == fold])
            for fold in folds}


# -- CSV writing -----------------------------------------------------------
def write_csv(path, rows, fieldnames):
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def write_state(path, value):
    write_json(path, value)


def load_resume_state(args, data, validation, config):
    out = Path(args.out)
    fits_path = out / 'fits.json'
    seed_path = out / 'seed_predictions.csv'
    provenance_path = out / 'provenance.json'
    failures_path = out / 'failures.json'
    for path in (fits_path, seed_path, provenance_path):
        if not path.is_file():
            raise FileNotFoundError(f'Resume artifact missing: {path}')
    fit_records = read_json(fits_path)
    provenance = read_json(provenance_path)
    failures = read_json(failures_path) if failures_path.is_file() else []
    if len(fit_records) != len(validation['rows']):
        raise ValueError('Resume fits.json does not contain all manifest fits')
    row_map = {(row['fold'], row['environment_index'], row['seed']): row
               for row in validation['rows']}
    seen = set()
    for record in fit_records:
        key = (int(record['fold']), int(record['environment_index']), int(record['seed']))
        if key not in row_map:
            raise ValueError(f'Resume fit is absent from manifest: {key}')
        row = row_map[key]
        if int(record['config_index']) != row['config_index'] \
                or int(record['epochs']) != row['epochs']:
            raise ValueError(f'Resume fit config/epochs mismatch: {key}')
        weight_path = Path(record['weight_path'])
        if not weight_path.is_file():
            raise FileNotFoundError(f'Resume weight missing: {weight_path}')
        if record['weight_sha256'] != sha256_of(weight_path):
            raise ValueError(f'Resume weight sha256 mismatch: {key}')
        if int(record['weight_bytes']) != weight_path.stat().st_size:
            raise ValueError(f'Resume weight size mismatch: {key}')
        if key in seen:
            raise ValueError(f'Duplicate resume fit: {key}')
        seen.add(key)
    if seen != set(row_map):
        raise ValueError('Resume fits do not cover every manifest row')

    environment_index = {label: index
                        for index, label in enumerate(validation['environments'])}
    expected = len(data.X) * len(validation['environments']) * len(validation['seeds'])
    with seed_path.open(newline='') as handle:
        seed_rows = list(csv.DictReader(handle))
    if len(seed_rows) != expected:
        raise ValueError(f'Resume seed prediction count {len(seed_rows)} != {expected}')
    grouped = {}
    seen_seed = set()
    for item in seed_rows:
        environment = str(item['environment'])
        fold = int(item['fold'])
        seed = int(item['seed'])
        material = int(item['material_index'])
        if environment not in environment_index or seed not in validation['seeds']:
            raise ValueError('Resume seed row has an unknown environment/seed')
        key = (fold, environment_index[environment], seed)
        if key not in row_map:
            raise ValueError('Resume seed row has an unknown fold')
        if material not in row_map[key]['test_ids']:
            raise ValueError('Resume seed row material is not a test ID')
        unique = (fold, environment, seed, material)
        if unique in seen_seed:
            raise ValueError('Resume seed row is duplicated')
        seen_seed.add(unique)
        y_true = float(item['y_true'])
        y_pred = float(item['y_pred'])
        if not np.isfinite(y_true) or not np.isfinite(y_pred):
            raise ValueError('Resume seed row is non-finite')
        truth = float(np.asarray(data.Y)[material, environment_index[environment]])
        if not np.allclose(y_true, truth, atol=1e-12, rtol=0):
            raise ValueError('Resume seed row truth does not match data')
        grouped.setdefault(key, {})[material] = y_pred
    if set(grouped) != set(row_map):
        raise ValueError('Resume seed predictions do not cover every fit')
    predictions = {}
    for key, values in grouped.items():
        test_ids = [int(value) for value in row_map[key]['test_ids']]
        if sorted(values) != sorted(test_ids):
            raise ValueError('Resume seed prediction IDs do not match test IDs')
        prediction = np.asarray([values[material] for material in test_ids],
                                dtype=np.float64)
        predictions[key] = {'prediction': prediction, 'test_ids': test_ids}
    fit_failures = sum(1 for failure in failures if 'row' in failure)
    if fit_failures:
        raise ValueError('Resume is not allowed when fit failures exist')
    original_summary_failures = len(failures) - fit_failures
    return fit_records, predictions, failures, provenance, fit_failures, original_summary_failures


# -- driver ----------------------------------------------------------------
def run(config, args, runner):
    manifest_path = Path(args.selection) if args.selection is not None \
        else Path(config.get('manifest', DEFAULT_MANIFEST))
    if bool(config.get('require_committed_manifest', True)):
        ensure_frozen_manifest(manifest_path)
    device = torch.device(config.get('device', 'cuda'))
    torch.set_num_threads(int(config.get('cpu_threads', 2)))
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(float(config.get('gpu_memory_fraction', 0.5)))
    data = load(Path(config['data']), config['data_sha256'])
    splits = read_json(Path(config['splits']))
    document, rows = load_manifest(manifest_path)
    if config.get('manifest_sha256') \
            and str(config['manifest_sha256']) != sha256_of(manifest_path):
        raise ValueError('Manifest file sha256 disagrees with config')
    if config.get('rows_sha256') \
            and str(config['rows_sha256']) != document.get('rows_sha256'):
        raise ValueError('Manifest rows_sha256 disagrees with config')
    if config.get('expected_fits') is not None \
            and int(config['expected_fits']) != len(rows):
        raise ValueError('Manifest fit count disagrees with expected_fits')
    for name, expected in (config.get('versions') or {}).items():
        found = package_version(name)
        if found != str(expected):
            raise RuntimeError(f'Pinned package {name} is {found}, expected {expected}')
    validation = validate_manifest(config, data, splits, document, rows, manifest_path)
    resume = bool(getattr(args, 'resume_summary', False))
    if resume:
        if not args.out.is_dir():
            raise FileNotFoundError('Resume requires an existing output directory')
        fit_records, predictions, failures, provenance, fit_failures, \
            original_summary_failures = load_resume_state(args, data, validation, config)
        weights_dir = args.out / 'weights'
        total_started = time.perf_counter()
        resume_info = {
            'initial_git_commit': provenance.get('git_commit'),
            'resume_git_commit': git_value('rev-parse', 'HEAD'),
            'resume_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
            'resume_command': sys.argv,
            'fits_verified': len(fit_records),
            'fit_failures': fit_failures,
            'original_summary_failures': original_summary_failures,
            'summary_recovery': 'success',
            'resume_source_sha256': {},
        }
        for path in (Path(__file__).parent / 'independent_final.py',
                     Path(__file__).parent / 'environment_tabm.py',
                     Path(__file__).parent / 'genomic_tabm.py',
                     Path(__file__).parent / 'data.py',
                     Path(__file__).parent / 'metrics.py',
                     Path(args.config), Path(runner)):
            if Path(path).exists():
                resume_info['resume_source_sha256'][str(path)] = sha256_of(Path(path))
    else:
        args.out.mkdir(parents=True, exist_ok=False)
        weights_dir = args.out / 'weights'
        weights_dir.mkdir()
        state = {'status': 'running', 'completed_fits': 0,
                 'expected_fits': len(validation['rows']),
                 'manifest': str(manifest_path), 'folds_complete': []}
        write_state(args.out / 'execution_status.json', state)
        write_state(args.out / 'manifest_validation.json', validation)
        provenance = {
            'config': config, 'manifest': {'path': str(manifest_path),
                                           'sha256': validation['manifest_sha256'],
                                           'rows_sha256': validation['rows_sha256'],
                                           'row_count': validation['row_count']},
            'validation': validation, 'data_sha256': config['data_sha256'],
            'splits_sha256': sha256_of(Path(config['splits'])),
            'git_commit': git_value('rev-parse', 'HEAD'),
            'source_sha256': {}, 'baselines': {},
            'environment': environment_provenance(device),
            'budget': len(validation['rows']),
            'protocol': 'CV1 outer test; full-dev refit; fixed manifest epochs; '
                        'member-mean then seed-mean; no test selection',
        }
        for path in (Path(__file__).parent / 'independent_final.py',
                     Path(__file__).parent / 'environment_tabm.py',
                     Path(__file__).parent / 'genomic_tabm.py',
                     Path(__file__).parent / 'data.py',
                     Path(__file__).parent / 'metrics.py',
                     Path(args.config), Path(runner)):
            if Path(path).exists():
                provenance['source_sha256'][str(path)] = sha256_of(Path(path))
        try:
            from wheat599.genomic_tabm_run import model_source_sha
            provenance['model_source'] = model_source_sha()
        except Exception as error:
            provenance['model_source'] = {'error': repr(error)}
        write_state(args.out / 'provenance.json', provenance)

        fit_records = []
        predictions = {}
        failures = []
        fit_failures = 0
        original_summary_failures = 0
        resume_info = None
        total_started = time.perf_counter()
    try:
        for row in ([] if resume else validation['rows']):
            try:
                result = fit_one(row, data, config, device, weights_dir,
                                 save_weights=bool(config.get('save_weights', True)))
                key = (row['fold'], row['environment_index'], row['seed'])
                predictions[key] = {'prediction': result['prediction'],
                                    'test_ids': row['test_ids'],
                                    'environment': row['environment'],
                                    'fold': row['fold']}
                fit_records.append({'fold': row['fold'],
                                    'environment_index': row['environment_index'],
                                    'environment': row['environment'],
                                    'seed': row['seed'],
                                    'config_index': row['config_index'],
                                    'epochs': row['epochs'],
                                    'inner_best_epochs': row['inner_best_epochs'],
                                    'parameters': result['parameters'],
                                    'weight_sha256': result['weight_sha256'],
                                    'weight_path': result['weight_path'],
                                    'weight_bytes': result['weight_bytes'],
                                    'preprocess_seconds': result['preprocess_seconds'],
                                    'train_seconds': result['train_seconds'],
                                    'predict_seconds': result['predict_seconds'],
                                    'memory': result['memory']})
                state['completed_fits'] += 1
            except Exception as error:
                failures.append({'row': row, 'reason': repr(error)})
                write_state(args.out / 'failures.json', failures)
                raise
            finally:
                write_state(args.out / 'execution_status.json', state)
                if fit_records:
                    write_state(args.out / 'fits.json', fit_records)
                    with (args.out / 'weights_sha256.txt').open('w', newline='') as handle:
                        writer = csv.writer(handle, lineterminator='\n')
                        writer.writerow(['fold', 'environment_index', 'environment', 'seed',
                                         'config_index', 'sha256', 'bytes', 'path'])
                        for record in fit_records:
                            writer.writerow([record['fold'], record['environment_index'],
                                             record['environment'], record['seed'],
                                             record['config_index'],
                                             record['weight_sha256'],
                                             record['weight_bytes'], record['weight_path']])
        if not resume:
            write_state(args.out / 'failures.json', failures)
            write_state(args.out / 'weights_manifest.json', fit_records)

        truth = {}
        seed_rows = []
        oof_rows = []
        for fold in validation['folds']:
            for environment_index, environment in enumerate(validation['environments']):
                for seed in validation['seeds']:
                    key = (fold, environment_index, seed)
                    if key not in predictions:
                        raise ValueError(f'Missing fit prediction {key}')
                    test_ids = predictions[key]['test_ids']
                    truth_values = np.asarray(data.Y[test_ids, environment_index],
                                              dtype=np.float64)
                    truth[(fold, environment_index)] = truth_values
                    for material, y_true, y_pred in zip(test_ids, truth_values,
                                                        predictions[key]['prediction']):
                        seed_rows.append({'environment': environment, 'fold': fold,
                                          'material_index': int(material), 'seed': seed,
                                          'y_true': float(y_true), 'y_pred': float(y_pred)})
                seeds = [predictions[(fold, environment_index, seed)]
                         for seed in validation['seeds']]
                test_ids = seeds[0]['test_ids']
                ensemble = np.mean([item['prediction'] for item in seeds], axis=0)
                truth_values = np.asarray(data.Y[test_ids, environment_index],
                                          dtype=np.float64)
                for material, y_true, y_pred in zip(test_ids, truth_values, ensemble):
                    oof_rows.append({'environment': environment, 'fold': fold,
                                     'material_index': int(material),
                                     'y_true': float(y_true), 'y_pred': float(y_pred)})

        seed_rows.sort(key=lambda row: (row['environment'], row['fold'], row['seed'],
                                        row['material_index']))
        oof_rows.sort(key=lambda row: (row['environment'], row['fold'], row['material_index']))
        if len(seed_rows) != len(data.X) * len(validation['environments']) * len(validation['seeds']):
            raise ValueError('Seed prediction row count mismatch')
        if len(oof_rows) != len(data.X) * len(validation['environments']):
            raise ValueError('OOF row count mismatch')
        if not resume:
            write_csv(args.out / 'seed_predictions.csv', seed_rows,
                      ['environment', 'fold', 'material_index', 'seed', 'y_true', 'y_pred'])
        write_csv(args.out / 'oof.csv', oof_rows,
                  ['environment', 'fold', 'material_index', 'y_true', 'y_pred'])

        candidate_groups = {}
        for row in oof_rows:
            key = (row['environment'], row['fold'])
            candidate_groups.setdefault(key, {'ids': [], 'truth': [], 'prediction': []})
            candidate_groups[key]['ids'].append(row['material_index'])
            candidate_groups[key]['truth'].append(row['y_true'])
            candidate_groups[key]['prediction'].append(row['y_pred'])
        per_fold = {key: metric_dict(value['truth'], value['prediction'])
                    for key, value in candidate_groups.items()}
        per_environment = {}
        for environment in validation['environments']:
            environment_keys = [key for key in candidate_groups if key[0] == environment]
            per_environment[environment] = metric_dict(
                [value for key in environment_keys for value in candidate_groups[key]['truth']],
                [value for key in environment_keys for value in candidate_groups[key]['prediction']])
        pooled = metric_dict([row['y_true'] for row in oof_rows],
                             [row['y_pred'] for row in oof_rows])
        candidate_summary = summary_block(per_fold, per_environment, pooled)
        per_fold_rows = [{'environment': key[0], 'fold': key[1], **per_fold[key]}
                         for key in sorted(per_fold, key=lambda value: (value[1], value[0]))]
        write_csv(args.out / 'per_fold.csv', per_fold_rows,
                  ['environment', 'fold', 'pcc', 'rmse', 'mae'])
        fold_rows = []
        for fold in validation['folds']:
            values = [per_fold[key] for key in per_fold if key[1] == fold]
            fold_rows.append({'fold': fold, **mean_metrics(values)})
        write_csv(args.out / 'fold_metrics.csv', fold_rows, ['fold', 'pcc', 'rmse', 'mae'])
        write_csv(args.out / 'oof_metrics.csv',
                  [{'environment': environment, **per_environment[environment]}
                   for environment in validation['environments']] +
                  [{'environment': 'macro', **mean_metrics(list(per_environment.values()))}],
                  ['environment', 'pcc', 'rmse', 'mae'])

        seed_metrics = {}
        for seed in validation['seeds']:
            rows_seed = [row for row in seed_rows if row['seed'] == seed]
            groups = {}
            for row in rows_seed:
                groups.setdefault((row['environment'], row['fold']), {'truth': [], 'prediction': []})
                groups[(row['environment'], row['fold'])]['truth'].append(row['y_true'])
                groups[(row['environment'], row['fold'])]['prediction'].append(row['y_pred'])
            metrics = {key: metric_dict(value['truth'], value['prediction'])
                       for key, value in groups.items()}
            seed_metrics[str(seed)] = summary_block(
                metrics,
                {environment: metric_dict(
                    [value for key in metrics if key[0] == environment
                     for value in groups[key]['truth']],
                    [value for key in metrics if key[0] == environment
                     for value in groups[key]['prediction']])
                 for environment in validation['environments']},
                metric_dict([row['y_true'] for row in rows_seed],
                            [row['y_pred'] for row in rows_seed]))

        baselines = {}
        comparison_rows = []
        baseline_values = {}
        for name, spec in baseline_specs(config).items():
            result = baseline_result(spec, candidate_groups)
            baseline_values[name] = result
            baselines[name] = {
                'source': str(spec['source']),
                'kind': spec['kind'],
                'derived': result.get('derived'),
                'sources': result.get('sources'),
                'summary': summary_block(result['per_fold'], result['per_environment'],
                                         result['pooled']),
            }
        for key in sorted(per_fold, key=lambda value: (value[1], value[0])):
            row = {'environment': key[0], 'fold': key[1],
                   'candidate_pcc': per_fold[key]['pcc'],
                   'candidate_rmse': per_fold[key]['rmse'],
                   'candidate_mae': per_fold[key]['mae']}
            for name in baseline_values:
                metrics = baseline_values[name]['per_fold'].get(key)
                for metric in METRICS:
                    value = None if metrics is None else metrics[metric]
                    row[f'{name}_{metric}'] = value
                    row[f'{name}_delta_{metric}'] = None if value is None \
                        else per_fold[key][metric] - value
            comparison_rows.append(row)
        fields = ['environment', 'fold'] + [f'candidate_{name}' for name in METRICS]
        for name in baseline_values:
            fields += [f'{name}_{metric}' for metric in METRICS]
            fields += [f'{name}_delta_{metric}' for metric in METRICS]
        write_csv(args.out / 'comparison.csv', comparison_rows, fields)

        candidate_fold = fold_macro(per_fold)
        baseline_folds = {name: fold_macro(result['per_fold'])
                          for name, result in baseline_values.items()}
        fold_comparison_rows = []
        for fold in validation['folds']:
            row = {'fold': fold}
            for metric in METRICS:
                row[f'candidate_{metric}'] = candidate_fold[fold][metric]
            for name, values in baseline_folds.items():
                for metric in METRICS:
                    row[f'{name}_{metric}'] = values[fold][metric]
                    row[f'{name}_delta_{metric}'] = (
                        candidate_fold[fold][metric] - values[fold][metric])
            fold_comparison_rows.append(row)
        fold_fields = ['fold'] + [f'candidate_{metric}' for metric in METRICS]
        for name in baseline_folds:
            fold_fields += [f'{name}_{metric}' for metric in METRICS]
            fold_fields += [f'{name}_delta_{metric}' for metric in METRICS]
        write_csv(args.out / 'fold_comparison.csv', fold_comparison_rows, fold_fields)

        primary_delta = {}
        fold_mean_delta = {}
        for name, result in baseline_values.items():
            summary = baselines[name]['summary']
            primary_delta[name] = {
                metric: candidate_summary['primary_40_envfold'][metric]
                - summary['primary_40_envfold'][metric] for metric in METRICS}
            fold_mean_delta[name] = {
                metric: candidate_summary['fold_mean'][metric]
                - summary['fold_mean'][metric] for metric in METRICS}
        comparison_summary = {'candidate': candidate_summary,
                              'baselines': {name: value['summary']
                                            for name, value in baselines.items()},
                              'sources': {name: value['sources']
                                          for name, value in baselines.items()},
                              'primary_delta': primary_delta,
                              'fold_mean_delta': fold_mean_delta,
                              'fold_macro': {
                                  'candidate': candidate_fold,
                                  'baselines': baseline_folds,
                              }}
        write_state(args.out / 'comparison_summary.json', comparison_summary)

        summary = {
            'n_runs': len(fit_records), 'n_failures': len(failures),
            'n_materials': len(data.X), 'n_environments': len(validation['environments']),
            'seeds': validation['seeds'], 'folds': validation['folds'],
            'primary': candidate_summary['primary_40_envfold'],
            'fold_mean': candidate_summary['fold_mean'],
            'pooled_oof': candidate_summary['pooled'],
            'per_environment': candidate_summary['per_environment'],
            'per_environment_fold_mean': candidate_summary['per_environment_fold_mean'],
            'per_fold': per_fold_rows,
            'seed_metrics': seed_metrics,
            'baselines': baselines,
            'manifest_validation': {'manifest_sha256': validation['manifest_sha256'],
                                    'rows_sha256': validation['rows_sha256'],
                                    'row_count': validation['row_count']},
            'cost': {
                'fits': len(fit_records),
                'epochs': sum(record['epochs'] for record in fit_records),
                'parameters': [record['parameters'] for record in fit_records],
                'preprocess_seconds': sum(record['preprocess_seconds'] for record in fit_records),
                'train_seconds': sum(record['train_seconds'] for record in fit_records),
                'predict_seconds': sum(record['predict_seconds'] for record in fit_records),
                'max_increment_megabytes': max(
                    (record['memory']['increment_megabytes'] for record in fit_records
                     if record['memory'] is not None), default=None),
            },
        }
        summary['failure_accounting'] = {
            'fit_failures': fit_failures,
            'original_summary_failures': original_summary_failures,
            'summary_recovery': 'success' if resume else None,
        }
        if resume:
            summary['resume'] = resume_info
            provenance['resume'] = dict(resume_info)
            write_state(args.out / 'provenance.json', provenance)
        write_state(args.out / 'summary.json', summary)
        status_payload = {
            'status': 'complete', 'completed_fits': len(fit_records),
            'expected_fits': len(validation['rows']), 'folds_complete': validation['folds'],
            'fit_failures': fit_failures,
            'original_summary_failures': original_summary_failures,
            'summary_recovery': 'success' if resume else None,
        }
        if resume:
            status_payload['resumed'] = True
        write_state(args.out / 'execution_status.json', status_payload)
        timing_payload = {'total_seconds': time.perf_counter() - total_started,
                          'fits': len(fit_records),
                          'epochs': summary['cost']['epochs'],
                          'train_seconds': summary['cost']['train_seconds'],
                          'predict_seconds': summary['cost']['predict_seconds']}
        write_state(args.out / ('resume_timings.json' if resume else 'timings.json'),
                    timing_payload)
    except Exception as error:
        write_state(args.out / 'execution_status.json',
                    {'status': 'failed', 'completed_fits': len(fit_records),
                     'expected_fits': len(validation['rows']), 'folds_complete': []})
        if not failures or failures[-1].get('reason') != repr(error):
            failures.append({'stage': 'run', 'reason': repr(error)})
        write_state(args.out / 'failures.json', failures)
        raise
