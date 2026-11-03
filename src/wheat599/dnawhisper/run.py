"""CV1 protocol driver for the DNAwhisper comparator.

In-training supervised pretraining, fine-tuning and per-environment phenotype
standardization are all fitted on the current fitting subset only.  The inner
folds select the fine-tune epoch count, the full development set is then refitted
with that fixed count, and the outer test materials are predicted once.  Upstream
model code is imported from a pinned external checkout; every difference from the
released configuration is written to provenance.json rather than absorbed.
"""

import copy
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch

from wheat599.data import load, sha256_of
from wheat599.dnawhisper.adapter import (finetune_curve, official_files, predict, pretrain,
                                         refit, standardize)
from wheat599.get_icm import write
from wheat599.metrics import mae, pcc, rmse


def git_blob_hash(path):
    raw = path.read_bytes()
    return hashlib.sha1(b'blob %d\x00' % len(raw) + raw).hexdigest()


def verify_upstream(checkout, manifest, commit):
    """Content check of every file the run imports or reads, against the pinned tree.

    A git working tree cannot be trusted as an identity claim here (upstream has no tag
    or release, and the checkout arrives as an archive), so each required file is hashed
    with git's own blob algorithm and compared to the blob id recorded from commit.
    """
    checked, mismatched = {}, []
    for path, expected in manifest.items():
        local = Path(checkout) / path
        if not local.is_file():
            mismatched.append(f'missing {path}')
            continue
        actual = git_blob_hash(local)
        checked[path] = actual
        if actual != expected:
            mismatched.append(f'{path}: local {actual} != pinned {expected}')
    if mismatched:
        raise ValueError(f'Upstream {commit} verification failed: ' + '; '.join(mismatched))
    return {'commit': commit, 'verified_blob_ids': checked,
            'method': 'git blob hash of each required file against the pinned commit tree'}


def unit_rows(fold, index, materials, inner):
    """Fit and scored rows for one inner unit, or for the outer development unit."""
    if inner:
        fit = np.array(fold['inner'][index]['train_idx'])
        target = np.array(fold['inner'][index]['val_idx'])
        if set(fit) & set(target) or set(fit) | set(target) != set(fold['dev_idx']):
            raise ValueError('Inner split does not partition the development set')
    else:
        fit = np.array(fold['dev_idx'])
        target = np.array(fold['test_idx'])
        if set(fit) & set(target) or set(fit) | set(target) != set(range(materials)):
            raise ValueError('Outer split does not cover all materials exactly once')
    return fit, target


def run(cfg, args, runner):
    # The YAML content is the protocol identity: it is what selection.json stores and what
    # the refit phase must match.  Resolved values (device, environment count) are added
    # afterwards for execution and provenance only, so they cannot make the phases disagree.
    protocol = copy.deepcopy(cfg)
    data = load(Path(cfg['data']), cfg['data_sha256'])
    folds = json.loads(Path(cfg['splits']).read_text())['folds']
    split_hash = sha256_of(Path(cfg['splits']))
    checkout = Path(cfg['source_checkout'])
    verification = verify_upstream(checkout, cfg['upstream_blobs'], cfg['source_commit'])
    files = official_files(checkout)
    frozen = json.loads(args.selection.read_text()) if args.selection else None
    if frozen and (frozen['config'] != protocol or frozen['splits_sha256'] != split_hash
                   or frozen['source_commit'] != cfg['source_commit']
                   or len(frozen['selected']) != len(folds)
                   or {item['fold'] for item in frozen['selected']} != {f['fold'] for f in folds}):
        raise ValueError('Frozen selection does not match this configuration or checkout')
    if frozen and any(item['epoch'] < 1 or item['epoch'] > cfg['finetune']['epochs']
                      for item in frozen['selected']):
        raise ValueError('Frozen epoch count is outside the fine-tune schedule')
    device = torch.device('cuda' if cfg['device'] == 'cuda' and torch.cuda.is_available() else 'cpu')
    if cfg['device'] == 'cuda' and device.type != 'cuda':
        print('CUDA unavailable; running on CPU with float32', flush=True)
        cfg = {**cfg, 'precision': '32-true', 'precision_fallback': {'from': cfg['precision'], 'to': '32-true',
                                             'reason': 'bf16-mixed is unavailable without CUDA'}}
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(cfg['gpu_memory_fraction'])
    cfg = {**cfg, 'device': device.type}
    args.out.mkdir(parents=True, exist_ok=False)
    environments = len(data.environments)
    cfg = {**cfg, 'environments': environments}
    (args.out / 'provenance.json').write_text(json.dumps({
        'config': cfg, 'data_sha256': cfg['data_sha256'], 'splits_sha256': split_hash,
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'source_sha256': {name: sha256_of(Path(__file__).with_name(name))
                          for name in ('run.py', 'adapter.py', '__init__.py')},
        'selection_sha256': sha256_of(args.selection) if args.selection else None,
        'upstream': {'url': cfg['source_url'], **verification,
                     'config_sha256': {name: digest for name, (_, digest) in files.items()}},
        'optimizer_passed_to_upstream': {'optimizer_config': cfg['optimizer_config'],
                                         'scheduler_config': cfg['scheduler_config']},
        'package_versions': {'python': __import__('sys').version.split()[0],
                             'torch': torch.__version__,
                             'pytorch_lightning': pl.__version__,
                             'numpy': np.__version__},
        'device': str(device),
        'selection_rule': 'mean macro validation PCC across the three inner units selects the '
                          'fine-tune epoch; earliest epoch wins ties; full development set is '
                          'refitted with that epoch count and test is evaluated once',
        'protocol': 'CV1; pretraining, fine-tuning and standardization fitted on the current '
                    'fitting subset only; test labels never used for fitting or selection',
        'deviations': cfg['deviations'],
        'runner_sha256': sha256_of(Path(runner))}, indent=2))
    rows, oof, failures, selected, curves = [], [], [], [], []
    parameters = None
    start = time.perf_counter()
    for fold in folds:
        k = fold['fold']
        print(f'fold {k}', flush=True)
        try:
            if frozen is None:
                curves_units = []
                for j in range(cfg['inner_units']):
                    fit, target = unit_rows(fold, j, data.n, inner=True)
                    mean, scale = standardize(data.Y, fit)
                    standardized = (data.Y - mean) / scale
                    seed = cfg['training_seed'] + 100 * k + j
                    pl.seed_everything(seed, workers=True)
                    weights, parameters, first = pretrain(checkout, cfg, data.X[fit],
                                                          standardized[fit], seed)
                    curves.append({'fold': k, 'unit': f'inner{j}', **first})
                    module, series, second = finetune_curve(
                        checkout, cfg, weights,
                        (data.X[fit], standardized[fit]), (data.X[target], standardized[target]), seed)
                    curves.append({'fold': k, 'unit': f'inner{j}', **second})
                    curves_units.append(series)
                    del module, weights
                limit = cfg['finetune']['epochs']
                if any(len(series) != limit for series in curves_units):
                    raise RuntimeError('Inner units produced unequal epoch curves')
                means = [float(np.mean([series[e] for series in curves_units])) for e in range(limit)]
                epoch = int(np.argmax(means)) + 1
                selected.append({'fold': k, 'epoch': epoch, 'inner_mean_pcc': means[epoch - 1],
                                 'inner_units': cfg['inner_units'], 'curve': means})
                (args.out / 'selection.json').write_text(json.dumps(
                    {'config': protocol, 'splits_sha256': split_hash,
                     'source_commit': cfg['source_commit'],
                     'parameters': parameters, 'selected': selected}, indent=2))
            else:
                entry = next(item for item in frozen['selected'] if item['fold'] == k)
                fit, target = unit_rows(fold, 0, data.n, inner=False)
                mean, scale = standardize(data.Y, fit)
                standardized = (data.Y - mean) / scale
                seed = cfg['training_seed'] + 100 * k + 99
                pl.seed_everything(seed, workers=True)
                weights, _, first = pretrain(checkout, cfg, data.X[fit], standardized[fit], seed)
                curves.append({'fold': k, 'unit': 'development', **first})
                module, second = refit(checkout, cfg, weights,
                                       (data.X[fit], standardized[fit]), seed, entry['epoch'],
                                       schedule_epochs=cfg['finetune']['epochs'])
                curves.append({'fold': k, 'unit': 'development', **second})
                prediction = predict(module, data.X[target], environments,
                                     cfg['predict_batch_size'],
                                     autocast=cfg['precision'] == 'bf16-mixed') * scale + mean
                truth = data.Y[target]
                if prediction.shape != truth.shape:
                    raise ValueError('Test prediction shape does not match the held-out rows')
                for e, env in enumerate(data.environments):
                    value = {'pcc': pcc(truth[:, e], prediction[:, e]),
                             'rmse': rmse(truth[:, e], prediction[:, e]),
                             'mae': mae(truth[:, e], prediction[:, e])}
                    if not all(np.isfinite(v) for v in value.values()):
                        raise ValueError('Non-finite test metric')
                    rows.append({'stage': 'test', 'fold': k, 'environment': env,
                                 'epoch': entry['epoch'], **value})
                    oof.extend({'environment': env, 'fold': k, 'material_index': int(index),
                                'y_true': float(t), 'y_pred': float(p)}
                               for index, t, p in zip(target, truth[:, e], prediction[:, e]))
                del module, weights
        except Exception as exc:
            failures.append({'fold': k, 'reason': repr(exc)})
            print(f'FAILED {k}: {exc}', flush=True)
        finally:
            (args.out / 'training.json').write_text(json.dumps(curves, indent=2))
            write(args.out / 'failures.csv', ['fold', 'reason'], failures)
            if frozen is None:
                write(args.out / 'inner_curve.csv', ['fold', 'epoch', 'mean_inner_val_pcc'],
                      [{'fold': item['fold'], 'epoch': e + 1, 'mean_inner_val_pcc': curve}
                       for item in selected for e, curve in enumerate(item['curve'])])
                (args.out / 'selection_summary.json').write_text(json.dumps(
                    {'parameters': parameters,
                     'selected_epochs': {item['fold']: item['epoch'] for item in selected}}, indent=2))
            else:
                write(args.out / 'per_fold.csv',
                      ['stage', 'fold', 'environment', 'epoch', 'pcc', 'rmse', 'mae'], rows)
                write(args.out / 'oof.csv',
                      ['environment', 'fold', 'material_index', 'y_true', 'y_pred'], oof)
    if failures:
        raise RuntimeError('Incomplete DNAwhisper run; inspect failures.csv')
    if frozen is None:
        print('Development selection complete; no test predictions were made.', flush=True)
        return
    if len([row for row in rows if row['stage'] == 'test']) != len(folds) * environments:
        raise RuntimeError('Unexpected number of test rows')
    per = {}
    for env in data.environments:
        values = [row for row in rows if row['environment'] == env]
        per[env] = {metric: float(np.mean([row[metric] for row in values]))
                    for metric in ('pcc', 'rmse', 'mae')}
    stages = [record for record in curves if 'seconds' in record]
    summary = {'n_runs': len([row for row in rows if row['stage'] == 'test']), 'n_failures': 0,
               'per_environment': per,
               'macro': {metric: float(np.mean([per[env][metric] for env in data.environments]))
                         for metric in ('pcc', 'rmse', 'mae')},
               'pooled_oof_pcc': {env: float(pcc(
                   np.array([row['y_true'] for row in oof if row['environment'] == env]),
                   np.array([row['y_pred'] for row in oof if row['environment'] == env])))
                    for env in data.environments},
               'selected_epochs': {item['fold']: item['epoch'] for item in frozen['selected']},
               'parameters': frozen.get('parameters'),
               'cost': {'stages': len(stages),
                        'stage_seconds': [record['seconds'] for record in stages],
                        'total_stage_seconds': float(np.sum([r['seconds'] for r in stages])),
                        'mean_stage_seconds': float(np.mean([r['seconds'] for r in stages])),
                        'peak_megabytes': float(np.max([r['peak_megabytes'] for r in stages])),
                        'optimizer_steps': int(np.sum([r['optimizer_steps'] for r in stages])),
                        'elapsed_seconds': round(time.perf_counter() - start, 3),
                        'device': str(device)}}
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)
