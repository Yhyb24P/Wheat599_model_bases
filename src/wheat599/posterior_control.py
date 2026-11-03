"""Matched-budget direct predictor control for the posterior correction study.

Trains the direct multi-output in-context model on the same fitting subset, the
same synthetic task stream, configuration distribution, step count and seed as
the posterior corrector, so the only difference is the prediction mechanism
(analytic base plus learned posterior increment versus direct amortized
prediction). Read-only with respect to the running study: this module is not
imported by non_gaussian_icm.
"""
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from wheat599.data import load, sha256_of
from wheat599.get_icm import write
from wheat599.metrics import pcc, rmse, mae
from wheat599.multienv_kernel import task_correlation
from wheat599.posterior_direct import DirectPredictor
from wheat599.posterior_synthetic import prepare_genotypes
from wheat599.posterior_training import train


def direct_prediction(model, markers, phenotypes, fit, target, base, environments, device):
    """Predict target rows from fit rows; standardization comes from fit rows only."""
    rows = torch.as_tensor(markers, dtype=torch.float64, device=device)
    y = torch.as_tensor(phenotypes[fit], dtype=torch.float64, device=device)
    mean, scale = y.mean(0), y.std(0, unbiased=False)
    if torch.any(scale <= 0):
        raise ValueError('Constant phenotype in fitting subset')
    response = (y - mean) / scale
    z, rbf, z_query, rbf_cross = prepare_genotypes(rows[fit], base['bandwidth'], target=rows[target])
    identity = torch.eye(environments, dtype=torch.float64, device=device)
    covariance = (1 - base['eta']) * identity + base['eta'] * torch.tensor(
        task_correlation(response.detach().cpu().numpy().astype(np.float64)),
        dtype=torch.float64, device=device)
    prediction = model(rows[fit], response, rows[target], covariance, base['fraction'],
                       base['ridge'][0], base['bandwidth'])
    return (prediction * scale + mean).detach().cpu().numpy()


def run(cfg, args, runner):
    data = load(Path(cfg['data']), cfg['data_sha256'])
    folds = json.loads(Path(cfg['splits']).read_text())['folds']
    split_hash = sha256_of(Path(cfg['splits']))
    base_records = json.loads((Path(cfg['base_source']) / 'base_choices.json').read_text())
    comparator = json.loads(Path(cfg['comparator_selection']).read_text())
    if comparator['splits_sha256'] != split_hash:
        raise ValueError('Comparator selection splits mismatch')
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'provenance.json').write_text(json.dumps({
        'config': cfg, 'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'splits_sha256': split_hash,
        'base_source_sha256': sha256_of(Path(cfg['base_source']) / 'base_choices.json'),
        'comparator_selection_sha256': sha256_of(Path(cfg['comparator_selection'])),
        'model_sha256': sha256_of(Path(__file__)),
        'training_sha256': sha256_of(Path(__file__).parent / 'posterior_training.py'),
        'runner_sha256': sha256_of(runner),
        'rule': 'matched task stream, configuration distribution, steps and seed with the corrector; '
                'inner-fold macro PCC is a conditionally selected score, not generalization',
        'protocol': 'CV1 development only; no test predictions'}, indent=2))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(cfg['gpu_memory_fraction'])
    rows, failures, curves = [], [], []
    start = time.perf_counter()
    for fold in folds:
        k = fold['fold']
        print(f'fold {k}', flush=True)
        try:
            eta = next(entry for entry in comparator['selected'] if entry['fold'] == k)['squared']['eta']
            for j, inner in enumerate(fold['inner']):
                fit, target = np.array(inner['train_idx']), np.array(inner['val_idx'])
                if set(fit) & set(target) or set(fit) | set(target) != set(fold['dev_idx']):
                    raise ValueError('Invalid development split')
                base = dict(next(record['homogeneous'] for record in base_records
                                 if record['fold'] == k and record['inner_fold'] == j))
                if len(set(base['ridge'])) != 1:
                    raise ValueError('The shared-ridge operator requires one ridge across environments')
                base['eta'] = eta
                seed = cfg['training']['seed'] + 100 * k + j
                torch.manual_seed(seed)
                # The shared network block names the iteration count 'steps' for the
                # corrector; the direct encoder names it 'layers'.
                network = dict(cfg['network'])
                model = DirectPredictor(markers=data.X.shape[1], layers=network.pop('steps'), **network)
                result = train(model, data.X[fit], {**cfg['training'], 'seed': seed,
                                                    'validation_seed': cfg['training']['validation_seed'] + seed},
                               device, mode='direct',
                               log=lambda record: curves.append({'fold': k, 'unit': f'inner{j}', **record}))
                model.load_state_dict(result['checkpoint']['state'])
                curves.append({'fold': k, 'unit': f'inner{j}', 'seed': seed, 'cost': result['cost'],
                               'checkpoint_step': result['checkpoint']['step']})
                prediction = direct_prediction(model.eval(), data.X, data.Y, fit, target, base,
                                               data.Y.shape[1], device)
                for e, env in enumerate(data.environments):
                    value = pcc(data.Y[target, e], prediction[:, e])
                    if not np.isfinite(value):
                        raise ValueError('Non-finite validation PCC')
                    rows.append({'fold': k, 'unit': j, 'environment': env, 'seed': seed,
                                 'pcc': value, 'rmse': rmse(data.Y[target, e], prediction[:, e]),
                                 'mae': mae(data.Y[target, e], prediction[:, e])})
                del model, result, prediction
        except Exception as exc:
            failures.append({'fold': k, 'reason': repr(exc)})
            print(f'FAILED {k}: {exc}', flush=True)
        finally:
            write(args.out / 'validation.csv',
                  ['fold', 'unit', 'environment', 'seed', 'pcc', 'rmse', 'mae'], rows)
            write(args.out / 'failures.csv', ['fold', 'reason'], failures)
            (args.out / 'training.json').write_text(json.dumps(curves, indent=2))
    expected = len(folds) * len(folds[0]['inner']) * data.Y.shape[1]
    if failures or len(rows) != expected:
        raise RuntimeError('Incomplete direct control run; inspect failures')
    per = {env: {metric: float(np.mean([row[metric] for row in rows if row['environment'] == env]))
                 for metric in ('pcc', 'rmse', 'mae')} for env in data.environments}
    summary = {'n_runs': len(rows), 'n_failures': 0, 'per_environment': per,
               'macro': {metric: float(np.mean([per[env][metric] for env in data.environments]))
                         for metric in ('pcc', 'rmse', 'mae')},
               'elapsed_seconds': time.perf_counter() - start}
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)
