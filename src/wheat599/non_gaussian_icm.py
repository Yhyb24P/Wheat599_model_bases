"""Non-Gaussian marker-effect posterior correction on the retained ICM base.

CV1 only. Every kernel, task covariance, standardization and network is fitted on
the current fitting subset; the correction strength gamma is chosen on the inner
folds of that subset. The gamma = 0 branch is the retained ICM path, so the
comparison inside one run uses identical base parameters.
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
from wheat599.posterior_network import PosteriorCorrector
from wheat599.posterior_operators import ConditionalLikelihood
from wheat599.posterior_synthetic import prepare_genotypes
from wheat599.posterior_training import train


def inference_subset(markers, phenotypes, fit, target, base, eta, device):
    """Build the operator and the ICM branch from fit rows only."""
    z, rbf, z_target, rbf_cross = prepare_genotypes(
        torch.as_tensor(markers[fit], dtype=torch.float64, device=device), base['bandwidth'],
        target=torch.as_tensor(markers[target], dtype=torch.float64, device=device))
    y = torch.as_tensor(phenotypes[fit], dtype=torch.float64, device=device)
    mean, scale = y.mean(0), y.std(0, unbiased=False)
    if torch.any(scale <= 0):
        raise ValueError('Constant phenotype in fitting subset')
    response = (y - mean) / scale
    identity = torch.eye(response.shape[1], dtype=torch.float64, device=device)
    task = (1 - eta) * identity + eta * torch.tensor(
        task_correlation(response.detach().cpu().numpy().astype(np.float64)),
        dtype=torch.float64, device=device)
    likelihood = ConditionalLikelihood(z, rbf, task, base['fraction'], base['ridge'][0])
    return {'likelihood': likelihood, 'response': response, 'mean': mean, 'scale': scale,
            'z_target': z_target, 'rbf_cross': rbf_cross, 'task': task}


def predict(prepared, model, gamma):
    """ICM prediction plus gamma times the learned posterior correction."""
    likelihood = prepared['likelihood']
    response = prepared['response']
    effects = likelihood.gaussian_posterior_mean(response)
    base = likelihood.predict_given_weights(response, effects, prepared['z_target'],
                                           prepared['rbf_cross'])
    if gamma == 0:
        return (base * prepared['scale'] + prepared['mean']).detach().cpu().numpy()
    correction = likelihood.prediction_correction(
        model(likelihood, likelihood.sufficient_score(response),
              likelihood.hessian_diagonal_blocks(batch_size=64), effects),
        prepared['z_target'], prepared['rbf_cross'])
    return ((base + gamma * correction) * prepared['scale'] + prepared['mean']).detach().cpu().numpy()


def trained_corrector(markers, cfg, device, log, seed):
    training = {**cfg['training'], 'seed': seed,
                'validation_seed': cfg['training']['validation_seed'] + seed}
    torch.manual_seed(seed)
    model = PosteriorCorrector(environments=cfg['training']['environments'], **cfg['network'])
    result = train(model, markers, training, device, mode='correction', log=log)
    model.load_state_dict(result['checkpoint']['state'])
    return model.eval(), result


def run(cfg, args, runner):
    data = load(Path(cfg['data']), cfg['data_sha256'])
    folds = json.loads(Path(cfg['splits']).read_text())['folds']
    split_hash = sha256_of(Path(cfg['splits']))
    base_source = Path(cfg['base_source'])
    base_records = json.loads((base_source / 'base_choices.json').read_text())
    comparator_path = Path(cfg['comparator_selection'])
    comparator = json.loads(comparator_path.read_text())
    if comparator['splits_sha256'] != split_hash:
        raise ValueError('Comparator selection splits mismatch')
    frozen = json.loads(args.selection.read_text()) if args.selection else None
    if frozen and (frozen['config'] != cfg or frozen['splits_sha256'] != split_hash
                   or len(frozen['selected']) != len(folds)
                   or {entry['fold'] for entry in frozen['selected']} != {fold['fold'] for fold in folds}):
        raise ValueError('Frozen selection mismatch')
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'provenance.json').write_text(json.dumps({
        'config': cfg, 'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'splits_sha256': split_hash, 'base_source_sha256': sha256_of(base_source / 'base_choices.json'),
        'comparator_selection_sha256': sha256_of(comparator_path),
        'model_sha256': sha256_of(Path(__file__)),
        'operator_sha256': sha256_of(Path(__file__).parent / 'posterior_operators.py'),
        'training_sha256': sha256_of(Path(__file__).parent / 'posterior_training.py'),
        'runner_sha256': sha256_of(runner),
        'selection_sha256': sha256_of(args.selection) if args.selection else None,
        'rule': 'inner-fold macro PCC across environments selects gamma; earliest grid value on tie',
        'protocol': 'CV1; gamma=0 branch is the retained ICM base; full development refit before one test pass'},
        indent=2))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(cfg['gpu_memory_fraction'])
    rows, oof, failures, selected, curves = [], [], [], [], []
    start = time.perf_counter()
    for fold in folds:
        k = fold['fold']
        print(f'fold {k}', flush=True)
        try:
            choice = next(entry for entry in comparator['selected'] if entry['fold'] == k)
            eta = choice['squared']['eta']
            if frozen is None:
                for j, inner in enumerate(fold['inner']):
                    fit, target = np.array(inner['train_idx']), np.array(inner['val_idx'])
                    if set(fit) & set(target) or set(fit) | set(target) != set(fold['dev_idx']):
                        raise ValueError('Invalid development split')
                    base = next(record['homogeneous'] for record in base_records
                                if record['fold'] == k and record['inner_fold'] == j)
                    unit_seed = cfg['training']['seed'] + 100 * k + j
                    model, result = trained_corrector(
                        data.X[fit], cfg, device,
                        lambda record, j=j: curves.append({'fold': k, 'unit': f'inner{j}', **record}),
                        unit_seed)
                    prepared = inference_subset(data.X, data.Y, fit, target, base, eta, device)
                    for gamma in cfg['gammas']:
                        prediction = predict(prepared, model, gamma)
                        for e, env in enumerate(data.environments):
                            score = pcc(data.Y[target, e], prediction[:, e])
                            if not np.isfinite(score):
                                raise ValueError('Non-finite validation PCC')
                            rows.append({'stage': 'inner', 'fold': k, 'unit': j, 'environment': env,
                                         'gamma': gamma, 'pcc': score,
                                         'rmse': rmse(data.Y[target, e], prediction[:, e]),
                                         'mae': mae(data.Y[target, e], prediction[:, e])})
                    curves.append({'fold': k, 'unit': f'inner{j}', 'seed': unit_seed,
                                   'cost': result['cost'],
                                   'checkpoint_step': result['checkpoint']['step']})
                    del model, prepared, result
                means = {gamma: float(np.mean([row['pcc'] for row in rows
                                               if row['stage'] == 'inner' and row['fold'] == k
                                               and row['gamma'] == gamma])) for gamma in cfg['gammas']}
                gamma = next(value for value in cfg['gammas'] if means[value] >= max(means.values()) - 1e-12)
                selected.append({'fold': k, 'eta': eta, 'gamma': gamma, 'inner_pcc': means[gamma],
                                 'inner_by_gamma': means,
                                 'icm_reference_pcc': means[0.0] if 0.0 in means else None})
                (args.out / 'selection.json').write_text(json.dumps(
                    {'config': cfg, 'splits_sha256': split_hash,
                     'base_source_sha256': sha256_of(base_source / 'base_choices.json'),
                     'selected': selected}, indent=2))
            else:
                entry = next(item for item in frozen['selected'] if item['fold'] == k)
                fit, target = np.array(fold['dev_idx']), np.array(fold['test_idx'])
                if set(fit) & set(target) or len(set(fit) | set(target)) != data.n:
                    raise ValueError('Invalid outer split')
                if entry['eta'] not in cfg['comparator_etas'] or entry['gamma'] not in cfg['gammas']:
                    raise ValueError('Invalid frozen gamma or eta')
                base = choice['base']
                if base.get('bandwidth') not in cfg['bandwidths'] \
                        or base.get('fraction') not in cfg['additive_fractions'] \
                        or base['ridge'][0] not in cfg['ridge']:
                    raise ValueError('Comparator base outside the configured grid')
                unit_seed = cfg['training']['seed'] + 100 * k + 99
                model, result = trained_corrector(
                    data.X[fit], cfg, device,
                    lambda record: curves.append({'fold': k, 'unit': 'development', **record}),
                    unit_seed)
                curves.append({'fold': k, 'unit': 'development', 'seed': unit_seed,
                               'cost': result['cost'],
                               'checkpoint_step': result['checkpoint']['step']})
                prepared = inference_subset(data.X, data.Y, fit, target, base, entry['eta'], device)
                prediction = predict(prepared, model, entry['gamma'])
                icm = predict(prepared, model, 0)
                for e, env in enumerate(data.environments):
                    metrics = {'pcc': pcc(data.Y[target, e], prediction[:, e]),
                               'rmse': rmse(data.Y[target, e], prediction[:, e]),
                               'mae': mae(data.Y[target, e], prediction[:, e])}
                    if not all(np.isfinite(value) for value in metrics.values()):
                        raise ValueError('Non-finite test metric')
                    rows.append({'stage': 'test', 'fold': k, 'unit': 'development', 'environment': env,
                                 'gamma': entry['gamma'], 'eta': entry['eta'],
                                 'bandwidth': base['bandwidth'], 'additive_fraction': base['fraction'],
                                 'ridge': base['ridge'][0],
                                 'icm_pcc': pcc(data.Y[target, e], icm[:, e]), **metrics})
                    oof.extend({'environment': env, 'fold': k, 'material_index': int(index),
                                'y_true': float(data.Y[index, e]), 'y_pred': float(value)}
                               for index, value in zip(target, prediction[:, e]))
                del model, prepared, prediction, icm
        except Exception as exc:
            failures.append({'fold': k, 'reason': repr(exc)})
            print(f'FAILED {k}: {exc}', flush=True)
        finally:
            fields = ['stage', 'fold', 'unit', 'environment', 'gamma', 'pcc', 'rmse', 'mae'] \
                if frozen is None else ['stage', 'fold', 'unit', 'environment', 'gamma', 'eta',
                                        'bandwidth', 'additive_fraction', 'ridge', 'pcc', 'rmse',
                                        'mae', 'icm_pcc']
            write(args.out / ('validation.csv' if frozen is None else 'per_fold.csv'), fields, rows)
            write(args.out / 'failures.csv', ['fold', 'reason'], failures)
            (args.out / 'training.json').write_text(json.dumps(curves, indent=2))
            if frozen:
                write(args.out / 'oof.csv', ['environment', 'fold', 'material_index', 'y_true', 'y_pred'], oof)
    expected = len(folds) * len(cfg['gammas']) * data.Y.shape[1] * len(folds[0]['inner']) \
        if frozen is None else len(folds) * data.Y.shape[1]
    if failures or len([row for row in rows if row['stage'] == ('inner' if frozen is None else 'test')]) != expected:
        raise RuntimeError('Incomplete non-Gaussian ICM run; inspect failures')
    if frozen is None:
        print('Development selection complete; no test predictions made.', flush=True)
        return
    per = {}
    for env in data.environments:
        values = [row for row in rows if row['stage'] == 'test' and row['environment'] == env]
        per[env] = {metric: float(np.mean([row[metric] for row in values]))
                    for metric in ('pcc', 'rmse', 'mae')}
        per[env]['icm_pcc'] = float(np.mean([row['icm_pcc'] for row in values]))
    summary = {'n_runs': len([row for row in rows if row['stage'] == 'test']), 'n_failures': 0,
               'per_environment': per,
               'macro': {metric: float(np.mean([per[env][metric] for env in data.environments]))
                         for metric in ('pcc', 'rmse', 'mae', 'icm_pcc')},
               'pooled_oof_pcc': {env: float(pcc(np.array([row['y_true'] for row in oof if row['environment'] == env]),
                                                 np.array([row['y_pred'] for row in oof if row['environment'] == env])))
                                  for env in data.environments},
               'elapsed_seconds': time.perf_counter() - start}
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)
