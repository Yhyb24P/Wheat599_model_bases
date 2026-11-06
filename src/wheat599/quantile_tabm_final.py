"""Frozen Wheat599 outer refits and fixed quantile–Huber fusion."""
import csv
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from wheat599.data import load, sha256_of
from wheat599.metrics import pcc, rmse, mae
from wheat599.quantile_tabm import build_model, update_epoch


def read(path):
    return json.loads(Path(path).read_text())


def validate_manifest(cfg, manifest, folds, settings, configs):
    if cfg['fusion_weight'] != .5 or manifest['data_sha256'] != cfg['data_sha256']:
        raise ValueError('Frozen fusion or data changed')
    for name, digest in manifest['sha256'].items():
        if sha256_of(Path(name)) != digest:
            raise ValueError(f'Frozen source changed: {name}')
    source = Path('results/quantile_tabm_development_20261106_v2')
    status = read(source/'execution_status.json')
    if status != {'status': 'complete', 'completed': 1920, 'expected': 1920} or read(source/'failures.json'):
        raise ValueError('Development incomplete')
    selections = {r['fold']: r['selected'] for r in read(source/'summary.json')['folds']}
    records = read(source/'trajectories.json')
    index = {(r['fold'], r['inner'], r['environment'], r['config_index'], r['seed']): r for r in records}
    if len(records) != 1920 or len(index) != 1920 or len(manifest['rows']) != cfg['expected_fits']:
        raise ValueError('Incomplete trajectories or manifest')
    seen = set()
    by_fold = {f['fold']: f for f in folds}
    for row in manifest['rows']:
        fold, e, seed, c = (row[k] for k in ('fold','environment_index','seed','config_index'))
        key = (fold, e, seed)
        if key in seen or seed not in settings['seeds'] or not 0 <= e < 4:
            raise ValueError('Duplicate or invalid fit')
        seen.add(key)
        f = by_fold[fold]
        if row['fit_ids'] != f['dev_idx'] or row['test_ids'] != f['test_idx'] or set(row['fit_ids']) & set(row['test_ids']):
            raise ValueError('Manifest split mismatch')
        if row['environment'] != ['1','2','4','5'][e] or c != selections[fold][e] or not 0 <= c < len(configs):
            raise ValueError('Manifest selected model mismatch')
        source_rows = [index[(fold, j, e, c, seed)] for j in range(3)]
        epochs = [r['best_epoch'] for r in source_rows]
        if row['inner_best_epochs'] != epochs or row['epochs'] != sorted(epochs)[1] or any(r['epochs_run'] >= 500 for r in source_rows):
            raise ValueError('Manifest epoch mismatch')
    expected = {(f['fold'], e, seed) for f in folds for e in range(4) for seed in settings['seeds']}
    if seen != expected or len(folds) != 10 or settings['n_quantiles'] != 99:
        raise ValueError('Invalid outer coverage or quantiles')


def fit_one(x_fit, y_fit, x_test, row, config, settings, device, weights):
    start = time.perf_counter()
    model = build_model(x_fit, y_fit, config, row['seed'], settings, device)
    x = torch.as_tensor(x_fit, dtype=torch.float32, device=device)
    xt = torch.as_tensor(x_test, dtype=torch.float32, device=device)
    y = model.standardize_y(torch.as_tensor(y_fit[:, None], dtype=torch.float32, device=device))
    q = torch.linspace(settings['quantile_min'], settings['quantile_max'], settings['n_quantiles'], device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    generator = torch.Generator().manual_seed(int(row['seed'])*1000 + row['config_index'])
    for _ in range(row['epochs']):
        update_epoch(model, x, y, q, opt, generator, settings['training'], device)
    model.eval()
    with torch.no_grad():
        prediction = model.inverse_y(model(xt).mean(dim=(1,2))[:, None])[:, 0].cpu().numpy()
    if not np.isfinite(prediction).all():
        raise ValueError('Nonfinite outer prediction')
    path = weights / f"f{row['fold']}_e{row['environment_index']}_s{row['seed']}.pt"
    torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()}, path)
    return prediction, {'fold': row['fold'], 'environment_index': row['environment_index'],
        'seed': row['seed'], 'config_index': row['config_index'], 'epochs': row['epochs'],
        'seconds': time.perf_counter()-start, 'weight_path': str(path), 'weight_sha256': sha256_of(path)}


def read_oof(path):
    rows = {}
    with Path(path).open() as handle:
        for row in csv.DictReader(handle):
            key = (int(row['fold']), str(row['environment']), int(row['material_index']))
            if key in rows:
                raise ValueError('Duplicate baseline prediction')
            rows[key] = (float(row['y_true']), float(row['y_pred']))
    return rows


def summarize(data, folds, seed_predictions, seeds, huber, mse):
    methods = ('quantile', 'mse_tabm', 'huber', 'quantile_huber')
    per_fold, oof = [], []
    used = set()
    pooled = {m: {str(e): [[], []] for e in data.environments} for m in methods}
    for fold in folds:
        f, ids = fold['fold'], fold['test_idx']
        for e, name in enumerate(data.environments):
            keys = [(f, str(name), int(i)) for i in ids]
            if used & set(keys):
                raise ValueError('Duplicate outer material')
            used.update(keys)
            truth = data.Y[ids, e]
            for bank in (huber, mse):
                if any(k not in bank for k in keys) or not np.allclose([bank[k][0] for k in keys], truth, atol=1e-12, rtol=0):
                    raise ValueError('Baseline material or truth mismatch')
            qp = np.mean([seed_predictions[f'f{f}_e{e}_s{seed}'] for seed in seeds], axis=0)
            hp = np.array([huber[k][1] for k in keys]); mp = np.array([mse[k][1] for k in keys])
            predictions = {'quantile': qp, 'mse_tabm': mp, 'huber': hp, 'quantile_huber': .5*qp+.5*hp}
            metrics = {}
            for method, pred in predictions.items():
                if len(pred) != len(truth) or not np.isfinite(pred).all():
                    raise ValueError('Invalid prediction')
                values = [fun(truth, pred) for fun in (pcc, rmse, mae)]
                if not np.isfinite(values).all():
                    raise ValueError('Nonfinite outer metrics')
                metrics[method] = dict(zip(('pcc','rmse','mae'), values))
                pooled[method][str(name)][0].extend(truth.tolist())
                pooled[method][str(name)][1].extend(pred.tolist())
            per_fold.append(dict(fold=f, environment=str(name), metrics=metrics))
            for i, material in enumerate(ids):
                oof.append(dict(fold=f, environment=str(name), material_index=int(material),
                                y_true=float(truth[i]), **{k: float(v[i]) for k,v in predictions.items()}))
    if len(used) != 599*4 or set(huber) != used or set(mse) != used:
        raise ValueError('Outer coverage mismatch')
    macro = {m: {metric: float(np.mean([r['metrics'][m][metric] for r in per_fold]))
                 for metric in ('pcc','rmse','mae')} for m in methods}
    by_environment = {m: {str(e): {metric: float(np.mean([r['metrics'][m][metric]
        for r in per_fold if r['environment'] == str(e)])) for metric in ('pcc','rmse','mae')}
        for e in data.environments} for m in methods}
    pooled_metrics = {m: {str(e): dict(zip(('pcc','rmse','mae'),
        [fun(*pooled[m][str(e)]) for fun in (pcc,rmse,mae)])) for e in data.environments} for m in methods}
    return {'macro': macro, 'per_fold': per_fold, 'per_environment': by_environment,
            'pooled_by_environment': pooled_metrics}, oof


def run(cfg, args, runner):
    if args.selection is not None:
        raise ValueError('Frozen manifest only')
    path = Path(cfg['manifest'])
    if sha256_of(path) != cfg['manifest_sha256']:
        raise ValueError('Frozen manifest hash mismatch')
    settings = yaml.safe_load(Path(cfg['development_config']).read_text())
    configs = yaml.safe_load(Path(cfg['architecture_config']).read_text())['configurations']
    folds = read(cfg['splits'])['folds']; manifest = read(path)
    validate_manifest(cfg, manifest, folds, settings, configs)
    torch.set_num_threads(cfg['cpu_threads'])
    device = torch.device(cfg['device'])
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(cfg['gpu_memory_fraction'])
    data = load(Path(cfg['data']), cfg['data_sha256'])
    args.out.mkdir(parents=True, exist_ok=False)
    weights = args.out/'weights'; weights.mkdir()
    def write(name, obj):
        (args.out/name).write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')
    write('provenance.json', {'config': cfg, 'manifest_sha256': sha256_of(path),
        'git_commit': subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
        'runner_sha256': sha256_of(Path(runner)), 'implementation_sha256': sha256_of(Path(__file__))})
    state = {'status':'running', 'completed':0, 'expected':cfg['expected_fits']}
    predictions, records, failures, start = {}, [], [], time.perf_counter()
    try:
        for row in manifest['rows']:
            e = row['environment_index']; tr, te = row['fit_ids'], row['test_ids']
            pred, info = fit_one(data.X[tr], data.Y[tr, e], data.X[te], row,
                                  configs[row['config_index']], settings, device, weights)
            predictions[f"f{row['fold']}_e{e}_s{row['seed']}"] = pred
            records.append(info); state['completed'] += 1
            write('execution_status.json', state)
            write('fits.json', records)
        if state['completed'] != cfg['expected_fits']:
            raise ValueError('Incomplete outer fits')
        np.savez_compressed(args.out/'seed_predictions.npz', **predictions)
        summary, oof = summarize(data, folds, predictions, settings['seeds'],
                                 read_oof(cfg['huber_oof']), read_oof(cfg['mse_oof']))
        with (args.out/'oof.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(oof[0]));writer.writeheader();writer.writerows(oof)
        write('summary.json', summary)
        state['status'] = 'complete'
    except BaseException as exc:
        state['status'] = 'failed'
        failures.append({'completed':state['completed'], 'error':repr(exc)})
        raise
    finally:
        write('execution_status.json', state);write('fits.json', records);write('failures.json', failures)
        write('timing.json', {'seconds':time.perf_counter()-start})
