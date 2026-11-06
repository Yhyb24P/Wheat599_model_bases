"""Wheat599 development evaluation using the official full-matrix Laplace RFM."""
import itertools
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from wheat599.data import load, sha256_of
from wheat599.metrics import pcc, rmse, mae


def train(x_fit, y_fit, x_val, bandwidth_factor, ridge, iterations, batch_size, device):
    from rfm import LaplaceRFM
    x = torch.as_tensor(x_fit, dtype=torch.float32, device=device)
    xv = torch.as_tensor(x_val, dtype=torch.float32, device=device)
    y = torch.as_tensor(y_fit[:, None], dtype=torch.float32, device=device)
    x_mean, x_std = x.mean(0), x.std(0, unbiased=False)
    x_std = torch.where(x_std > 0, x_std, torch.ones_like(x_std))
    x, xv = (x-x_mean)/x_std, (xv-x_mean)/x_std
    y_mean, y_std = y.mean(0), y.std(0, unbiased=False)
    if float(y_std.item()) <= 0:
        raise ValueError('Constant fit phenotype')
    y = (y-y_mean)/y_std
    bandwidth = float(torch.median(torch.pdist(x)).item()) * bandwidth_factor
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError('Invalid fit bandwidth')
    model = LaplaceRFM(bandwidth=bandwidth, reg=ridge, diag=False, centering=False)
    model.device, model.fit_using_eigenpro, model.keep_device = device, False, True
    predictions = []
    with torch.no_grad():
        for i in range(iterations + 1):
            model.fit_predictor(x, y, verbose=False)
            pred = (model.predict(xv)*y_std+y_mean)[:, 0].cpu().numpy()
            if not np.isfinite(pred).all():
                raise ValueError('Nonfinite RFM prediction')
            predictions.append(pred)
            if i < iterations:
                model.fit_M(x, y, verbose=False, M_batch_size=batch_size)
                if not torch.isfinite(model.M).all():
                    raise ValueError('Nonfinite RFM feature matrix')
    return np.stack(predictions), bandwidth


def read(path):
    return json.loads(Path(path).read_text())


def run(cfg, args, runner):
    if args.selection is not None:
        raise ValueError('Development only')
    import rfm
    upstream = Path(rfm.__file__).resolve().parent.parent
    commit = subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != cfg['upstream_commit']:
        raise ValueError('Incorrect official RFM source version')
    torch.set_num_threads(cfg['cpu_threads'])
    device = torch.device(cfg['device'])
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(cfg['gpu_memory_fraction'])
    data = load(Path(cfg['data']), cfg['data_sha256'])
    folds = read(cfg['splits'])['folds']
    source = Path(cfg['reference_source'])
    if len(folds) != 10 or any(len(f['inner']) != 3 for f in folds):
        raise ValueError('Expected 10 folds and 3 inner splits')
    if read(source/'provenance.json')['sha256'][cfg['splits']] != sha256_of(Path(cfg['splits'])):
        raise ValueError('Reference split mismatch')
    choices = list(itertools.product(cfg['bandwidth_multipliers'], cfg['ridge']))
    args.out.mkdir(parents=True, exist_ok=False)
    def write(name, obj):
        (args.out/name).write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')
    sources = [Path(cfg['splits']), Path(__file__), Path(runner), Path(args.config),
               source/'provenance.json'] + [source/f"references_{f['fold']}_{j}.npz"
                                           for f in folds for j in range(3)]
    sources += sorted((upstream/'rfm').glob('*.py'))
    write('provenance.json', {'config': cfg, 'git_commit': subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], text=True).strip(), 'upstream_commit': commit,
        'sha256': {str(p): sha256_of(p) for p in sources}})
    expected = len(folds)*3*4*len(choices)
    state = dict(status='running', completed=0, expected=expected)
    records, failures, start = [], [], time.perf_counter()
    try:
        for fold in folds:
            f = fold['fold']
            for j, split in enumerate(fold['inner']):
                tr, val = np.array(split['train_idx']), np.array(split['val_idx'])
                if (set(tr) & set(val) or set(tr) | set(val) != set(fold['dev_idx'])
                        or (set(tr) | set(val)) & set(fold['test_idx'])):
                    raise ValueError('Invalid split')
                predictions = {}
                for e in range(4):
                    for c, (factor, ridge) in enumerate(choices):
                        started = time.perf_counter()
                        pred, bandwidth = train(data.X[tr], data.Y[tr, e], data.X[val],
                            factor, ridge, cfg['iterations'], cfg['agop_batch_size'], device)
                        predictions[f'e{e}_c{c}'] = pred
                        records.append(dict(fold=f, inner=j, environment=e, config_index=c,
                                            bandwidth=bandwidth, seconds=time.perf_counter()-started))
                        state['completed'] += 1
                        write('execution_status.json', state)
                np.savez_compressed(args.out/f'predictions_{f}_{j}.npz', **predictions)
                write('trajectories.json', records)
        if state['completed'] != expected:
            raise ValueError('Incomplete experiment')
        results = []
        for fold in folds:
            f = fold['fold']
            arrays = [np.load(args.out/f'predictions_{f}_{j}.npz') for j in range(3)]
            refs = [np.load(source/f'references_{f}_{j}.npz') for j in range(3)]
            chosen = []
            for e in range(4):
                scores = np.array([[np.mean([pcc(data.Y[split['val_idx'], e], arrays[j][f'e{e}_c{c}'][i])
                    for j, split in enumerate(fold['inner'])]) for i in range(cfg['iterations']+1)]
                    for c in range(len(choices))])
                if not np.isfinite(scores).all():
                    raise ValueError('Nonfinite RFM selection score')
                c, i = np.unravel_index(np.argmax(scores), scores.shape)
                chosen.append({'config_index': int(c), 'iteration': int(i)})
            per_method = {k: [] for k in ('rfm','huber','independent_tabm','fixed_fusion')}
            for j, split in enumerate(fold['inner']):
                truth = data.Y[split['val_idx']]
                pred_rfm = np.column_stack([arrays[j][f"e{e}_c{v['config_index']}"][v['iteration']]
                                            for e, v in enumerate(chosen)])
                for method in per_method:
                    pred = pred_rfm if method == 'rfm' else refs[j][method]
                    values = [[fun(truth[:, e], pred[:, e]) for fun in (pcc, rmse, mae)] for e in range(4)]
                    if not np.isfinite(values).all():
                        raise ValueError('Nonfinite metrics')
                    per_method[method].append(values)
            results.append(dict(fold=f, selected=chosen,
                                metrics={k: np.mean(v, axis=0).tolist() for k, v in per_method.items()}))
        summary = {method: dict(zip(('pcc','rmse','mae'),
            np.mean([r['metrics'][method] for r in results], axis=(0, 1)).tolist()))
            for method in ('rfm','huber','independent_tabm','fixed_fusion')}
        write('summary.json', {'macro': summary, 'folds': results, 'configurations': choices})
        state['status'] = 'complete'
    except BaseException as exc:
        state['status'] = 'failed'
        failures.append({'completed': state['completed'], 'error': repr(exc)})
        raise
    finally:
        write('execution_status.json', state)
        write('trajectories.json', records)
        write('failures.json', failures)
        write('timing.json', {'seconds': time.perf_counter()-start})
