"""Fixed-epoch outer evaluation for the shape-quantile comparisons."""
import csv
import json
import multiprocessing
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml

from genopred.data.wheat599 import load, sha256_of
from genopred.models.tabular.quantile_tabm import _init_fit_worker, mixture_scores
from genopred.models.tabular.quantile_tabm_final import summarize


def _fit_outer(task):
    from genopred.models.tabular import quantile_tabm
    fold, environment, config, seed, epochs = task
    x, y, folds, configs, settings, device = quantile_tabm._FIT_CONTEXT
    split = folds[fold]
    tr, te = split['dev_idx'], split['test_idx']
    prediction, info = quantile_tabm.train(x[tr], y[tr, environment], x[te], None,
        configs[config], seed, config, {**settings, 'fixed_epochs': epochs}, device)
    return task, prediction, info


def manifest(folds, seeds, configs, summary, records, max_epochs):
    index = {(r['fold'], r['inner'], r['environment'], r['config_index'], r['seed']): r for r in records}
    expected = {(f['fold'], j, e, c, s) for f in folds for j in range(3)
                for e in range(4) for c in range(len(configs)) for s in seeds}
    if len(index) != len(records) or set(index) != expected:
        raise ValueError('Incomplete development trajectories')
    rows = []
    for fold in folds:
        f = fold['fold']
        choice = next(row['selected'] for row in summary['folds'] if row['fold'] == f)
        for e, c in enumerate(choice):
            if not 0 <= c < len(configs):
                raise ValueError('Invalid selected configuration')
            for seed in seeds:
                epochs = [index[(f, j, e, c, seed)]['best_epoch'] for j in range(3)]
                if any(not isinstance(n, int) or not 0 <= n <= max_epochs for n in epochs):
                    raise ValueError('Invalid selected epoch')
                rows.append((f, e, c, seed, sorted(epochs)[1]))
    if len(rows) != 10 * 4 * len(seeds):
        raise ValueError('Incomplete selected configurations')
    return rows


def reference(path):
    original, huber, mse = {}, {}, {}
    with path.open(newline='') as handle:
        for row in csv.DictReader(handle):
            key = (int(row['fold']), row['environment'], int(row['material_index']))
            if key in original:
                raise ValueError('Duplicate reference material')
            y = float(row['y_true'])
            original[key] = (y, float(row['quantile']), float(row['quantile_huber']))
            huber[key] = (y, float(row['huber']))
            mse[key] = (y, float(row['mse_tabm']))
    return original, huber, mse


def run(cfg, args, runner):
    source = args.selection.parent
    read = lambda path: json.loads(Path(path).read_text())
    configs = yaml.safe_load(Path(cfg['architecture_config']).read_text())['configurations']
    folds = read(cfg['splits'])['folds']
    if len(folds) != 10 or sorted(i for f in folds for i in f['test_idx']) != list(range(599)):
        raise ValueError('Invalid outer material coverage')
    for f in folds:
        dev, test = set(f['dev_idx']), set(f['test_idx'])
        if dev & test or dev | test != set(range(599)) or len(f['inner']) != 3:
            raise ValueError('Invalid outer split')
        if any(set(s['train_idx']) & set(s['val_idx']) or set(s['train_idx']) | set(s['val_idx']) != dev for s in f['inner']):
            raise ValueError('Invalid inner split')
    expected = len(folds) * 3 * 4 * len(configs) * len(cfg['seeds'])
    if read(source/'execution_status.json') != {'status': 'complete', 'completed': expected, 'expected': expected} or read(source/'failures.json'):
        raise ValueError('Incomplete development experiment')
    provenance = read(source/'provenance.json')
    paths = [Path(args.config), Path(cfg['architecture_config']), Path(cfg['splits']), Path('src/genopred/models/tabular/quantile_tabm.py')]
    if cfg['optimizer_variant'] != 'shape_midpoint':
        paths.append(Path('src/genopred/models/tabular/shape_quantile.py'))
    def matches(p):
        hashes = [v for k, v in provenance['sha256'].items() if k == str(p) or k.endswith('/' + str(p))]
        return len(hashes) == 1 and hashes[0] == sha256_of(p)
    if provenance['config'] != cfg or any(not matches(p) for p in paths):
        raise ValueError('Development configuration or source changed')
    selections, records = read(args.selection), read(source/'trajectories.json')
    tasks = manifest(folds, cfg['seeds'], configs, selections, records, cfg['training']['max_epochs'])
    data = load(Path(cfg['data']), cfg['data_sha256'])
    reference_path = Path('results/wheat599/tabm/quantile_tabm_final_20261106/oof.csv')
    args.out.mkdir(parents=True, exist_ok=False)
    def write(name, value):
        (args.out/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    write('manifest.json', {'rule': 'development-selected configuration; per-seed median inner epoch', 'tasks': tasks})
    paths += [Path(runner), Path(__file__), args.selection, source/'trajectories.json', reference_path]
    write('provenance.json', {'config': cfg, 'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                              'sha256': {str(p): sha256_of(p) for p in paths}})
    state, records, predictions, distributions, failures = {'status':'running','completed':0,'expected':len(tasks)}, [], {}, {}, []
    write('execution_status.json', state)
    start = time.perf_counter()
    try:
        with ProcessPoolExecutor(max_workers=cfg['workers'], mp_context=multiprocessing.get_context('spawn'),
                initializer=_init_fit_worker, initargs=(data.X, data.Y, folds, configs, cfg, cfg['workers'])) as pool:
            for task, pred, info in pool.map(_fit_outer, tasks, chunksize=1):
                f, e, c, seed, epochs = task
                dist = info.pop('_distribution')
                if not np.isfinite(pred).all() or not np.isfinite(dist).all():
                    raise ValueError('Nonfinite outer prediction')
                predictions[f'f{f}_e{e}_s{seed}'] = pred
                distributions.setdefault((f, e), []).append(dist)
                records.append({'fold':f, 'environment':e, 'config_index':c, 'seed':seed, 'epochs':epochs, **info})
                state['completed'] += 1
                write('execution_status.json', state)
        if state['completed'] != state['expected']:
            raise ValueError('Incomplete outer fits')
        original, huber, mse = reference(reference_path)
        summary, oof = summarize(data, folds, predictions, cfg['seeds'], huber, mse, original, candidate_fusion=False)
        probabilities, per_fold = {}, []
        for fold in folds:
            for e, name in enumerate(data.environments):
                f, ids = fold['fold'], fold['test_idx']
                parts = distributions[(f, e)]
                if len(parts) != len(cfg['seeds']):
                    raise ValueError('Incomplete predictive mixture')
                scores = mixture_scores(parts, data.Y[ids, e])
                atoms = np.concatenate(parts, axis=1).reshape(len(ids), -1)
                bounds = np.quantile(atoms, [.05, .95], axis=1, method='inverted_cdf').T
                per_fold.append({'fold': f, 'environment': name, 'values': scores.mean(axis=0).tolist()})
                probabilities.update({(f, name, int(i)): np.r_[scores[j], bounds[j]] for j, i in enumerate(ids)})
        values = np.array([r['values'] for r in per_fold])
        summary['probabilistic'] = {'metrics':['crps','coverage90','width90'], 'macro':values.mean(axis=0).tolist(),
            'per_environment':{name:np.array([r['values'] for r in per_fold if r['environment']==name]).mean(axis=0).tolist() for name in data.environments},
            'per_fold':per_fold}
        for row in oof:
            row.update(dict(zip(('crps','coverage90','width90','lower90','upper90'), probabilities[(row['fold'], row['environment'], row['material_index'])].tolist())))
        with (args.out/'oof.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(oof[0])); writer.writeheader(); writer.writerows(oof)
        np.savez_compressed(args.out/'seed_predictions.npz', **predictions)
        write('summary.json', summary)
        state['status'] = 'complete'
    except BaseException as error:
        state['status'] = 'failed'; failures.append({'completed':state['completed'], 'error':repr(error)})
        raise
    finally:
        write('execution_status.json', state); write('fits.json', records); write('failures.json', failures)
        write('timing.json', {'seconds':time.perf_counter()-start})
