"""Fixed-parameter, development-only comparison of grouped Huber kernels."""
import csv
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from wheat599.data import load, sha256_of
from wheat599.grouped_kernel import grouped_rbf_kernels, stratified_random_assignments
from wheat599.kernel_baselines import kernels
from wheat599.robust_icm import features, fit_predict
from wheat599.metrics import pcc, rmse, mae


def read(path):
    return json.loads(Path(path).read_text())


def summarize(rows, seeds):
    arms = ['original', 'correlation'] + [f'random_{s}' for s in seeds]
    indexed = {(r['fold'], r['inner'], r['arm']): r for r in rows}
    expected = {(f, j, a) for f in range(10) for j in range(3) for a in arms}
    if len(indexed) != len(rows) or set(indexed) != expected:
        raise ValueError('Incomplete or duplicate comparison grid')
    scores = {}
    for arm in arms:
        values = np.array([[[indexed[f, j, arm]['per_environment'][e][k]
                             for k in ('pcc', 'rmse', 'mae')] for e in range(4)]
                           for f in range(10) for j in range(3)])
        scores[arm] = values.reshape(10, 3, 4, 3).mean(axis=1)
    scores['random_mean'] = np.mean([scores[a] for a in arms[2:]], axis=0)
    if not all(np.isfinite(v).all() for v in scores.values()):
        raise ValueError('Nonfinite summary')
    result = {'completed_fits': len(rows), 'metrics': {}, 'comparisons': {}}
    for arm, values in scores.items():
        result['metrics'][arm] = {
            'macro': dict(zip(('pcc', 'rmse', 'mae'), values.mean(axis=(0, 1)).tolist())),
            'per_environment': values.mean(axis=0).tolist(),
            'fold_macro_pcc': values[:, :, 0].mean(axis=1).tolist()}
    for reference in ('original', 'random_mean'):
        delta = (scores['correlation'][:, :, 0] - scores[reference][:, :, 0]).mean(axis=1)
        result['comparisons'][reference] = {'pcc_gain': float(delta.mean()),
                                           'fold_wins': int((delta > 0).sum()),
                                           'fold_deltas': delta.tolist()}
    result['stage2_eligible'] = all(c['pcc_gain'] >= .001 and c['fold_wins'] >= 7
                                    for c in result['comparisons'].values())
    result['metric_order'] = ['pcc', 'rmse', 'mae']
    result['random_rule'] = 'Mean metrics of six separately fitted random kernels'
    return result


def run(cfg, args, runner):
    if args.selection is not None:
        raise ValueError('This stage permits development only')
    torch.set_num_threads(cfg['cpu_threads'])
    data = load(Path(cfg['data']), cfg['data_sha256'])
    folds = read(cfg['splits'])['folds']
    source = Path(cfg['huber_source'])
    selection = read(source / 'selection.json')
    solver = selection['config']
    provenance = read(source / 'provenance.json')
    if (provenance['splits_sha256'] != sha256_of(Path(cfg['splits'])) or
            solver['data_sha256'] != cfg['data_sha256']):
        raise ValueError('Historical data/splits mismatch')
    if (cfg['tau'] != .25 or cfg['groups'] != 64 or cfg['frequency_bins'] != 10 or
            cfg['random_seeds'] != [101, 202, 303, 404, 505, 606] or
            len(folds) != 10 or any(len(f['inner']) != 3 for f in folds)):
        raise ValueError('Configuration differs from the fixed 240-fit protocol')
    bases = read(Path(solver['base_source']) / 'base_choices.json')
    diagnostics = read(source / 'optimization.json')
    with (source / 'validation.csv').open() as handle:
        recorded = list(csv.DictReader(handle))
    args.out.mkdir(parents=True, exist_ok=False)

    def write(name, value):
        (args.out / name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')

    paths = [Path(__file__), Path(runner), Path(args.config), Path(cfg['splits']),
             source/'selection.json', source/'optimization.json', source/'validation.csv',
             Path(solver['base_source'])/'base_choices.json',
             Path(__file__).with_name('grouped_kernel.py'),
             Path(__file__).with_name('robust_icm.py')]
    write('provenance.json', {'config': cfg, 'sha256': {str(p): sha256_of(p) for p in paths},
                             'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                             'environment_order': [str(e) for e in data.environments],
                             'scope': 'inner fit/validation only; original Huber thresholds fixed'})
    state = {'status': 'running', 'completed_fits': 0, 'expected_fits': 240}
    rows, metadata, failures = [], [], []
    started = time.perf_counter()
    write('execution_status.json', state)
    try:
        for fold in folds:
            f = fold['fold']
            choice = next(s['huber'] for s in selection['selected'] if s['fold'] == f)
            for j, inner in enumerate(fold['inner']):
                fit, val = np.array(inner['train_idx']), np.array(inner['val_idx'])
                if set(fit) & set(val) or set(fit) | set(val) != set(fold['dev_idx']):
                    raise ValueError('Invalid inner partition')
                if (set(fit) | set(val)) & set(fold['test_idx']):
                    raise ValueError('Outer test rows in development')
                base = next(b['homogeneous'] for b in bases if b['fold'] == f and b['inner_fold'] == j)
                def matches(r):
                    return (int(r['fold']) == f and int(r['inner_fold']) == j and
                            r['model'] == 'huber' and all(float(r[k]) == choice[k]
                            for k in ('eta', 'delta', 'multiplier')))
                old = [r for r in diagnostics if matches(r)]
                if len(old) != 1:
                    raise ValueError('Missing or duplicate selected Huber diagnostics')
                thresholds = old[0]['thresholds']
                a, ac, r, rc = kernels(data.X, fit, val, base['bandwidth'], 'cpu')
                grouped = grouped_rbf_kernels(data.X, fit, val, bandwidth=base['bandwidth'],
                                             tau=cfg['tau'], groups=cfg['groups'])
                assignment = grouped.metadata['assignment']
                random = stratified_random_assignments(
                    assignment, data.X[fit].mean(0), cfg['random_seeds'],
                    bins=cfg['frequency_bins'], constant_mask=data.X[fit].std(0) == 0)
                arms = [('original', r, rc, None), ('correlation', grouped.train,
                         grouped.cross, grouped.metadata)]
                for seed, labels in random.items():
                    g = grouped_rbf_kernels(data.X, fit, val, bandwidth=base['bandwidth'],
                                           tau=cfg['tau'], groups=cfg['groups'], assignment=labels)
                    arms.append((f'random_{seed}', g.train, g.cross, g.metadata))
                for arm, kr, kc, group_info in arms:
                    fraction = base['fraction']
                    gram = fraction*a + (1-fraction)*torch.as_tensor(kr)
                    cross = fraction*ac + (1-fraction)*torch.as_tensor(kc)
                    prepared = features(data.X, data.Y, fit, val, base, material_kernels=(gram, cross))
                    pred, diag = fit_predict(prepared, base['ridge'][0]*choice['multiplier'],
                                            choice['eta'], choice['delta'], solver,
                                            fixed_thresholds=thresholds)
                    if arm == 'original' and not np.allclose(
                            np.array(diag['residual_mad_scale'])*choice['delta'], thresholds,
                            atol=1e-10, rtol=1e-10):
                        raise ValueError('Original Huber threshold reconstruction mismatch')
                    metrics = []
                    for e, env in enumerate(data.environments):
                        values = dict(pcc=pcc(data.Y[val, e], pred[:, e]),
                                      rmse=rmse(data.Y[val, e], pred[:, e]),
                                      mae=mae(data.Y[val, e], pred[:, e]))
                        if not np.isfinite(list(values.values())).all():
                            raise ValueError('Nonfinite metric')
                        if arm == 'original':
                            ref = [v for v in recorded if matches(v) and str(v['environment']) == str(env)]
                            if len(ref) != 1 or max(abs(values[k]-float(ref[0][k])) for k in values) > 1e-7:
                                raise ValueError('Original Huber reconstruction differs from saved validation')
                        metrics.append(values)
                    rows.append({'fold': f, 'inner': j, 'arm': arm, 'per_environment': metrics,
                                 'macro': {k: float(np.mean([m[k] for m in metrics])) for k in metrics[0]}})
                    np.savez_compressed(args.out/f'f{f}_i{j}_{arm}.npz', ids=val, prediction=pred)
                    metadata.append({'fold': f, 'inner': j, 'arm': arm, 'groups': group_info,
                                     'solver': diag})
                    state['completed_fits'] += 1
                    write('execution_status.json', state)
                write('units.json', rows)
        if state['completed_fits'] != state['expected_fits']:
            raise ValueError('Incomplete fit grid')
        write('summary.json', summarize(rows, cfg['random_seeds']))
        state['status'] = 'complete'
    except BaseException as error:
        state['status'] = 'failed'
        failures.append({'reason': repr(error), 'completed_fits': state['completed_fits']})
        raise
    finally:
        write('units.json', rows)
        write('metadata.json', metadata)
        write('failures.json', failures)
        write('execution_status.json', state)
        write('timing.json', {'seconds': time.perf_counter()-started})
