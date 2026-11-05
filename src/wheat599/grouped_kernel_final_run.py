"""Frozen-parameter, outer evaluation of the fixed grouped Huber kernel."""
import csv
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from wheat599.data import load, sha256_of
from wheat599.get_icm import write as write_csv
from wheat599.grouped_kernel import grouped_rbf_kernels
from wheat599.kernel_baselines import kernels
from wheat599.metrics import mae, pcc, rmse
from wheat599.robust_icm import features, fit_predict


def read(path):
    return json.loads(Path(path).read_text())


def summarize(rows, expected_original_macro):
    arms = ('original', 'grouped')
    indexed = {(r['fold'], r['arm']): r for r in rows}
    expected = {(f, a) for f in range(10) for a in arms}
    if len(indexed) != len(rows) or set(indexed) != expected:
        raise ValueError('Incomplete or duplicate outer grid')
    scores = {}
    for arm in arms:
        values = np.array([[[indexed[f, arm]['per_environment'][e][k]
                             for k in ('pcc', 'rmse', 'mae')] for e in range(4)]
                           for f in range(10)])
        scores[arm] = values.reshape(10, 4, 3).mean(axis=1)
    result = {'completed_fits': len(rows), 'metrics': {}}
    for arm, values in scores.items():
        result['metrics'][arm] = {
            'macro': dict(zip(('pcc', 'rmse', 'mae'), values.mean(axis=0).tolist())),
            'per_environment': values.mean(axis=0).tolist(),
            'fold_macro_pcc': values[:, :, 0].mean(axis=1).tolist()}
    delta = scores['grouped'][:, :, 0] - scores['original'][:, :, 0]
    result['comparison'] = {'pcc_gain': float(delta.mean()),
                            'fold_wins': int((delta > 0).sum()),
                            'fold_deltas': delta.tolist()}
    if not all(np.isfinite(v).all() for v in scores.values()):
        raise ValueError('Nonfinite summary')
    result['metric_order'] = ['pcc', 'rmse', 'mae']
    result['original_macro_pcc'] = result['metrics']['original']['macro']['pcc']
    result['saved_final_macro_pcc'] = expected_original_macro
    result['scope'] = 'outer test evaluated once; frozen development selections and final Huber thresholds'
    return result


def run(cfg, args, runner):
    if args.selection is not None:
        raise ValueError('This stage is a frozen outer evaluation')
    if cfg['tau'] != .25 or cfg['groups'] != 64:
        raise ValueError('Configuration differs from the fixed grouped protocol')
    torch.set_num_threads(cfg['cpu_threads'])
    data = load(Path(cfg['data']), cfg['data_sha256'])
    folds = read(cfg['splits'])['folds']
    if len(folds) != 10:
        raise ValueError('Expected the fixed ten outer folds')
    dev = Path(cfg['huber_source'])
    selection = read(dev / 'selection.json')
    solver = selection['config']
    dev_provenance = read(dev / 'provenance.json')
    if (dev_provenance['splits_sha256'] != sha256_of(Path(cfg['splits'])) or
            solver['data_sha256'] != cfg['data_sha256']):
        raise ValueError('Historical development data/splits mismatch')
    final = Path(cfg['final_source'])
    final_provenance = read(final / 'provenance.json')
    if (final_provenance['config'] != solver or
            final_provenance['splits_sha256'] != sha256_of(Path(cfg['splits'])) or
            final_provenance['base_source_sha256'] != sha256_of(Path(solver['base_source']) / 'base_choices.json') or
            final_provenance['selection_sha256'] != sha256_of(dev / 'selection.json')):
        raise ValueError('Final source was not produced from this frozen development selection')
    records = read(final / 'optimization.json')
    thresholds = {e['fold']: e['thresholds'] for e in records}
    if (len(records) != 10 or set(thresholds) != {f['fold'] for f in folds} or
            not all(e['optimizer_success'] for e in records) or
            not all(np.all(np.asarray(v) > 0) for v in thresholds.values())):
        raise ValueError('Final Huber threshold records are incomplete')
    saved_oof = {}
    with (final / 'oof.csv').open() as handle:
        for row in csv.DictReader(handle):
            saved_oof[(row['environment'], int(row['fold']), int(row['material_index']))] = \
                (float(row['y_true']), float(row['y_pred']))
    expected_original_macro = read(final / 'summary.json')['macro']['pcc']
    args.out.mkdir(parents=True, exist_ok=False)

    def write(name, value):
        (args.out / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

    paths = [Path(__file__), Path(runner), Path(args.config), Path(cfg['splits']),
             dev / 'selection.json', dev / 'provenance.json',
             Path(solver['base_source']) / 'base_choices.json',
             final / 'provenance.json', final / 'optimization.json', final / 'oof.csv',
             final / 'summary.json', Path(__file__).with_name('grouped_kernel.py'),
             Path(__file__).with_name('robust_icm.py')]
    write('provenance.json', {
        'config': cfg, 'sha256': {str(p): sha256_of(p) for p in paths},
        'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'environment_order': [str(e) for e in data.environments],
        'scope': 'outer test once; frozen development selections; original Huber thresholds reused for both arms'})
    state = {'status': 'running', 'completed_fits': 0, 'expected_fits': 20}
    rows, oof, failures = [], [], []
    oof_max = 0.
    started = time.perf_counter()
    write('execution_status.json', state)
    try:
        for fold in folds:
            f = fold['fold']
            entry = next(s for s in selection['selected'] if s['fold'] == f)
            choice, base = entry['huber'], entry['base']
            fit, test = np.array(fold['dev_idx']), np.array(fold['test_idx'])
            if set(fit) & set(test) or len(set(fit) | set(test)) != data.n:
                raise ValueError('Invalid outer split')
            fixed = np.asarray(thresholds[f])
            a, ac, r, rc = kernels(data.X, fit, test, base['bandwidth'], 'cpu')
            grouped = grouped_rbf_kernels(data.X, fit, test, bandwidth=base['bandwidth'],
                                          tau=cfg['tau'], groups=cfg['groups'])
            arms = [('original', r, rc), ('grouped', grouped.train, grouped.cross)]
            for arm, kr, kc in arms:
                gram = base['fraction'] * a + (1 - base['fraction']) * torch.as_tensor(kr)
                cross = base['fraction'] * ac + (1 - base['fraction']) * torch.as_tensor(kc)
                prepared = features(data.X, data.Y, fit, test, base, material_kernels=(gram, cross))
                pred, diag = fit_predict(prepared, base['ridge'][0] * choice['multiplier'],
                                        choice['eta'], choice['delta'], solver,
                                        fixed_thresholds=fixed)
                if arm == 'original':
                    if not np.allclose(np.array(diag['residual_mad_scale']) * choice['delta'],
                                       fixed, atol=1e-10, rtol=1e-10):
                        raise ValueError('Original Huber threshold reconstruction mismatch')
                    for e, env in enumerate(data.environments):
                        for ix, value in zip(test, pred[:, e]):
                            true, saved = saved_oof[(str(env), f, int(ix))]
                            if abs(true - float(data.Y[ix, e])) > 1e-12:
                                raise ValueError('Saved OOF row does not match data')
                            oof_max = max(oof_max, abs(saved - float(value)))
                metrics = []
                for e, env in enumerate(data.environments):
                    values = dict(pcc=pcc(data.Y[test, e], pred[:, e]),
                                  rmse=rmse(data.Y[test, e], pred[:, e]),
                                  mae=mae(data.Y[test, e], pred[:, e]))
                    if not np.isfinite(list(values.values())).all():
                        raise ValueError('Nonfinite metric')
                    metrics.append(values)
                    oof.extend({'arm': arm, 'environment': env, 'fold': f, 'material_index': int(ix),
                                'y_true': float(data.Y[ix, e]), 'y_pred': float(v)}
                               for ix, v in zip(test, pred[:, e]))
                rows.append({'fold': f, 'arm': arm, 'per_environment': metrics,
                             'macro': {k: float(np.mean([m[k] for m in metrics])) for k in metrics[0]}})
                state['completed_fits'] += 1
                write('execution_status.json', state)
            write('units.json', rows)
        if state['completed_fits'] != state['expected_fits']:
            raise ValueError('Incomplete outer grid')
        summary = summarize(rows, expected_original_macro)
        summary['self_check'] = {
            'oof_max_difference': oof_max,
            'macro_pcc_difference': abs(summary['original_macro_pcc'] - expected_original_macro),
            'threshold_reconstruction': 'atol=rtol=1e-10 per original arm'}
        write('summary.json', summary)
        write_csv(args.out / 'oof.csv',
                  ['arm', 'environment', 'fold', 'material_index', 'y_true', 'y_pred'], oof)
        state['status'] = 'complete'
    except BaseException as error:
        state['status'] = 'failed'
        failures.append({'reason': repr(error), 'completed_fits': state['completed_fits']})
        raise
    finally:
        write('units.json', rows)
        write('failures.json', failures)
        write('execution_status.json', state)
        write('timing.json', {'seconds': time.perf_counter() - started})
