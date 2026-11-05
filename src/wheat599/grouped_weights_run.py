"""Development comparison of fit-only grouped kernel weight learners."""
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from wheat599.data import load, sha256_of
from wheat599.grouped_kernel_run import read
from wheat599.metrics import pcc, rmse, mae


def summarize(rows):
    arms = ('identity', 'scalar', 'network')
    indexed = {(r['fold'], r['inner'], r['arm']): r for r in rows}
    expected = {(f, j, a) for f in range(10) for j in range(3) for a in arms}
    if len(indexed) != len(rows) or set(indexed) != expected:
        raise ValueError('Incomplete or duplicate weights comparison')
    scores = {}
    for arm in arms:
        values = np.array([indexed[f, j, arm]['metrics'] for f in range(10) for j in range(3)])
        scores[arm] = values.reshape(10, 3, 4, 3).mean(axis=1)
    if not all(np.isfinite(v).all() for v in scores.values()):
        raise ValueError('Nonfinite summary')
    result = {'metrics': {}, 'comparisons': {}, 'metric_order': ['pcc', 'rmse', 'mae']}
    for arm, values in scores.items():
        result['metrics'][arm] = {'macro': dict(zip(result['metric_order'], values.mean(axis=(0, 1)).tolist())),
                                  'per_environment': values.mean(axis=0).tolist(),
                                  'fold_macro_pcc': values[:, :, 0].mean(axis=1).tolist()}
    for arm in ('identity', 'scalar'):
        delta = (scores['network'][:, :, 0] - scores[arm][:, :, 0]).mean(axis=1)
        result['comparisons'][arm] = dict(pcc_gain=float(delta.mean()), fold_wins=int((delta > 0).sum()),
                                           fold_deltas=delta.tolist())
    result['mechanism_pass'] = all(c['pcc_gain'] >= .001 and c['fold_wins'] >= 7
                                   for c in result['comparisons'].values())
    result['independent_tabm_development_pcc'] = .49187397138978695
    result['outer_eligible'] = (result['mechanism_pass'] and
        result['metrics']['network']['macro']['pcc'] > result['independent_tabm_development_pcc'])
    return result


def run(cfg, args, runner):
    from wheat599.grouped_weights import run_unit
    if args.selection is not None:
        raise ValueError('Development only')
    torch.set_num_threads(cfg['cpu_threads'])
    if cfg['device'] == 'cuda':
        torch.cuda.set_per_process_memory_fraction(.25)
    data = load(Path(cfg['data']), cfg['data_sha256'])
    folds = read(cfg['splits'])['folds']
    source = Path(cfg['huber_source'])
    selection = read(source/'selection.json')
    solver = selection['config']
    base_path = Path(solver['base_source'])/'base_choices.json'
    bases = read(base_path)
    old_diagnostics = read(source/'optimization.json')
    identity = Path(cfg['identity_source'])
    if not read(identity/'summary.json')['stage2_eligible']:
        raise ValueError('Fixed grouping did not pass')
    if (read(source/'provenance.json')['splits_sha256'] != sha256_of(Path(cfg['splits'])) or
            solver['data_sha256'] != cfg['data_sha256'] or
            selection['base_source_sha256'] != sha256_of(base_path)):
        raise ValueError('Historical source mismatch')
    args.out.mkdir(parents=True, exist_ok=False)
    def write(name, value):
        (args.out/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    paths = [Path(__file__), Path(__file__).with_name('grouped_weights.py'), Path(runner),
             Path(args.config), Path(cfg['splits']), base_path, source/'selection.json',
             source/'optimization.json', identity/'summary.json']
    write('provenance.json', dict(config=cfg, code_commit=subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], text=True).strip(), sha256={str(p): sha256_of(p) for p in paths},
        environment_order=list(data.environments)))
    rows, failures, diagnostics = [], [], []
    state = dict(status='running', completed_fits=0, expected_fits=90)
    started = time.perf_counter()
    try:
        for fold in folds:
            f = fold['fold']
            choice = next(s['huber'] for s in selection['selected'] if s['fold'] == f)
            for j, inner in enumerate(fold['inner']):
                state['active_unit'] = [f, j]
                write('execution_status.json', state)
                fit, val = np.array(inner['train_idx']), np.array(inner['val_idx'])
                if (set(fit) & set(val) or set(fit) | set(val) != set(fold['dev_idx']) or
                        (set(fit) | set(val)) & set(fold['test_idx'])):
                    raise ValueError('Invalid inner partition')
                base = next(b['homogeneous'] for b in bases if b['fold'] == f and b['inner_fold'] == j)
                selected = [d for d in old_diagnostics if d['fold'] == f and d['inner_fold'] == j
                    and d['model'] == 'huber' and all(d[k] == choice[k] for k in ('eta', 'delta', 'multiplier'))]
                if len(selected) != 1:
                    raise ValueError('Missing or duplicate original thresholds')
                result = run_unit(data.X[fit], data.Y[fit], data.X[val], base, choice,
                                  selected[0]['thresholds'], solver, device=cfg['device'])
                prior = np.load(identity/f'f{f}_i{j}_correlation.npz')
                if not np.array_equal(prior['ids'], val):
                    raise ValueError('Identity reference material IDs differ')
                identity_error = float(np.max(abs(result['arms']['identity']['prediction'] - prior['prediction'])))
                if identity_error > 1e-4:
                    raise ValueError(f'Generic Huber identity mismatch: {identity_error}')
                for arm in ('identity', 'scalar', 'network'):
                    item = result['arms'][arm]
                    pred = np.asarray(item.pop('prediction'))
                    values = [[float(metric(data.Y[val, e], pred[:, e])) for metric in (pcc, rmse, mae)]
                              for e in range(4)]
                    if not np.isfinite(values).all():
                        raise ValueError('Nonfinite metrics')
                    rows.append(dict(fold=f, inner=j, arm=arm, metrics=values))
                    np.savez_compressed(args.out/f'f{f}_i{j}_{arm}.npz', ids=val, prediction=pred)
                    state['completed_fits'] += 1
                diagnostics.append(dict(fold=f, inner=j, identity_max_error=identity_error, **result))
                write('units.json', rows)
                write('execution_status.json', state)
        write('summary.json', summarize(rows))
        state['status'] = 'complete'
    except BaseException as error:
        state['status'] = 'failed'
        failures.append(dict(reason=repr(error), completed_fits=state['completed_fits'],
                             active_unit=state.get('active_unit'),
                             partial_unit_fits='not counted unless run_unit returns successfully'))
        raise
    finally:
        write('units.json', rows)
        write('diagnostics.json', diagnostics)
        write('failures.json', failures)
        write('execution_status.json', state)
        write('timing.json', dict(seconds=time.perf_counter()-started))
