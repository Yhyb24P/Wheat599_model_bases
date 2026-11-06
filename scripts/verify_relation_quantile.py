"""Independent audit for the relation-regularized quantile TabM development run.

Read-only with respect to the frozen checkout; writes only to --out.
Recomputes all metrics from raw saved predictions and truth, independent of the
runner's summary.  Partial mode audits only complete units (all lambda/seed
keys present); strict mode additionally requires full completion (180/180,
zero failures) and cross-checks the runner's summary/selection files.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

LAMBDA_KEYS = {0.1: 'l0.1', 1.0: 'l1.0', 10.0: 'l10.0'}
CANDIDATES = ['0.0', '0.1', '1.0', '10.0']


def pcc(t, p):
    t = np.asarray(t, float).ravel(); p = np.asarray(p, float).ravel()
    if t.shape != p.shape or t.size == 0:
        raise ValueError('shape mismatch')
    if t.std() == 0 or p.std() == 0:
        return float('nan')
    return float(np.corrcoef(t, p)[0, 1])


def rmse(t, p):
    t = np.asarray(t, float); p = np.asarray(p, float)
    return float(np.sqrt(np.mean((t - p) ** 2)))


def mae(t, p):
    t = np.asarray(t, float); p = np.asarray(p, float)
    return float(np.mean(np.abs(t - p)))


def env_metrics(truth, pred):
    out = [[pcc(truth[:, e], pred[:, e]), rmse(truth[:, e], pred[:, e]),
            mae(truth[:, e], pred[:, e])] for e in range(truth.shape[1])]
    if not np.isfinite(out).all():
        raise ValueError('nonfinite metric')
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='frozen checkout root')
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--strict', action='store_true',
                    help='require full completion and cross-check runner summary/selection')
    args = ap.parse_args()
    root = Path(args.root)
    run = Path(args.run_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(root / 'src'))

    cfg = yaml.safe_load((root / 'configs/relation_quantile.yaml').read_text())
    qcfg = yaml.safe_load((root / Path(cfg['quantile_config'])).read_text())
    seeds = [int(s) for s in qcfg['seeds']]
    lambdas = [float(v) for v in cfg['lambdas']]
    if set(lambdas) != set(LAMBDA_KEYS):
        raise ValueError(f'Unexpected lambdas {lambdas}')
    if len(seeds) != 2:
        raise ValueError('Expected two seeds')

    from wheat599.data import load
    data = load(root / qcfg['data'], qcfg['data_sha256'])
    Y = data.Y
    folds = json.loads((root / qcfg['splits']).read_text())['folds']
    if len(folds) != 10 or any(len(f['inner']) != 3 for f in folds):
        raise ValueError('Unexpected split structure')
    baseline = root / cfg['baseline_source']
    reference = root / cfg['reference_source']
    chosen = {v['fold']: v['selected'] for v in json.loads((baseline / 'summary.json').read_text())['folds']}

    expected_units = 30
    expected_records = expected_units * len(lambdas) * len(seeds)
    checks = {}

    def check(name, ok, detail):
        checks[name] = {'pass': bool(ok), **detail}
        if not ok:
            raise SystemExit(json.dumps({'verdict': 'FAIL', 'failed_check': name,
                                         'checks': checks}, indent=2))

    # ---- unit inventory -------------------------------------------------
    unit_files = {}
    for f in folds:
        for j in range(3):
            path = run / f'predictions_{f["fold"]}_{j}.npz'
            if not path.exists():
                continue
            z = np.load(path)
            keys = sorted(z.keys())
            needed = {f'{LAMBDA_KEYS[l]}_s{s}' for l in lambdas for s in seeds}
            if set(keys) == needed:
                unit_files[(f['fold'], j)] = z
    n_complete = len(unit_files)
    status = json.loads((run / 'status.json').read_text())
    check('unit_files_complete_shape',
          all(np.isfinite(z[k]).all() and z[k].shape[1] == 4
              for z in unit_files.values() for k in z.keys()),
          {'complete_units': n_complete, 'expected_units': expected_units,
           'status_file': status})

    if args.strict:
        check('run_complete', status.get('status') == 'complete'
              and status.get('completed') == status.get('expected') == expected_records,
              {'status': status})
        failures = json.loads((run / 'failures.json').read_text())
        check('zero_failures', failures == [], {'failures': failures})
        records = json.loads((run / 'trajectories.json').read_text())
        cells = {(r['fold'], r['inner'], float(r['lambda']), int(r['seed'])) for r in records}
        want = {(f['fold'], j, l, s) for f in folds for j in range(3)
                for l in lambdas for s in seeds}
        check('trajectory_records', len(records) == expected_records and cells == want
              and all(np.isfinite(r['best_pcc']).all() for r in records),
              {'records': len(records), 'missing': sorted(map(str, want - cells))[:8],
               'duplicates': len(records) - len(cells)})
        for f in folds:
            idxs = {tuple(r['config_indices']) for r in records if r['fold'] == f['fold']}
            if idxs != {tuple(chosen[f['fold']])}:
                raise SystemExit(json.dumps({'verdict': 'FAIL', 'failed_check': 'config_indices',
                                             'checks': checks}, indent=2))
        checks['config_indices_match_baseline_selection'] = {'pass': True}

    # ---- independent metric recomputation --------------------------------
    # Per complete unit and candidate: average the two seeds (or reuse the
    # frozen baseline predictions for 0.0), then per-environment metrics vs
    # validation truth.  Fold level averages the three inner units; macro
    # averages all folds x environments x inner units, mirroring the runner.
    per = {}
    for f in folds:
        for j, inner in enumerate(f['inner']):
            key = (f['fold'], j)
            if key not in unit_files:
                continue
            val = np.array(inner['val_idx'])
            truth = Y[val]
            z = unit_files[key]
            base = np.load(baseline / f'predictions_{f["fold"]}_{j}.npz')
            ref = np.load(reference / f'references_{f["fold"]}_{j}.npz')
            sel = chosen[f['fold']]
            orig = np.column_stack([np.mean([base[f'e{e}_c{sel[e]}_s{s}'] for s in seeds], axis=0)
                                    for e in range(4)])
            fusion = 0.5 * (orig + ref['huber'])
            cand = {'0.0': orig, 'fusion': fusion}
            for l in lambdas:
                cand[str(l)] = np.mean([z[f'{LAMBDA_KEYS[l]}_s{s}'] for s in seeds], axis=0)
            per[key] = {k: env_metrics(truth, v) for k, v in cand.items()}
            per[key]['huber_check'] = env_metrics(truth, ref['huber'])

    complete_folds = {k[0] for k in per if all((k[0], j) in per for j in range(3))}
    fold_metrics = {}
    for fid in sorted(complete_folds):
        stack = {k: np.mean(np.stack([per[(fid, j)][k] for j in range(3)]), axis=0)
                 for k in ('0.0', '0.1', '1.0', '10.0', 'fusion', 'huber_check')}
        fold_metrics[fid] = stack

    fold_dump = {str(fid): {k: fold_metrics[fid][k].tolist() for k in
                            ('0.0', '0.1', '1.0', '10.0', 'fusion', 'huber_check')}
                 for fid in sorted(fold_metrics)} if fold_metrics else {}
    audit = {
        'run_dir': str(run), 'complete_units': n_complete, 'complete_folds': sorted(complete_folds),
        'strict': args.strict, 'checks': checks, 'fold_metrics': fold_dump,
        'note': ('Some development scores were computed on data also used to select lambda; '
                 'selection bias applies and no outer-generalization claim is supported.'),
    }

    if len(complete_folds) == 10:
        macro = {}
        for k in ('0.0', '0.1', '1.0', '10.0', 'fusion'):
            arr = np.stack([fold_metrics[f][k] for f in sorted(fold_metrics)])
            macro[k] = [float(v) for v in np.mean(arr, axis=(0, 1)).tolist()]
            # per-environment PCC means over all folds, for reporting
            macro[k + '_per_env_pcc'] = [float(v) for v in np.mean(arr[:, :, 0], axis=0).tolist()]
        paired = {}
        for l in ('0.1', '1.0', '10.0'):
            entry = {}
            for ref_name in ('0.0', 'fusion'):
                d = [fold_metrics[f][l][:, 0].mean() - fold_metrics[f][ref_name][:, 0].mean()
                     for f in sorted(fold_metrics)]
                d_env = [ (fold_metrics[f][l] - fold_metrics[f][ref_name])[:, 0].ravel()
                          for f in sorted(fold_metrics) ]
                d_env = np.concatenate(d_env)
                entry[ref_name] = {
                    'pcc_fold_wins': int(np.sum(np.array(d) > 0)),
                    'pcc_macro_diff': float(np.mean(d)),
                    'pcc_foldenv_wins': int(np.sum(d_env > 0)),
                    'pcc_foldenv_total': int(d_env.size),
                    'rmse_macro_diff': float(np.mean([
                        fold_metrics[f][l][:, 1].mean() - fold_metrics[f][ref_name][:, 1].mean()
                        for f in sorted(fold_metrics)])),
                    'mae_macro_diff': float(np.mean([
                        fold_metrics[f][l][:, 2].mean() - fold_metrics[f][ref_name][:, 2].mean()
                        for f in sorted(fold_metrics)])),
                }
            paired[l] = entry
        audit['macro'] = macro
        audit['paired_vs_references'] = paired

        # selection recomputation (fold-level mean env PCC per candidate; argmax, tie -> zero)
        selection = []
        for fid in sorted(fold_metrics):
            scores = {k: float(fold_metrics[fid][k][:, 0].mean())
                      for k in CANDIDATES + ['fusion']}
            idx = max(range(len(CANDIDATES)), key=lambda i: scores[CANDIDATES[i]])
            selection.append({'fold': fid, 'lambda': CANDIDATES[idx], 'scores': scores})
        audit['selection_recomputed'] = selection
        if args.strict:
            saved = json.loads((run / 'selection.json').read_text())
            saved_map = {v['fold']: (float(v['lambda']), [float(x) for x in v['scores']])
                         for v in saved['folds']}
            ok = all(abs(saved_map[s['fold']][0] - float(s['lambda'])) < 1e-12 and
                     np.allclose(saved_map[s['fold']][1],
                                 [float(s['scores'][k]) for k in CANDIDATES], atol=1e-9)
                     for s in selection)
            check('selection_matches', ok,
                  {'recomputed': [(s['fold'], s['lambda']) for s in selection]})
            summary = json.loads((run / 'summary.json').read_text())
            smax = 0.0
            for fid in sorted(fold_metrics):
                entry = next(e for e in summary['folds'] if e['fold'] == fid)
                for k in CANDIDATES + ['fusion']:
                    smax = max(smax, float(np.max(np.abs(np.array(entry['metrics'][k]) - fold_metrics[fid][k]))))
            mm = max(abs(float(summary['macro'][k][metric]) - float(macro[k][i]))
                     for k in CANDIDATES + ['fusion']
                     for i, metric in enumerate(('pcc', 'rmse', 'mae')))
            check('summary_recomputed', max(smax, mm) < 1e-9, {'max_metric_diff': float(max(smax, mm))})

    (out / 'audit.json').write_text(json.dumps(audit, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: audit[k] for k in ('run_dir', 'complete_units', 'complete_folds', 'strict')}, indent=2))
    if len(complete_folds) == 10:
        print(json.dumps({'macro_pcc': {k: macro[k][0] for k in CANDIDATES + ['fusion']},
                          'paired': audit['paired_vs_references'],
                          'selected': [(s['fold'], s['lambda']) for s in audit['selection_recomputed']]},
                         indent=2, allow_nan=False))
    print('AUDIT_WRITTEN', out / 'audit.json')


if __name__ == '__main__':
    main()
