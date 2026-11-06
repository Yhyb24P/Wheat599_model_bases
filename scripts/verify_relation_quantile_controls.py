"""Independent final audit for the relation controls (permutation + whole-curve) run.

Read-only on the frozen checkout and run directory; writes only to --out.
Strict mode requires 120/120 completion and zero failures, then recomputes all
five arms (baseline, fusion, center, shuffled, curve) from raw saved predictions
and validation truth, re-verifies every recorded kernel permutation together
with full-spectrum preservation on the real fit kernels, and cross-checks the
runner's summary/selection artifacts.
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

MODES = ('shuffled', 'curve')
ARMS = ('baseline', 'fusion', 'center') + MODES


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
    root = Path(sys.argv[1])
    run = Path(sys.argv[2])
    out = Path(sys.argv[3])
    expected_commit = sys.argv[4] if len(sys.argv) > 4 else None
    original_root = Path(sys.argv[5]) if len(sys.argv) > 5 else root
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(root / 'src'))

    checks = {}

    def check(name, ok, detail):
        checks[name] = {'pass': bool(ok), **detail}
        if not ok:
            raise SystemExit(json.dumps({'verdict': 'FAIL', 'failed_check': name,
                                         'checks': checks}, indent=2))

    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    check('checkout_commit', expected_commit is None or commit.startswith(expected_commit),
          {'commit': commit})

    cfg = yaml.safe_load((root / 'configs/relation_quantile_controls.yaml').read_text())
    base = yaml.safe_load((root / cfg['original_config']).read_text())
    settings = yaml.safe_load((root / base['quantile_config']).read_text())
    seeds = [int(s) for s in settings['seeds']]
    strength = cfg['strength']
    if strength not in [float(v) for v in base['lambdas']]:
        raise ValueError('strength not in original grid')
    strength_key = f'l{strength}'

    from wheat599.data import load
    from wheat599.kernel_baselines import kernels
    from wheat599.relation_regularizer import build_kernel
    data = load(root / settings['data'], settings['data_sha256'])
    Y = data.Y
    folds = json.loads((root / settings['splits']).read_text())['folds']
    original = original_root / cfg['original_result']
    baseline = original_root / base['baseline_source']
    ref = original_root / base['reference_source']
    bases = json.loads((root / base['base_choices']).read_text())
    chosen = {v['fold']: v['selected'] for v in json.loads((baseline / 'summary.json').read_text())['folds']}

    # ---- completion -------------------------------------------------------
    status = json.loads((run / 'status.json').read_text())
    expected = len(folds) * 3 * len(seeds) * 2
    check('run_complete', status.get('status') == 'complete'
          and status.get('completed') == status.get('expected') == expected, {'status': status})
    failures = json.loads((run / 'failures.json').read_text())
    check('zero_failures', failures == [], {'failures': failures})
    records = json.loads((run / 'trajectories.json').read_text())
    cells = {(r['fold'], r['inner'], r['mode'], int(r['seed'])) for r in records}
    want = {(f['fold'], j, m, s) for f in folds for j in range(3) for m in MODES for s in seeds}
    check('trajectory_records', len(records) == expected and cells == want
          and all(np.isfinite(r['best_pcc']).all() for r in records),
          {'records': len(records), 'duplicates': len(records) - len(cells)})

    # ---- unit files ---------------------------------------------------------
    unit_files = {}
    for f in folds:
        for j in range(3):
            path = run / f'predictions_{f["fold"]}_{j}.npz'
            if not path.exists():
                continue
            z = np.load(path)
            needed = {f'{m}_s{s}' for m in MODES for s in seeds}
            if set(z.keys()) != needed:
                raise SystemExit(json.dumps({'verdict': 'FAIL', 'failed_check': 'unit_file_keys',
                                             'file': str(path), 'checks': checks}, indent=2))
            unit_files[(f['fold'], j)] = z
    check('unit_files', len(unit_files) == len(folds) * 3
          and all(np.isfinite(z[k]).all() and z[k].shape[1] == 4
                  for z in unit_files.values() for k in z.keys()),
          {'units': len(unit_files)})

    # ---- permutation reproducibility + full spectrum on real kernels -------
    device = torch.device('cpu')
    perm_bad, spectrum_bad = [], []
    perm_hashes = { (e['fold'], e['inner']): e['sha256']
                    for e in json.loads((run / 'permutations.json').read_text()) }
    if set(perm_hashes) != set(unit_files):
        raise SystemExit(json.dumps({'verdict': 'FAIL', 'failed_check': 'permutation_coverage',
                                     'checks': checks}, indent=2))
    for idx_f, f in enumerate(folds):
        fid = f['fold']
        for j, inner in enumerate(f['inner']):
            tr = np.array(inner['train_idx'])
            b = next(v['homogeneous'] for v in bases if v['fold'] == fid and v['inner_fold'] == j)
            gram = build_kernel(data.X[tr], bandwidth=b['bandwidth'], fraction=b['fraction'], device=device)
            gen = torch.Generator(device='cpu').manual_seed(int(cfg['shuffle_seed']) + 3 * fid + j)
            perm = torch.randperm(len(tr), generator=gen)
            digest = hashlib.sha256(perm.numpy().astype('<i8').tobytes()).hexdigest()
            if digest != perm_hashes[(fid, j)]:
                perm_bad.append((fid, j))
            shuffled = gram[perm][:, perm]
            if not torch.allclose(shuffled, shuffled.T, atol=1e-12, rtol=1e-12):
                spectrum_bad.append((fid, j, 'symmetry'))
                continue
            ev_k = torch.linalg.eigvalsh(gram.double())
            ev_p = torch.linalg.eigvalsh(shuffled.double())
            diff = float(torch.abs(ev_k - ev_p).max().item())
            scale = max(1.0, float(torch.abs(ev_k).max().item()))
            if diff > 1e-8 * scale:
                spectrum_bad.append((fid, j, diff))
    check('permutation_reproducible', not perm_bad, {'mismatches': perm_bad})
    check('permutation_spectrum_preserved', not spectrum_bad, {'mismatches': spectrum_bad[:5],
                                                               'units_checked': len(unit_files)})

    # ---- five-arm independent recomputation ---------------------------------
    per = {}
    for f in folds:
        fid = f['fold']
        for j, inner in enumerate(f['inner']):
            val = np.array(inner['val_idx'])
            truth = Y[val]
            z = unit_files[(fid, j)]
            base_z = np.load(baseline / f'predictions_{fid}_{j}.npz')
            ref_z = np.load(ref / f'references_{fid}_{j}.npz')
            orig_z = np.load(original / f'predictions_{fid}_{j}.npz')
            sel = chosen[fid]
            baseline_arm = np.column_stack([
                np.mean([base_z[f'e{e}_c{sel[e]}_s{s}'] for s in seeds], axis=0) for e in range(4)])
            huber = ref_z['huber']
            center = np.mean([orig_z[f'{strength_key}_s{s}'] for s in seeds], axis=0)
            arms = {
                'baseline': baseline_arm,
                'fusion': 0.5 * (baseline_arm + huber),
                'center': center,
                'shuffled': np.mean([z[f'shuffled_s{s}'] for s in seeds], axis=0),
                'curve': np.mean([z[f'curve_s{s}'] for s in seeds], axis=0),
            }
            per[(fid, j)] = {k: env_metrics(truth, v) for k, v in arms.items()}
    fold_metrics = {}
    for fid in sorted({k[0] for k in per}):
        fold_metrics[fid] = {k: np.mean(np.stack([per[(fid, j)][k] for j in range(3)]), axis=0)
                             for k in ARMS}
    macro = {k: [float(v) for v in np.mean(
        np.stack([fold_metrics[f][k] for f in sorted(fold_metrics)]), axis=(0, 1)).tolist()]
        for k in ARMS}

    audit = {
        'run_dir': str(run), 'checkout_commit': commit, 'checks': checks,
        'arms': {k: {'pcc': macro[k][0], 'rmse': macro[k][1], 'mae': macro[k][2]} for k in ARMS},
        'fold_metrics': {str(fid): {k: fold_metrics[fid][k].tolist() for k in ARMS}
                         for fid in sorted(fold_metrics)},
        'note': ('Center/shuffled/curve were trained and evaluated on development data whose splits '
                 'also selected the architecture and (post hoc) lambda=10; baseline and fusion reuse '
                 'those same splits. All comparisons are development diagnostics with selection bias '
                 'and provide no outer-generalization evidence.'),
    }

    # ---- paired differences vs center ---------------------------------------
    paired = {}
    for mode in MODES:
        entry = {}
        for arm in ARMS:
            d = [fold_metrics[f][mode][:, 0].mean() - fold_metrics[f][arm][:, 0].mean()
                 for f in sorted(fold_metrics)]
            d_env = np.concatenate([(fold_metrics[f][mode] - fold_metrics[f][arm])[:, 0].ravel()
                                    for f in sorted(fold_metrics)])
            entry[arm] = {
                'pcc_macro_diff': float(np.mean(d)),
                'pcc_fold_wins': int(np.sum(np.array(d) > 0)),
                'pcc_foldenv_wins': int(np.sum(d_env > 0)),
                'rmse_macro_diff': float(np.mean([
                    fold_metrics[f][mode][:, 1].mean() - fold_metrics[f][arm][:, 1].mean()
                    for f in sorted(fold_metrics)])),
                'mae_macro_diff': float(np.mean([
                    fold_metrics[f][mode][:, 2].mean() - fold_metrics[f][arm][:, 2].mean()
                    for f in sorted(fold_metrics)])),
            }
        paired[mode] = entry
    audit['paired'] = paired

    # ---- runner artifacts cross-check ----------------------------------------
    summary = json.loads((run / 'summary.json').read_text())
    smax = 0.0
    for fid in sorted(fold_metrics):
        entry = next(e for e in summary['folds'] if e['fold'] == fid)
        for k in ARMS:
            smax = max(smax, float(np.max(np.abs(np.array(entry['metrics'][k]) - fold_metrics[fid][k]))))
    mm = max(abs(float(summary['macro'][k][metric]) - macro[k][i])
             for k in ARMS for i, metric in enumerate(('pcc', 'rmse', 'mae')))
    check('summary_recomputed', max(smax, mm) < 1e-9, {'max_metric_diff': float(max(smax, mm))})
    prior = json.loads((original / 'summary.json').read_text())['macro'][str(strength)]
    cmax = max(abs(float(prior[k]) - macro['center'][i])
               for i, k in enumerate(('pcc', 'rmse', 'mae')))
    check('center_matches_original_run', cmax < 1e-12, {'max_center_diff': float(cmax)})
    sel = json.loads((original / 'selection.json').read_text())
    audit['original_lambda_selection_note'] = [
        {'fold': v['fold'], 'lambda': v['lambda']} for v in sel['folds']]

    (out / 'audit.json').write_text(json.dumps(audit, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'verdict': 'PASS', 'arms': audit['arms'], 'paired': paired},
                     indent=2, allow_nan=False))
    print('AUDIT_WRITTEN', out / 'audit.json')


if __name__ == '__main__':
    main()
