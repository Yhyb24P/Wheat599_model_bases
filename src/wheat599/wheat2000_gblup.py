"""Centered marker-kernel GBLUP/RR-BLUP, with fold-local phenotype intercepts.

A raw pairwise dot product may be cached for a fixed published marker matrix.
Centering uses only the current fit IDs. Each output has its own ridge selection;
sharing an eigendecomposition does not share fitted phenotype coefficients.
"""
import numpy as np


def raw_marker_kernel(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all() or x.shape[1] < 1:
        raise ValueError('Expected a finite material-by-marker matrix')
    return x @ x.T / x.shape[1]


def predict_grid(raw_kernel, y, fit_ids, query_ids, lambdas):
    """Return lambda x query x trait predictions without reading query phenotypes.

    The modeled kernel is (X-X_fit_mean)(X-X_fit_mean)^T / p. ``lambda``
    is in this kernel's units. The intercept is the fitted trait mean. A positive
    fixed lambda gives exactly the centered primal ridge prediction with penalty
    p*lambda; no feature scaling or phenotype relationship is learned globally.
    """
    gram = np.asarray(raw_kernel, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        y = y[:, None]
    fit = np.asarray(fit_ids, dtype=np.int64)
    query = np.asarray(query_ids, dtype=np.int64)
    ridge = np.asarray(lambdas, dtype=np.float64)
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1] or len(y) != len(gram):
        raise ValueError('Kernel/phenotype dimensions differ')
    if fit.ndim != 1 or query.ndim != 1 or len(fit) < 2 or len(query) < 1 \
            or len(np.unique(fit)) != len(fit) or len(np.unique(query)) != len(query):
        raise ValueError('Invalid or duplicate material IDs')
    if set(fit) & set(query) or np.any(fit < 0) or np.any(query < 0) \
            or np.any(fit >= len(y)) or np.any(query >= len(y)):
        raise ValueError('Fit/query IDs must be disjoint and in range')
    if ridge.ndim != 1 or not len(ridge) or not np.isfinite(ridge).all() or np.any(ridge <= 0):
        raise ValueError('Ridge values must be finite and positive')
    yy = y[fit]
    if not np.isfinite(yy).all():
        raise ValueError('Nonfinite fit phenotypes')
    train = gram[np.ix_(fit, fit)]
    cross = gram[np.ix_(query, fit)]
    if not np.isfinite(train).all() or not np.isfinite(cross).all():
        raise ValueError('Nonfinite fit/query kernel')
    train_means, grand = train.mean(0), float(train.mean())
    centered = train - train_means[:, None] - train_means[None, :] + grand
    cross = cross - cross.mean(1)[:, None] - train_means[None, :] + grand
    values, vectors = np.linalg.eigh(centered)
    if float(values.min()) < -1e-8 * max(1., float(abs(values).max())):
        raise ValueError('Centered marker kernel is not positive semidefinite')
    values = np.maximum(values, 0.)  # Roundoff at the centered constant direction.
    mean = yy.mean(0)
    coefficients = vectors.T @ (yy - mean)
    projected = cross @ vectors
    shrunk = coefficients[None] / (values[None, :, None] + ridge[:, None, None])
    predictions = np.einsum('qi,lit->lqt', projected, shrunk) + mean
    if not np.isfinite(predictions).all():
        raise FloatingPointError('Nonfinite GBLUP prediction')
    return predictions


def run(config, args, runner):
    """Three-inner-fold ridge selection, then a separate frozen final phase."""
    import csv
    import hashlib
    import json
    import subprocess
    import time
    from pathlib import Path
    from threadpoolctl import threadpool_limits
    from wheat599.data import sha256_of
    from wheat599.environment_selection import metric_matrix
    from wheat599.wheat2000_data import load_wheat2000

    def write(name, value):
        (args.out / name).write_text(json.dumps(value, indent=2) + '\n')

    data = load_wheat2000(config['data'], config['data_sha256'],
                         config['cache'], config['cache_sha256'])
    split_path = Path(config['splits'])
    if sha256_of(split_path) != config['splits_sha256']:
        raise ValueError('Frozen Wheat2000 splits SHA differs')
    splits = json.loads(split_path.read_text())
    if list(data.trait_names) != config['traits'] or splits['trait_names'] != config['traits']:
        raise ValueError('All eight traits and their order must match the frozen protocol')
    ridge = np.logspace(**config['ridge_logspace'])
    config_sha = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    sources = {str(p): sha256_of(Path(p)) for p in
               (Path(__file__), Path(runner), Path(args.config),
                Path(__file__).with_name('wheat2000_data.py'),
                Path(__file__).with_name('environment_selection.py'),
                Path(__file__).with_name('metrics.py'))}
    selected = None
    if args.selection is not None:
        from wheat599.independent_final import ensure_frozen_manifest
        ensure_frozen_manifest(args.selection)
        payload = json.loads(args.selection.read_text())
        if payload['config_sha256'] != config_sha:
            raise ValueError('GBLUP frozen selection/config differ')
        if payload.get('source_sha256') != sources:
            raise ValueError('GBLUP frozen selection/source differ')
        selected = payload['selected']
        expected = {(f['fold'], t) for f in splits['folds'] for t in data.trait_names}
        if {(r['fold'], r['trait']) for r in selected} != expected or len(selected) != len(expected):
            raise ValueError('GBLUP frozen selection is incomplete or duplicated')
    args.out.mkdir(parents=True, exist_ok=False)
    write('provenance.json', {
        'config': config, 'config_sha256': config_sha,
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'source_sha256': sources,
        'selection_sha256': sha256_of(args.selection) if args.selection is not None else None,
        'data_sha256': config['data_sha256'], 'cache_sha256': config['cache_sha256'],
        'splits_sha256': config['splits_sha256'],
        'numpy_version': np.__version__,
        'kernel': 'fit-centered XX.T / markers, no global phenotype or feature preprocessing',
        'rule': 'mean three-inner PCC selects lambda for each trait/fold; first tie',
    })
    start = time.perf_counter()
    failures, selection, rows, predictions = [], [], [], []
    state = {'status': 'running', 'folds_complete': [],
             'phase': 'final' if selected is not None else 'development'}
    write('execution_status.json', state)
    try:
        with threadpool_limits(limits=int(config['cpu_threads'])):
            raw = raw_marker_kernel(data.X)
            for fold in splits['folds']:
                f = int(fold['fold'])
                if selected is None:
                    scores = []
                    for inner in fold['inner']:
                        val = inner['val_idx']
                        pred = predict_grid(raw, data.Y, inner['train_idx'], val, ridge)
                        scores.append([metric_matrix(p, data.Y[val]) for p in pred])
                    means = np.mean(scores, axis=0)  # lambda x trait x metric.
                    for e, trait in enumerate(data.trait_names):
                        pccs = means[:, e, 0]
                        c = int(np.flatnonzero(pccs >= pccs.max() - config['tie_tolerance'])[0])
                        selection.append({'fold': f, 'trait': trait, 'lambda': float(ridge[c]),
                                          'inner_mean_metrics': dict(zip(
                                              ('pcc', 'rmse', 'mae'), means[c, e].tolist())),
                                          'grid_mean_pcc': pccs.tolist()})
                    write('selection.json', {'config_sha256': config_sha,
                                             'source_sha256': sources, 'selected': selection})
                else:
                    chosen = {r['trait']: r['lambda'] for r in selected if r['fold'] == f}
                    # One eigendecomposition supplies all selected lambdas/traits; each
                    # trait uses only its own independently chosen ridge prediction.
                    unique = sorted(set(chosen.values()))
                    pred = predict_grid(raw, data.Y, fold['dev_idx'], fold['test_idx'], unique)
                    ensemble = np.column_stack([pred[unique.index(chosen[t]), :, e]
                                                for e, t in enumerate(data.trait_names)])
                    predictions.append((f, fold['test_idx'], ensemble, chosen))
                state['folds_complete'].append(f)
                write('execution_status.json', state)
        if selected is not None:
            oof = []
            for f, ids, pred, chosen in predictions:
                metrics = metric_matrix(pred, data.Y[ids])
                for e, trait in enumerate(data.trait_names):
                    rows.append({'fold': f, 'trait': trait, 'lambda': chosen[trait],
                                 **dict(zip(('pcc', 'rmse', 'mae'), metrics[e].tolist()))})
                    for i, value in zip(ids, pred[:, e]):
                        oof.append({'trait': trait, 'fold': f, 'material_index': int(i),
                                    'genotype_id': str(data.genotype_ids[i]),
                                    'phenotype_id': str(data.phenotype_ids[i]),
                                    'y_true': float(data.Y[i, e]), 'y_pred': float(value)})
            with (args.out / 'oof.csv').open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(oof[0]), lineterminator='\n')
                writer.writeheader()
                writer.writerows(oof)
            write('per_fold.json', rows)
            write('summary.json', {
                'primary': {m: float(np.mean([r[m] for r in rows]))
                            for m in ('pcc', 'rmse', 'mae')},
                'per_trait': {t: {m: float(np.mean([r[m] for r in rows if r['trait'] == t]))
                                  for m in ('pcc', 'rmse', 'mae')} for t in data.trait_names},
                'n_trait_folds': len(rows), 'oof_rows': len(oof),
                'scope': 'Frozen same-material outer comparison; no test selection'})
        else:
            write('summary.json', {'n_trait_folds': len(selection),
                                  'macro_development_pcc': float(np.mean(
                                      [r['inner_mean_metrics']['pcc'] for r in selection])),
                                  'scope': 'Conditional inner validation only; no test predictions'})
        state['status'] = 'complete'
        write('execution_status.json', state)
        write('timings.json', {'seconds': time.perf_counter() - start})
        write('failures.json', failures)
    except Exception as error:
        failures.append({'stage': state['phase'], 'reason': repr(error)})
        state['status'] = 'failed'
        write('execution_status.json', state)
        write('failures.json', failures)
        raise
