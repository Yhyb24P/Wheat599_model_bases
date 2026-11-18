"""SoyBase regression: fit-only preprocessing, joint splits, NaN-masked nested loop."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from genopred.data.soybase import make_regression_splits, REG_TRAITS
from genopred.experiments.soybase.regression import fit_only_preprocess


def test_fit_only_preprocess_uses_fit_rows_only():
    rng = np.random.default_rng(0)
    X = rng.integers(0, 3, (40, 6)).astype(np.float32)
    X[:, 0] = np.nan                      # all-missing -> dropped
    X[:, 1] = 0.0
    X[0, 2] = np.nan
    X[30:, 2] = np.nan                    # test-only missingness pattern
    fit = np.arange(30)
    out = fit_only_preprocess(X, fit, 0.9, 0.05)
    assert out.shape[1] < 6
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out[fit].mean(0), 0.0, atol=1e-5)
    np.testing.assert_allclose(out[fit].std(0), 1.0, atol=1e-5)
    # changing test rows must not change the preprocessing of fit rows
    X2 = X.copy()
    X2[30:, 3] = 2.0
    out2 = fit_only_preprocess(X2, fit, 0.9, 0.05)
    np.testing.assert_array_equal(out[fit], out2[fit])


def test_joint_splits_cover_all_and_are_label_independent():
    rng = np.random.default_rng(1)
    Y = rng.normal(size=(60, 7))
    Y[rng.random((60, 7)) < 0.3] = np.nan
    data = SimpleNamespace(X=np.zeros((60, 5), dtype=np.float32), Y=Y)
    folds, info = make_regression_splits(data, n_splits=5, seed=42, inner_splits=3)
    covered = sorted(i for f in folds for i in f['test_idx'])
    assert covered == list(range(60))
    for f in folds:
        assert not set(f['dev_idx']) & set(f['test_idx'])
        for v in f['inner']:
            assert not set(v['train_idx']) & set(v['val_idx'])
            assert set(v['train_idx']) | set(v['val_idx']) == set(f['dev_idx'])
    Y2 = Y.copy()
    Y2[:] = np.nan
    folds2, _ = make_regression_splits(SimpleNamespace(X=data.X, Y=Y2), n_splits=5, seed=42, inner_splits=3)
    assert folds == folds2
    assert info['traits'] == list(REG_TRAITS)


@pytest.mark.parametrize('missing', [False, True])
def test_nested_classical_kernel_models_and_fit_sets(tmp_path, missing):
    from genopred.experiments.wheat2000.classical import nested_classical
    rng = np.random.default_rng(2)
    n, p = 48, 8
    X = rng.normal(size=(n, p)).astype(np.float32)
    beta = rng.normal(size=(p, 2))
    Y = X @ beta + rng.normal(scale=0.5, size=(n, 2))
    # deterministic missingness: keep every inner split well-conditioned
    if missing:
        Y[::7, 0] = np.nan
        Y[3::11, 1] = np.nan
    data = SimpleNamespace(X=X, Y=Y)
    folds, _ = make_regression_splits(SimpleNamespace(X=X, Y=np.hstack([Y, np.zeros((n, 5))])),
                                      n_splits=3, seed=42, inner_splits=2)
    # min_samples_leaf-style guards keep tiny smoke fits from degenerating to
    # constant predictions (undefined PCC); real grids run on thousands of rows
    cfg = {'search': {'rf': {'n_estimators': [40], 'max_features': [1.0],
                             'min_samples_leaf': [1]}},
           'seeds': [42], 'jobs': 1, 'splits_sha256': 'x', 'data_sha256': 'x',
           'cache_sha256': 'x', 'source_git_commit': 'x',
           'bandwidths': [1.0], 'ridge': [0.1, 1.0], 'additive_fractions': [1.0]}
    fit_sets = []

    def prepare(X, fit):
        fit_sets.append(fit.tolist())
        return X

    labeled = np.isfinite(Y) if missing else None
    options = dict(sources={}, config_sha='x', prepare=prepare, labeled=labeled)
    args = SimpleNamespace(out=tmp_path / 'sel', selection=None, models=None, config=None)
    nested_classical(cfg, args, data, folds, ['t0', 't1'], **options)
    assert fit_sets == [v['train_idx'] for f in folds for v in f['inner']]
    sel = json.loads((args.out / 'selection.json').read_text())
    assert len(sel['selected']) == 3 * 2 * (1 + 2)  # folds x traits x (lgbm + 2 kernel models)
    args2 = SimpleNamespace(out=tmp_path / 'outer', selection=args.out / 'selection.json',
                            models=None, config=None)
    fit_sets.clear()
    nested_classical(cfg, args2, data, folds, ['t0', 't1'], **options)
    assert fit_sets == [f['dev_idx'] for f in folds]
    summary = json.loads((args2.out / 'summary.json').read_text())
    assert summary['n_runs'] == 3 * 2 * 3
    import csv
    with (args2.out / 'oof.csv').open() as handle:
        oof = list(csv.DictReader(handle))
    for row in oof:
        assert np.isfinite(float(row['y_true'])) and np.isfinite(float(row['y_pred']))
    assert ([r['y_pred'] for r in oof if r['model'] == 'gaussian']
            != [r['y_pred'] for r in oof if r['model'] == 'additive_rbf'])
    for t, e in (('t0', 0), ('t1', 1)):
        idx = sorted(int(r['material_index']) for r in oof if r['model'] == 'rf' and r['trait'] == t)
        assert idx == np.flatnonzero(np.isfinite(Y[:, e])).tolist()
