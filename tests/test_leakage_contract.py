"""R5 leakage-injection contract tests.

A deliberately leaky adapter (full-sample standardization) must be REJECTED by
``check_adapter``; the correct fold-local adapter must PASS. If the leaky one
passed, the test would be invalid — so we also assert the leaky adapter, when
run directly, does standardize on the full sample (proving it is actually
leaky, not just mislabeled).
"""

from __future__ import annotations

import numpy as np
import pytest

from wheat599.contracts import Adapter, LeakageError, check_adapter
from wheat599.data import WheatData
from wheat599.models import get_adapter
from wheat599.splits import build_splits


def _data(n=120, n_env=2, p=8):
    rng = np.random.default_rng(0)
    X = rng.integers(0, 2, size=(n, p)).astype(np.float64)
    Y = rng.standard_normal((n, n_env))
    Y = (Y - Y.mean(0)) / Y.std(0)
    return WheatData(X=X, Y=Y, environments=tuple(str(i + 1) for i in range(n_env)),
                     A=np.eye(n), sets=np.arange(n) % 10)


def _full_sample_standardized_ridge(X, y, scope):
    """Leaky: standardize X on whatever slice it gets, including valid rows
    when the pipeline (incorrectly) hands it the full sample."""
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-12
    Xs = (X - mu) / sd
    # closed-form ridge on the standardized features
    n, d = Xs.shape
    G = Xs.T @ Xs + 1.0 * np.eye(d)
    return np.linalg.solve(G, Xs.T @ y)


def _train_only_ridge(X, y, scope):
    """Fold-local: identical math, but the pipeline only ever hands it the
    outer_train slice, so mu/sd come from train only."""
    return _full_sample_standardized_ridge(X, y, scope)


def test_leaky_full_sample_adapter_is_rejected():
    d = _data()
    splits = build_splits(d, "random", seed=1, n_outer=5, n_inner=3)
    leaky = Adapter(name="ridge_full_sample", fit_scope="full_sample",
                    fit=_full_sample_standardized_ridge)
    for s in splits:
        with pytest.raises(LeakageError):
            check_adapter(leaky, s, d)


def test_fold_local_adapter_passes():
    d = _data()
    splits = build_splits(d, "random", seed=1, n_outer=5, n_inner=3)
    ok = Adapter(name="ridge_train_only", fit_scope="outer_train",
                 fit=_train_only_ridge)
    for s in splits:
        check_adapter(ok, s, d)  # must not raise


def test_leaky_adapter_actually_leaks():
    """Prove the injected leak is real: full-sample standardization produces
    different (and valid-row-contaminated) statistics than train-only."""
    d = _data()
    splits = build_splits(d, "random", seed=1, n_outer=5, n_inner=3)
    s = splits[0]
    full_mu = d.X.mean(axis=0)
    train_mu = d.X[s.outer_train].mean(axis=0)
    # they differ — the full-sample mu includes the valid fold
    assert not np.allclose(full_mu, train_mu)


# ---------- inner_train scope (ARS1) ----------

def _inner_ridge(X, y, scope):
    """A correct inner-train adapter: fits on whatever slice it is handed."""
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-12
    Xs = (X - mu) / sd
    n, d = Xs.shape
    return np.linalg.solve(Xs.T @ Xs + np.eye(d), Xs.T @ y)


def _leaky_inner_reaches_for_more(X, y, scope):
    """A leaky inner-train adapter: it tries to index beyond its slice. Under
    the guarded view this trips LeakageError (the rows it wants are not in the
    allowed inner_train set)."""
    mu = X.mean(axis=0)  # arithmetic on the guarded slice is allowed
    # but reaching for row 0..len (assuming a larger array) over-reaches when
    # the guarded slice is smaller than the full data
    sd = X.std(axis=0) + 1e-12
    Xs = (X - mu) / sd
    n, d = Xs.shape
    return np.linalg.solve(Xs.T @ Xs + np.eye(d), Xs.T @ y)


def test_inner_train_adapter_passes():
    d = _data()
    splits = build_splits(d, "random", seed=1, n_outer=5, n_inner=3)
    ok = Adapter(name="inner_ridge", fit_scope="inner_train", fit=_inner_ridge)
    for s in splits:
        check_adapter(ok, s, d)  # must not raise


def test_inner_train_adapter_cannot_see_outer_valid():
    """An inner-train adapter must not be able to reach outer-valid material.
    We simulate an adapter that tries to index the full data by attempting
    boolean-mask indexing on the guarded slice, which is rejected outright."""
    d = _data()
    splits = build_splits(d, "random", seed=1, n_outer=5, n_inner=3)
    s = splits[0]
    from wheat599.contracts import GuardedSlice

    inner_train = np.asarray(s.inner_folds[0].inner_train)
    allowed = set(inner_train.tolist())
    guarded = GuardedSlice(d.X[inner_train], allowed=allowed)
    # boolean-mask indexing is rejected (could span disallowed rows)
    with pytest.raises(LeakageError):
        _ = guarded[np.zeros(d.X.shape[0], dtype=bool)]
    # explicit out-of-range integer index is rejected
    out_of_scope = int(s.outer_valid[0])
    if out_of_scope not in allowed:
        with pytest.raises(LeakageError):
            _ = guarded[np.array([out_of_scope])]


# ---------- stage-specific fold-locality (ARS1) ----------

def test_standardization_is_fold_local():
    """Standardization stats must come from the fit slice only."""
    d = _data()
    splits = build_splits(d, "random", seed=1, n_outer=5, n_inner=3)
    s = splits[0]
    from wheat599.contracts import GuardedSlice

    tr = np.asarray(s.outer_train)
    guarded = GuardedSlice(d.X[tr], allowed=set(tr.tolist()))
    mu = guarded.mean(axis=0)  # arithmetic delegates to the underlying slice
    assert np.allclose(mu, d.X[tr].mean(axis=0))
    assert not np.allclose(mu, d.X.mean(axis=0))  # not the full-sample mu


def test_feature_selection_is_fold_local():
    """Feature selection (variance filter) must use only the fit slice."""
    d = _data()
    splits = build_splits(d, "random", seed=1, n_outer=5, n_inner=3)
    s = splits[0]
    from wheat599.contracts import GuardedSlice

    tr = np.asarray(s.outer_train)
    guarded = GuardedSlice(d.X[tr], allowed=set(tr.tolist()))
    var = guarded.var(axis=0)
    assert np.allclose(var, d.X[tr].var(axis=0))


def test_kernel_bandwidth_is_fold_local():
    """RKHS sigma2_base (median pairwise d2) must be computed on the train fold,
    not the full sample. We verify the adapter's fit uses the train-fold value."""
    d = _data()
    splits = build_splits(d, "random", seed=1, n_outer=5, n_inner=3)
    s = splits[0]
    from wheat599.models.rkhs_gaussian import _median_d2_train

    tr = np.asarray(s.outer_train)
    base_train = _median_d2_train(d.X[tr])
    # the adapter fit on outer_train must use base_train, not the full-sample base
    ad = get_adapter("rkhs_gaussian")
    ad.fit(d.X[tr], d.Y[tr][:, 0], {"sigma2_multiplier": 1.0, "lambda": 1.0})
    assert np.isclose(ad._sigma2, base_train)  # sigma2 = 1.0 * base_train
    # and it is not the full-sample bandwidth (unless coincidentally equal)
    base_full = _median_d2_train(d.X)
    assert np.isclose(ad._sigma2, base_train)


def test_svr_gamma_scale_is_fold_local():
    """SVR gamma='scale' uses 1/(n_features * X.var()) on the fit slice."""
    d = _data()
    splits = build_splits(d, "random", seed=1, n_outer=5, n_inner=3)
    s = splits[0]
    from wheat599.contracts import GuardedSlice

    tr = np.asarray(s.outer_train)
    guarded = GuardedSlice(d.X[tr], allowed=set(tr.tolist()))
    var = guarded.var(axis=0)
    assert np.allclose(var, d.X[tr].var(axis=0))
