"""Fault-isolation tests for the Stage-1 runner (2026-10-29 incident).

Per-candidate / per-fold / per-model-seed isolation: a failing candidate is
skipped and recorded; a fold whose refit fails is left NaN and recorded; the
benchmark never aborts as a whole. CPU-only, synthetic data, no torch.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from wheat599.models.base import Adapter, ParamGrid
from wheat599.splits import build_splits
from scripts.run_stage1 import run_adapter_oof  # noqa: E402


def _fake_data(n=60, n_env=2, p=6, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 2, size=(n, p)).astype(np.float64)
    Y = rng.standard_normal((n, n_env))
    Y = (Y - Y.mean(0)) / Y.std(0)
    return SimpleNamespace(X=X, Y=Y, n=n, n_env=n_env,
                           environments=tuple(str(i + 1) for i in range(n_env)))


class _Flaky(Adapter):
    name = "flaky"
    calls = {"n": 0}

    def param_grid(self, smoke=False):
        return ParamGrid({"a": [1, 2]})

    def fit(self, X, y, params):
        type(self).calls["n"] += 1
        if params.get("a") == 2 and type(self).calls["n"] % 2 == 0:
            raise RuntimeError("boom (injected)")
        self._fill = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(X.shape[0], self._fill)


class _AlwaysFail(Adapter):
    name = "always_fail"

    def param_grid(self, smoke=False):
        return ParamGrid({"a": [1]})

    def fit(self, X, y, params):
        raise RuntimeError("always fails (injected)")

    def predict(self, X):
        return np.zeros(X.shape[0])


def test_flaky_candidate_is_skipped_and_run_completes():
    d = _fake_data()
    splits = build_splits(d, "random", seed=7, n_outer=3, n_inner=2)
    errors: list = []
    preds, fold_of, info = run_adapter_oof(_Flaky(), d, splits, False, 7, errors)
    assert np.isfinite(preds).all()          # a=1 always succeeds -> predictions exist
    assert len(errors) > 0                   # injected failures recorded
    assert all("error" in e for e in errors)
    assert any(e["stage"] == "fit_or_predict" for e in errors)
    assert all(not r.get("failed", False) for r in info["records"])


def test_all_failing_adapter_marks_folds_and_does_not_raise():
    d = _fake_data()
    splits = build_splits(d, "random", seed=7, n_outer=3, n_inner=2)
    errors: list = []
    preds, fold_of, info = run_adapter_oof(_AlwaysFail(), d, splits, False, 7, errors)
    assert np.isnan(preds).all()             # failed folds stay explicitly NaN
    assert len(errors) > 0
    assert any(e["stage"] in ("selection_all_failed", "fold_failed") for e in errors)
    assert all(r.get("failed", False) for r in info["records"])


def test_selection_skips_nan_scoring_candidates():
    class _NanScorer(_Flaky):
        name = "nan_scorer"

        def param_grid(self, smoke=False):
            return ParamGrid({"a": [1, 2]})

        def fit(self, X, y, params):
            if params.get("a") == 2:
                raise FloatingPointError("nan path")
            self._fill = float(np.mean(y))
            return self

    d = _fake_data()
    splits = build_splits(d, "random", seed=3, n_outer=2, n_inner=2)
    errors: list = []
    preds, _, info = run_adapter_oof(_NanScorer(), d, splits, False, 3, errors)
    assert np.isfinite(preds).all()
    assert len(errors) > 0
