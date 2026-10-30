"""Real-output tests for the torch-based adapters (FT-Transformer, TabPFN).

These require torch + rtdl-revisiting-models + tabpfn (the wheat599-bench env).
They are skipped cleanly when those deps are absent. They verify the adapters
produce REAL outputs (not train-mean placeholders): FT-T can overfit a small
synthetic set, TabPFN predictions are not constant, and the runner end-to-end
smoke produces non-mean predictions.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
rtdl = pytest.importorskip("rtdl_revisiting_models")
tabpfn = pytest.importorskip("tabpfn")

from wheat599.models import get_adapter


def _synthetic(n=120, p=10, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 2, size=(n, p)).astype(np.float64)
    y = X @ rng.normal(0, 1, p) + rng.normal(0, 0.1, n)
    y = (y - y.mean()) / y.std()
    return X, y


def test_ft_transformer_overfits_synthetic():
    """FT-Transformer must be able to overfit a small synthetic set (loss drops
    well below the variance of y), proving it is not a train-mean placeholder."""
    X, y = _synthetic(n=120, p=10, seed=1)
    ad = get_adapter("ft_transformer")
    params = {"d_block": 16, "n_blocks": 1, "dropout": 0.0,
              "learning_rate": 0.003, "weight_decay": 0.0,
              "max_epochs": 60, "patience": 60, "batch_size": 32}
    ad.fit(X, y, params)
    pred = ad.predict(X)
    assert pred.shape == y.shape
    assert np.isfinite(pred).all()
    # not constant
    assert pred.std() > 1e-3
    # overfit: training PCC high
    from wheat599.metrics import pcc
    assert pcc(y, pred) > 0.8


def test_ft_transformer_predict_non_mean():
    """Predictions must differ from the train mean (not a placeholder)."""
    X, y = _synthetic(n=100, p=8, seed=2)
    ad = get_adapter("ft_transformer")
    ad.fit(X, y, {"d_block": 16, "n_blocks": 1, "dropout": 0.0,
                   "learning_rate": 0.003, "weight_decay": 0.0,
                   "max_epochs": 40, "patience": 40, "batch_size": 32})
    pred = ad.predict(X)
    assert abs(float(pred.mean()) - float(y.mean())) < 1.0  # roughly calibrated
    assert pred.std() > 1e-3  # but not constant


def test_tabpfn_predict_non_constant():
    """TabPFN frozen-ICL predictions must not be constant."""
    X, y = _synthetic(n=100, p=8, seed=3)
    ad = get_adapter("tabpfn_v3_5")
    try:
        ad.fit(X, y, {"categorization": "auto_default"})
    except Exception as exc:  # noqa: BLE001 - license is an external prerequisite
        if type(exc).__name__ == "TabPFNLicenseError":
            pytest.skip(f"TabPFN token/weights unavailable: {exc}")
        raise
    pred = ad.predict(X)
    assert pred.shape == y.shape
    assert np.isfinite(pred).all()
    assert pred.std() > 1e-4  # not constant
