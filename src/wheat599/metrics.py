"""Evaluation metrics (docs/RESEARCH_PLAN.md §4.3).

PCC and RMSE are the co-primary metrics; MAE and R2 are reported alongside.
Every metric takes 1-D arrays of equal length (truth, prediction) for one
environment.
"""

from __future__ import annotations

import numpy as np


def _check(truth: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    truth = np.asarray(truth, dtype=np.float64).ravel()
    pred = np.asarray(pred, dtype=np.float64).ravel()
    if truth.shape != pred.shape:
        raise ValueError(f"shape mismatch: {truth.shape} vs {pred.shape}")
    if truth.size == 0:
        raise ValueError("empty arrays")
    return truth, pred


def pcc(truth: np.ndarray, pred: np.ndarray) -> float:
    """Pearson correlation coefficient."""
    truth, pred = _check(truth, pred)
    if truth.std() == 0 or pred.std() == 0:
        return float("nan")
    return float(np.corrcoef(truth, pred)[0, 1])


def rmse(truth: np.ndarray, pred: np.ndarray) -> float:
    truth, pred = _check(truth, pred)
    return float(np.sqrt(np.mean((truth - pred) ** 2)))


def mae(truth: np.ndarray, pred: np.ndarray) -> float:
    truth, pred = _check(truth, pred)
    return float(np.mean(np.abs(truth - pred)))


def r2(truth: np.ndarray, pred: np.ndarray) -> float:
    """Coefficient of determination (1 - SSE/SST)."""
    truth, pred = _check(truth, pred)
    sst = np.sum((truth - truth.mean()) ** 2)
    if sst == 0:
        return float("nan")
    return float(1.0 - np.sum((truth - pred) ** 2) / sst)


ALL_METRICS = (pcc, rmse, mae, r2)


def compute_all(truth: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {fn.__name__: fn(truth, pred) for fn in ALL_METRICS}
