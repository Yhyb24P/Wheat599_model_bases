"""Evaluation metrics (docs/history/RESEARCH_PLAN.md §4.3).

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


# --- classification metrics (SoyBase qualitative traits) ---

def balanced_accuracy(truth: np.ndarray, pred: np.ndarray) -> float:
    """Mean of per-class recall (sklearn balanced_accuracy_score)."""
    from sklearn.metrics import balanced_accuracy_score
    truth, pred = _check(truth, pred)
    return float(balanced_accuracy_score(truth.astype(int), pred.astype(int)))


def macro_f1(truth: np.ndarray, pred: np.ndarray) -> float:
    from sklearn.metrics import f1_score
    truth, pred = _check(truth, pred)
    return float(f1_score(truth.astype(int), pred.astype(int), average='macro'))


def compute_classification(truth: np.ndarray, pred: np.ndarray, n_classes: int) -> dict:
    """BA, macro-F1, per-class precision/recall/f1/support, confusion matrix."""
    from sklearn.metrics import (balanced_accuracy_score, confusion_matrix,
                                 f1_score, precision_recall_fscore_support)
    truth, pred = _check(truth, pred)
    truth, pred = truth.astype(int), pred.astype(int)
    labels = list(range(n_classes))
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, pred, labels=labels, zero_division=0)
    return {
        'balanced_accuracy': float(balanced_accuracy_score(truth, pred)),
        'macro_f1': float(f1_score(truth, pred, labels=labels, average='macro', zero_division=0)),
        'per_class': {
            str(c): {'precision': float(precision[c]), 'recall': float(recall[c]),
                     'f1': float(f1[c]), 'support': int(support[c])}
            for c in labels
        },
        'confusion_matrix': confusion_matrix(truth, pred, labels=labels).tolist(),
    }
