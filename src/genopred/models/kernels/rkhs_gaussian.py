"""Gaussian RKHS adapter — numpy closed form.

Kernel K = exp(-d2 / (2 sigma2)), where d2 is the mean squared marker difference
between material pairs. sigma2_base is the median pairwise d2 *on the training
fold* (never the full sample); the grid multiplies it by sigma2_multipliers.
lambda is selected in the inner 3-fold. bandwidth and lambda stay fold-local.

Closed form mirrors the GBLUP adapter: alpha = (K + lambda I)^{-1} y.
"""

from __future__ import annotations

import numpy as np

from ..base import Adapter, ParamGrid, register


def _pairwise_d2(A: np.ndarray, B: np.ndarray | None = None) -> np.ndarray:
    """Mean squared marker difference.

    For A (n, p) vs B (m, p): d2[i,j] = mean_k (A[i,k] - B[j,k])^2.
    If B is None, B = A (pairwise within A).
    """
    if B is None:
        B = A
    # (n, 1, p) - (1, m, p) -> (n, m, p); mean over p
    diff = A[:, None, :] - B[None, :, :]
    return np.mean(diff ** 2, axis=2)


def _median_d2_train(X: np.ndarray) -> float:
    d2 = _pairwise_d2(X)
    iu = np.triu_indices(X.shape[0], k=1)
    return float(np.median(d2[iu]))


@register
class GaussianRKHSAdapter(Adapter):
    name = "rkhs_gaussian"

    def param_grid(self, smoke: bool = False) -> ParamGrid:
        grid = {
            "sigma2_multiplier": [0.25, 0.5, 1.0, 2.0, 4.0],
            "lambda": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        }
        pg = ParamGrid(grid)
        return pg.smoke() if smoke else pg

    def fit(self, X: np.ndarray, y: np.ndarray, params: dict) -> "GaussianRKHSAdapter":
        mult = float(params.get("sigma2_multiplier", 1.0))
        lam = float(params.get("lambda", 1.0))
        # sigma2_base computed on the TRAINING fold only (fold-local bandwidth)
        self._sigma2 = mult * _median_d2_train(X)
        K = np.exp(-_pairwise_d2(X) / (2.0 * self._sigma2))
        n = K.shape[0]
        A = K + lam * np.eye(n)
        self._alpha = np.linalg.solve(A, y)
        self._X_train = X
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        d2 = _pairwise_d2_cross(self._X_train, X)
        K = np.exp(-d2 / (2.0 * self._sigma2))
        return K.T @ self._alpha


def _pairwise_d2_cross(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """d2[i,j] = mean_k (A[i,k] - B[j,k])^2, shape (n_A, n_B)."""
    diff = A[:, None, :] - B[None, :, :]
    return np.mean(diff ** 2, axis=2)
