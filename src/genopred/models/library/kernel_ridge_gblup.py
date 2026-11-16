"""Kernel ridge / GBLUP adapter — numpy closed form.

GBLUP with the marker-based kernel G = X X^T / p. This is the same kernel-ridge
machinery as RR-BLUP in the GBLUP parameterization; the lead confirmed one
implementation covers both. lambda is selected in the inner 3-fold.

Closed form: alpha = (G + lambda I)^{-1} y, predictions = K_test_train @ alpha.
G is (n_train x n_train); K_test_train is (n_test x n_train).
"""

from __future__ import annotations

import numpy as np

from .base import Adapter, ParamGrid, register


def _kernel_gblup(X: np.ndarray) -> np.ndarray:
    p = X.shape[1]
    return (X @ X.T) / p


@register
class KernelRidgeGBLUPAdapter(Adapter):
    name = "kernel_ridge_gblup"

    def param_grid(self, smoke: bool = False) -> ParamGrid:
        grid = {
            "lambda": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
        }
        pg = ParamGrid(grid)
        return pg.smoke() if smoke else pg

    def fit(self, X: np.ndarray, y: np.ndarray, params: dict) -> "KernelRidgeGBLUPAdapter":
        lam = float(params.get("lambda", 1.0))
        G = _kernel_gblup(X)
        n = G.shape[0]
        A = G + lam * np.eye(n)
        self._alpha = np.linalg.solve(A, y)
        self._X_train = X
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        K = _kernel_gblup_cross(self._X_train, X)
        return K.T @ self._alpha


def _kernel_gblup_cross(Xa: np.ndarray, Xb: np.ndarray) -> np.ndarray:
    """Cross kernel K[i,j] = Xa[i] . Xb[j] / p (GBLUP marker kernel)."""
    p = Xa.shape[1]
    return (Xa @ Xb.T) / p
