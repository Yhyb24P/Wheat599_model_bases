"""Linear shrinkage (Ridge) adapter — sklearn Ridge.

The primary linear baseline (RESEARCH_PLAN §3). Fit scope is outer_train only;
alpha is selected in the inner 3-fold by the benchmark loop.
"""

from __future__ import annotations

import numpy as np

from .base import Adapter, ParamGrid, register


@register
class RidgeAdapter(Adapter):
    name = "linear"

    def param_grid(self, smoke: bool = False) -> ParamGrid:
        grid = {
            "alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
            "fit_intercept": [True],
        }
        pg = ParamGrid(grid)
        return pg.smoke() if smoke else pg

    def fit(self, X: np.ndarray, y: np.ndarray, params: dict) -> "RidgeAdapter":
        from sklearn.linear_model import Ridge

        self._model = Ridge(
            alpha=float(params.get("alpha", 1.0)),
            fit_intercept=bool(params.get("fit_intercept", True)),
        )
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)
