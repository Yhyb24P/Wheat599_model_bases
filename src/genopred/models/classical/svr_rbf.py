"""RBF-SVR adapter — sklearn SVR.

Traditional strong baseline (RESEARCH_PLAN §3). C / gamma / epsilon are selected
in the inner 3-fold. ``gamma='scale'`` is the sklearn sentinel (1/(n_features *
X.var())) and is kept as a distinct grid entry; the parser resolves it inside
fit. All selection is fold-local.
"""

from __future__ import annotations

import numpy as np

from ..base import Adapter, ParamGrid, register


@register
class SVRRBFAdapter(Adapter):
    name = "svr_rbf"

    def param_grid(self, smoke: bool = False) -> ParamGrid:
        grid = {
            "C": [0.1, 1.0, 10.0, 100.0],
            "gamma": ["scale", 0.001, 0.01, 0.1],
            "epsilon": [0.01, 0.05, 0.1],
        }
        pg = ParamGrid(grid)
        return pg.smoke() if smoke else pg

    def fit(self, X: np.ndarray, y: np.ndarray, params: dict) -> "SVRRBFAdapter":
        from sklearn.svm import SVR

        self._model = SVR(
            C=float(params.get("C", 1.0)),
            kernel="rbf",
            gamma=params.get("gamma", "scale"),
            epsilon=float(params.get("epsilon", 0.1)),
        )
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)
