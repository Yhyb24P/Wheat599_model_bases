"""TabM adapter — yandex-research/tabm (skeleton + lazy import).

A strong non-attention neural baseline (RESEARCH_PLAN §3). k members in {8,16},
width in {64,128}, AdamW, early stopping on inner valid. Depends on torch +
the tabm package; skipped cleanly when absent.
"""

from __future__ import annotations

import numpy as np

from ..base import Adapter, ParamGrid, register


@register
class TabMAdapter(Adapter):
    name = "tabm"

    @staticmethod
    def check_deps() -> None:
        try:
            import torch  # noqa: F401
            import tabm  # noqa: F401
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise ImportError(
                "tabm requires torch + yandex-research/tabm in the wheat599-bench "
                "env (see env/requirements-bench.txt)"
            ) from exc

    def param_grid(self, smoke: bool = False) -> ParamGrid:
        grid = {
            "k": [8, 16],
            "width": [64, 128],
            "learning_rate": [0.001, 0.003],
            "weight_decay": [0.00001, 0.0001],
            "max_epochs": [200],
        }
        pg = ParamGrid(grid)
        return pg.smoke() if smoke else pg

    def fit(self, X: np.ndarray, y: np.ndarray, params: dict) -> "TabMAdapter":
        self.check_deps()
        # Real tabm training loop (batch ensemble, AdamW, inner-valid early
        # stopping) is wired when the bench env is live. Skeleton records inputs.
        self._X_train = X
        self._y_train = y
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Skeleton placeholder; real forward lands with the bench env.
        return np.full(X.shape[0], float(np.mean(self._y_train)), dtype=np.float64)
