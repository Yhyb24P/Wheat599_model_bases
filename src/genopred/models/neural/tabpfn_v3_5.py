"""TabPFN-3.5 frozen-ICL adapter (real inference).

The primary pretrained tabular prior (RESEARCH_PLAN §3, hard boundary: no
fine-tuning, fixed checkpoint). tabpfn==9.2.0, ModelVersion.V3_5,
checkpoint tabpfn-v3.5-20260909.safetensors. Two categorization modes
(core.yaml ``sensitivity.tabpfn_categorization``):
  * ``auto_default``      — categorical_features_indices=None; TabPFN infers
                            categorical columns (binary markers become
                            categorical via MIN_UNIQUE_FOR_NUMERICAL_FEATURES=4).
  * ``forced_numerical``  — categorical_features_indices=[] so binary markers
                            are treated as numeric.
The main result fixes ``auto_default``; ``forced_numerical`` is a separate
ablation, never mixed into the main result. No hyperparameter tuning (frozen ICL).

Device is recorded per run (cuda/cpu); GPU is not required (CPU fallback works).
Weights are not vendored; they are fetched by tabpfn on first use and SHA-verified
by the lead before any formal run.
"""

from __future__ import annotations

import numpy as np

from ..base import Adapter, ParamGrid, register


@register
class TabPFNAdapter(Adapter):
    name = "tabpfn_v3_5"

    @staticmethod
    def check_deps() -> None:
        try:
            import torch  # noqa: F401
            import tabpfn  # noqa: F401
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise ImportError(
                "tabpfn_v3_5 requires torch + tabpfn==9.2.0 in the wheat599-bench "
                "env (see env/requirements-bench.txt)"
            ) from exc

    def param_grid(self, smoke: bool = False) -> ParamGrid:
        # Frozen ICL: no tuning. The only knob is the categorization mode, a
        # sensitivity ablation, not a tuned param. Main result: auto_default.
        return ParamGrid({"categorization": ["auto_default"]})

    def fit(self, X: np.ndarray, y: np.ndarray, params: dict) -> "TabPFNAdapter":
        self.check_deps()
        import torch
        from tabpfn import TabPFNRegressor
        from tabpfn.constants import ModelVersion

        mode = params.get("categorization", "auto_default")
        self._mode = mode
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device
        cat_idx = None if mode == "auto_default" else []
        self._model = TabPFNRegressor.create_default_for_version(
            ModelVersion.V3_5,
            device=device,
            categorical_features_indices=cat_idx,
        )
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self._model.predict(X), dtype=np.float64)
