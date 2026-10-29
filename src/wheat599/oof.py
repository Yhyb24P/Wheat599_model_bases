"""Out-of-fold prediction container.

Each material receives exactly one prediction per (repeat, fold, environment)
when the full CV completes. We store OOF predictions as a flat array indexed by
material, plus a provenance dict, so metrics can be recomputed from OOF alone
(RESEARCH_PLAN §4.5) without re-running models.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class OOF:
    """Out-of-fold predictions for one environment across one repeat."""

    environment: str
    y_true: np.ndarray  # (n,) observed yields
    y_pred: np.ndarray  # (n,) out-of-fold predictions
    repeat: int
    fold_of: np.ndarray  # (n,) which outer fold produced each prediction
    provenance: dict = field(default_factory=dict)

    def metrics(self) -> dict[str, float]:
        from .metrics import compute_all

        return compute_all(self.y_true, self.y_pred)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            environment=self.environment,
            y_true=self.y_true,
            y_pred=self.y_pred,
            repeat=self.repeat,
            fold_of=self.fold_of,
            provenance=json.dumps(self.provenance),
        )

    @classmethod
    def load(cls, path: Path) -> "OOF":
        blob = np.load(Path(path), allow_pickle=False)
        return cls(
            environment=str(blob["environment"]),
            y_true=blob["y_true"],
            y_pred=blob["y_pred"],
            repeat=int(blob["repeat"]),
            fold_of=blob["fold_of"],
            provenance=json.loads(str(blob["provenance"])),
        )
