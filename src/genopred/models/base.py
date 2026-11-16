"""Base adapter protocol and registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ParamGrid:
    """A small hyperparameter grid. Each key maps to a list of candidate values.

    For SMOKE_ONLY runs the grid is truncated to its first value per key, so the
    pipeline exercises the code path without a full search.
    """

    grid: dict[str, list[Any]]

    def smoke(self) -> "ParamGrid":
        """First value per key — the SMOKE_ONLY grid."""
        return ParamGrid({k: [v[0]] for k, v in self.grid.items()})

    def __iter__(self):
        return iter(self.grid.items())

    def keys(self):
        return self.grid.keys()


@dataclass
class Adapter:
    """A fold-local model adapter.

    Subclasses set ``name`` and implement ``fit``/``predict``/``param_grid``.
    ``fit_scope`` is always ``'outer_train'`` — an adapter must never fit on
    outer-valid or full-sample material.
    """

    name: str = "base"
    fit_scope: str = "outer_train"

    def param_grid(self, smoke: bool = False) -> ParamGrid:  # pragma: no cover
        raise NotImplementedError

    def fit(self, X: np.ndarray, y: np.ndarray, params: dict) -> "Adapter":  # pragma: no cover
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    @staticmethod
    def check_deps() -> None:
        """Raise ImportError if the adapter's runtime deps are missing."""


REGISTRY: dict[str, type[Adapter]] = {}


def register(cls: type[Adapter]) -> type[Adapter]:
    REGISTRY[cls.name] = cls
    return cls


def get_adapter(name: str) -> Adapter:
    if name not in REGISTRY:
        raise KeyError(f"unknown adapter {name!r}; registered: {sorted(REGISTRY)}")
    return REGISTRY[name]()
