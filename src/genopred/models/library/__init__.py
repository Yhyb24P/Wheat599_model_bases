"""Model adapter framework for the Wheat599 benchmark (M1).

Each adapter is a fold-local estimator: it fits only on the outer-training
materials handed to it and declares its fit scope through the contracts module
(``fit_scope='outer_train'``). The inner 3-fold model selection loop lives in
``run_benchmark``; an adapter exposes ``fit(X_train, y_train, params)`` and
``predict(X)``.

Adapters are registered by name in ``REGISTRY``. The four numpy/sklearn adapters
(linear, kernel_ridge_gblup, rkhs_gaussian, svr_rbf) are dependency-light and
run in the base env. The torch-based adapters (tabm, ft_transformer,
tabpfn_v3_5) are imported lazily and skipped cleanly when their deps are
absent, so the SMOKE_ONLY pipeline degrades gracefully.
"""

from __future__ import annotations

from .base import Adapter, ParamGrid, REGISTRY, get_adapter, register
from .linear import RidgeAdapter
from .kernel_ridge_gblup import KernelRidgeGBLUPAdapter
from .rkhs_gaussian import GaussianRKHSAdapter
from .svr_rbf import SVRRBFAdapter

# Register the dependency-light adapters eagerly.
for _cls in (RidgeAdapter, KernelRidgeGBLUPAdapter, GaussianRKHSAdapter, SVRRBFAdapter):
    register(_cls)

# Torch-based adapters are registered lazily so importing this package never
# fails when torch/tabpfn/rtdl/tabm are absent (base env). They self-register
# only if their deps import; otherwise available_adapters() omits them.
for _lazy in ("ft_transformer", "tabm", "tabpfn_v3_5"):
    try:
        __import__(f"{__name__}.{_lazy}")
    except Exception:
        pass


def available_adapters() -> list[str]:
    """Names of adapters whose runtime deps are importable right now."""
    names = []
    for name in REGISTRY:
        try:
            get_adapter(name).check_deps()
            names.append(name)
        except Exception:
            continue
    return names


__all__ = [
    "Adapter",
    "ParamGrid",
    "REGISTRY",
    "get_adapter",
    "register",
    "available_adapters",
]
