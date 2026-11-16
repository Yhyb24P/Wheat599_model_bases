"""Model adapter framework for the genomic prediction benchmark.

Each adapter is a fold-local estimator: it fits only on the outer-training
materials handed to it and declares ``fit_scope='outer_train'``. Adapters are
registered by name in ``REGISTRY``. The numpy/sklearn adapters (classical
linear/kernel_ridge_gblup/svr_rbf, kernels rkhs_gaussian) are dependency-light
and run in the base env. The torch-based adapters (neural tabm/ft_transformer/
tabpfn_v3_5) are imported lazily and skipped cleanly when their deps are absent.
"""

from __future__ import annotations

from .base import Adapter, ParamGrid, REGISTRY, get_adapter, register
from .classical.linear import RidgeAdapter
from .classical.kernel_ridge_gblup import KernelRidgeGBLUPAdapter
from .kernels.rkhs_gaussian import GaussianRKHSAdapter
from .classical.svr_rbf import SVRRBFAdapter

for _cls in (RidgeAdapter, KernelRidgeGBLUPAdapter, GaussianRKHSAdapter, SVRRBFAdapter):
    register(_cls)

for _lazy in ("neural.ft_transformer", "neural.tabm", "neural.tabpfn_v3_5"):
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