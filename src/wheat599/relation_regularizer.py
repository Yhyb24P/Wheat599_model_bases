"""Exact material-relation regularizer for standardized multi-environment predictions.

For fit materials the kernel is the standardized additive/RBF mixture used by
``kernel_baselines.kernels``.  With

    H = I - 11^T/n
    B = 0.5 I + 0.5 corr(Y_fit)
    M = mean over model members and quantile points

the penalty is

    R = tr(B^{-1} (HM)^T (H K H + kappa I)^{-1} HM) / (n * environments).

All solves use full Cholesky factorizations in float64 internally; no random-batch
approximation is made, and the scalar is returned in the input dtype.  ``M`` is already on the standardized phenotype scale.  The function keeps
the autograd graph through ``M`` and does not change module modes or dropout;
callers own deterministic evaluation and gradient retention.
"""
from __future__ import annotations

import numpy as np
import torch

from wheat599.kernel_baselines import kernels

DEFAULT_KAPPA = 0.1


def build_kernel(x_fit, *, bandwidth, fraction, device='cpu'):
    """Fit-only standardized additive/RBF kernel used by the existing baselines."""
    x_fit = torch.as_tensor(x_fit, dtype=torch.float64, device=device)
    if x_fit.ndim != 2 or x_fit.shape[0] < 2 or not torch.isfinite(x_fit).all():
        raise ValueError('x_fit must be a finite material-by-marker matrix')
    bandwidth = float(bandwidth)
    fraction = float(fraction)
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError('bandwidth must be finite and positive')
    if not 0 <= fraction <= 1:
        raise ValueError('fraction must be in [0, 1]')
    indices = torch.arange(x_fit.shape[0], device=device)
    additive, _, rbf, _ = kernels(x_fit, indices, indices, bandwidth, device)
    return fraction * additive + (1.0 - fraction) * rbf


def _center(matrix):
    return matrix - matrix.mean(dim=0, keepdim=True)


def _task_covariance(y_fit, dtype, device, eta=0.5):
    y = torch.as_tensor(y_fit, dtype=dtype, device=device)
    if y.ndim != 2 or y.shape[0] < 1 or not torch.isfinite(y).all():
        raise ValueError('y_fit must be a finite material-by-environment matrix')
    mean = y.mean(dim=0)
    scale = y.std(dim=0, unbiased=False)
    if torch.any(scale <= 0):
        raise ValueError('Constant fit phenotype')
    normalized = (y - mean) / scale
    norm = torch.linalg.vector_norm(normalized, dim=0)
    if torch.any(norm <= 0):
        raise ValueError('Zero phenotype norm')
    eta = float(eta)
    if not np.isfinite(eta) or not 0 <= eta <= 1:
        raise ValueError('eta must be in [0, 1]')
    correlation = ((normalized / norm).T @ (normalized / norm))
    environments = y.shape[1]
    return (1.0 - eta) * torch.eye(environments, dtype=dtype, device=device) \
        + eta * correlation


def _validate_kernel(kernel, materials, dtype, device):
    kernel = torch.as_tensor(kernel, dtype=dtype, device=device)
    if kernel.ndim != 2 or kernel.shape != (materials, materials) \
            or not torch.isfinite(kernel).all():
        raise ValueError('kernel must be a finite square fit-by-fit matrix')
    if not torch.allclose(kernel, kernel.T, atol=1e-10, rtol=1e-10):
        raise ValueError('kernel must be symmetric')
    return kernel


def relation_penalty(predictions, kernel, y_fit, *, eta=0.5, kappa=DEFAULT_KAPPA):
    """Return the exact scalar relation penalty with gradients through predictions.

    ``predictions`` has shape ``(n_fit, environments, ...)``; all trailing axes
    (for example model members and 99 quantiles) are averaged into M.  ``kernel``
    is the fit-by-fit material kernel.  ``y_fit`` supplies the fit-only
    environment correlation.
    """
    if not torch.is_tensor(predictions):
        predictions = torch.as_tensor(predictions)
    if not predictions.is_floating_point():
        raise ValueError('predictions must be a floating tensor')
    if predictions.ndim < 2:
        raise ValueError('predictions must have material and environment axes')
    materials, environments = predictions.shape[:2]
    if materials < 2 or environments < 1:
        raise ValueError('At least two materials and one environment are required')
    if not torch.isfinite(predictions).all():
        raise ValueError('predictions must be finite')
    kappa = float(kappa)
    if not np.isfinite(kappa) or kappa <= 0:
        raise ValueError('kappa must be finite and positive')
    input_dtype = predictions.dtype
    device = predictions.device
    predictions = predictions.double()
    dtype = predictions.dtype
    kernel = _validate_kernel(kernel, materials, dtype, device)
    b_matrix = _task_covariance(y_fit, dtype, device, eta=eta)
    if b_matrix.shape != (environments, environments):
        raise ValueError('y_fit environment count disagrees with predictions')

    if predictions.ndim > 2:
        m = predictions.mean(dim=tuple(range(2, predictions.ndim)))
    else:
        m = predictions
    if m.shape != (materials, environments):
        raise ValueError('Prediction mean has an unexpected shape')
    centered = _center(m)

    # H K H without forming H: double centering of the material kernel.
    kernel_centered = kernel - kernel.mean(dim=0, keepdim=True) \
        - kernel.mean(dim=1, keepdim=True) + kernel.mean()
    system = kernel_centered + kappa * torch.eye(
        materials, dtype=dtype, device=device)
    try:
        system_factor = torch.linalg.cholesky(system)
        b_factor = torch.linalg.cholesky(b_matrix)
    except RuntimeError as error:
        raise ValueError('Cholesky factorization failed') from error
    solved = torch.cholesky_solve(centered, system_factor)
    gram = centered.T @ solved
    b_inverse_gram = torch.cholesky_solve(gram, b_factor)
    trace = torch.diagonal(b_inverse_gram).sum()
    penalty = trace / (materials * environments)
    if not torch.isfinite(penalty):
        raise FloatingPointError('Nonfinite relation penalty')
    return penalty.to(input_dtype)


def make_relation(gram, y_fit, *, eta=0.5, kappa=DEFAULT_KAPPA, device='cpu'):
    """Cache fit-only K/B Cholesky factors and return a differentiable relation(M).

    ``M`` must be a ``[n, environments]`` tensor on the same device; it is centered
    on materials and may carry autograd history.  The returned closure performs
    only solves, never a new factorization.
    """
    kernel = torch.as_tensor(gram, dtype=torch.float64, device=device)
    if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]             or not torch.isfinite(kernel).all():
        raise ValueError('gram must be a finite square material kernel')
    if not torch.allclose(kernel, kernel.T, atol=1e-10, rtol=1e-10):
        raise ValueError('gram must be symmetric')
    materials = kernel.shape[0]
    if materials < 2:
        raise ValueError('At least two fit materials are required')
    kappa = float(kappa)
    if not np.isfinite(kappa) or kappa <= 0:
        raise ValueError('kappa must be finite and positive')
    b_matrix = _task_covariance(y_fit, torch.float64, torch.device(device), eta=eta)
    environments = b_matrix.shape[0]
    kernel_centered = kernel - kernel.mean(dim=0, keepdim=True) \
        - kernel.mean(dim=1, keepdim=True) + kernel.mean()
    system = kernel_centered + kappa * torch.eye(
        materials, dtype=torch.float64, device=device)
    try:
        system_factor = torch.linalg.cholesky(system)
        b_factor = torch.linalg.cholesky(b_matrix)
    except RuntimeError as error:
        raise ValueError('Cholesky factorization failed') from error

    def relation(m):
        if not torch.is_tensor(m):
            m = torch.as_tensor(m, dtype=torch.float64, device=device)
        if not m.is_floating_point() or m.ndim != 2:
            raise ValueError('M must be a floating [n, environments] tensor')
        if tuple(m.shape) != (materials, environments):
            raise ValueError('M shape disagrees with the cached fit kernel')
        if not torch.isfinite(m).all():
            raise ValueError('M must be finite')
        input_dtype = m.dtype
        work = m.to(device=device, dtype=torch.float64)
        centered = _center(work)
        solved = torch.cholesky_solve(centered, system_factor)
        gram_matrix = centered.T @ solved
        trace = torch.diagonal(torch.cholesky_solve(gram_matrix, b_factor)).sum()
        value = trace / (materials * environments)
        if not torch.isfinite(value):
            raise FloatingPointError('Nonfinite relation penalty')
        return value.to(input_dtype)

    relation.materials = materials
    relation.environments = environments
    relation.kappa = kappa
    relation.eta = float(eta)
    return relation


def relation_regularizer(x_fit, y_fit, predictions, *, bandwidth, fraction,
                         eta=0.5, kappa=DEFAULT_KAPPA, device='cpu'):
    """Convenience wrapper that builds the fit-only kernel before the penalty."""
    kernel = build_kernel(x_fit, bandwidth=bandwidth, fraction=fraction,
                          device=device)
    return relation_penalty(predictions, kernel, y_fit, eta=eta, kappa=kappa)
