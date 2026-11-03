"""Enumerated reference posterior for small multi-environment marker sets."""

import itertools
import math

import numpy as np
import torch


def posterior_mean(z, y, background, task, fraction, ridge, rho, inclusion,
                   return_log_evidence=False):
    """Integrate inclusion states and a flat intercept using exact enumeration.

    This exponential reference is intended for small p. The Gaussian background
    covariance is ``(1-fraction)*background ⊗ task + ridge*I``.
    """
    n, p = z.shape
    q = y.shape[1]
    if (y.shape[0] != n or background.shape != (n, n) or task.shape != (q, q)
            or not 0 <= fraction <= 1 or ridge <= 0):
        raise ValueError('Invalid data dimensions or covariance parameters')
    if p > 16 or not 0 < inclusion <= 1 or not 0 <= rho <= 1:
        raise ValueError('Enumeration requires p <= 16, 0 < inclusion <= 1, 0 <= rho <= 1')
    eye = torch.eye(n * q, dtype=z.dtype, device=z.device)
    design = torch.eye(q, dtype=z.dtype, device=z.device).repeat(n, 1)
    response = y.reshape(-1, 1)
    residual = (1 - fraction) * torch.kron(background.contiguous(), task.contiguous()) + ridge * eye
    log_weights, means = [], []
    states = [(1,) * p] if inclusion == 1 else itertools.product((0, 1), repeat=p)
    for state in states:
        active = torch.tensor(state, dtype=z.dtype, device=z.device)
        scales = 1 - rho + rho * active / inclusion
        kernel = (z * scales) @ z.T
        covariance = residual + fraction * torch.kron(kernel.contiguous(), task.contiguous())
        chol = torch.linalg.cholesky(covariance)
        solved = torch.cholesky_solve(torch.cat((response, design), dim=1), chol)
        fsolve = solved[:, 1:]
        fixed_chol = torch.linalg.cholesky(design.T @ fsolve)
        fixed = torch.cholesky_solve(design.T @ solved[:, :1], fixed_chol)
        centered = solved[:, :1] - fsolve @ fixed
        logdet = 2 * torch.log(chol.diagonal()).sum()
        logdet_fixed = 2 * torch.log(fixed_chol.diagonal()).sum()
        prior = 0. if inclusion == 1 else (sum(state) * math.log(inclusion)
                                            + (p - sum(state)) * math.log1p(-inclusion))
        log_weights.append(-0.5 * (logdet + logdet_fixed + (response.T @ centered).squeeze()) + prior)
        means.append(fraction ** 0.5 * scales[:, None] * (z.T @ centered.reshape(n, q) @ task))
    log_weights = torch.stack(log_weights)
    probabilities = torch.softmax(log_weights, dim=0)
    mean = torch.einsum('s,spq->pq', probabilities, torch.stack(means))
    if return_log_evidence:
        return mean, torch.logsumexp(log_weights, dim=0)
    return mean


def hyperprior_posterior_mean(z, y, background, task, fraction, ridge, order=8):
    """Integrate rho's point/uniform mixture and log-uniform inclusion by quadrature."""
    if order < 1 or z.shape[1] < 2:
        raise ValueError('Quadrature order must be positive and p must be at least 2')
    nodes, weights = np.polynomial.legendre.leggauss(order)
    nodes, weights = (nodes + 1) / 2, weights / 2
    means, log_weights = [], []
    # At rho=0 the likelihood is independent of inclusion.
    mean, evidence = posterior_mean(z, y, background, task, fraction, ridge,
                                    0., .5, return_log_evidence=True)
    means.append(mean)
    log_weights.append(evidence + math.log(.5))
    for r, wr in zip(nodes, weights):
        for t, wt in zip(nodes, weights):
            inclusion = math.exp(-math.log(z.shape[1]) * t)
            mean, evidence = posterior_mean(z, y, background, task, fraction, ridge,
                                            float(r), inclusion, return_log_evidence=True)
            means.append(mean)
            log_weights.append(evidence + math.log(.5 * wr * wt))
    probabilities = torch.softmax(torch.stack(log_weights), dim=0)
    return torch.einsum('s,spq->pq', probabilities, torch.stack(means))


def independent_scale_posterior_mean(z, y, background, task, fraction, ridge, rho, inclusion):
    """Exact fixed-prior reference when environments have independent scales."""
    if not torch.allclose(task, torch.diag(task.diagonal())):
        raise ValueError('Independent scales require diagonal task covariance')
    return torch.cat([
        posterior_mean(z, y[:, k:k + 1], background, task[k:k + 1, k:k + 1],
                       fraction, ridge, rho, inclusion)
        for k in range(y.shape[1])
    ], dim=1)
