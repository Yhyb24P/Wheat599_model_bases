"""Synthetic multi-environment tasks under the shared marker-scale prior."""

import math

import torch


def prepare_genotypes(markers, bandwidth, target=None, return_target_kernel=False):
    """Reproduce fit-only additive and RBF normalization, optionally for target rows."""
    if markers.ndim != 2 or markers.shape[0] < 2 or bandwidth <= 0:
        raise ValueError('Invalid genotype matrix or bandwidth')
    n, p = markers.shape
    centered = markers - markers.mean(dim=0)
    scale = centered.square().sum() / (n * p)
    if scale <= 0:
        raise ValueError('Zero additive kernel scale')
    z = centered / (p * scale).sqrt()
    norms = markers.square().sum(dim=1)
    distance = (norms[:, None] + norms[None, :] - 2 * markers @ markers.T).clamp_min(0)
    upper = torch.triu_indices(n, n, 1, device=markers.device)
    median = torch.quantile(distance[upper[0], upper[1]], .5)
    if median <= 0:
        raise ValueError('Zero median genetic distance')
    rbf = torch.exp(-bandwidth * distance / median)
    if target is None:
        if return_target_kernel:
            raise ValueError('Target kernel requires target markers')
        return z, rbf
    if target.ndim != 2 or target.shape[1] != p:
        raise ValueError('Target marker dimensions do not match fit rows')
    z_target = (target - markers.mean(dim=0)) / (p * scale).sqrt()
    cross_distance = (target.square().sum(dim=1)[:, None] + norms[None, :]
                      - 2 * target @ markers.T).clamp_min(0)
    cross = torch.exp(-bandwidth * cross_distance / median)
    if not return_target_kernel:
        return z, rbf, z_target, cross
    target_norms = target.square().sum(dim=1)
    target_distance = (target_norms[:, None] + target_norms[None, :]
                       - 2 * target @ target.T).clamp_min(0)
    return z, rbf, z_target, cross, torch.exp(-bandwidth * target_distance / median)


def psd_root(matrix):
    """Return a factor R with R R.T equal to the PSD input covariance."""
    eigenvalues, eigenvectors = torch.linalg.eigh((matrix + matrix.T) / 2)
    tolerance = torch.finfo(matrix.dtype).eps * matrix.shape[0] * eigenvalues.abs().max().clamp_min(1)
    if torch.any(eigenvalues < -100 * tolerance):
        raise ValueError('Covariance is not positive semidefinite')
    return eigenvectors * eigenvalues.clamp_min(0).sqrt().unsqueeze(0)


def sample_task(z, rbf, task, fraction, ridge, generator, shared_scales=True,
                rho=None, inclusion=None):
    """Draw latent effects and phenotypes for one fixed genotype design.

    Rows of z and rbf must be confined to the current fitting subset. Splitting
    those rows into context and query is the caller's responsibility.
    """
    n, p = z.shape
    q = task.shape[0]
    if p < 2 or not 0 <= fraction <= 1 or ridge <= 0:
        raise ValueError('Invalid task dimensions or covariance parameters')
    if rho is None:
        uniform = torch.rand((), generator=generator, device=z.device, dtype=z.dtype)
        rho = torch.where(uniform < .5, torch.zeros_like(uniform), 2 * uniform - 1)
    else:
        rho = torch.as_tensor(rho, device=z.device, dtype=z.dtype)
    if inclusion is None:
        log_inclusion = -math.log(p) * torch.rand((), generator=generator, device=z.device, dtype=z.dtype)
        inclusion = log_inclusion.exp()
    else:
        inclusion = torch.as_tensor(inclusion, device=z.device, dtype=z.dtype)
    if not 0 <= rho <= 1 or not 0 < inclusion <= 1:
        raise ValueError('Invalid mixture prior parameters')
    if not shared_scales and not torch.allclose(task, torch.diag(task.diagonal())):
        raise ValueError('Independent scales require diagonal task covariance')
    shape = (p,) if shared_scales else (p, q)
    active = torch.rand(shape, generator=generator, device=z.device, dtype=z.dtype) < inclusion
    scales = 1 - rho + rho * active.to(z.dtype) / inclusion
    task_root = psd_root(task)
    effects = (torch.randn((p, q), generator=generator, device=z.device, dtype=z.dtype)
               @ task_root.T)
    effects = effects * (scales.sqrt()[:, None] if shared_scales else scales.sqrt())
    background = (psd_root(rbf) @ torch.randn((n, q), generator=generator, device=z.device, dtype=z.dtype)
                  @ task_root.T)
    noise = torch.randn((n, q), generator=generator, device=z.device, dtype=z.dtype)
    y = fraction ** .5 * z @ effects + (1 - fraction) ** .5 * background + ridge ** .5 * noise
    return {'y': y, 'effects': effects, 'rho': rho, 'inclusion': inclusion, 'active': active}
