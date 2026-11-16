"""Shape-shrink quantile transform: mean / scale / standardized shape.

Location-scale-shape decomposition of the shape-shrink quantile TabM
candidate; see ShapeTransform for the parameterization and exact moment
identities. This module only transforms — training, configuration and
runs live elsewhere.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def standardized_shape(logits: torch.Tensor) -> torch.Tensor:
    """Map spacing logits to an ordered, standardized shape.

    logits: [..., M-1]. Softmax gives non-negative spacings summing to 1;
    their cumulative sum u starts at 0 and ends at 1. Centering u and
    dividing by its root second moment yields r: [..., M], non-decreasing,
    mean 0 and (1/M) sum r**2 = 1. The denominator cannot vanish because
    u always spans [0, 1].
    """
    g = torch.softmax(logits, dim=-1)
    u = torch.cat([torch.zeros_like(g[..., :1]), torch.cumsum(g, dim=-1)], dim=-1)
    centered = u - u.mean(dim=-1, keepdim=True)
    denom = torch.sqrt((centered * centered).mean(dim=-1, keepdim=True))
    return centered / denom


def _normal_shape_logits(n_quantiles: int, dtype: torch.dtype) -> torch.Tensor:
    """Spacing logits whose standardized shape is the normal quantile shape."""
    tau = (torch.arange(n_quantiles, dtype=dtype) + 0.5) / n_quantiles
    z = torch.special.ndtri(tau)
    centered = z - z.mean()
    r = centered / torch.sqrt((centered * centered).mean())
    u = (r - r[0]) / (r[-1] - r[0])
    return torch.log(u[1:] - u[:-1])


class ShapeTransform(nn.Module):
    """Turn backbone outputs into ordered quantiles and a shape penalty.

    mode:
      free   raw[..., M+1] = [mu, scale_logit, M-1 spacing logits]; each
             member carries its own shape; no template; penalty 0.
      shrink raw[..., M+1] as free, plus a learnable shared template; the
             penalty is the mean squared shape gap to the template over
             all samples, members and quantile points.
      shared raw[..., 2] = [mu, scale_logit]; every member uses the shared
             template shape; penalty 0.

    scale = softplus(scale_logit) > 0, mu is the location. forward returns
    (q, penalty) with q[..., M]; q.mean(-1) == mu and the per-member second
    central moment == scale**2. Only shared/shrink hold template parameters.
    """

    MODES = ("free", "shared", "shrink")

    def __init__(self, mode: str, n_quantiles: int = 99, *,
                 normal_init: bool = False, dtype: torch.dtype = torch.float64):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}")
        if n_quantiles < 2:
            raise ValueError("n_quantiles must be >= 2")
        self.mode = mode
        self.n_quantiles = int(n_quantiles)
        self._raw_dim = 2 if mode == "shared" else self.n_quantiles + 1
        if mode in ("shared", "shrink"):
            logits = (_normal_shape_logits(self.n_quantiles, dtype) if normal_init
                      else torch.zeros(self.n_quantiles - 1, dtype=dtype))
            self.template_logits = nn.Parameter(logits)

    def _template_shape(self, dtype: torch.dtype) -> torch.Tensor:
        return standardized_shape(self.template_logits.to(dtype))

    def forward(self, raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if raw.shape[-1] != self._raw_dim:
            raise ValueError(
                f"{self.mode} expects raw[..., {self._raw_dim}], got {tuple(raw.shape)}")
        mu = raw[..., 0]
        scale = F.softplus(raw[..., 1]) + torch.finfo(raw.dtype).tiny
        if self.mode == "shared":
            r = self._template_shape(raw.dtype)
            q = mu[..., None] + scale[..., None] * r
            return q, q.new_zeros(())
        r = standardized_shape(raw[..., 2:])
        q = mu[..., None] + scale[..., None] * r
        if self.mode == "shrink":
            penalty = ((r - self._template_shape(raw.dtype)) ** 2).mean()
        else:
            penalty = q.new_zeros(())
        return q, penalty


def normalized_crps(q: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Per-element normalized CRPS of the ordered equal-weight distribution.

    q[..., M] are sorted locations, y[...] the matching observations. With
    1-indexed j,
      S = (1/M) sum_j |q_j - y| - (1/M**2) sum_j (2j - M - 1) q_j,
    the O(M) energy form. Differentiable; average over members and samples
    for the training loss. Equals midpoint_pinball and the pairwise energy
    form for sorted q.
    """
    m = q.shape[-1]
    weight = 2 * torch.arange(m, dtype=q.dtype, device=q.device) + 1 - m
    abs_term = (q - y[..., None]).abs().mean(dim=-1)
    spread_term = (q * weight).sum(dim=-1) / (m * m)
    return abs_term - spread_term


def midpoint_pinball(q: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Equivalent 2/M midpoint-level pinball loss, per-element.

    tau_j = (j - 1/2)/M (1-indexed), rho_tau(u) = u * (tau - 1[u < 0]),
    S = (2/M) sum_j rho_{tau_j}(y - q_j). Differentiable; average over
    members and samples for the training loss.
    """
    m = q.shape[-1]
    tau = (torch.arange(m, dtype=q.dtype, device=q.device) + 0.5) / m
    u = y[..., None] - q
    rho = u * (tau - (u < 0).to(q.dtype))
    return 2.0 / m * rho.sum(dim=-1)
