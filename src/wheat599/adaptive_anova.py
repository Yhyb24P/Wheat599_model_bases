"""Adaptive order-wise ANOVA-kernel Gaussian process for binary markers.

Implements the candidate specified in docs/methods (gpt66 attachment, 2026-11-14):
a learnable interaction representation with a GP posterior readout. Markers are
encoded as s = 2x - 1 in {-1, +1}. The complete order-d combination space is kept
via the ANOVA kernel k_d; a few learnable templates per order (d = 2, 3) define
directions whose prior variance is adjusted bidirectionally by exp(delta), while
the orthogonal complement keeps the base shrinkage. Orders >= 4 form a shared
Hamming-RBF tail. A single environment is fitted per model instance.

All computation is float64. The order-d kernels are evaluated through power sums
of t_j = s_j s'_j (never by enumerating combinations).
"""

import math

import numpy as np
import torch


def _denom(p, d):
    if d == 1:
        return float(p)
    if d == 2:
        return float(p * (p - 1))
    if d == 3:
        return float(p * (p - 1) * (p - 2))
    raise ValueError(f'unsupported order {d}')


def base_kernels(S, T, p):
    """Complete order-1/2/3 ANOVA kernels between binary rows S[m,p], T[k,p].

    For +/-1 inputs the power sums are P1 = sum_j s_j t_j, P2 = p, P3 = P1, so
    k_d = e_d(t) / C(p,d) reduces to closed forms in P1 alone. Diagonal is 1.
    Returns (K1, K2, K3), each [m, k].
    """
    P1 = S @ T.T
    K1 = P1 / _denom(p, 1)
    K2 = (P1.square() - p) / _denom(p, 2)
    K3 = (P1.pow(3) - 3 * p * P1 + 2 * P1) / _denom(p, 3)
    return K1, K2, K3


def base_kernel(S, T, p, d):
    return base_kernels(S, T, p)[d - 1]


def template_kernel(S, V, p, d):
    """k_d(v_r, s_i) for templates V[R,p] (real) and binary rows S[m,p] -> [m,R].

    Here t_j = v_{r,j} s_{i,j}, so P1 = S V^T, P2 = ||v_r||^2, P3 = S (v_r^3)^T.
    """
    P1 = S @ V.T
    P2 = V.square().sum(1)
    P3 = S @ V.pow(3).T
    if d == 2:
        return (P1.square() - P2) / _denom(p, 2)
    if d == 3:
        return (P1.pow(3) - 3 * P1 * P2 + 2 * P3) / _denom(p, 3)
    raise ValueError(f'unsupported order {d}')


def gram_matrix(V, p, d):
    """G_d[r,t] = k_d(v_r, v_t) for real templates V[R,p] -> [R,R]."""
    P1 = V @ V.T
    P2 = V.square() @ V.square().T
    P3 = V.pow(3) @ V.pow(3).T
    if d == 2:
        return (P1.square() - P2) / _denom(p, 2)
    if d == 3:
        return (P1.pow(3) - 3 * P1 * P2 + 2 * P3) / _denom(p, 3)
    raise ValueError(f'unsupported order {d}')


def _inv_sqrt(G, rel_floor=1e-10):
    """Symmetric inverse square root G^{-1/2} via eigendecomposition.

    Exact for well-conditioned G (the assumed independent-direction case). If
    optimization drifts templates toward linear dependence, G becomes near
    singular; a relative eigenvalue floor (rel_floor * lambda_max) caps the
    inverse square root so the transform stays finite and bounded instead of
    diverging. The floored near-null direction is regularized, not amplified;
    this is a numerical guard, not a change to the independent-direction math.
    """
    eig, U = torch.linalg.eigh(G)
    scale = eig.max().clamp_min(1e-30)
    eig = eig.clamp_min(rel_floor * scale)
    return (U * eig.rsqrt()) @ U.T


def adjusted_kernel(S, T, V, delta, p, d):
    """Order-d kernel with bidirectional variance adjustment on template directions.

    k~_d = (k_d - h h'^T) + h diag(exp(delta)) h'^T, where h = G_d^{-1/2} k_d(v, .).
    At delta = 0 this equals k_d exactly. Both terms are PSD for S = T.
    """
    G = gram_matrix(V, p, d)
    Gis = _inv_sqrt(G)
    Hs = template_kernel(S, V, p, d) @ Gis
    Ht = template_kernel(T, V, p, d) @ Gis
    Kd = base_kernel(S, T, p, d)
    weight = torch.exp(delta)
    return Kd - Hs @ Ht.T + (Hs * weight) @ Ht.T


def tail_kernel(S, T, rho, p):
    """Normalized order->=4 Hamming-RBF tail k_{>3} between binary rows.

    k_RBF = ((1-rho)/(1+rho))^Hamming = sum_d a_d k_d with
    a_d = C(p,d) rho^d / (1+rho)^p. Subtracting orders 0..3 and dividing by the
    (constant) diagonal 1 - sum_{d<=3} a_d leaves a PSD order->=4 kernel with
    unit diagonal. float64 and log-space a_d limit cancellation for small rho.
    """
    P1 = S @ T.T
    hamming = (p - P1) / 2
    rho = torch.as_tensor(rho, dtype=S.dtype, device=S.device)
    log_ratio = torch.log1p(-rho) - torch.log1p(rho)
    rbf = torch.exp(hamming * log_ratio)
    log_norm = p * torch.log1p(rho)
    a = []
    for d in range(4):
        log_comb = math.lgamma(p + 1) - math.lgamma(d + 1) - math.lgamma(p - d + 1)
        a.append(torch.exp(torch.as_tensor(log_comb + d * math.log(float(rho)),
                                           dtype=S.dtype, device=S.device) - log_norm))
    K1, K2, K3 = base_kernels(S, T, p)
    low = a[0] + a[1] * K1 + a[2] * K2 + a[3] * K3
    norm = 1 - (a[0] + a[1] + a[2] + a[3])
    return (rbf - low) / norm


def _select_templates(s_fit, rank, p, d, generator, rel_floor=1e-8):
    """Greedily pick `rank` fit materials giving a full-rank order-d Gram.

    Uses only current fit materials (as +/-1 rows, unit-normalized). A candidate
    is accepted only if the growing template Gram stays full rank, so dependent
    directions (including v and -v, which share an order-2 direction) are never
    kept. Degenerate fit sets that cannot supply `rank` independent directions
    raise rather than padding with duplicates.
    """
    n = s_fit.shape[0]
    if n < rank:
        raise ValueError(f'fit set has {n} materials; cannot place {rank} independent order-{d} templates')
    perm = torch.randperm(n, generator=generator).tolist()
    chosen = []
    for idx in perm:
        cand = s_fit[idx]
        trial = torch.stack(chosen + [cand])
        trial = trial / trial.norm(dim=1, keepdim=True).clamp_min(1e-12)
        eig = torch.linalg.eigvalsh(gram_matrix(trial, p, d))
        if eig.min() > rel_floor * eig.max().clamp_min(1e-30):
            chosen.append(cand)
        if len(chosen) == rank:
            break
    if len(chosen) < rank:
        raise ValueError(f'fit set spans fewer than {rank} independent order-{d} directions; degenerate')
    V = torch.stack(chosen)
    return V / V.norm(dim=1, keepdim=True).clamp_min(1e-12)


class AdaptiveAnovaGP(torch.nn.Module):
    """Single-environment adaptive ANOVA-kernel GP.

    Parameters
    ----------
    x_fit : array [n, p]
        Binary 0/1 DArT markers of the fitting materials.
    y_fit : array [n] or [n, 1]
        Phenotype of one environment for the fitting materials.
    rho : float
        Hamming-RBF bandwidth parameter, 0 < rho < 1 (fixed hyperparameter).
    rank : int
        Number of learnable templates per order (d = 2, 3).
    seed : int
        Seed for deterministic template initialization from the fit set.

    The constructor standardizes y on the fit set and initializes templates from
    fit materials only (non-duplicate directions), delta = 0. ``loss`` returns the
    regularized marginal likelihood; ``forward`` returns predictions on the
    original y scale.
    """

    def __init__(self, x_fit, y_fit, rho=0.01, rank=8, seed=0):
        super().__init__()
        if not 0 < rho < 1:
            raise ValueError('rho must lie in (0, 1)')
        x = np.asarray(x_fit, dtype=np.float64)
        y = np.asarray(y_fit, dtype=np.float64).reshape(-1)
        if x.ndim != 2:
            raise ValueError('x_fit must be [n, p]')
        n, p = x.shape
        if p < 3:
            raise ValueError('need at least 3 markers for the order-3 kernel')
        if y.shape[0] != n:
            raise ValueError('x_fit and y_fit row counts differ')
        if not np.all((x == 0) | (x == 1)):
            raise ValueError('x_fit must be binary 0/1 markers')
        y_mean = float(y.mean())
        y_scale = float(y.std())
        if y_scale == 0:
            raise ValueError('Constant target')

        self.p = p
        self.rank = int(rank)
        self.register_buffer('rho', torch.tensor(float(rho), dtype=torch.float64))
        self.register_buffer('s_fit', torch.as_tensor(2 * x - 1, dtype=torch.float64))
        self.register_buffer('y_mean', torch.tensor(y_mean, dtype=torch.float64))
        self.register_buffer('y_scale', torch.tensor(y_scale, dtype=torch.float64))
        self.register_buffer('y_std', torch.as_tensor((y - y_mean) / y_scale, dtype=torch.float64))
        self.register_buffer('eye', torch.eye(n, dtype=torch.float64))

        gen = torch.Generator().manual_seed(int(seed))
        s_cpu = self.s_fit.cpu()
        V2 = _select_templates(s_cpu, self.rank, p, 2, gen)
        V3 = _select_templates(s_cpu, self.rank, p, 3, gen)
        self.V2 = torch.nn.Parameter(V2.to(self.s_fit.device))
        self.V3 = torch.nn.Parameter(V3.to(self.s_fit.device))
        self.delta2 = torch.nn.Parameter(torch.zeros(self.rank, dtype=torch.float64))
        self.delta3 = torch.nn.Parameter(torch.zeros(self.rank, dtype=torch.float64))
        self.log_sigma = torch.nn.Parameter(torch.zeros(4, dtype=torch.float64))
        self.log_noise = torch.nn.Parameter(torch.tensor(math.log(0.1), dtype=torch.float64))
        self.mu = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))

    def _templates(self):
        # Fixed unit norm: only the direction is learnable, not the scale.
        V2 = self.V2 / self.V2.norm(dim=1, keepdim=True).clamp_min(1e-12)
        V3 = self.V3 / self.V3.norm(dim=1, keepdim=True).clamp_min(1e-12)
        return V2, V3

    def kernel(self, S, T):
        """Full covariance k_theta between binary rows S[m,p] and T[k,p]."""
        V2, V3 = self._templates()
        p, rho = self.p, self.rho
        K1 = base_kernel(S, T, p, 1)
        K2 = adjusted_kernel(S, T, V2, self.delta2, p, 2)
        K3 = adjusted_kernel(S, T, V3, self.delta3, p, 3)
        Kt = tail_kernel(S, T, rho, p)
        sigma = torch.exp(self.log_sigma)
        return sigma[0] * K1 + sigma[1] * K2 + sigma[2] * K3 + sigma[3] * Kt

    def _precision(self):
        K = self.kernel(self.s_fit, self.s_fit)
        noise = torch.exp(self.log_noise).clamp_min(1e-8)
        C = K + noise * self.eye
        L = torch.linalg.cholesky(C)
        return C, L

    def loss(self, delta_penalty):
        """Regularized GP marginal likelihood (scalar, differentiable).

        L = 0.5 r^T C^{-1} r + 0.5 log|C| + (delta_penalty/2) sum_d ||delta_d||^2,
        with r = y_std - mu and C = K_theta(X,X) + sigma_eps^2 I.
        """
        _, L = self._precision()
        r = self.y_std - self.mu
        alpha = torch.cholesky_solve(r.unsqueeze(1), L).squeeze(1)
        quad = (r * alpha).sum()
        logdet = 2 * torch.log(torch.diagonal(L)).sum()
        penalty = 0.5 * delta_penalty * (self.delta2.square().sum() + self.delta3.square().sum())
        return 0.5 * quad + 0.5 * logdet + penalty

    def forward(self, x_query):
        """Posterior mean prediction on the original y scale, shape [m]."""
        T = torch.as_tensor(np.asarray(x_query, dtype=np.float64),
                            dtype=self.s_fit.dtype, device=self.s_fit.device)
        T = 2 * T - 1
        _, L = self._precision()
        r = self.y_std - self.mu
        alpha = torch.cholesky_solve(r.unsqueeze(1), L).squeeze(1)
        cross = self.kernel(T, self.s_fit)
        pred_std = self.mu + cross @ alpha
        return self.y_mean + self.y_scale * pred_std