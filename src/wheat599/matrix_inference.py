"""Material-site bidirectional inference over the raw marker matrix.

The network keeps the observed matrices instead of collapsing markers into a single
token.  Site states ``H`` live per marker and are exchanged with material states at
every layer through the context matrix ``Z``, with an explicit environment axis:

    m0        = Z^T Y / n                           (default moment_scale='n')
    H0[i,j,:] = W0^T [ s * query_x[i,j] ; m0[j] ]   (bias-free, s set by query_scale)
    A         = Z H / sqrt(p)
    V         = psi([A, Y, A (x) Y])                (A (x) Y is d-by-q, never summed)
    M         = Z^T V / n
    g         = sigmoid(gate([H^2, M^2, H (.) M, rms(M)]))
    H        <- (1 - g) (.) H + g (.) update(M)      (bias-free update)
    r         = query_x^T H / sqrt(p)
    y_hat     = r W_out + b_out

Sign covariance.  Every site-coupled weight is bias-free and the gate sees only even
functions of the sign-covariant states (squares, products, an RMS amplitude), so
flipping any marker column in ``context_x`` and ``query_x`` together leaves the
prediction unchanged exactly.  Marker permutation permutes the site axis, material
permutation leaves predictions unchanged, and queries are independent, so chunked and
single-query evaluation agree up to floating-point reduction order.

Scale limitations:
* With ``query_scale='one'`` (the literal formula) the readout contains
  ``||query_x||^2 / sqrt(p)`` from the query channel of ``H0``, which grows like
  sqrt(p) for standardised markers; ``A`` carries the analogue autocorrelation term
  through ``Z^T Y``.  This is a property of the given formula, not a numerical defect.
* With ``query_scale='inverse_sqrt_p'`` the query channel is divided by sqrt(p); the
  quadratic term becomes ``||query_x||^2 / p``. This controls that initialization
  term only; it does not guarantee the scale of the nonlinear iteration.
* ``sqrt_n`` divides both the initial cross-moment and later messages by sqrt(n)
  instead of n. Neither option automatically calibrates signals across n or p.
* No per-site or global amplitude matching is applied. Message magnitudes remain
  available to the gate instead of weak evidence being forced to unit magnitude.
* H depends on the same context used by ZH. In particular the initial moment
  contributes a self-material term proportional to ||z_i||^2 y_i. Independence
  arguments for the variance of ZH therefore do not hold in general.

Special case with an explicit form: take ``H = query_x``, ``V = Y * (Z H / sqrt(p))``,
``M = Z^T V / n`` and ``r = query_x^T M / sqrt(p)``.  Then

    r = 1/(n p) * sum_i y_i (z_i . query_x)^2,

i.e. a p-normalised quadratic-similarity statistic, the acknowledged quadratic-kernel
relation.  This is the only kernel-equivalence statement made here; the full iteration
mixes the query channel, the label-weighted cross-moment ``Z^T diag(y) Z`` (which does
not commute with ``Z^T Z``) and the nonlinear gate, so no global polynomial or kernel
ridge claim is made.

Relation to enumerated pair statistics: LinearPFN's optional pair-statistics heads
build explicit per-pair statistics, whereas this operator keeps
O(p) site states and never materialises p^2; the distinction is implicit iterative
inference versus explicit pair enumeration.
"""

import math

import torch
from torch import nn


class MatrixInference(nn.Module):
    """Per-query material-site message passing with an explicit environment axis."""

    def __init__(self, environments=4, width=32, layers=4, psi_hidden=64, gate_hidden=64,
                 moment_scale='n', query_scale='one', dtype=torch.float32):
        super().__init__()
        if min(environments, width, layers, psi_hidden, gate_hidden) < 1:
            raise ValueError('Network dimensions and layers must be positive')
        if moment_scale not in ('sqrt_n', 'n'):
            raise ValueError("moment_scale must be 'sqrt_n' or 'n'")
        if query_scale not in ('one', 'inverse_sqrt_p'):
            raise ValueError("query_scale must be 'one' or 'inverse_sqrt_p'")
        self.environments, self.width, self.layers = int(environments), int(width), int(layers)
        self.moment_scale, self.query_scale = moment_scale, query_scale
        self.initial = nn.Linear(1 + self.environments, self.width, bias=False)
        self.psi = nn.Sequential(
            nn.Linear(self.width + self.environments + self.width * self.environments, psi_hidden),
            nn.SiLU(),
            nn.Linear(psi_hidden, self.width),
        )
        self.gate = nn.Sequential(
            nn.Linear(3 * self.width + 1, gate_hidden),
            nn.SiLU(),
            nn.Linear(gate_hidden, self.width),
        )
        self.update = nn.Linear(self.width, self.width, bias=False)
        self.readout = nn.Linear(self.width, self.environments)
        self.to(dtype=dtype)

    def _check(self, context_x, context_y, query_x):
        if context_x.ndim != 2 or context_y.ndim != 2 or query_x.ndim != 2:
            raise ValueError('context_x, context_y and query_x must be matrices')
        if context_y.shape != (context_x.shape[0], self.environments):
            raise ValueError('context_y must be materials by environments')
        if query_x.shape[1] != context_x.shape[1]:
            raise ValueError('query_x must use the same markers as context_x')
        if context_x.shape[0] < 2 or query_x.shape[0] < 1 or context_x.shape[1] < 1:
            raise ValueError('Require two context materials, one marker and one query')
        for value in (context_x, context_y, query_x):
            if not torch.isfinite(value).all():
                raise ValueError('Non-finite network input')

    def forward(self, context_x, context_y, query_x):
        self._check(context_x, context_y, query_x)
        dtype, device = self.initial.weight.dtype, self.initial.weight.device
        z = context_x.to(dtype=dtype, device=device)
        y = context_y.to(dtype=dtype, device=device)
        q = query_x.to(dtype=dtype, device=device)
        materials, markers = z.shape
        queries = q.shape[0]

        denominator = math.sqrt(materials) if self.moment_scale == 'sqrt_n' else materials
        moment = (z.T @ y) / denominator
        query_channel = q.unsqueeze(-1)
        if self.query_scale == 'inverse_sqrt_p':
            query_channel = query_channel / math.sqrt(markers)
        state = self.initial(torch.cat([query_channel,
                                        moment.unsqueeze(0).expand(queries, -1, -1)], dim=-1))
        for _ in range(self.layers):
            sites = torch.einsum('np,mpd->mnd', z, state) / math.sqrt(markers)
            interaction = sites.unsqueeze(-1) * y.unsqueeze(0).unsqueeze(2)
            features = torch.cat([sites,
                                  y.unsqueeze(0).expand(queries, -1, -1),
                                  interaction.reshape(queries, materials, -1)], dim=-1)
            material_state = self.psi(features)
            update = torch.einsum('np,mnd->mpd', z, material_state) / denominator
            amplitude = update.square().mean(dim=-1, keepdim=True).clamp_min(1e-24).sqrt()
            gate = torch.sigmoid(self.gate(torch.cat([state.square(), update.square(),
                                                      state * update, amplitude], dim=-1)))
            state = (1 - gate) * state + gate * self.update(update)
        summary = torch.einsum('mp,mpd->md', q, state) / math.sqrt(markers)
        return self.readout(summary)

    def parameter_count(self):
        return int(sum(p.numel() for p in self.parameters()))

    def flops_per_forward(self, materials, markers, queries):
        """Dominant terms: the two site-material contractions per layer."""
        return int(2 * self.layers * 2 * queries * materials * markers * self.width)
