"""Continuous quantile queries read marker evidence before scalar regression."""
import math

import torch
from torch import nn


from wheat599.induced_ftt import PreLNCrossAttention


class QueryQuantile(nn.Module):
    """One environment, one seed; forward returns standardized [batch, 1, queries]."""
    def __init__(self, fit_x, fit_y, *, width=64, heads=4, dropout=0.1,
                 quantiles=99, quantile_min=0.001, quantile_max=0.999):
        super().__init__()
        x = torch.as_tensor(fit_x, dtype=torch.float32).detach().cpu()
        y = torch.as_tensor(fit_y, dtype=torch.float32).detach().cpu().reshape(-1, 1)
        if x.ndim != 2 or x.shape[0] != len(y) or len(y) < 2:
            raise ValueError('Expected matching fit materials and one target environment')
        if not torch.isfinite(x).all() or not torch.isfinite(y).all():
            raise ValueError('Nonfinite fitting data')
        if width < 1 or heads < 1 or width % heads or not 0 <= dropout < 1:
            raise ValueError('Invalid attention dimensions or dropout')
        if quantiles < 2 or not 0 < quantile_min < quantile_max < 1:
            raise ValueError('Invalid quantile grid')
        std = x.std(0, unbiased=False)
        self.register_buffer('x_mean', x.mean(0))
        self.register_buffer('x_constant', std == 0)
        self.register_buffer('x_std', torch.where(std == 0, torch.ones_like(std), std))
        self.register_buffer('y_mean', y.mean(0))
        scale = y.std(0, unbiased=False)
        if torch.any(scale <= 0):
            raise ValueError('Constant fitting target')
        self.register_buffer('y_std', scale)
        self.register_buffer('quantiles', torch.linspace(quantile_min, quantile_max, quantiles))
        self.register_buffer('frequencies', torch.arange(33, dtype=torch.float32))
        self.identity = nn.Parameter(torch.empty(x.shape[1], width))
        self.state = nn.Parameter(torch.empty(x.shape[1], width))
        nn.init.uniform_(self.identity, -width ** -0.5, width ** -0.5)
        nn.init.uniform_(self.state, -width ** -0.5, width ** -0.5)
        self.token_encoder = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(),
                                           nn.Linear(2 * width, width))
        self.query_encoder = nn.Sequential(nn.Linear(33, width), nn.GELU(),
                                           nn.Linear(width, width))
        self.reads = nn.ModuleList([PreLNCrossAttention(width, heads, 2 * width, dropout, dropout, dropout) for _ in range(2)])
        self.readout = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width),
                                     nn.GELU(), nn.Linear(width, 1))

    def standardize_y(self, y):
        return (y - self.y_mean) / self.y_std

    def inverse_y(self, prediction):
        return prediction * self.y_std + self.y_mean

    def forward(self, x, tau=None):
        if x.ndim != 2 or x.shape[1] != self.identity.shape[0] or not torch.isfinite(x).all():
            raise ValueError('Expected finite material-by-marker queries')
        tau = self.quantiles if tau is None else torch.as_tensor(tau, dtype=self.quantiles.dtype, device=self.quantiles.device)
        if tau.ndim != 1 or tau.numel() == 0 or not torch.isfinite(tau).all() or torch.any((tau <= 0) | (tau >= 1)):
            raise ValueError('Quantile queries must be a nonempty vector inside (0,1)')
        z = ((x - self.x_mean) / self.x_std).masked_fill(self.x_constant, 0)
        memory = self.token_encoder(self.identity[None] + z[..., None] * self.state[None])
        basis = torch.cos(math.pi * tau[:, None] * self.frequencies[None])
        query = self.query_encoder(basis)[None].expand(len(x), -1, -1)
        for block in self.reads:
            query = block(query, memory)
        return self.readout(query).squeeze(-1).unsqueeze(1)
