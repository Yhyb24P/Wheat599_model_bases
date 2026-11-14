"""Supervised conditional spin means; exact enumeration, fixed temperature T=1."""
import torch
from torch import nn


class MarkerRegressor(nn.Module):
    """Fit-only scaling and the additive path shared by the three comparisons."""
    def __init__(self, fit_x, fit_y):
        super().__init__()
        x = torch.as_tensor(fit_x, dtype=torch.float32).detach().cpu()
        y = torch.as_tensor(fit_y, dtype=torch.float32).detach().cpu().reshape(-1, 1)
        if x.ndim != 2 or len(x) != len(y) or len(x) < 2:
            raise ValueError('Expected matching fit materials and scalar targets')
        if not torch.isfinite(x).all() or not torch.isfinite(y).all():
            raise ValueError('Nonfinite fitting data')
        std = x.std(0, unbiased=False)
        self.register_buffer('x_mean', x.mean(0))
        self.register_buffer('x_constant', std == 0)
        self.register_buffer('x_std', torch.where(std == 0, torch.ones_like(std), std))
        self.register_buffer('y_mean', y.mean(0))
        self.register_buffer('y_std', y.std(0, unbiased=False))
        if not torch.isfinite(self.y_std).all() or torch.any(self.y_std <= 0):
            raise ValueError('Invalid fitting phenotype scale')
        self.beta = nn.Parameter(torch.zeros(x.shape[1]))
        self.intercept = nn.Parameter(torch.zeros(()))

    def standardize_x(self, x):
        if x.ndim != 2 or x.shape[1] != len(self.beta):
            raise ValueError('Material-by-marker shape mismatch')
        return ((x - self.x_mean) / self.x_std).masked_fill(self.x_constant, 0)

    def standardize_y(self, y):
        return (y - self.y_mean) / self.y_std

    def inverse_y(self, y):
        return y * self.y_std + self.y_mean


class SupervisedIsing(MarkerRegressor):
    """E(h|x)=-a(x)·h-sum_{k<l} J_kl h_k h_l; return [batch,1,1]."""
    def __init__(self, fit_x, fit_y, *, spins=12, coupled=True):
        super().__init__(fit_x, fit_y)
        if not isinstance(spins, int) or not 1 <= spins <= 12:
            raise ValueError('Exact enumeration budget permits 1..12 hidden spins')
        self.coupled = coupled
        self.field = nn.Linear(len(self.beta), spins)
        self.readout = nn.Parameter(torch.empty(spins))
        # A nonzero readout lets the hidden parameters learn from the first step.
        nn.init.normal_(self.readout, std=0.02)
        left, right = torch.triu_indices(spins, spins, offset=1)
        if coupled:
            self.coupling = nn.Parameter(torch.zeros(len(left)))
            states = 2 * ((torch.arange(1 << spins)[:, None] >> torch.arange(spins)) & 1) - 1
            self.register_buffer('states', states.float())
            self.register_buffer('pair_states', (states[:, left] * states[:, right]).float())
        else:
            self.register_buffer('coupling', torch.zeros(len(left)))

    def hidden_mean(self, field):
        if not self.coupled:
            return torch.tanh(field)
        # Stable softmax evaluates the full finite-temperature distribution, not MAP.
        logits = field @ self.states.T + self.pair_states @ self.coupling
        return logits.softmax(dim=-1) @ self.states

    def forward(self, x):
        z = self.standardize_x(x)
        prediction = self.intercept + z @ self.beta + self.hidden_mean(self.field(z)) @ self.readout
        return prediction[:, None, None]


class AdditiveMLP(MarkerRegressor):
    """Ordinary two-hidden-layer network with the same additive prediction path."""
    def __init__(self, fit_x, fit_y, *, width=128):
        super().__init__(fit_x, fit_y)
        self.network = nn.Sequential(nn.Linear(len(self.beta), width), nn.ReLU(),
                                     nn.Linear(width, width), nn.ReLU(),
                                     nn.Linear(width, 1, bias=False))

    def forward(self, x):
        z = self.standardize_x(x)
        prediction = self.intercept + z @ self.beta + self.network(z).squeeze(-1)
        return prediction[:, None, None]
