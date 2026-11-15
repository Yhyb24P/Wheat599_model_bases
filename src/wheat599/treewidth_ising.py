"""Exact first/second spin moments on a fixed width-two chain."""
import torch
from torch import nn

from wheat599.supervised_ising import MarkerRegressor


class TreewidthIsing(MarkerRegressor):
    """Edges (k,k+1)/(k,k+2); node order carries no genomic interpretation."""
    def __init__(self, fit_x, fit_y, *, spins=64, coupled=True):
        super().__init__(fit_x, fit_y)
        if not isinstance(spins, int) or spins < 3:
            raise ValueError('Width-two inference requires at least three spins')
        self.coupled = coupled
        self.field = nn.Linear(len(self.beta), spins)
        self.readout = nn.Parameter(torch.empty(spins))
        self.pair_readout = nn.Parameter(torch.empty(2 * spins - 3))
        nn.init.normal_(self.readout, std=0.02)
        nn.init.normal_(self.pair_readout, std=0.02)
        self.coupling = nn.Parameter(torch.zeros(2 * spins - 3), requires_grad=coupled)
        left = torch.cat((torch.arange(spins - 1), torch.arange(spins - 2)))
        self.register_buffer('left', left)
        self.register_buffer('right', torch.cat((left[:spins - 1] + 1, left[spins - 1:] + 2)))
        self.register_buffer('states', torch.tensor([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]]))

    def moments(self, field):
        """Return node means and edge products, adjacent edges before distance-two."""
        if not self.coupled:
            mean = field.tanh()
            return mean, mean[:, self.left] * mean[:, self.right]
        n = field.shape[1]
        u, v = self.states[:, 0], self.states[:, 1]
        # A 4x4 transfer joins (h[t-2],h[t-1]) to (h[t-1],h[t]).
        factors = (field[:, 2:, None, None] * v
                   + self.coupling[1:n-1][None, :, None, None] * v[:, None] * v
                   + self.coupling[n-1:][None, :, None, None] * u[:, None] * v)
        factors = factors.masked_fill(v[:, None] != u, -torch.inf)
        padded = 1 << (n - 3).bit_length()
        # Dense zero tail factors add only a constant partition multiplier.
        levels = [torch.nn.functional.pad(factors, (0, 0, 0, 0, 0, padded - (n - 2)))]
        while levels[-1].shape[1] > 1:
            blocks = levels[-1]
            merged = (blocks[:, ::2, :, :, None] + blocks[:, 1::2, None, :, :]).logsumexp(3)
            levels.append(merged - merged.logsumexp((2, 3), keepdim=True))
        alpha = (field[:, :1, None] * u + field[:, 1:2, None] * v + self.coupling[0] * u * v)
        alpha = alpha - alpha.logsumexp(2, keepdim=True)
        beta = torch.zeros_like(alpha)
        # Balanced contraction retains O(R) work and gives O(log R) GPU depth.
        for blocks in reversed(levels[:-1]):
            right_alpha = (alpha[:, :, :, None] + blocks[:, ::2]).logsumexp(2)
            left_beta = (blocks[:, 1::2] + beta[:, :, None, :]).logsumexp(3)
            alpha = torch.stack((alpha, right_alpha), 2).flatten(1, 2)
            beta = torch.stack((left_beta, beta), 2).flatten(1, 2)
            alpha = alpha - alpha.logsumexp(2, keepdim=True)
            beta = beta - beta.logsumexp(2, keepdim=True)
        logits = alpha[:, :n-2, :, None] + factors + beta[:, :n-2, None, :]
        joint = logits.flatten(2).softmax(2).reshape_as(logits)
        first = joint[:, 0].sum(2)
        mean = torch.cat(((first * u).sum(1, keepdim=True), (first * v).sum(1, keepdim=True),
                          (joint * v).sum((2, 3))), 1)
        pairs = torch.cat(((first * u * v).sum(1, keepdim=True),
                           (joint * v[:, None] * v).sum((2, 3)),
                           (joint * u[:, None] * v).sum((2, 3))), 1)
        return mean, pairs

    def forward(self, x):
        z = self.standardize_x(x)
        mean, pairs = self.moments(self.field(z))
        prediction = (self.intercept + z @ self.beta
                      + mean @ self.readout + pairs @ self.pair_readout)
        return prediction[:, None, None]
