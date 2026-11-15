"""Centered connection changes in the first official TabM input layer."""
import torch
from torch import nn

from wheat599.quantile_tabm import quantile_loss


class ConnectionBasis(nn.Module):
    def __init__(self, base, rank, *, shared=True):
        super().__init__()
        if not isinstance(rank, int) or rank < 1:
            raise ValueError('Connection rank must be positive')
        self.base, self.shared = base, shared
        members, markers = base.r.shape
        width = base.weight.shape[0]
        shape = () if shared else (members,)
        self.u = nn.Parameter(torch.randn(*shape, width, rank, dtype=base.weight.dtype, device='cpu'))
        self.v = nn.Parameter(torch.randn(*shape, markers, rank, dtype=base.weight.dtype, device='cpu')
                              if shared else torch.zeros(members, markers, rank, dtype=base.weight.dtype, device='cpu'))
        if shared:
            self.coefficients = nn.Parameter(torch.zeros(members, rank, dtype=base.weight.dtype, device='cpu'))

    def delta(self):
        if self.shared:
            c = self.coefficients - self.coefficients.mean(0, keepdim=True)
            modulation = (c[:, None, :] * self.u[None]) @ self.v.T
            return self.base.weight[None] * modulation
        change = self.base.weight[None] * (self.u @ self.v.transpose(-1, -2))
        # Center actual changes; independent factors need not share directions.
        return change - change.mean(0, keepdim=True)

    def penalty(self):
        return self.delta().square().mean()

    def forward(self, x):
        if self.shared:
            c = self.coefficients - self.coefficients.mean(0, keepdim=True)
            # GenomicTabM repeats the same standardized markers across members.
            g = torch.nn.functional.linear(x[:, :1, :] * self.v.T[None], self.base.weight)
            extra = torch.einsum('brh,mr->bmh', g * self.u.T[None], c)
        else:
            extra = torch.einsum('bmp,mhp->bmh', x, self.delta())
        return self.base(x) + extra


def basis_loss(model, x, y, quantiles, strength):
    loss = quantile_loss(model(x), y, quantiles)
    if strength:
        loss = loss + strength * model.backbone.backbone.blocks[0][0].penalty()
    return loss
