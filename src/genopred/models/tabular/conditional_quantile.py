"""Independent deep conditional units with continuous entmax connections."""
import torch
from torch import nn


def entmax15(x):
    x = x / 2
    ordered = x.sort(dim=-1, descending=True).values
    rho = torch.arange(1, x.shape[-1] + 1, device=x.device, dtype=x.dtype)
    mean = ordered.cumsum(-1) / rho
    variance = ordered.square().cumsum(-1) / rho - mean.square()
    tau = mean - ((1 - rho * variance).clamp_min(0) / rho).sqrt()
    support = (tau <= ordered).sum(-1, keepdim=True).clamp_min(1)
    threshold = tau.gather(-1, support - 1)
    return (x - threshold).clamp_min(0).square()


class ConditionalQuantile(nn.Module):
    def __init__(self, fit_x, fit_y, members=4, width=32, layers=3, quantiles=99):
        super().__init__()
        x = torch.as_tensor(fit_x, dtype=torch.float32)
        y = torch.as_tensor(fit_y, dtype=torch.float32).reshape(-1, 1)
        self.register_buffer('x_mean', x.mean(0))
        std = x.std(0, unbiased=False)
        self.register_buffer('x_constant', std == 0)
        self.register_buffer('x_std', torch.where(std == 0, torch.ones_like(std), std))
        self.register_buffer('y_mean', y.mean(0))
        self.register_buffer('y_std', y.std(0, unbiased=False).clamp_min(1e-12))
        p = x.shape[1]
        self.connections = nn.ParameterList()
        self.tables = nn.ParameterList()
        for layer in range(layers):
            d = p + layer * width
            self.connections.append(nn.Parameter(torch.randn(members, width, 2, d)))
            # Near-copy of first input, with independent perturbations per member.
            self.tables.append(nn.Parameter(torch.tensor([-2., -2., 2., 2.]).expand(members, width, 4).clone() + .05 * torch.randn(members, width, 4)))
        self.linear = nn.Parameter(torch.randn(members, p) * .001)
        self.readout = nn.Parameter(torch.randn(members, layers * width, quantiles) * .01)
        self.bias = nn.Parameter(torch.zeros(members, quantiles))

    def standardize_y(self, y):
        return (y - self.y_mean) / self.y_std

    def inverse_y(self, y):
        return y * self.y_std + self.y_mean

    def forward(self, x):
        if not torch.isfinite(x).all() or torch.any((x < 0) | (x > 1)):
            raise ValueError('Conditional units require finite marker inputs in [0,1]')
        features = x[:, None, :].expand(-1, self.linear.shape[0], -1)
        hidden = []
        for logits, table in zip(self.connections, self.tables):
            choice = entmax15(logits)
            uv = torch.einsum('bmd,musd->bmus', features, choice)
            u, v = uv.unbind(-1)
            c = table.sigmoid()
            h = (c[..., 0]*(1-u)*(1-v) + c[..., 1]*(1-u)*v +
                 c[..., 2]*u*(1-v) + c[..., 3]*u*v)
            hidden.append(h)
            features = torch.cat((features, h), dim=-1)
        z = ((x - self.x_mean) / self.x_std).masked_fill(self.x_constant, 0)
        return torch.einsum('bmp,mpq->bmq', torch.cat(hidden, dim=-1), self.readout) + (z @ self.linear.T)[..., None] + self.bias
