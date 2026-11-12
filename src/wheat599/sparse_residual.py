"""Full-marker sparse residual quantile TabM with a ProxGEN L1/2 proximal step.

Backbone: a dense full-marker entry, ``n_blocks`` bilinear residual blocks, and a
per-member ``quantiles``-wide output. Each residual block holds TWO fully-connected
``LinearBatchEnsemble`` transforms (shared matrix + ``members`` multiplicative r/s
adapters) around a ReLU/dropout, wrapped in a per-member RMSNorm pre-norm residual.
``forward`` returns ``(batch, members, quantiles)``.

Sparsity: only the residual-block shared matrices (``proximal_parameters``, two per
block) receive the coordinate-wise L1/2 proximal step; every other parameter uses
ordinary AdamW decay (``decay_parameters``). ``proximal_step`` is called AFTER
``optimizer.step()`` and reads the Adam bias-corrected preconditioner
D = lr / (sqrt(v_hat) + eps); the per-coordinate penalty is tau = lambda * D. Proximal
parameters must sit in a weight_decay=0 group (see ``make_optimizer``) so that ``p.data``
after the step equals the pure Adam update u = w - D * m_hat. Zeros are never frozen, so
a coordinate can reactivate on a later step.
"""
import math

import torch
from torch import nn


def l12_prox(u, tau):
    """Global-optimal proximal operator of tau * |.|^(1/2), coordinate-wise.

    Minimises .5*(z-u)^2 + tau*sqrt(|z|). Zero for |u| <= 1.5*tau^(2/3); otherwise the
    largest root t of t^3 - |u| t + tau/2 = 0 gives z = sign(u)*t^2 (half-thresholding).
    """
    abs_u = u.abs()
    tau = tau.clamp_min(0.0)
    threshold = 1.5 * tau.pow(2.0 / 3.0)
    active = abs_u > threshold
    # Clamp guards the arccos argument against round-off; inactive lanes are masked out.
    safe = abs_u.clamp_min(1e-30)
    cos_arg = (-3.0 * math.sqrt(3.0) * tau / (4.0 * safe.pow(1.5))).clamp(-1.0, 1.0)
    t = 2.0 * torch.sqrt(safe / 3.0) * torch.cos(torch.arccos(cos_arg) / 3.0)
    return torch.where(active, torch.sign(u) * t.square(), torch.zeros_like(u))


def proximal_step(model, optimizer, lam):
    """Apply the L1/2 proximal step to the residual-block shared matrices in place."""
    prox = {id(p) for p in model.proximal_parameters()}
    with torch.no_grad():
        for group in optimizer.param_groups:
            beta1, beta2 = group['betas']
            lr, eps = group['lr'], group['eps']
            for p in group['params']:
                if id(p) not in prox:
                    continue
                state = optimizer.state[p]
                if not state:
                    continue
                step = float(state['step'])
                m_hat = state['exp_avg'] / (1.0 - beta1 ** step)
                v_hat = state['exp_avg_sq'] / (1.0 - beta2 ** step)
                preconditioner = lr / (v_hat.sqrt() + eps)
                p.copy_(l12_prox(p, lam * preconditioner))


def make_optimizer(model, learning_rate, weight_decay, **adam_kwargs):
    """AdamW with proximal shared matrices (no decay) separated from decaying params."""
    prox = model.proximal_parameters()
    prox_ids = {id(p) for p in prox}
    decay = [p for p in model.parameters() if id(p) not in prox_ids]
    return torch.optim.AdamW(
        [{'params': prox, 'weight_decay': 0.0},
         {'params': decay, 'weight_decay': float(weight_decay)}],
        lr=float(learning_rate), **adam_kwargs)


class BilinearResidualBlock(nn.Module):
    """Two fully-connected bilinear transforms in a per-member pre-norm residual."""

    def __init__(self, width, members, dropout):
        super().__init__()
        import tabm
        self.norm = nn.RMSNorm(width)
        self.fc1 = tabm.LinearBatchEnsemble(width, width, k=members, scaling_init='ones')
        self.fc2 = tabm.LinearBatchEnsemble(width, width, k=members, scaling_init='ones')
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, h):
        return h + self.fc2(self.drop(self.act(self.fc1(self.norm(h)))))

    def shared_matrices(self):
        return [self.fc1.weight, self.fc2.weight]


class SparseResidualQuantile(nn.Module):
    """Marker-only quantile ensemble with sparse bilinear residual blocks."""

    def __init__(self, train_x, train_y, *, width, members=32, quantiles=99,
                 n_blocks=3, dropout=0.1, dtype=torch.float32):
        super().__init__()
        import tabm
        x = torch.as_tensor(train_x, dtype=dtype).detach().cpu()
        y = torch.as_tensor(train_y, dtype=dtype).detach().cpu().reshape(-1, 1)
        if x.ndim != 2 or x.shape[0] != y.shape[0]:
            raise ValueError('train_x and train_y must share the material axis')
        self.width, self.members, self.quantiles = int(width), int(members), int(quantiles)
        self.n_blocks, self.markers = int(n_blocks), int(x.shape[1])
        mean = x.mean(dim=0)
        std = x.std(dim=0, unbiased=False)
        constant = std <= 0
        self.register_buffer('x_mean', mean)
        self.register_buffer('x_std', torch.where(constant, torch.ones_like(std), std))
        self.register_buffer('x_constant', constant)
        self.register_buffer('y_mean', y.mean(dim=0))
        self.register_buffer('y_std', y.std(dim=0, unbiased=False).clamp_min(1e-12))
        self.entry = tabm.LinearBatchEnsemble(
            self.markers, self.width, k=self.members, scaling_init=('random-signs', 'ones'))
        self.blocks = nn.ModuleList(
            [BilinearResidualBlock(self.width, self.members, dropout) for _ in range(self.n_blocks)])
        self.output = tabm.LinearEnsemble(self.width, self.quantiles, k=self.members)
        self.to(dtype=dtype)

    def standardize_x(self, x):
        x = torch.as_tensor(x, dtype=self.x_mean.dtype)
        block = (x - self.x_mean) / self.x_std
        return torch.where(self.x_constant.unsqueeze(0), torch.zeros_like(block), block)

    def standardize_y(self, y):
        y = torch.as_tensor(y, dtype=self.y_mean.dtype)
        return (y - self.y_mean) / self.y_std

    def inverse_y(self, prediction):
        return prediction * self.y_std + self.y_mean

    def forward(self, x):
        z = self.standardize_x(x)
        h = self.entry(z[:, None, :].expand(-1, self.members, -1))
        for block in self.blocks:
            h = block(h)
        return self.output(h)

    def proximal_parameters(self):
        """Residual-block shared matrices (two per block): proximal only, no decay."""
        return [w for block in self.blocks for w in block.shared_matrices()]

    def decay_parameters(self):
        """All remaining parameters: ordinary AdamW decay."""
        prox_ids = {id(p) for p in self.proximal_parameters()}
        return [p for p in self.parameters() if id(p) not in prox_ids]

    def parameter_count(self):
        return int(sum(p.numel() for p in self.parameters()))