"""Chr-ResMLP: chromosome-grouped local MLP plus a direct SNP main-effect path.

Local path: per-chromosome Linear(group->d_group)+ReLU, concatenate, then
Linear(21*d_group->d_fuse)+ReLU+Linear(d_fuse->classes). Main path: one
Linear(all SNPs->classes). Logits add. Fit-only x scaling; no y statistics.
Historical AdditiveMLP is the plain-MLP control (same file family).
"""
import torch
from torch import nn


class ChrResMLP(nn.Module):
    def __init__(self, fit_x, chrom_labels, n_classes, *, d_group=8, d_fuse=64):
        super().__init__()
        x = torch.as_tensor(fit_x, dtype=torch.float32).detach().cpu()
        if x.ndim != 2 or len(x) < 2 or not torch.isfinite(x).all():
            raise ValueError('Expected finite fit materials by markers')
        if len(chrom_labels) != x.shape[1]:
            raise ValueError('chromosome labels must cover every marker')
        std = x.std(0, unbiased=False)
        self.register_buffer('x_mean', x.mean(0))
        self.register_buffer('x_constant', std == 0)
        self.register_buffer('x_std', torch.where(std == 0, torch.ones_like(std), std))
        self.groups = sorted(set(str(c) for c in chrom_labels))
        self.slices = []
        self.local = nn.ModuleList()
        for name in self.groups:
            idx = torch.as_tensor([i for i, c in enumerate(chrom_labels) if str(c) == name])
            self.register_buffer(f'idx_{name}', idx)
            self.local.append(nn.Linear(len(idx), d_group))
        self.fuse = nn.Sequential(nn.Linear(len(self.groups) * d_group, d_fuse), nn.ReLU(),
                                  nn.Linear(d_fuse, n_classes))
        self.main = nn.Linear(x.shape[1], n_classes)

    def forward(self, x):
        z = ((x - self.x_mean) / self.x_std).masked_fill(self.x_constant, 0)
        parts = [torch.relu(layer(z[:, getattr(self, f'idx_{name}')]))
                 for name, layer in zip(self.groups, self.local)]
        return self.fuse(torch.cat(parts, dim=1)) + self.main(z)
