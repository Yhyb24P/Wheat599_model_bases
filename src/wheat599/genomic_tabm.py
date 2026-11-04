"""Thin genomic adaptation of the official TabM ensemble (tabm==0.0.3).

Two variants keep the official backbone untouched:

* ``raw``: ``TabM.make(n_num_features=p, d_out=q, ...)`` on the column-standardised
  markers, official initialisation, no linear branch.
* ``grouped_additive``: each marker is assigned to one of at most ``groups`` anchor
  clusters chosen greedily on the training rows (largest raw-variance column first, then
  the column with the smallest maximum absolute correlation to the current anchors); group
  tokens ``sum_{j in g} z_j W_jr / sqrt(n_g)`` (rank ``rank``) are flattened and fed to
  the same official TabM, and a full-marker linear branch ``z beta + b`` is broadcast to
  all ``k`` ensemble members.  The TabM output layer is zero-initialised so the initial
  prediction equals the ridge fit exactly; every parameter is then trained jointly.

Preprocessing is fitted on the training rows only: per-column mean/std for markers
(constant columns become zero), per-environment mean/std for phenotypes.  The stored
statistics are buffers; passing different data never changes them.  Grouping is O(n p G)
and never forms a p-by-p matrix, needs no marker coordinates, and reports the realised
number of groups when fewer candidates exist.
"""

import torch
from torch import nn


class GenomicTabM(nn.Module):
    """Official TabM backbone with a genomic input adaptation."""

    def __init__(self, train_x, train_y, *, variant='grouped_additive', beta=None,
                 bias=None, groups=64, rank=4, n_blocks=3, d_block=256, dropout=0.1,
                 k=32, beta_ridge=0.0, dtype=torch.float32):
        super().__init__()
        try:
            import tabm  # provided by the official tabm==0.0.3 package
        except ImportError as error:  # pragma: no cover - depends on the environment
            raise ImportError('tabm==0.0.3 is required; put its directory on PYTHONPATH') \
                from error
        if variant not in ('raw', 'grouped_additive'):
            raise ValueError("variant must be 'raw' or 'grouped_additive'")
        # Statistics and grouping are fitted on CPU with the runner's dtype; call
        # ``model.to(device)`` afterwards.
        x = torch.as_tensor(train_x, dtype=dtype).detach().cpu()
        y = torch.as_tensor(train_y, dtype=dtype).detach().cpu()
        if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
            raise ValueError('train_x and train_y must share the material axis')
        self.variant, self.dtype = variant, dtype
        self.environments = int(y.shape[1])
        self.markers = int(x.shape[1])
        self.groups_requested, self.rank = int(groups), int(rank)
        self.beta_ridge = float(beta_ridge)
        mean = x.mean(dim=0)
        std = x.std(dim=0, unbiased=False)
        constant = std <= 0
        safe = torch.where(constant, torch.ones_like(std), std)
        self.register_buffer('x_mean', mean)
        self.register_buffer('x_std', safe)
        self.register_buffer('x_constant', constant)
        self.register_buffer('y_mean', y.mean(dim=0))
        self.register_buffer('y_std', y.std(dim=0, unbiased=False).clamp_min(1e-12))
        standardised = (x - mean) / safe
        standardised = torch.where(constant.unsqueeze(0), torch.zeros_like(standardised),
                                   standardised)
        if variant == 'grouped_additive':
            anchors, assignment, sizes = self._group(standardised)
            self.register_buffer('anchors', anchors)
            self.register_buffer('assignment', assignment)
            self.register_buffer('group_sizes', sizes)
            if beta is None:
                raise ValueError('grouped_additive requires the ridge beta from the runner')
            beta = torch.as_tensor(beta, dtype=dtype).detach().cpu()
            if beta.shape != (self.markers, self.environments):
                raise ValueError('beta must be markers by environments')
            bias = torch.zeros(self.environments, dtype=dtype) if bias is None \
                else torch.as_tensor(bias, dtype=dtype).detach().cpu()
            self.beta = nn.Parameter(beta.clone())
            self.bias = nn.Parameter(bias.clone())
            # The group fan-in is already normalised by 1/sqrt(n_g); standard normal
            # weights keep the token variance near one, matching the TabM input scale.
            self.weights = nn.Parameter(torch.randn(self.markers, self.rank, dtype=dtype))
            self.backbone = tabm.TabM.make(n_num_features=self.actual_groups * self.rank,
                                           d_out=self.environments, n_blocks=n_blocks,
                                           d_block=d_block, dropout=dropout, k=k)
            with torch.no_grad():
                self.backbone.output.weight.zero_()
                self.backbone.output.bias.zero_()
        else:
            self.anchors = None
            self.backbone = tabm.TabM.make(n_num_features=self.markers,
                                           d_out=self.environments, n_blocks=n_blocks,
                                           d_block=d_block, dropout=dropout, k=k)
        self.backbone.to(dtype=dtype)

    @property
    def actual_groups(self):
        """Realised group count, derived from the saved anchors (state-dict safe)."""
        return 0 if self.anchors is None else int(self.anchors.numel())

    # -- grouping ----------------------------------------------------------
    @torch.no_grad()
    def _group(self, standardised):
        rows, markers = standardised.shape
        candidates = (~self.x_constant).nonzero(as_tuple=False).reshape(-1)
        # Standardised columns all have variance one, so the first anchor must use the
        # raw fitting variance instead.
        variance = self.x_std.square()
        if candidates.numel() == 0:
            empty = torch.empty(0, dtype=torch.long)
            assignment = torch.zeros(markers, dtype=torch.long)
            sizes = torch.zeros(0, dtype=torch.long)
            return empty, assignment, sizes
        best = torch.zeros(markers, dtype=standardised.dtype)
        anchors = []
        remaining = candidates.clone()
        for _ in range(min(self.groups_requested, candidates.numel())):
            if not anchors:
                score = variance[remaining]
                pick = remaining[torch.argmax(score)]
            else:
                pick = remaining[torch.argmin(best[remaining])]
            anchors.append(int(pick))
            remaining = remaining[remaining != pick]
            correlation = (standardised.T @ standardised[:, pick]) / rows
            best = torch.maximum(best, correlation.abs())
        anchors = torch.tensor(sorted(set(anchors)), dtype=torch.long)
        anchor_corr = (standardised.T @ standardised[:, anchors]) / rows
        assignment = anchor_corr.abs().argmax(dim=1)
        assignment[anchors] = torch.arange(anchors.numel())
        assignment[self.x_constant] = 0
        valid = ~self.x_constant
        sizes = torch.zeros(anchors.numel(), dtype=torch.long)
        for position in range(anchors.numel()):
            sizes[position] = int(((assignment == position) & valid).sum())
        return anchors, assignment, sizes

    # -- preprocessing -----------------------------------------------------
    def standardize_x(self, x):
        x = torch.as_tensor(x, dtype=self.dtype)
        block = (x - self.x_mean) / self.x_std
        return torch.where(self.x_constant.unsqueeze(0), torch.zeros_like(block), block)

    def standardize_y(self, y):
        y = torch.as_tensor(y, dtype=self.dtype)
        return (y - self.y_mean) / self.y_std

    def inverse_y(self, prediction):
        return prediction * self.y_std + self.y_mean

    def _tokens(self, z):
        contribution = z.unsqueeze(-1) * self.weights.unsqueeze(0)
        tokens = torch.zeros(z.shape[0], self.actual_groups, self.rank, dtype=z.dtype,
                             device=z.device)
        tokens.index_add_(1, self.assignment, contribution)
        scale = self.group_sizes.to(z.dtype).clamp_min(1.0).sqrt()
        tokens = tokens / scale.unsqueeze(0).unsqueeze(-1)
        return tokens.reshape(z.shape[0], self.actual_groups * self.rank)

    # -- forward -----------------------------------------------------------
    def forward(self, x):
        z = self.standardize_x(x)
        if self.variant == 'raw':
            return self.backbone(z)
        ensemble = self.backbone(self._tokens(z))
        linear = z @ self.beta + self.bias
        return ensemble + linear.unsqueeze(1)

    # -- bookkeeping -------------------------------------------------------
    def parameter_count(self):
        return int(sum(parameter.numel() for parameter in self.parameters()))

    def ridge_penalty(self):
        if self.variant != 'grouped_additive':
            return torch.zeros((), dtype=self.dtype)
        return self.beta_ridge * self.beta.square().sum()

    def group_metadata(self):
        return {'variant': self.variant, 'markers': self.markers,
                'groups_requested': self.groups_requested,
                'groups_actual': self.actual_groups,
                'anchors': None if self.anchors is None else self.anchors.tolist(),
                'group_sizes': None if self.anchors is None else self.group_sizes.tolist(),
                'constant_markers': int(self.x_constant.sum()),
                'rank': self.rank,
                'token_dimension': (None if self.variant == 'raw'
                                    else self.actual_groups * self.rank)}
