"""Genomic TabM, environment-sharing backbones and induced marker attention."""
from __future__ import annotations

import math
import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


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


class EnvironmentLinearBatchEnsemble(nn.Module):
    """Official LinearBatchEnsemble with per-environment input/output diagonal scales."""

    def __init__(self, linear, environments):
        super().__init__()
        required = ('weight', 'r', 's', 'in_features', 'out_features', 'k')
        if any(not hasattr(linear, name) for name in required):
            raise TypeError('linear must be an official tabm.LinearBatchEnsemble')
        self.linear = linear
        self.environments = int(environments)
        if self.environments < 1:
            raise ValueError('environments must be positive')
        self.in_features = int(linear.in_features)
        self.out_features = int(linear.out_features)
        self.k = int(linear.k)
        self.u = nn.Parameter(torch.ones(self.environments, self.in_features,
                                         dtype=linear.weight.dtype))
        self.v = nn.Parameter(torch.ones(self.environments, self.out_features,
                                         dtype=linear.weight.dtype))

    def forward(self, x, environment):
        environment = int(environment)
        if not 0 <= environment < self.environments:
            raise ValueError('environment index out of range')
        if x.ndim != 3 or x.shape[-2] != self.k or x.shape[-1] != self.in_features:
            raise ValueError('x must have shape (batch, members, in_features)')
        linear = self.linear
        out = x * linear.r * self.u[environment]
        out = out @ linear.weight.T
        out = out * linear.s * self.v[environment]
        if linear.bias is not None:
            out = out + linear.bias
        return out


class SharedTabM(GenomicTabM):
    """The original four-output TabM arm."""

    def __init__(self, train_x, train_y, *, n_blocks=3, d_block=256, dropout=0.1,
                 k=32, dtype=torch.float32):
        super().__init__(train_x, train_y, variant='raw', n_blocks=n_blocks,
                         d_block=d_block, dropout=dropout, k=k, dtype=dtype)
        self.members = int(k)
        self.n_blocks = int(n_blocks)
        self.d_block = int(d_block)
        self.dropout = float(dropout)
        self.environment_count = int(self.environments)

    def forward_environment(self, x, environment):
        environment = int(environment)
        if not 0 <= environment < self.environment_count:
            raise ValueError('environment index out of range')
        return self.forward(x)[..., environment]

    def environment_parameters(self):
        return []

    def environment_penalty(self):
        return self.backbone.output.weight.new_zeros(())

    def parameter_groups(self, weight_decay):
        return [{'params': list(self.parameters()),
                 'weight_decay': float(weight_decay)}]

    def metadata(self):
        return {
            'variant': 'shared',
            'markers': self.markers,
            'environments': self.environment_count,
            'members': self.members,
            'n_blocks': self.n_blocks,
            'd_block': self.d_block,
            'dropout': self.dropout,
            'environment_parameters': 0,
            'environment_penalty_coefficient': 0.0,
            'parameters': self.parameter_count(),
        }


class IndependentTabM(nn.Module):
    """Four independent single-output TabMs, each with its own fit scaler and seed."""

    def __init__(self, train_x, train_y, *, n_blocks=3, d_block=256, dropout=0.1,
                 k=32, seed=0, environment_seeds=None, dtype=torch.float32):
        super().__init__()
        x = torch.as_tensor(train_x, dtype=dtype).detach().cpu()
        y = torch.as_tensor(train_y, dtype=dtype).detach().cpu()
        if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
            raise ValueError('train_x and train_y must share the material axis')
        environments = int(y.shape[1])
        if environments < 1:
            raise ValueError('At least one environment is required')
        if environment_seeds is None:
            seeds = [int(seed)] * environments
        else:
            seeds = [int(value) for value in environment_seeds]
            if len(seeds) != environments:
                raise ValueError('environment_seeds must provide one seed per environment')
        self.dtype = dtype
        self.environment_count = environments
        self.environment_seeds = tuple(seeds)
        self.members = int(k)
        self.n_blocks = int(n_blocks)
        self.d_block = int(d_block)
        self.dropout = float(dropout)
        cpu_state = torch.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
        models = []
        try:
            for environment in range(environments):
                torch.manual_seed(seeds[environment])
                models.append(GenomicTabM(
                    x, y[:, environment:environment + 1], variant='raw',
                    n_blocks=self.n_blocks, d_block=self.d_block,
                    dropout=self.dropout, k=self.members, dtype=dtype))
        finally:
            torch.set_rng_state(cpu_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)
        self.models = nn.ModuleList(models)

    def forward(self, x):
        return torch.cat([model(x) for model in self.models], dim=-1)

    def forward_environment(self, x, environment):
        environment = int(environment)
        if not 0 <= environment < self.environment_count:
            raise ValueError('environment index out of range')
        return self.models[environment](x)[..., 0]

    def environment_model(self, environment):
        return self.models[int(environment)]

    def standardize_x(self, x):
        return self.models[0].standardize_x(x)

    def standardize_y(self, y):
        y = torch.as_tensor(y, dtype=self.dtype, device=self.models[0].y_mean.device)
        if y.ndim != 2 or y.shape[1] != self.environment_count:
            raise ValueError('y must be a material-by-environment matrix')
        return torch.cat([
            self.models[environment].standardize_y(y[:, environment:environment + 1])
            for environment in range(self.environment_count)], dim=-1)

    def inverse_y(self, prediction):
        prediction = torch.as_tensor(prediction)
        if prediction.shape[-1] != self.environment_count:
            raise ValueError('prediction must have one column per environment')
        return torch.cat([
            self.models[environment].inverse_y(
                prediction[..., environment:environment + 1])
            for environment in range(self.environment_count)], dim=-1)

    def environment_parameters(self):
        return []

    def environment_penalty(self):
        return self.models[0].backbone.output.weight.new_zeros(())

    def parameter_groups(self, weight_decay):
        return [{'params': list(self.parameters()),
                 'weight_decay': float(weight_decay)}]

    def parameter_count(self):
        return int(sum(parameter.numel() for parameter in self.parameters()))

    def metadata(self):
        return {
            'variant': 'independent',
            'markers': self.models[0].markers,
            'environments': self.environment_count,
            'environment_seeds': list(self.environment_seeds),
            'members': self.members,
            'n_blocks': self.n_blocks,
            'd_block': self.d_block,
            'dropout': self.dropout,
            'environment_parameters': 0,
            'environment_penalty_coefficient': 0.0,
            'parameters': self.parameter_count(),
            'parameters_per_environment': [model.parameter_count() for model in self.models],
        }


class EnvironmentTabM(SharedTabM):
    """TabM with per-environment multiplicative modulation of every shared linear map."""

    def __init__(self, train_x, train_y, *, n_blocks=3, d_block=256, dropout=0.1,
                 k=32, environment_penalty=1e-3, dtype=torch.float32):
        super().__init__(train_x, train_y, n_blocks=n_blocks, d_block=d_block,
                         dropout=dropout, k=k, dtype=dtype)
        try:
            import tabm
        except ImportError as error:  # pragma: no cover - depends on the environment
            raise ImportError('tabm==0.0.3 is required') from error
        for block in self.backbone.backbone.blocks:
            if len(block) != 3 or not isinstance(block[0], tabm.LinearBatchEnsemble) \
                    or not isinstance(block[2], nn.Dropout):
                raise ValueError('Unexpected official TabM block structure')
            block[0] = EnvironmentLinearBatchEnsemble(block[0], self.environment_count)
        self.environment_penalty_coefficient = float(environment_penalty)
        if self.environment_penalty_coefficient < 0:
            raise ValueError('environment_penalty must be non-negative')

    def _forward_hidden(self, features):
        hidden = [self.backbone.ensemble_view(features)] * self.environment_count
        for block in self.backbone.backbone.blocks:
            hidden = [block[0](state, environment)
                      for environment, state in enumerate(hidden)]
            hidden = [block[1](state) for state in hidden]
            dropout = block[2]
            if self.training and dropout.training and dropout.p > 0:
                if dropout.p >= 1:
                    hidden = [torch.zeros_like(state) for state in hidden]
                else:
                    keep = torch.rand(hidden[0].shape, device=hidden[0].device,
                                      dtype=hidden[0].dtype) >= dropout.p
                    mask = keep.to(hidden[0].dtype) / (1.0 - dropout.p)
                    hidden = [state * mask for state in hidden]
        return hidden

    def forward(self, x):
        features = self.standardize_x(x)
        hidden = self._forward_hidden(features)
        return torch.stack([self.backbone.output(state)[..., environment]
                            for environment, state in enumerate(hidden)], dim=-1)

    def forward_environment(self, x, environment):
        environment = int(environment)
        if not 0 <= environment < self.environment_count:
            raise ValueError('environment index out of range')
        return self.forward(x)[..., environment]

    def environment_parameters(self):
        return [parameter for name, parameter in self.named_parameters()
                if name.endswith('.u') or name.endswith('.v')]

    def environment_penalty(self):
        parameters = self.environment_parameters()
        if not parameters or self.environment_penalty_coefficient == 0:
            return self.backbone.output.weight.new_zeros(())
        deviations = torch.cat([(parameter - 1.0).reshape(-1)
                                for parameter in parameters])
        return self.environment_penalty_coefficient * deviations.square().mean()

    def parameter_groups(self, weight_decay):
        environment = self.environment_parameters()
        environment_ids = {id(parameter) for parameter in environment}
        base = [parameter for parameter in self.parameters()
                if id(parameter) not in environment_ids]
        groups = [{'params': base, 'weight_decay': float(weight_decay)}]
        if environment:
            groups.append({'params': environment, 'weight_decay': 0.0})
        return groups

    def metadata(self):
        metadata = super().metadata()
        environment = self.environment_parameters()
        metadata.update({
            'variant': 'conditioned',
            'environment_parameters': int(sum(parameter.numel() for parameter in environment)),
            'environment_parameter_tensors': len(environment),
            'environment_penalty_coefficient': self.environment_penalty_coefficient,
            'dropout_mask': 'shared across environments within a training batch',
            'all_ones_restores_shared_eval': True,
        })
        return metadata


class ReGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = x.chunk(2, dim=-1)
        return a * F.relu(b)


class PreLNCrossAttention(nn.Module):
    """Pre-LN cross-attention + residual, followed by a Pre-LN ReGLU FFN."""

    def __init__(self, d: int, heads: int, ffn_hidden: int,
                 attention_dropout: float, ffn_dropout: float,
                 residual_dropout: float):
        super().__init__()
        self.norm_q = nn.LayerNorm(d)
        self.norm_kv = nn.LayerNorm(d)
        self.attention = nn.MultiheadAttention(
            d, heads, dropout=attention_dropout, batch_first=True)
        self.attention_dropout = nn.Dropout(residual_dropout)
        self.norm_ffn = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, 2 * ffn_hidden),
            ReGLU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(ffn_hidden, d),
        )
        self.ffn_dropout = nn.Dropout(residual_dropout)

    def forward(self, query: torch.Tensor,
                key_value: torch.Tensor) -> torch.Tensor:
        q = self.norm_q(query)
        kv = self.norm_kv(key_value)
        attention, _ = self.attention(q, kv, kv, need_weights=False)
        updated = query + self.attention_dropout(attention)
        return updated + self.ffn_dropout(self.ffn(self.norm_ffn(updated)))


class InducedBlock(nn.Module):
    """One bidirectional induced-attention block."""

    def __init__(self, d: int, n_latents: int, heads: int,
                 ffn_hidden: int, attention_dropout: float,
                 ffn_dropout: float, residual_dropout: float):
        super().__init__()
        self.latent = nn.Parameter(torch.randn(1, n_latents, d) * 0.02)
        self.latent_read = PreLNCrossAttention(
            d, heads, ffn_hidden, attention_dropout, ffn_dropout,
            residual_dropout)
        self.token_update = PreLNCrossAttention(
            d, heads, ffn_hidden, attention_dropout, ffn_dropout,
            residual_dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch = tokens.shape[0]
        context = self.latent_read(
            self.latent.expand(batch, -1, -1), tokens)
        return self.token_update(tokens, context)


class SharedResponseTokenizer(nn.Module):
    """Marker identity plus a shared low-rank state-response direction."""

    def __init__(self, n_markers: int, d: int, rank: int):
        super().__init__()
        self.identity = nn.Parameter(torch.empty(n_markers, d))
        self.coeff = nn.Parameter(torch.empty(n_markers, rank))
        self.directions = nn.Parameter(torch.empty(d, rank))
        # Identity starts small; coefficients and U are full-scale so the
        # shared response is active at initialization (U must not be zero).
        nn.init.normal_(self.identity, std=0.01)
        nn.init.normal_(self.coeff, std=1.0)
        nn.init.kaiming_uniform_(self.directions, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        response = self.coeff @ self.directions.T       # (p, d)
        return self.identity.unsqueeze(0) + x.unsqueeze(-1) * response.unsqueeze(0)

    def token_norm_mean(self) -> float:
        return float(self.identity.norm(dim=1).mean().detach().cpu())


class MultiQueryReadout(nn.Module):
    """K learnable queries pool the final token representations."""

    def __init__(self, d: int, n_queries: int, heads: int,
                 attention_dropout: float):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, n_queries, d) * 0.02)
        self.norm_q = nn.LayerNorm(d)
        self.norm_kv = nn.LayerNorm(d)
        self.attention = nn.MultiheadAttention(
            d, heads, dropout=attention_dropout, batch_first=True)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch = tokens.shape[0]
        queries = self.queries.expand(batch, -1, -1)
        pooled, _ = self.attention(
            self.norm_q(queries), self.norm_kv(tokens),
            self.norm_kv(tokens), need_weights=False)
        return queries + pooled


class InducedFTTransformer(nn.Module):
    def __init__(
        self,
        n_markers: int,
        d: int = 64,
        rank: int = 16,
        n_blocks: int = 2,
        n_latents: int = 32,
        heads: int = 4,
        ffn_hidden: int = 128,
        n_queries: int = 4,
        readout_hidden: int = 32,
        attention_dropout: float = 0.1,
        ffn_dropout: float = 0.1,
        residual_dropout: float = 0.0,
    ):
        super().__init__()
        self.n_markers = int(n_markers)
        self.d = int(d)
        self.rank = int(rank)
        self.n_blocks = int(n_blocks)
        self.n_latents = int(n_latents)
        self.n_queries = int(n_queries)
        self.heads = int(heads)
        self.tokenizer = SharedResponseTokenizer(n_markers, d, rank)
        self.blocks = nn.ModuleList([
            InducedBlock(d, n_latents, heads, ffn_hidden,
                         attention_dropout, ffn_dropout, residual_dropout)
            for _ in range(n_blocks)
        ])
        self.readout = MultiQueryReadout(
            d, n_queries, heads, attention_dropout)
        self.head = nn.Sequential(
            nn.Linear(n_queries * d, readout_hidden),
            nn.ReLU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(readout_hidden, 1),
        )

    def forward(self, x_cont: torch.Tensor,
                x_cat: torch.Tensor | None = None) -> torch.Tensor:
        if x_cat is not None:
            raise ValueError("InducedFTTransformer is marker-only; x_cat must be None")
        tokens = self.tokenizer(x_cont)
        for block in self.blocks:
            tokens = block(tokens)
        pooled = self.readout(tokens)
        return self.head(pooled.flatten(1))

    def make_parameter_groups(self) -> list[dict]:
        zero_decay: set[int] = set()
        for module in self.modules():
            if isinstance(module, nn.LayerNorm):
                zero_decay.update(id(p) for p in module.parameters())
        for block in self.blocks:
            zero_decay.add(id(block.latent))
        zero_decay.add(id(self.readout.queries))
        for name, parameter in self.named_parameters():
            if name.endswith("bias"):
                zero_decay.add(id(parameter))
        main = [p for p in self.parameters() if id(p) not in zero_decay]
        no_decay = [p for p in self.parameters() if id(p) in zero_decay]
        return [{"params": main}, {"params": no_decay}]

    def parameter_count(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))

    def tokenizer_parameter_count(self) -> int:
        return int(sum(p.numel() for p in self.tokenizer.parameters()))

    def parameter_breakdown(self) -> dict:
        return {
            "tokenizer": self.tokenizer_parameter_count(),
            "induced_blocks": int(sum(
                p.numel() for block in self.blocks for p in block.parameters())),
            "readout": int(sum(p.numel() for p in self.readout.parameters())),
            "head": int(sum(p.numel() for p in self.head.parameters())),
            "total": self.parameter_count(),
        }

    def attention_score_count(self) -> int:
        """Attention scores per sample across all heads (not parameters)."""
        return int(self.n_blocks * 2 * self.n_markers * self.n_latents
                   * self.heads + self.n_queries * self.n_markers * self.heads)


def build_induced_model(n_markers: int, cfg: dict, seed: int) -> InducedFTTransformer:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    return InducedFTTransformer(
        n_markers=n_markers,
        d=int(cfg["d_block"]),
        rank=int(cfg["response_rank"]),
        n_blocks=int(cfg["n_blocks"]),
        n_latents=int(cfg["n_latents"]),
        heads=int(cfg["attention_n_heads"]),
        ffn_hidden=int(cfg["ffn_d_hidden"]),
        n_queries=int(cfg["readout_queries"]),
        readout_hidden=int(cfg["readout_mlp_hidden"]),
        attention_dropout=float(cfg["attention_dropout"]),
        ffn_dropout=float(cfg["ffn_dropout"]),
        residual_dropout=float(cfg["residual_dropout"]),
    )
