"""Environment-sharing TabM variants for Wheat599.

All arms preserve the official tabm==0.0.3 backbone, all 1279 markers, the member axis
and the per-member loss. ``SharedTabM`` is the original four-output TabM.
``IndependentTabM`` is four independent single-output TabMs with their own phenotype
scalers and seeded construction. ``EnvironmentTabM`` adds per-environment input/output
multiplicative scales ``u[e]``/``v[e]`` around every official ``LinearBatchEnsemble``:

    y = ((x * r_member * u[e]) @ W.T) * (s_member * v[e]) + bias

``u`` and ``v`` start at one, while ``W``, ``r``, ``s`` and ``bias`` stay official. The
bias is added after the output scaling and is not multiplied by ``v[e]``. With all ones
and eval mode every environment's hidden state equals the shared hidden state, and
environment e reads column e of the original output head; stacking the columns recovers
the shared ``(n, k, q)`` prediction. The shared-dropout-mask training path keeps one
mask per block across environments, so the all-ones stochastic mechanism stays close to
the original TabM.

The environment scales carry a weak mean-normalised L2 prior around one
(``environment_scale_l2``, fixed by the caller). ``parameter_groups`` puts them in a
zero-weight-decay group so ordinary weight decay cannot pull them to zero. The
conditioned output is still ``(n, k, q)``; no extra head or member axis is introduced.
"""

import torch
from torch import nn

from wheat599.genomic_tabm import GenomicTabM


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
