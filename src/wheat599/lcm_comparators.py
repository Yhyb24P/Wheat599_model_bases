"""Parameter-matched neural comparators for locus-complement modulation.

Inputs are centred using the fitting materials by the experiment runner.
Both models retain a complete additive path and predict all environments.
"""

import math

import torch
from torch import nn


class _Comparator(nn.Module):
    """Expose the shared additive initialization and optimizer contract."""

    @property
    def coefficients(self):
        return self.additive.weight.T

    @property
    def bias(self):
        return self.additive.bias

    def initialize(self):
        return self

    def parameter_groups(self, lambda_a, lambda_n):
        other = [parameter for name, parameter in self.named_parameters()
                 if not name.startswith('additive.')]
        return [
            {'params': [self.additive.weight], 'weight_decay': 2 * lambda_a},
            {'params': [self.additive.bias], 'weight_decay': 0.0},
            {'params': other, 'weight_decay': 2 * lambda_n},
        ]

    def proximal(self, threshold):
        pass

    def zero_rows(self):
        return 0

    def count(self):
        return sum(parameter.numel() for parameter in self.parameters())


class AdditiveMLP(_Comparator):
    """Add a shared two-layer nonlinear predictor to marker effects."""

    def __init__(self, markers: int, environments: int = 4, rank: int = 8, variant=None):
        super().__init__()
        width = 2 * rank
        self.additive = nn.Linear(markers, environments)
        self.network = nn.Sequential(
            nn.Linear(markers, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, environments),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.additive(z) + self.network(z)


class ContextModulation(_Comparator):
    """Use a shared nonlinear full-input context to gate marker effects."""

    def __init__(self, markers: int, environments: int = 4, rank: int = 8, variant=None):
        super().__init__()
        self.additive = nn.Linear(markers, environments)
        self.encoder = nn.Linear(markers, rank, bias=False)
        self.response = nn.Sequential(
            nn.Linear(rank, rank), nn.SiLU(), nn.Linear(rank, rank, bias=False)
        )
        self.gates = nn.Parameter(torch.zeros(markers, rank))
        self.scale = math.sqrt(rank)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        context = self.encoder(z)
        response = self.response(context) - self.response(torch.zeros_like(context))
        gate = torch.tanh(response @ self.gates.T / self.scale)
        return self.additive(z * (1 + gate))
