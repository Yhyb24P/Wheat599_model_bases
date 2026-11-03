"""Permutation-equivariant iterative marker posterior correction."""

import torch
from torch import nn


class PosteriorCorrector(nn.Module):
    """Update four-environment effect estimates using the full likelihood."""

    def __init__(self, environments=4, width=64, heads=4, hidden=128, steps=4):
        super().__init__()
        self.environments = environments
        self.steps = steps
        features = 4 * environments + 2 * environments * environments
        self.input = nn.Linear(features, width)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.norm_attention = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, width))
        self.norm_ffn = nn.LayerNorm(width)
        self.output = nn.Linear(width, environments)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, likelihood, score, diagonal, gaussian_mean):
        p, q = score.shape
        if q != self.environments or diagonal.shape != (p, q, q):
            raise ValueError('Incompatible marker or environment dimensions')
        correction = torch.zeros_like(gaussian_mean)
        task = likelihood.b.reshape(1, q * q).expand(p, -1)
        # The prior supports only the range of B, including when B is singular.
        task_support = torch.linalg.pinv(likelihood.b, hermitian=True) @ likelihood.b
        for _ in range(self.steps):
            residual = score - likelihood.hessian_apply(gaussian_mean + correction)
            features = torch.cat((score, diagonal.reshape(p, -1), task,
                                  gaussian_mean, correction, residual), dim=1)
            tokens = self.input(features).unsqueeze(0)
            context, _ = self.attention(tokens, tokens, tokens, need_weights=False)
            tokens = self.norm_attention(tokens + context)
            tokens = self.norm_ffn(tokens + self.ffn(tokens))
            correction = correction + self.output(tokens.squeeze(0)) @ task_support
        return correction
