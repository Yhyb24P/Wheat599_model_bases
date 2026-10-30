"""FT-T-derived candidate: original affine tokenizer + one full marker
interaction block + position-dependent factorized linear readout.

The tokenizer restores the original FT parameterization and initialization:
    t_j = b_j + x_j w_j,
with both b_j and w_j initialized uniformly in [-1/sqrt(d), 1/sqrt(d)].

The interaction block uses full self-attention on all markers, with no
LayerNorm before attention; the FFN is preceded by LayerNorm:
    U  = T0 + MHA(T0, T0, T0)
    T1 = U + FFN(LN(U))

The readout is a rank-k factorization of a per-marker linear map:
    V = G H,  G in R^{p x k}, H in R^{k x d},  y = bias + <H, G^T T1>.
It has no output-range restriction and no query-key pooling step.

Marker-only; no pedigree wheat.A.  "A" in comparisons is the previous
FT-Transformer architecture label.
"""

from __future__ import annotations

import math
import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class ReGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = x.chunk(2, dim=-1)
        return a * F.relu(b)


class FTAffineTokenizer(nn.Module):
    """Original FT continuous-feature embedding, with equal b/w scale."""

    def __init__(self, n_markers: int, d: int):
        super().__init__()
        self.bias = nn.Parameter(torch.empty(n_markers, d))
        self.weight = nn.Parameter(torch.empty(n_markers, d))
        bound = d ** -0.5
        nn.init.uniform_(self.bias, -bound, bound)
        nn.init.uniform_(self.weight, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bias.unsqueeze(0) + x.unsqueeze(-1) * self.weight.unsqueeze(0)

    def token_norm_mean(self) -> float:
        return float(self.bias.norm(dim=1).mean().detach().cpu())


class SDPAMultiheadAttention(nn.Module):
    """Full self-attention with PyTorch SDPA and optional attention dropout."""

    def __init__(self, d: int, heads: int, attention_dropout: float):
        super().__init__()
        if d % heads != 0:
            raise ValueError("d must be divisible by heads")
        self.d = d
        self.heads = heads
        self.head_dim = d // heads
        self.attention_dropout = float(attention_dropout)
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        qkv = self.qkv(x).reshape(batch, length, 3, self.heads,
                                  self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        dropout = self.attention_dropout if self.training else 0.0
        attended = F.scaled_dot_product_attention(
            query, key, value, dropout_p=dropout, is_causal=False)
        attended = attended.transpose(1, 2).reshape(batch, length, self.d)
        return self.out(attended)


class FullMarkerBlock(nn.Module):
    """One full marker self-attention block; no LN before attention."""

    def __init__(self, d: int, heads: int, ffn_hidden: int,
                 attention_dropout: float, ffn_dropout: float,
                 residual_dropout: float):
        super().__init__()
        self.attention = SDPAMultiheadAttention(d, heads, attention_dropout)
        self.norm_ffn = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, 2 * ffn_hidden),
            ReGLU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(ffn_hidden, d),
        )
        self.residual_dropout = nn.Dropout(residual_dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        updated = tokens + self.attention(tokens)
        return updated + self.residual_dropout(self.ffn(self.norm_ffn(updated)))


class FactorizedLinearReadout(nn.Module):
    """y = bias + <H, G^T T>; V = G H is a rank-k marker readout."""

    def __init__(self, n_markers: int, d: int, rank: int,
                 init_scale: float = 0.5):
        super().__init__()
        self.G = nn.Parameter(torch.empty(n_markers, rank))
        self.H = nn.Parameter(torch.empty(rank, d))
        self.bias = nn.Parameter(torch.zeros(1))
        # Keep the initial output scale O(token scale) for a sum over p markers.
        nn.init.normal_(self.G, std=n_markers ** -0.5)
        nn.init.normal_(self.H, std=init_scale * d ** -0.5)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        pooled = torch.einsum("pk,bpd->bkd", self.G, tokens)
        output = torch.einsum("bkd,kd->b", pooled, self.H) + self.bias
        return output.unsqueeze(-1)


class DirectFTTransformer(nn.Module):
    def __init__(
        self,
        n_markers: int,
        d: int = 64,
        heads: int = 4,
        ffn_hidden: int = 128,
        readout_rank: int = 4,
        readout_init_scale: float = 0.5,
        attention_dropout: float = 0.1,
        ffn_dropout: float = 0.1,
        residual_dropout: float = 0.0,
    ):
        super().__init__()
        self.n_markers = int(n_markers)
        self.d = int(d)
        self.heads = int(heads)
        self.readout_rank = int(readout_rank)
        self.tokenizer = FTAffineTokenizer(n_markers, d)
        self.block = FullMarkerBlock(
            d, heads, ffn_hidden, attention_dropout, ffn_dropout,
            residual_dropout)
        self.readout = FactorizedLinearReadout(
            n_markers, d, readout_rank, init_scale=readout_init_scale)

    def forward(self, x_cont: torch.Tensor,
                x_cat: torch.Tensor | None = None) -> torch.Tensor:
        if x_cat is not None:
            raise ValueError("DirectFTTransformer is marker-only; x_cat must be None")
        return self.readout(self.block(self.tokenizer(x_cont)))

    def make_parameter_groups(self) -> list[dict]:
        zero_decay: set[int] = set()
        zero_decay.update(id(p) for p in self.tokenizer.parameters())
        for module in self.modules():
            if isinstance(module, nn.LayerNorm):
                zero_decay.update(id(p) for p in module.parameters())
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
            "interaction_block": int(sum(
                p.numel() for p in self.block.parameters())),
            "readout": int(sum(p.numel() for p in self.readout.parameters())),
            "total": self.parameter_count(),
        }

    def attention_score_count(self) -> int:
        return int(self.heads * self.n_markers * self.n_markers)


def build_direct_model(n_markers: int, cfg: dict, seed: int) -> DirectFTTransformer:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    return DirectFTTransformer(
        n_markers=n_markers,
        d=int(cfg["d_block"]),
        heads=int(cfg["attention_n_heads"]),
        ffn_hidden=int(cfg["ffn_d_hidden"]),
        readout_rank=int(cfg["readout_rank"]),
        readout_init_scale=float(cfg.get("readout_init_scale", 0.5)),
        attention_dropout=float(cfg["attention_dropout"]),
        ffn_dropout=float(cfg["ffn_dropout"]),
        residual_dropout=float(cfg["residual_dropout"]),
    )
