"""FT-Transformer-derived architecture with shared tokenizer, induced attention,
and multi-query readout.

Design (first complete model, no component ablation):
* tokenizer: t_j = b_j + x_j * U c_j (marker identity + low-rank shared response)
* two bidirectional induced-attention blocks:
    H = MAB(I, T), T_next = MAB(T, H)
  with Pre-LN residual cross-attention and a ReGLU feed-forward block.
* readout: K learnable queries pool the final token representations, then a
  small MLP maps the flattened K x d representation to one phenotype.

No positional encoding is used; marker identity is carried by b_j.  The latent
units and readout queries are not assumed to correspond to LD blocks, genomic
regions, or biological pathways.
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
