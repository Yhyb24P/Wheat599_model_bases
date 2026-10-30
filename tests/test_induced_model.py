"""Shape, parameter, and reproducibility tests for the induced FT-T model."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from wheat599.induced_ftt import InducedFTTransformer, build_induced_model


def _tiny_cfg():
    return {
        "d_block": 16,
        "response_rank": 4,
        "n_blocks": 1,
        "n_latents": 4,
        "attention_n_heads": 2,
        "ffn_d_hidden": 8,
        "readout_queries": 2,
        "readout_mlp_hidden": 4,
        "attention_dropout": 0.0,
        "ffn_dropout": 0.0,
        "residual_dropout": 0.0,
    }


def test_forward_shape_and_parameter_counts():
    p = 23
    cfg = _tiny_cfg()
    model = build_induced_model(p, cfg, seed=3)
    model.eval()
    with torch.no_grad():
        out = model(x_cont=torch.randn(3, p))
    assert out.shape == (3, 1)
    assert torch.isfinite(out).all()
    expected_tokenizer = p * cfg["d_block"] + p * cfg["response_rank"] \
        + cfg["d_block"] * cfg["response_rank"]
    assert model.tokenizer_parameter_count() == expected_tokenizer
    assert model.parameter_count() > expected_tokenizer
    assert model.attention_score_count() == (
        cfg["n_blocks"] * 2 * p * cfg["n_latents"] * cfg["attention_n_heads"]
        + cfg["readout_queries"] * p * cfg["attention_n_heads"]
    )


def test_parameter_groups_are_a_partition():
    model = InducedFTTransformer(29, d=16, rank=4, n_blocks=1,
                                 n_latents=4, heads=2, ffn_hidden=8,
                                 n_queries=2, readout_hidden=4,
                                 attention_dropout=0.0, ffn_dropout=0.0,
                                 residual_dropout=0.0)
    groups = model.make_parameter_groups()
    all_ids = [id(p) for p in model.parameters()]
    group_ids = [id(p) for g in groups for p in g["params"]]
    assert len(all_ids) == len(set(all_ids))
    assert len(group_ids) == len(set(group_ids))
    assert set(group_ids) == set(all_ids)


def test_seeded_initialization_is_reproducible():
    p = 17
    cfg = _tiny_cfg()
    a = build_induced_model(p, cfg, seed=7)
    b = build_induced_model(p, cfg, seed=7)
    c = build_induced_model(p, cfg, seed=8)
    sa, sb, sc = a.state_dict(), b.state_dict(), c.state_dict()
    assert all(torch.equal(sa[k], sb[k]) for k in sa)
    assert any(not torch.equal(sa[k], sc[k]) for k in sa)
