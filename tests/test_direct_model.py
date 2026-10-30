"""Shape, parameter, readout-equivalence, and seeding tests for the direct FT-T
candidate (original tokenizer + full attention + factorized readout)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from wheat599.direct_ftt import DirectFTTransformer, build_direct_model


def _tiny_cfg():
    return {
        "d_block": 16,
        "attention_n_heads": 2,
        "ffn_d_hidden": 8,
        "readout_rank": 2,
        "readout_init_scale": 0.5,
        "attention_dropout": 0.0,
        "ffn_dropout": 0.0,
        "residual_dropout": 0.0,
    }


def test_forward_shape_and_parameter_breakdown():
    p = 23
    cfg = _tiny_cfg()
    model = build_direct_model(p, cfg, seed=3)
    model.eval()
    with torch.no_grad():
        out = model(x_cont=torch.randn(3, p))
    assert out.shape == (3, 1)
    assert torch.isfinite(out).all()
    d = cfg["d_block"]
    breakdown = model.parameter_breakdown()
    assert breakdown["tokenizer"] == 2 * p * d
    assert breakdown["readout"] == p * cfg["readout_rank"] + cfg["readout_rank"] * d + 1
    assert breakdown["total"] == model.parameter_count()
    assert model.attention_score_count() == (
        cfg["attention_n_heads"] * p * p)


def test_tokenizer_uses_equal_scale_init():
    p = 40
    model = build_direct_model(p, _tiny_cfg(), seed=5)
    d = _tiny_cfg()["d_block"]
    expected_std = (2 * d ** -0.5) / (12 ** 0.5)
    assert model.tokenizer.bias.std().item() == pytest.approx(expected_std, rel=0.15)
    assert model.tokenizer.weight.std().item() == pytest.approx(expected_std, rel=0.15)


def test_readout_equals_explicit_marker_sum():
    p = 17
    model = build_direct_model(p, _tiny_cfg(), seed=11)
    tokens = torch.randn(1, p, _tiny_cfg()["d_block"])
    with torch.no_grad():
        y = model.readout(tokens).squeeze(-1)
    explicit = (model.readout.G @ model.readout.H)
    manual = float((tokens[0] * explicit).sum())
    assert float(y[0]) == pytest.approx(manual, rel=1e-4, abs=1e-5)


def test_attention_has_no_input_layernorm():
    model = build_direct_model(13, _tiny_cfg(), seed=2)
    norms = [name for name, module in model.block.attention.named_modules()
             if isinstance(module, torch.nn.LayerNorm)]
    assert norms == []
    assert isinstance(model.block.norm_ffn, torch.nn.LayerNorm)


def test_parameter_groups_partition_and_seeding():
    model = build_direct_model(29, _tiny_cfg(), seed=7)
    groups = model.make_parameter_groups()
    all_ids = [id(p) for p in model.parameters()]
    group_ids = [id(p) for g in groups for p in g["params"]]
    assert len(group_ids) == len(set(group_ids))
    assert set(group_ids) == set(all_ids)
    a = build_direct_model(29, _tiny_cfg(), seed=9)
    b = build_direct_model(29, _tiny_cfg(), seed=9)
    c = build_direct_model(29, _tiny_cfg(), seed=10)
    sa, sb, sc = a.state_dict(), b.state_dict(), c.state_dict()
    assert all(torch.equal(sa[k], sb[k]) for k in sa)
    assert any(not torch.equal(sa[k], sc[k]) for k in sa)


def test_eval_mode_is_deterministic():
    model = build_direct_model(19, _tiny_cfg(), seed=1)
    model.eval()
    x = torch.randn(2, 19)
    with torch.no_grad():
        first = model(x_cont=x)
        second = model(x_cont=x)
    assert torch.allclose(first, second)
