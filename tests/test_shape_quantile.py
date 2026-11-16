import pytest
import torch
from torch.nn import functional as F

from genopred.models.tabm.shape import (
    ShapeTransform, midpoint_pinball, normalized_crps, standardized_shape,
)

DTYPE = torch.float64


def _pairwise_crps(q, y):
    m = q.shape[-1]
    abs_term = (q - y[..., None]).abs().mean(dim=-1)
    diff = (q[..., :, None] - q[..., None, :]).abs()
    spread = diff.sum(dim=(-1, -2)) / (2 * m * m)
    return abs_term - spread


def _sorted_quantiles(*shape):
    return torch.randn(*shape, dtype=DTYPE).sort(dim=-1).values


def test_crps_pinball_and_pairwise_agree():
    torch.manual_seed(0)
    q = _sorted_quantiles(6, 4, 99)
    y = torch.randn(6, 4, dtype=DTYPE)
    crps = normalized_crps(q, y)
    pinball = midpoint_pinball(q, y)
    pairwise = _pairwise_crps(q, y)
    assert torch.allclose(crps, pinball, atol=1e-12)
    assert torch.allclose(crps, pairwise, atol=1e-12)
    assert (crps >= -1e-12).all()


def test_crps_is_differentiable():
    q = _sorted_quantiles(5, 99).requires_grad_(True)
    y = torch.randn(5, dtype=DTYPE)
    normalized_crps(q, y).mean().backward()
    assert q.grad is not None and torch.isfinite(q.grad).all()
    midpoint_pinball(q, y).mean().backward()


def test_shape_is_ordered_and_standardized():
    torch.manual_seed(1)
    r = standardized_shape(torch.randn(7, 98, dtype=DTYPE))
    assert r.shape == (7, 99)
    assert torch.allclose(r.mean(dim=-1), torch.zeros(7, dtype=DTYPE), atol=1e-12)
    assert torch.allclose((r * r).mean(dim=-1), torch.ones(7, dtype=DTYPE), atol=1e-12)
    assert (r[..., 1:] - r[..., :-1] >= -1e-12).all()


def test_free_moments_ordering_and_zero_penalty():
    torch.manual_seed(2)
    model = ShapeTransform("free")
    raw = torch.randn(4, 3, 100, dtype=DTYPE)
    q, penalty = model(raw)
    mu, scale = raw[..., 0], F.softplus(raw[..., 1])
    assert q.shape == (4, 3, 99)
    assert torch.allclose(q.mean(dim=-1), mu, atol=1e-10)
    assert torch.allclose(((q - mu[..., None]) ** 2).mean(dim=-1), scale ** 2, atol=1e-10)
    assert (q[..., 1:] - q[..., :-1] >= -1e-10).all()
    assert penalty.item() == 0.0


def test_shared_template_is_truly_shared():
    torch.manual_seed(3)
    model = ShapeTransform("shared")
    raw = torch.randn(4, 3, 2, dtype=DTYPE)
    q, penalty = model(raw)
    mu, scale = raw[..., 0], F.softplus(raw[..., 1])
    r = (q - mu[..., None]) / scale[..., None]
    template = standardized_shape(model.template_logits)
    assert torch.allclose(r, template.expand_as(r), atol=1e-12)
    assert penalty.item() == 0.0
    other = torch.randn(4, 3, 2, dtype=DTYPE)
    q2, _ = model(other)
    r2 = (q2 - other[..., 0][..., None]) / F.softplus(other[..., 1])[..., None]
    assert torch.allclose(r2, template.expand_as(r2), atol=1e-12)


def test_template_parameters_only_in_shared_and_shrink():
    free = ShapeTransform("free")
    assert not hasattr(free, "template_logits")
    assert list(free.parameters()) == []
    for mode in ("shared", "shrink"):
        model = ShapeTransform(mode)
        assert model.template_logits.shape == (98,)
        assert [p.numel() for p in model.parameters()] == [98]


def test_shrink_penalty_matches_template_gap():
    torch.manual_seed(4)
    model = ShapeTransform("shrink")
    raw = torch.randn(4, 3, 100, dtype=DTYPE)
    _, penalty = model(raw)
    r = standardized_shape(raw[..., 2:])
    r0 = standardized_shape(model.template_logits)
    assert torch.allclose(penalty, ((r - r0) ** 2).mean(), atol=1e-12)
    assert penalty.item() > 0.0
    matched = raw.clone()
    matched[..., 2:] = model.template_logits.detach()
    _, zero = model(matched)
    assert zero.item() < 1e-24


def test_finite_lambda_changes_loss_and_point_identity():
    torch.manual_seed(5)
    model = ShapeTransform("shrink")
    raw = torch.randn(4, 3, 100, dtype=DTYPE, requires_grad=True)
    y = torch.randn(4, 3, dtype=DTYPE)
    q, penalty = model(raw)
    assert torch.allclose(q.mean(dim=-1), raw[..., 0], atol=1e-10)
    loss = normalized_crps(q, y).mean() + 0.5 * penalty
    loss.backward()
    assert model.template_logits.grad is not None
    assert torch.isfinite(model.template_logits.grad).all()
    assert model.template_logits.grad.abs().sum() > 0


def test_extreme_logits_finite_and_differentiable():
    for value in (1e6, -1e6, 1e300):
        model = ShapeTransform("shrink")
        raw = torch.randn(2, 100, dtype=DTYPE, requires_grad=True)
        with torch.no_grad():
            raw[..., 2:] = value
        q, penalty = model(raw)
        assert torch.isfinite(q).all() and torch.isfinite(penalty).all()
        loss = normalized_crps(q, torch.zeros(2, dtype=DTYPE)).mean() + penalty
        loss.backward()
        assert torch.isfinite(raw.grad).all()
    shared = ShapeTransform("shared")
    with torch.no_grad():
        shared.template_logits.fill_(1e6)
    q, _ = shared(torch.randn(2, 2, dtype=DTYPE))
    assert torch.isfinite(q).all()
    raw = torch.tensor([[0., -1000.]], dtype=torch.float32, requires_grad=True)
    q, _ = ShapeTransform('shared', dtype=torch.float32)(raw)
    assert torch.isfinite(q).all()
    assert torch.any(q != 0)


def test_no_input_leakage_between_location_scale_and_shape():
    torch.manual_seed(6)
    model = ShapeTransform("free")
    raw = torch.randn(3, 100, dtype=DTYPE)
    q, _ = model(raw)
    r = (q - raw[..., 0][..., None]) / F.softplus(raw[..., 1])[..., None]
    shifted = raw.clone()
    shifted[..., 0] += 5.0
    shifted[..., 1] += 2.0
    q2, _ = model(shifted)
    r2 = (q2 - shifted[..., 0][..., None]) / F.softplus(shifted[..., 1])[..., None]
    assert torch.allclose(r, r2, atol=1e-12)
    relit = raw.clone()
    relit[..., 2:] += 1.0
    q3, _ = model(relit)
    assert torch.allclose(q3.mean(dim=-1), raw[..., 0], atol=1e-10)


def test_normal_init_recovers_normal_quantile_shape():
    model = ShapeTransform("shared", normal_init=True)
    r0 = standardized_shape(model.template_logits)
    tau = (torch.arange(99, dtype=DTYPE) + 0.5) / 99
    z = torch.special.ndtri(tau)
    centered = z - z.mean()
    target = centered / torch.sqrt((centered * centered).mean())
    assert torch.allclose(r0, target, atol=1e-10)


def test_mode_and_dimension_validation():
    with pytest.raises(ValueError):
        ShapeTransform("unknown")
    with pytest.raises(ValueError):
        ShapeTransform("free")(torch.randn(3, 2, dtype=DTYPE))
    with pytest.raises(ValueError):
        ShapeTransform("shared")(torch.randn(3, 100, dtype=DTYPE))
