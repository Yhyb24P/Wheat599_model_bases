import copy

import numpy as np
import pytest
import torch

from genopred.models.tabm.quantile import build_model, quantile_loss, train
from genopred.models.tabm.interactions import ConnectionBasis, basis_loss


def model_inputs():
    rng = np.random.default_rng(83)
    x = rng.normal(size=(24, 9)).astype('float32')
    y = rng.normal(size=24).astype('float32')
    config = dict(n_blocks=2, d_block=8, dropout=.1, k=3, learning_rate=.001, weight_decay=.001)
    settings = dict(n_quantiles=5, quantile_min=.001, quantile_max=.999,
                    training=dict(max_epochs=2, patience=2, batch_size=8, gradient_clip=5, tie_tolerance=1e-12))
    return x, y, config, settings


@pytest.mark.parametrize('shared', [True, False])
def test_zero_branch_preserves_original_outputs_gradients_and_rng(shared):
    x, y, config, settings = model_inputs()
    model = build_model(x[:16], y[:16], config, 42, settings, 'cpu')
    reference = copy.deepcopy(model)
    rng = torch.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]):
        layer = ConnectionBasis(model.backbone.backbone.blocks[0][0], 2, shared=shared)
    assert torch.equal(rng, torch.get_rng_state())
    model.backbone.backbone.blocks[0][0] = layer
    added = sum(p.numel() for p in model.parameters()) - sum(p.numel() for p in reference.parameters())
    assert added == (2 * (9 + 8 + 3) if shared else 3 * 2 * (9 + 8))
    sample = torch.tensor(x[16:])
    torch.manual_seed(51)
    actual = model(sample)
    torch.manual_seed(51)
    expected = reference(sample)
    torch.testing.assert_close(actual, expected)
    actual.square().mean().backward()
    expected.square().mean().backward()
    torch.testing.assert_close(layer.base.weight.grad, reference.backbone.backbone.blocks[0][0].weight.grad)
    moving = layer.coefficients if shared else layer.v
    assert moving.grad.norm() > 0
    assert layer.u.grad.norm() == 0


@pytest.mark.parametrize('shared', [True, False])
def test_explicit_effective_weights_centering_and_weight_penalty(shared):
    x, y, config, settings = model_inputs()
    model = build_model(x[:16], y[:16], config, 42, settings, 'cpu').double()
    layer = ConnectionBasis(model.backbone.backbone.blocks[0][0], 2, shared=shared)
    with torch.no_grad():
        (layer.coefficients if shared else layer.v).normal_(std=.1)
    model.backbone.backbone.blocks[0][0] = layer
    z = model.standardize_x(torch.tensor(x[:4], dtype=torch.float64))
    repeated = z[:, None, :].expand(-1, 3, -1)
    delta = layer.delta()
    torch.testing.assert_close(delta.mean(0), torch.zeros_like(delta[0]), atol=1e-6, rtol=0)
    weight = layer.base.s[:, :, None] * layer.base.weight * layer.base.r[:, None, :] + delta
    expected = torch.einsum('bmp,mhp->bmh', repeated, weight) + layer.base.bias
    torch.testing.assert_close(layer(repeated), expected)
    torch.testing.assert_close(layer.penalty(), delta.square().sum() / delta.numel())
    target = model.standardize_y(torch.tensor(y[:4, None], dtype=torch.float64))
    levels = torch.linspace(.001, .999, 5, dtype=torch.float64)
    model.eval()
    torch.testing.assert_close(basis_loss(model, torch.tensor(x[:4], dtype=torch.float64), target, levels, 3.),
                               quantile_loss(model(torch.tensor(x[:4], dtype=torch.float64)), target, levels) + 3 * layer.penalty())
    layer(repeated).square().mean().backward()
    assert layer.u.grad.norm() > 0 and layer.v.grad.norm() > 0


@pytest.mark.parametrize('variant', ['shared_basis', 'independent_basis'])
def test_training_preserves_seed_and_fixed_epoch_refit(variant):
    x, y, config, settings = model_inputs()
    config.update(basis_rank=2, basis_penalty=1e5)
    settings['optimizer_variant'] = variant
    first, info = train(x[:16], y[:16], x[16:], y[16:], config, 42, 0, settings, 'cpu')
    torch.randn(10)
    repeated, _ = train(x[:16], y[:16], x[16:], y[16:], config, 42, 0, settings, 'cpu')
    np.testing.assert_array_equal(first, repeated)
    fixed, _ = train(x[:16], y[:16], x[16:], None, config, 42, 0,
                      {**settings, 'fixed_epochs': info['best_epoch']}, 'cpu')
    np.testing.assert_array_equal(first, fixed)
