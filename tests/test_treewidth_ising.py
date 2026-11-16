import numpy as np
import pytest
import torch

from genopred.models.structured.supervised_ising import AdditiveMLP
from genopred.models.structured.treewidth_ising import TreewidthIsing


def data(p=9):
    rng = np.random.default_rng(5)
    x = rng.integers(0, 2, (24, p)).astype('float32')
    x[:, 0] = 0
    return x, rng.normal(size=len(x)).astype('float32')


@pytest.mark.parametrize('spins', [3, 5, 7])
def test_exact_moments_and_readout_gradients(spins):
    x, y = data()
    model = TreewidthIsing(x, y, spins=spins).double()
    torch.manual_seed(8)
    model.coupling.data.normal_(std=.5)
    field = torch.randn(3, spins, dtype=torch.float64, requires_grad=True)
    states = (2 * ((torch.arange(1 << spins)[:, None] >> torch.arange(spins)) & 1) - 1).double()
    products = states[:, model.left] * states[:, model.right]
    probabilities = (field @ states.T + products @ model.coupling).softmax(1)
    actual = model.moments(field)
    expected = probabilities @ states, probabilities @ products
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=2e-14, rtol=2e-14)
    weights = [torch.randn_like(a) for a in actual]
    grad = lambda values: torch.autograd.grad(sum((v * w).sum() for v, w in zip(values, weights)),
                                             (field, model.coupling), retain_graph=True)
    for a, b in zip(grad(actual), grad(expected)):
        torch.testing.assert_close(a, b, atol=3e-14, rtol=3e-14)


def test_zero_coupling_matched_initialization_and_parameter_counts():
    x, y = data(1279)
    torch.manual_seed(42)
    model = TreewidthIsing(x, y)
    torch.manual_seed(42)
    control = TreewidthIsing(x, y, coupled=False)
    assert model.left.numel() == 125
    assert sum(p.numel() for p in model.parameters()) == 83514
    assert sum(p.numel() for p in control.parameters() if p.requires_grad) == 83389
    assert sum(p.numel() for p in AdditiveMLP(x, y, width=64).parameters()) == 87424
    for (name, a), (other, b) in zip(model.named_parameters(), control.named_parameters()):
        assert name == other
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    sample = torch.tensor(x[:3])
    torch.testing.assert_close(model(sample), control(sample), atol=1e-6, rtol=1e-5)
    changed = sample.clone()
    changed[:, 0] = 100
    torch.testing.assert_close(model(sample), model(changed))
    torch.testing.assert_close(model(sample[:1]), model(sample)[:1], atol=1e-6, rtol=1e-5)
    model(sample).square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.coupling.grad.norm() > 0


def test_large_fields_are_finite():
    x, y = data()
    model = TreewidthIsing(x, y, spins=5).double()
    model.coupling.data.fill_(10)
    field = torch.tensor([[1000., -1000., 1000., -1000., 1000.]], requires_grad=True, dtype=torch.float64)
    mean, pairs = model.moments(field)
    (mean.sum() + pairs.sum()).backward()
    torch.testing.assert_close(mean, field.sign())
    assert torch.isfinite(field.grad).all() and torch.isfinite(model.coupling.grad).all()


@pytest.mark.parametrize('variant', ['treewidth_ising', 'treewidth_independent'])
def test_training_seed_and_fixed_epoch_refit(variant):
    from genopred.models.tabm.quantile import train
    x, y = data()
    config = dict(spins=5, learning_rate=.002, weight_decay=.01)
    settings = dict(optimizer_variant=variant, n_quantiles=1, quantile_min=.001, quantile_max=.999,
                    training=dict(objective='mse', max_epochs=2, patience=1, batch_size=8,
                                  gradient_clip=5, tie_tolerance=1e-12))
    first, info = train(x[:16], y[:16], x[16:], y[16:], config, 42, 0, settings, 'cpu')
    torch.randn(30)
    second, _ = train(x[:16], y[:16], x[16:], y[16:], config, 42, 0, settings, 'cpu')
    np.testing.assert_array_equal(first, second)
    pred, refit = train(x[:16], y[:16], x[16:], None, config, 42, 0,
                        {**settings, 'fixed_epochs': info['best_epoch']}, 'cpu')
    np.testing.assert_array_equal(first, pred)
    assert refit['best_epoch'] == info['best_epoch'] and refit['best_pcc'] is None
