"""Checks for the environment-sharing TabM variants."""

import copy
import sys

import pytest
import torch

TABM_DEPENDENCIES = '/tmp/wheat599_tabm_deps'
if TABM_DEPENDENCIES not in sys.path:
    sys.path.insert(0, TABM_DEPENDENCIES)

import tabm  # noqa: E402

from wheat599.environment_tabm import (EnvironmentLinearBatchEnsemble,  # noqa: E402
                                        EnvironmentTabM, IndependentTabM, SharedTabM)

torch.set_num_threads(1)

TINY = {'n_blocks': 2, 'd_block': 8, 'dropout': 0.0, 'k': 2, 'dtype': torch.float64}


def data(seed=3, materials=30, markers=8, environments=4):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(materials, markers, generator=generator, dtype=torch.float64)
    y = torch.randn(materials, environments, generator=generator, dtype=torch.float64)
    y[:, 1] = y[:, 1] * 2.0 + 3.0
    y[:, 2] = y[:, 2] * 0.5 - 2.0
    return x, y


def build_shared(x, y, seed=7, **overrides):
    settings = dict(TINY)
    settings.update(overrides)
    torch.manual_seed(seed)
    return SharedTabM(x, y, **settings)


def build_conditioned(x, y, seed=7, **overrides):
    settings = dict(TINY)
    settings.update(overrides)
    torch.manual_seed(seed)
    return EnvironmentTabM(x, y, **settings)


def build_independent(x, y, seed=7, seeds=None, **overrides):
    settings = dict(TINY)
    settings.update(overrides)
    return IndependentTabM(x, y, seed=seed, environment_seeds=seeds, **settings)


def test_environment_layer_all_ones_bias_and_environment_gradient():
    torch.manual_seed(1)
    layer = tabm.LinearBatchEnsemble(4, 3, bias=True, k=2, scaling_init='ones',
                                     dtype=torch.float64)
    x = torch.randn(5, 2, 4, dtype=torch.float64)
    wrapper = EnvironmentLinearBatchEnsemble(layer, 2)
    assert torch.equal(wrapper.u, torch.ones_like(wrapper.u))
    assert torch.equal(wrapper.v, torch.ones_like(wrapper.v))
    expected = layer(x)
    assert torch.allclose(wrapper(x, 0), expected, atol=1e-12, rtol=0)

    base_environment_one = wrapper(x, 1).clone()
    with torch.no_grad():
        wrapper.u[0].add_(0.5)
    assert not torch.allclose(wrapper(x, 0), expected)
    assert torch.allclose(wrapper(x, 1), base_environment_one, atol=1e-12, rtol=0)

    bias_layer = tabm.LinearBatchEnsemble(4, 3, bias=True, k=2, scaling_init='ones',
                                          dtype=torch.float64)
    bias_wrapper = EnvironmentLinearBatchEnsemble(bias_layer, 2)
    with torch.no_grad():
        bias_wrapper.u.zero_()
        bias_wrapper.v.zero_()
    got = bias_wrapper(x, 0)
    expected_bias = bias_layer.bias.unsqueeze(0).expand_as(got)
    assert torch.allclose(got, expected_bias, atol=1e-12, rtol=0)

    gradient_layer = tabm.LinearBatchEnsemble(4, 3, bias=True, k=2, scaling_init='ones',
                                              dtype=torch.float64)
    gradient_wrapper = EnvironmentLinearBatchEnsemble(gradient_layer, 2)
    gradient_wrapper(x, 0).square().mean().backward()
    assert gradient_wrapper.u.grad is not None
    assert gradient_wrapper.v.grad is not None
    assert gradient_wrapper.u.grad[0].abs().sum() > 0
    assert gradient_wrapper.v.grad[0].abs().sum() > 0
    assert torch.equal(gradient_wrapper.u.grad[1], torch.zeros_like(
        gradient_wrapper.u.grad[1]))


def test_conditioned_all_ones_matches_shared_and_copies_parameters():
    x, y = data()
    shared = build_shared(x, y, seed=11)
    conditioned = build_conditioned(x, y, seed=11)
    shared.eval()
    conditioned.eval()
    with torch.no_grad():
        shared_output = shared(x)
        conditioned_output = conditioned(x)
    assert shared_output.shape == (x.shape[0], TINY['k'], y.shape[1])
    assert conditioned_output.shape == shared_output.shape
    assert torch.allclose(conditioned_output, shared_output, atol=1e-12, rtol=0)
    assert torch.allclose(conditioned.forward_environment(x, 2),
                          shared_output[..., 2], atol=1e-12, rtol=0)
    assert torch.allclose(shared.forward_environment(x, 2),
                          shared_output[..., 2], atol=1e-12, rtol=0)
    assert torch.allclose(conditioned.environment_penalty(),
                          torch.zeros((), dtype=torch.float64), atol=1e-12, rtol=0)

    for shared_block, conditioned_block in zip(shared.backbone.backbone.blocks,
                                                conditioned.backbone.backbone.blocks):
        official = shared_block[0]
        adapted = conditioned_block[0].linear
        for name in ('weight', 'r', 's', 'bias'):
            assert torch.allclose(getattr(adapted, name), getattr(official, name),
                                  atol=1e-12, rtol=0)
        assert torch.equal(conditioned_block[0].u,
                           torch.ones_like(conditioned_block[0].u))
        assert torch.equal(conditioned_block[0].v,
                           torch.ones_like(conditioned_block[0].v))
    assert torch.allclose(conditioned.backbone.output.weight,
                          shared.backbone.output.weight, atol=1e-12, rtol=0)


def test_conditioned_environment_effect_penalty_and_weight_decay_groups():
    x, y = data()
    conditioned = build_conditioned(x, y, seed=12, environment_penalty=1e-3)
    conditioned.eval()
    with torch.no_grad():
        before = conditioned(x)
        first = conditioned.backbone.backbone.blocks[0][0]
        first.u[0].add_(0.5)
        after = conditioned(x)
    assert not torch.allclose(before[..., 0], after[..., 0])
    assert torch.allclose(before[..., 1:], after[..., 1:], atol=1e-12, rtol=0)
    assert float(conditioned.environment_penalty()) > 0
    parameters = conditioned.environment_parameters()
    assert sum(parameter.numel() for parameter in parameters) == 4 * (8 + 8) * 2
    groups = conditioned.parameter_groups(0.1)
    environment_ids = {id(parameter) for parameter in parameters}
    assert len(groups) == 2
    for group in groups:
        if any(id(parameter) in environment_ids for parameter in group['params']):
            assert group['weight_decay'] == 0.0
        else:
            assert group['weight_decay'] == 0.1
    metadata = conditioned.metadata()
    assert metadata['variant'] == 'conditioned'
    assert metadata['environment_parameters'] == sum(
        parameter.numel() for parameter in parameters)
    assert metadata['environment_penalty_coefficient'] == 1e-3


def test_conditioned_shared_dropout_mask_and_zero_dropout_equivalence():
    x, y = data()
    dropout_model = build_conditioned(x, y, seed=21, dropout=0.5)
    dropout_model.train()
    hidden = dropout_model._forward_hidden(dropout_model.standardize_x(x))
    for environment in range(1, len(hidden)):
        assert torch.allclose(hidden[0], hidden[environment], atol=1e-12, rtol=0)

    shared = build_shared(x, y, seed=22, dropout=0.0)
    conditioned = build_conditioned(x, y, seed=22, dropout=0.0)
    shared.train()
    conditioned.train()
    with torch.no_grad():
        assert torch.allclose(conditioned(x), shared(x), atol=1e-12, rtol=0)


def test_conditioned_fit_statistics_fixed_and_state_reload():
    x, y = data()
    conditioned = build_conditioned(x, y, seed=13)
    conditioned.eval()
    statistics = {name: getattr(conditioned, name).clone()
                  for name in ('x_mean', 'x_std', 'y_mean', 'y_std')}
    with torch.no_grad():
        conditioned(torch.randn_like(x))
    for name, value in statistics.items():
        assert torch.equal(getattr(conditioned, name), value)
    with torch.no_grad():
        expected = conditioned(x)
    state = copy.deepcopy(conditioned.state_dict())
    restored = build_conditioned(x, y, seed=99)
    restored.load_state_dict(state)
    restored.eval()
    with torch.no_grad():
        assert torch.allclose(restored(x), expected, atol=1e-12, rtol=0)


def test_independent_scalers_seed_order_and_output():
    x, y = data()
    model = build_independent(x, y, seed=3, seeds=[3, 3, 3, 3])
    model.eval()
    with torch.no_grad():
        output = model(x)
    assert output.shape == (x.shape[0], TINY['k'], y.shape[1])
    standardised = model.standardize_y(y)
    assert torch.allclose(standardised.mean(0), torch.zeros(y.shape[1], dtype=torch.float64),
                          atol=1e-12, rtol=0)
    assert torch.allclose(standardised.std(0, unbiased=False),
                          torch.ones(y.shape[1], dtype=torch.float64), atol=1e-12, rtol=0)
    assert torch.allclose(model.inverse_y(standardised), y, atol=1e-12, rtol=0)
    for environment in range(y.shape[1]):
        assert torch.allclose(model.forward_environment(x, environment),
                              output[..., environment], atol=1e-12, rtol=0)
    torch.rand(100)
    other = build_independent(x, y, seed=3, seeds=[3, 3, 3, 3])
    for first, second in zip(model.models, other.models):
        for key in first.state_dict():
            assert torch.allclose(first.state_dict()[key], second.state_dict()[key],
                                  atol=1e-12, rtol=0)
    statistics = {name: getattr(model.models[0], name).clone()
                  for name in ('x_mean', 'x_std', 'y_mean', 'y_std')}
    with torch.no_grad():
        model(torch.randn_like(x))
    for name, value in statistics.items():
        assert torch.equal(getattr(model.models[0], name), value)


def test_independent_parameters_are_not_shared_and_reload():
    x, y = data()
    model = build_independent(x, y, seed=31, seeds=[31, 32, 33, 34])
    model.eval()
    with torch.no_grad():
        expected = model(x)
    state = copy.deepcopy(model.state_dict())
    identifiers = [id(parameter) for submodel in model.models
                   for parameter in submodel.parameters()]
    assert len(identifiers) == len(set(identifiers))
    unchanged = copy.deepcopy(model.models[1].state_dict())
    with torch.no_grad():
        model.models[0].backbone.output.weight.add_(1.0)
    for key in unchanged:
        assert torch.equal(model.models[1].state_dict()[key], unchanged[key])
    restored = build_independent(x, y, seed=99, seeds=[99, 99, 99, 99])
    restored.load_state_dict(state)
    restored.eval()
    with torch.no_grad():
        assert torch.allclose(restored(x), expected, atol=1e-12, rtol=0)
    assert model.environment_parameters() == []
    assert float(model.environment_penalty()) == 0.0
    assert len(model.parameter_groups(0.1)) == 1
    assert model.metadata()['variant'] == 'independent'


def test_shared_fit_statistics_penalty_and_groups():
    x, y = data()
    shared = build_shared(x, y, seed=41)
    statistics = {name: getattr(shared, name).clone()
                  for name in ('x_mean', 'x_std', 'y_mean', 'y_std')}
    with torch.no_grad():
        shared(torch.randn_like(x))
    for name, value in statistics.items():
        assert torch.equal(getattr(shared, name), value)
    assert shared.environment_parameters() == []
    assert float(shared.environment_penalty()) == 0.0
    assert len(shared.parameter_groups(0.1)) == 1
    assert shared.metadata()['variant'] == 'shared'
    assert shared.metadata()['parameters'] == shared.parameter_count()


def test_conditioned_environment_scales_receive_gradients():
    x, y = data()
    conditioned = build_conditioned(x, y, seed=51, dropout=0.0)
    conditioned.train()
    output = conditioned(x)
    output.square().mean().backward()
    parameters = conditioned.environment_parameters()
    assert parameters
    assert all(parameter.grad is not None for parameter in parameters)
    assert any(parameter.grad.abs().sum() > 0 for parameter in parameters)


def test_shared_seed_before_build_and_state_reload():
    x, y = data()
    first = build_shared(x, y, seed=41)
    second = build_shared(x, y, seed=41)
    for key in first.state_dict():
        assert torch.allclose(first.state_dict()[key], second.state_dict()[key],
                              atol=1e-12, rtol=0)
    first.eval()
    with torch.no_grad():
        expected = first(x)
    state = copy.deepcopy(first.state_dict())
    restored = build_shared(x, y, seed=99)
    restored.load_state_dict(state)
    restored.eval()
    with torch.no_grad():
        assert torch.allclose(restored(x), expected, atol=1e-12, rtol=0)
