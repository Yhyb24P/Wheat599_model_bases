"""Checks for the official-TabM genomic adaptation."""

import copy
import sys

import torch

TABM_DEPENDENCIES = '/tmp/wheat599_tabm_deps'
if TABM_DEPENDENCIES not in sys.path:
    sys.path.insert(0, TABM_DEPENDENCIES)

from wheat599.genomic_tabm import GenomicTabM  # noqa: E402

torch.set_num_threads(1)

TINY = {'n_blocks': 1, 'd_block': 8, 'dropout': 0.0, 'k': 2, 'rank': 2, 'groups': 4}


def data(seed=3, materials=30, markers=9, environments=4):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(materials, markers, generator=generator, dtype=torch.float64)
    x[:, 3] = 1.0  # constant column
    y = torch.randn(materials, environments, generator=generator, dtype=torch.float64)
    beta = torch.randn(markers, environments, generator=generator, dtype=torch.float64) * 0.01
    return x, y, beta


def build(x, y, beta=None, variant='grouped_additive', **overrides):
    settings = {'variant': variant, 'dtype': torch.float64, **TINY}
    settings.update(overrides)
    if beta is not None:
        return GenomicTabM(x, y, beta=beta, bias=torch.zeros(y.shape[1], dtype=torch.float64),
                           **settings)
    return GenomicTabM(x, y, **settings)


def test_fit_only_statistics_and_grouping():
    x, y, beta = data()
    model = build(x, y, beta)
    assert torch.allclose(model.x_mean, x.mean(dim=0), atol=1e-12, rtol=0)
    assert torch.allclose(model.x_std[~model.x_constant],
                          x.std(dim=0, unbiased=False)[~model.x_constant], atol=1e-12, rtol=0)
    assert bool(model.x_constant[3])
    anchors_before = model.anchors.clone()
    different = torch.randn(20, x.shape[1], generator=torch.Generator().manual_seed(7),
                            dtype=torch.float64)
    model(different)
    assert torch.equal(model.anchors, anchors_before)
    assert torch.allclose(model.x_mean, x.mean(dim=0), atol=1e-12, rtol=0)
    x2, y2, beta2 = data(seed=11)
    other = build(x2, y2, beta2)
    assert not torch.equal(other.anchors, anchors_before)


def test_every_marker_is_covered():
    x, y, beta = data()
    model = build(x, y, beta)
    assignment = model.assignment
    assert int(assignment.min()) >= 0
    assert int(assignment.max()) < model.actual_groups
    for position, anchor in enumerate(model.anchors.tolist()):
        assert int(assignment[anchor]) == position
    assert int(assignment[model.x_constant].eq(0).all())
    valid = (~model.x_constant).sum().item()
    assert int(model.group_sizes.sum()) == valid
    metadata = model.group_metadata()
    assert metadata['groups_actual'] == model.actual_groups
    assert metadata['token_dimension'] == model.actual_groups * model.rank
    assert metadata['constant_markers'] == int(model.x_constant.sum())


def test_zero_branch_is_exactly_the_ridge_fit():
    x, y, beta = data()
    model = build(x, y, beta)
    model.eval()
    with torch.no_grad():
        output = model(x)
        expected = model.standardize_x(x) @ beta
    assert output.shape == (x.shape[0], TINY['k'], y.shape[1])
    assert torch.allclose(output, expected.unsqueeze(1).expand_as(output),
                          atol=1e-12, rtol=0)


def test_group_count_shrinks_when_candidates_are_few():
    x, y, beta = data(markers=4)
    x[:, 1] = 0.5  # a second constant column
    model = build(x, y, beta, groups=64)
    assert model.actual_groups == int((~model.x_constant).sum())
    assert model.group_metadata()['groups_actual'] == model.actual_groups


def test_official_backbone_receives_gradients():
    x, y, beta = data()
    model = build(x, y, beta)
    model(x).square().mean().backward()
    backbone_gradients = [parameter.grad for parameter in model.backbone.parameters()
                          if parameter.grad is not None]
    assert backbone_gradients
    assert any(float(gradient.abs().sum()) > 0 for gradient in backbone_gradients)
    assert float(model.beta.grad.abs().sum()) > 0
    assert float(model.backbone.output.weight.grad.abs().sum()) > 0
    # With a zero-initialised head the tokenizer is frozen at step zero by design;
    # the first update moves the head and the tokenizer then receives gradients.
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    model(x).square().mean().backward()
    assert model.weights.grad is not None
    assert float(model.weights.grad.abs().sum()) > 0


def test_eval_predictions_are_batch_consistent():
    x, y, beta = data(seed=13)
    model = build(x, y, beta).eval()
    with torch.no_grad():
        full = model(x)
        chunks = torch.cat([model(x[index:index + 7]) for index in range(0, x.shape[0], 7)], 0)
    assert torch.allclose(full, chunks, atol=1e-12, rtol=0)


def test_constant_markers_are_zeroed_for_any_input():
    x, y, beta = data(seed=29)
    model = build(x, y, beta)
    probe = x.clone()
    probe[:, 3] = torch.linspace(-5.0, 5.0, probe.shape[0], dtype=torch.float64)
    standardised = model.standardize_x(probe)
    assert torch.equal(standardised[:, 3], torch.zeros(probe.shape[0], dtype=torch.float64))
    assert torch.equal(model.standardize_x(x)[:, 3],
                       torch.zeros(x.shape[0], dtype=torch.float64))


def test_state_dict_is_tensor_only_and_restores_statistics_across_instances():
    x, y, beta = data(seed=31)
    source = build(x, y, beta).eval()
    state = copy.deepcopy(source.state_dict())
    assert all(torch.is_tensor(value) for value in state.values())
    x2, y2, beta2 = data(seed=37)
    target = build(x2, y2, beta2).eval()
    assert not torch.equal(target.x_mean, source.x_mean)
    target.load_state_dict(state)
    assert torch.equal(target.x_mean, source.x_mean)
    assert torch.equal(target.anchors, source.anchors)
    assert target.actual_groups == source.actual_groups
    assert target.group_metadata() == source.group_metadata()
    with torch.no_grad():
        assert torch.equal(target(x), source(x))


def test_state_dict_roundtrip_reproduces_predictions():
    x, y, beta = data(seed=17)
    model = build(x, y, beta).eval()
    clone = build(x, y, beta).eval()
    clone.load_state_dict(copy.deepcopy(model.state_dict()))
    with torch.no_grad():
        assert torch.equal(model(x), clone(x))


def test_raw_variant_keeps_official_initialisation():
    x, y, _ = data(seed=19)
    model = build(x, y, variant='raw')
    assert model.variant == 'raw'
    assert float(model.backbone.output.weight.abs().sum()) > 0
    model.eval()
    with torch.no_grad():
        output = model(x)
    assert output.shape == (x.shape[0], TINY['k'], y.shape[1])


def test_ridge_penalty_matches_the_parameter():
    x, y, beta = data(seed=23)
    model = build(x, y, beta, beta_ridge=0.7)
    expected = 0.7 * model.beta.square().sum()
    assert torch.allclose(model.ridge_penalty(), expected, atol=1e-12, rtol=0)
    raw = build(x, y, variant='raw', beta_ridge=0.7)
    assert float(raw.ridge_penalty()) == 0.0


def test_first_anchor_uses_raw_fitting_variance():
    generator = torch.Generator().manual_seed(41)
    x = torch.randn(40, 6, generator=generator, dtype=torch.float64)
    x[:, 4] *= 7.0  # largest raw variance, but standardisation hides it
    y = torch.randn(40, 4, generator=generator, dtype=torch.float64)
    beta = torch.zeros(6, 4, dtype=torch.float64)
    model = build(x, y, beta, groups=1)
    assert model.actual_groups == 1
    assert int(model.anchors[0]) == 4
    assert model.anchors[0] == int(torch.argmax(x.std(dim=0, unbiased=False)))


def test_token_variance_matches_backbone_input_scale():
    generator = torch.Generator().manual_seed(43)
    materials, markers = 2000, 500
    x = torch.randn(materials, markers, generator=generator, dtype=torch.float32)
    y = torch.randn(materials, 4, generator=generator, dtype=torch.float32)
    beta = torch.zeros(markers, 4, dtype=torch.float32)
    model = build(x, y, beta, groups=8, rank=4, dtype=torch.float32)
    with torch.no_grad():
        tokens = model._tokens(model.standardize_x(x))
    variance = float(tokens.var(unbiased=False))
    assert 0.5 < variance < 2.0, variance


def test_float32_construction_and_forward():
    generator = torch.Generator().manual_seed(47)
    x = torch.randn(32, 7, generator=generator, dtype=torch.float32)
    y = torch.randn(32, 4, generator=generator, dtype=torch.float32)
    beta = torch.randn(7, 4, generator=generator, dtype=torch.float32) * 0.1
    model = build(x, y, beta, dtype=torch.float32)
    output = model(x)
    assert output.dtype == torch.float32
    assert output.shape == (32, 2, 4)
    assert torch.isfinite(output).all()


def test_cuda_construction_from_cuda_inputs_and_forward():
    if not torch.cuda.is_available():
        return
    generator = torch.Generator().manual_seed(53)
    x = torch.randn(32, 9, generator=generator, dtype=torch.float32)
    y = torch.randn(32, 4, generator=generator, dtype=torch.float32)
    beta = torch.randn(9, 4, generator=generator, dtype=torch.float32) * 0.1
    model = build(x.cuda(), y.cuda(), beta, dtype=torch.float32)
    assert model.x_mean.device.type == 'cpu'
    assert model.anchors.device.type == 'cpu'
    model = model.to('cuda')
    output = model(x.cuda())
    assert output.device.type == 'cuda'
    assert torch.isfinite(output).all()
    prediction = model(x.cuda())
    assert torch.allclose(output, prediction, atol=1e-6, rtol=0)


def test_cuda_eval_matches_cpu_eval():
    if not torch.cuda.is_available():
        return
    generator = torch.Generator().manual_seed(59)
    x = torch.randn(24, 8, generator=generator, dtype=torch.float32)
    y = torch.randn(24, 4, generator=generator, dtype=torch.float32)
    beta = torch.randn(8, 4, generator=generator, dtype=torch.float32) * 0.1
    model = build(x, y, beta, dtype=torch.float32).eval()
    with torch.no_grad():
        cpu = model(x)
        gpu = model.to('cuda')(x.cuda()).cpu()
    assert torch.allclose(cpu, gpu, atol=1e-4, rtol=0)
