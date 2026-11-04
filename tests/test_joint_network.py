"""Checks for the joint-evidence candidate network."""

import math

import numpy as np
import pytest
import torch

from wheat599.joint_evidence import Evidence
from wheat599.joint_network import JointInference

torch.set_num_threads(1)


def toy(seed=3, materials=24, markers=5, environments=2, lambda_a=3.0):
    generator = torch.Generator().manual_seed(seed)
    z = torch.randn(materials, markers, generator=generator, dtype=torch.float64)
    z = z - z.mean(dim=0)
    z = z / z.square().mean().sqrt()
    y = torch.randn(materials, environments, generator=generator, dtype=torch.float64)
    query = torch.randn(4, markers, generator=generator, dtype=torch.float64)
    return Evidence(z, y, lambda_a=lambda_a), query


def model(evidence, **overrides):
    settings = {'environments': evidence.environments, 'width': 8, 'rounds': 2,
                'add_per_round': 3, 'max_candidates': 6}
    settings.update(overrides)
    torch.manual_seed(11)
    return JointInference(**settings)


def test_forward_shapes_and_gate_dimension():
    evidence, query = toy()
    net = model(evidence)
    out = net(evidence, query)
    expected = (query.shape[0], evidence.environments)
    assert out['prediction'].shape == expected
    assert out['additive_prediction'].shape == expected
    assert out['gate_features'].shape == (net.gate_feature_dim,)
    assert net.gate_feature_dim == 12 + evidence.environments ** 2
    assert out['d'].shape == out['indices'].shape
    assert out['coefficients'].shape == (out['indices'].numel(), evidence.environments)
    assert len(out['trace']) == net.rounds
    assert torch.isfinite(out['prediction']).all()


def test_marker_sign_flip_is_invariant():
    evidence, query = toy(seed=5)
    marker = 2
    signs = torch.ones(evidence.markers, dtype=evidence.dtype)
    signs[marker] = -1.0
    flipped = Evidence(evidence.z * signs, evidence.y, lambda_a=evidence.lambda_a)
    net = model(evidence)
    base = net(evidence, query)
    other = net(flipped, query * signs)
    assert torch.equal(base['indices'].sort().values, other['indices'].sort().values)
    assert torch.allclose(base['prediction'], other['prediction'], atol=1e-9, rtol=0)
    assert torch.allclose(base['gate_features'], other['gate_features'], atol=1e-9, rtol=0)


def test_material_permutation_is_invariant():
    evidence, query = toy(seed=7)
    order = torch.tensor([3, 0, 5, 1, 4, 2, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                          18, 19, 20, 21, 22, 23])
    permuted = Evidence(evidence.z[order], evidence.y[order], lambda_a=evidence.lambda_a)
    net = model(evidence)
    assert torch.allclose(net(evidence, query)['prediction'],
                          net(permuted, query)['prediction'], atol=1e-9, rtol=0)


def test_marker_permutation_is_invariant():
    evidence, query = toy(seed=9)
    order = torch.tensor([2, 0, 4, 1, 3])
    permuted = Evidence(evidence.z[:, order], evidence.y, lambda_a=evidence.lambda_a)
    net = model(evidence)
    assert torch.allclose(net(evidence, query)['prediction'],
                          net(permuted, query[:, order])['prediction'], atol=1e-9, rtol=0)


def test_query_isolation():
    evidence, query = toy(seed=13)
    net = model(evidence)
    first = net(evidence, query)
    second = net(evidence, query + 1.5)
    assert torch.equal(first['indices'], second['indices'])
    assert torch.equal(first['d'], second['d'])
    assert torch.equal(first['gate_features'], second['gate_features'])
    assert not torch.allclose(first['prediction'], second['prediction'])


def test_gradients_reach_network_parameters():
    evidence, query = toy(seed=17)
    net = model(evidence)
    out = net(evidence, query)
    loss = out['prediction'].square().mean()
    loss.backward()
    gradients = {name: parameter.grad for name, parameter in net.named_parameters()}
    assert all(gradient is not None and torch.isfinite(gradient).all()
               for gradient in gradients.values())
    assert any(float(gradient.abs().sum()) > 0 for gradient in gradients.values())
    assert net.encoder[0].weight.grad is not None
    assert net.head[0].weight.grad is not None


class Counter:
    def __init__(self, evidence):
        self.original = evidence.score
        self.calls = []

    def __call__(self, indices=None, coefficients=None):
        self.calls.append((indices is None, coefficients is None))
        return self.original(indices, coefficients)


def test_rounds_use_full_space_conditional_scores(monkeypatch):
    evidence, query = toy(seed=19)
    counter = Counter(evidence)
    monkeypatch.setattr(evidence, 'score', counter)
    net = model(evidence, rounds=3)
    net(evidence, query)
    assert len(counter.calls) == 1 + net.rounds
    assert counter.calls[0] == (True, True)
    assert sum(1 for indices_none, coefficients_none in counter.calls[1:]
               if not indices_none and not coefficients_none) >= 2


def test_relation_variant_shares_parameters_and_differs():
    evidence, query = toy(seed=23)
    joint = model(evidence, variant='joint')
    independent = model(evidence, variant='independent')
    independent.load_state_dict(joint.state_dict())
    assert sum(p.numel() for p in joint.parameters()) == \
        sum(p.numel() for p in independent.parameters())
    first = joint(evidence, query)
    second = independent(evidence, query)
    assert torch.isfinite(first['prediction']).all()
    assert torch.isfinite(second['prediction']).all()
    assert first['indices'].numel() == second['indices'].numel()
    assert not torch.allclose(first['d'], second['d'])


def test_empty_candidate_set_falls_back_to_additive():
    evidence, query = toy(seed=29)
    net = model(evidence, add_per_round=1, max_candidates=1)
    net.head[-1].bias.data.fill_(-1e4)
    out = net(evidence, query)
    assert torch.allclose(out['prediction'], out['additive_prediction'], atol=1e-6, rtol=0)


def test_initial_variance_is_absolute_point_one():
    evidence, _ = toy(seed=31)
    net = model(evidence, max_variance=100.0)
    bias = net.head[-1].bias.detach()
    assert torch.allclose(torch.sigmoid(bias) * net.max_variance,
                          torch.full_like(bias, 0.1), atol=1e-6, rtol=0)
    small = model(evidence, max_variance=0.2)
    assert abs(float(torch.sigmoid(small.head[-1].bias) * small.max_variance) - 0.1) < 1e-6


def test_independent_variant_still_uses_the_message_path():
    evidence, query = toy(seed=37)
    joint = model(evidence, variant='joint')
    independent = model(evidence, variant='independent')
    independent.load_state_dict(joint.state_dict())
    baseline = independent(evidence, query)['prediction'].detach().clone()
    with torch.no_grad():
        for parameter in independent.message.parameters():
            parameter.add_(0.5)
    changed = independent(evidence, query)['prediction'].detach()
    assert not torch.allclose(baseline, changed), \
        'the shared message path must remain active in the independent variant'
    independent.zero_grad(set_to_none=True)
    out = independent(evidence, query)
    out['prediction'].square().mean().backward()
    assert any(parameter.grad is not None and float(parameter.grad.abs().sum()) > 0
               for parameter in independent.message.parameters())


def test_every_round_adds_a_full_batch_before_trimming():
    evidence, query = toy(seed=41)
    net = model(evidence, rounds=3, add_per_round=3, max_candidates=6)
    out = net(evidence, query)
    added = [entry['added'] for entry in out['trace']]
    selected = [entry['selected'] for entry in out['trace']]
    assert added == [3, 3, 3], added
    assert all(value <= 6 for value in selected), selected
    assert out['indices'].numel() <= 6
