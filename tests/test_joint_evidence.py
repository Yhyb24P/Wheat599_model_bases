"""Checks for the exact evidence operators in ``wheat599.joint_evidence``."""

import math

import numpy as np
import pytest
import torch

from wheat599.joint_evidence import Evidence

torch.set_num_threads(1)


def toy(seed=3, materials=30, markers=7, environments=2, lambda_a=3.0):
    generator = torch.Generator().manual_seed(seed)
    z = torch.randn(materials, markers, generator=generator, dtype=torch.float64)
    z = z - z.mean(dim=0)
    z = z / z.square().mean().sqrt()
    y = torch.randn(materials, environments, generator=generator, dtype=torch.float64)
    return Evidence(z, y, lambda_a=lambda_a), z, y


def dense_joint(evidence, indices, d):
    """Dense solution of the weighted joint ridge problem, multiplied through by n."""
    z, y = evidence.z, evidence.y
    a = evidence.a
    columns = evidence.phi(indices)
    p, q = a.shape[1], y.shape[1]
    k = columns.shape[1]
    identity_p = torch.eye(p, dtype=z.dtype)
    identity_k = torch.eye(k, dtype=z.dtype)
    top = torch.cat([a.T @ a + evidence.lambda_a * identity_p, a.T @ columns], dim=1)
    bottom = torch.cat([columns.T @ a, columns.T @ columns
                        + y.shape[0] * torch.diag(1.0 / d)], dim=1)
    system = torch.cat([top, bottom], dim=0)
    right = torch.cat([a.T @ y, columns.T @ y], dim=0)
    solution = torch.linalg.solve(system, right)
    return solution[:p], solution[p:]


def test_readout_matches_dense_joint_solution():
    evidence, _, _ = toy()
    indices = torch.tensor([0, 4, 11], dtype=torch.long)
    d = torch.tensor([0.7, 2.0, 0.3], dtype=torch.float64)
    b, c = evidence.readout(indices, d)
    expected_b, expected_c = dense_joint(evidence, indices, d)
    assert torch.allclose(b, expected_b, atol=1e-10, rtol=0)
    assert torch.allclose(c, expected_c, atol=1e-10, rtol=0)


def test_score_and_statistics_are_consistent():
    evidence, _, _ = toy()
    indices = torch.tensor([1, 3, 6, 9], dtype=torch.long)
    stats = evidence.statistics(indices)
    full = evidence.score()
    assert full.shape == (evidence.pair_count, evidence.environments)
    assert torch.allclose(stats['S'], full[indices], atol=1e-12, rtol=0)
    assert stats['H'].shape == (4, 4)
    assert torch.allclose(stats['H'], stats['H'].T, atol=1e-12, rtol=0)
    assert bool((stats['noise_diag'] > 0).all())
    assert bool((stats['additive_diag'] > 0).all())
    energy = stats['energy']
    assert energy.shape == (evidence.environments, evidence.environments)
    assert torch.allclose(energy, energy.T, atol=1e-12, rtol=0)
    assert torch.allclose(energy, (evidence.y.T @ (evidence.m @ evidence.y))
                          / evidence.materials, atol=1e-12, rtol=0)
    c = torch.randn(4, evidence.environments, dtype=torch.float64)
    columns = evidence.phi(indices)
    direct = columns.T @ (evidence.m @ (evidence.y - columns @ c)) / evidence.materials
    assert torch.allclose(evidence.restricted_score(indices, c), direct, atol=1e-12, rtol=0)
    conditional = evidence.score(indices, c)
    assert conditional.shape == (evidence.pair_count, evidence.environments)
    assert torch.allclose(conditional[indices], direct, atol=1e-12, rtol=0)
    elsewhere = torch.tensor([2, 5], dtype=torch.long)
    shifted = evidence.m @ (evidence.y - columns @ c) / evidence.materials
    assert torch.allclose(conditional[elsewhere], evidence.phi(elsewhere).T @ shifted,
                          atol=1e-12, rtol=0)


def test_covariance_identities_for_m_squared():
    evidence, _, _ = toy(seed=5, materials=60, markers=6)
    indices = torch.tensor([0, 2, 5], dtype=torch.long)
    stats = evidence.statistics(indices)
    columns = evidence.phi(indices)
    generator = torch.Generator().manual_seed(11)
    draws = 8000
    scores = []
    additive = []
    beta = torch.randn(evidence.markers, 20000, generator=generator, dtype=torch.float64)
    additive_scores = (columns.T @ (evidence.m @ (evidence.a @ beta))) / evidence.materials
    for _ in range(draws):
        noise = torch.randn(evidence.materials, evidence.environments, generator=generator,
                            dtype=torch.float64)
        scores.append((columns.T @ (evidence.m @ noise)) / evidence.materials)
    scores = torch.stack(scores)
    empirical = scores.var(dim=0, unbiased=False).mean(dim=1)
    assert torch.allclose(empirical, stats['noise_diag'], rtol=0.08, atol=1e-12)
    empirical_additive = additive_scores.var(dim=1, unbiased=False)
    assert torch.allclose(empirical_additive, stats['additive_diag'], rtol=0.10, atol=1e-12)


def test_zero_evidence_is_exactly_additive():
    evidence, z, _ = toy(seed=7)
    indices = torch.tensor([0, 5, 10], dtype=torch.long)
    d = torch.zeros(3, dtype=torch.float64)
    b, c = evidence.readout(indices, d)
    assert torch.equal(b, evidence._ridge_b)
    assert torch.equal(c, torch.zeros_like(c))
    prediction = evidence.predict(z, indices, b, c)
    assert torch.equal(prediction, evidence.ridge(z))


def test_readout_gradient_is_finite_at_zero_and_matches_finite_difference():
    evidence, z, _ = toy(seed=9)
    indices = torch.tensor([2, 4, 8], dtype=torch.long)
    theta = torch.tensor([-2.0, 0.4, 1.1], dtype=torch.float64, requires_grad=True)

    def objective(values):
        d = torch.nn.functional.softplus(values)
        b, c = evidence.readout(indices, d)
        return evidence.predict(z, indices, b, c).square().mean()

    loss = objective(theta)
    loss.backward()
    analytic = theta.grad.clone()
    assert torch.isfinite(analytic).all()
    epsilon = 1e-6
    numeric = torch.zeros_like(theta)
    for position in range(theta.numel()):
        step = torch.zeros_like(theta)
        step[position] = epsilon
        numeric[position] = (objective(theta.detach() + step)
                             - objective(theta.detach() - step)) / (2 * epsilon)
    assert torch.allclose(analytic, numeric, rtol=2e-5, atol=1e-8)


def test_gradient_at_exact_zero_evidence_is_finite():
    evidence, z, _ = toy(seed=13)
    indices = torch.tensor([1, 2], dtype=torch.long)
    d = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    b, c = evidence.readout(indices, d)
    loss = evidence.predict(z, indices, b, c).square().mean()
    loss.backward()
    assert d.grad is not None and torch.isfinite(d.grad).all()
    assert float(d.grad.abs().max()) > 0


def test_shape_and_input_validation():
    evidence, z, _ = toy()
    with pytest.raises(ValueError):
        evidence.readout(torch.tensor([0, 1]), torch.tensor([-0.1, 1.0], dtype=torch.float64))
    with pytest.raises(ValueError):
        evidence.readout(torch.tensor([0, 1]), torch.tensor([1.0], dtype=torch.float64))
    with pytest.raises(ValueError):
        evidence.phi(torch.tensor([evidence.pair_count]))
    empty_b, empty_c = evidence.readout(torch.tensor([], dtype=torch.long),
                                        torch.tensor([], dtype=torch.float64))
    assert empty_c.shape == (0, evidence.environments)
    assert torch.equal(empty_b, evidence._ridge_b)


def test_root_factorisation_and_two_readout_forms_agree():
    evidence, _, _ = toy(seed=17)
    assert torch.allclose(evidence._root.T @ evidence._root, evidence.m, atol=1e-12, rtol=0)
    indices = torch.tensor([0, 3, 7], dtype=torch.long)
    d = torch.tensor([0.4, 1.5, 0.9], dtype=torch.float64)
    _, woodbury = evidence.readout(indices, d)
    block = evidence._block(indices)
    identity = torch.eye(indices.numel(), dtype=torch.float64)
    direct = torch.linalg.solve(identity + d.unsqueeze(1) * block['H'],
                                d.unsqueeze(1) * block['S'])
    assert torch.allclose(woodbury, direct, atol=1e-11, rtol=0)
