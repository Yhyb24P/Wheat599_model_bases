"""Checks for the vector readout and the score-geometry shapes."""

import math

import torch

from wheat599.joint_evidence import Evidence
from wheat599.vector_readout import covariance_shapes, readout, residual_shapes

torch.set_num_threads(1)


def toy(seed=3, materials=26, markers=6, environments=3, lambda_a=3.0):
    generator = torch.Generator().manual_seed(seed)
    z = torch.randn(materials, markers, generator=generator, dtype=torch.float64)
    z = z - z.mean(dim=0)
    z = z / z.square().mean().sqrt()
    y = torch.randn(materials, environments, generator=generator, dtype=torch.float64)
    return Evidence(z, y, lambda_a=lambda_a)


def random_omega(environments, seed=5):
    generator = torch.Generator().manual_seed(seed)
    factor = torch.randn(environments, environments, generator=generator, dtype=torch.float64)
    return factor @ factor.T + 0.1 * torch.eye(environments, dtype=torch.float64)


def test_identity_omega_matches_scalar_readout():
    evidence = toy()
    indices = torch.tensor([0, 4, 9], dtype=torch.long)
    d = torch.tensor([0.3, 1.7, 0.8], dtype=torch.float64)
    identity = torch.eye(evidence.environments, dtype=torch.float64)
    b_vector, c_vector = readout(evidence, indices, d, identity)
    b_scalar, c_scalar = evidence.readout(indices, d)
    assert torch.allclose(c_vector, c_scalar, atol=1e-10, rtol=0)
    assert torch.allclose(b_vector, b_scalar, atol=1e-10, rtol=0)


def test_zero_scale_is_exactly_additive():
    evidence = toy(seed=7)
    indices = torch.tensor([1, 2, 5], dtype=torch.long)
    d = torch.zeros(3, dtype=torch.float64)
    b, c = readout(evidence, indices, d, random_omega(evidence.environments))
    assert torch.equal(b, evidence._ridge_b)
    assert torch.equal(c, torch.zeros_like(c))


def test_singular_omega_drops_the_null_direction():
    evidence = toy(seed=9, environments=2)
    indices = torch.tensor([0, 3, 6, 8], dtype=torch.long)
    d = torch.tensor([0.5, 2.0, 0.7, 1.1], dtype=torch.float64)
    omega = torch.tensor([[1.0, 0.6], [0.6, 1.0]], dtype=torch.float64)
    values, vectors = torch.linalg.eigh((omega + omega.T) / 2)
    singular = vectors @ torch.diag(torch.stack([values[-1],
                                                torch.zeros((), dtype=torch.float64)])) @ vectors.T
    b, c = readout(evidence, indices, d, singular)
    assert torch.isfinite(b).all() and torch.isfinite(c).all()
    dropped = vectors[:, 1]  # diag values were [lambda_max, 0] on this basis
    assert torch.allclose(c @ dropped, torch.zeros(indices.numel(), dtype=torch.float64),
                          atol=1e-12, rtol=0)
    # the surviving direction equals the single-direction system
    statistics = evidence.statistics(indices)
    scaled = d * values[-1]
    single = torch.linalg.solve(torch.eye(indices.numel(), dtype=torch.float64)
                                + scaled.unsqueeze(1) * statistics['H'],
                                scaled * (statistics['S'] @ vectors[:, 0]))
    assert torch.allclose(c @ vectors[:, 0], single, atol=1e-10, rtol=0)


def test_two_environment_kronecker_system_agrees():
    evidence = toy(seed=11, environments=2, markers=5)
    indices = torch.tensor([2, 6, 7], dtype=torch.long)
    d = torch.tensor([0.9, 0.4, 1.6], dtype=torch.float64)
    omega = torch.tensor([[1.2, 0.5], [0.5, 0.8]], dtype=torch.float64)
    _, c = readout(evidence, indices, d, omega)
    statistics = evidence.statistics(indices)
    hessian, score = statistics['H'], statistics['S']
    k, q = indices.numel(), evidence.environments
    system = torch.eye(k * q, dtype=torch.float64) \
        + torch.kron((d.unsqueeze(1) * hessian).contiguous(), omega.T.contiguous())
    right = (d.unsqueeze(1) * score) @ omega
    expected = torch.linalg.solve(system, right.reshape(-1)).reshape(k, q)
    assert torch.allclose(c, expected, atol=1e-10, rtol=0)


def test_material_and_candidate_permutations():
    evidence = toy(seed=13, materials=24, markers=5, environments=2)
    indices = torch.tensor([1, 4, 7], dtype=torch.long)
    d = torch.tensor([0.6, 1.2, 0.9], dtype=torch.float64)
    omega = random_omega(evidence.environments, seed=17)
    b, c = readout(evidence, indices, d, omega)
    order = torch.randperm(24, generator=torch.Generator().manual_seed(19))
    permuted = Evidence(evidence.z[order], evidence.y[order], lambda_a=evidence.lambda_a)
    b_perm, c_perm = readout(permuted, indices, d, omega)
    assert torch.allclose(b, b_perm, atol=1e-10, rtol=0)
    assert torch.allclose(c, c_perm, atol=1e-10, rtol=0)
    shuffle = torch.tensor([2, 0, 1])
    b_shuffled, c_shuffled = readout(evidence, indices[shuffle], d[shuffle], omega)
    assert torch.allclose(b, b_shuffled, atol=1e-10, rtol=0)
    assert torch.allclose(c[shuffle], c_shuffled, atol=1e-10, rtol=0)


def test_empty_candidates_fall_back_to_ridge():
    evidence = toy(seed=23)
    b, c = readout(evidence, torch.empty(0, dtype=torch.long),
                   torch.empty(0, dtype=torch.float64),
                   torch.eye(evidence.environments, dtype=torch.float64))
    assert torch.equal(b, evidence._ridge_b)
    assert c.shape == (0, evidence.environments)


def test_covariance_shapes_are_psd_and_normalised():
    evidence = toy(seed=29, environments=4)
    indices = torch.tensor([0, 2, 3, 5, 8], dtype=torch.long)
    identity, diagonal, half, full = covariance_shapes(evidence, indices)
    q = evidence.environments
    for matrix in (identity, diagonal, half, full):
        assert matrix.shape == (q, q)
        assert torch.allclose(matrix, matrix.T, atol=1e-12, rtol=0)
        eigenvalues = torch.linalg.eigvalsh((matrix + matrix.T) / 2)
        assert float(eigenvalues.min()) >= -1e-10 * max(1.0, float(eigenvalues.abs().max()))
    score = evidence.restricted_score(indices)
    gram = score.T @ score / score.shape[0]
    assert torch.allclose(full, gram / (gram.diagonal().sum() / q), atol=1e-10, rtol=0)
    assert abs(float(torch.trace(full)) - q) < 1e-9
    assert torch.allclose(diagonal, torch.diag(full.diagonal()), atol=1e-12, rtol=0)
    assert torch.allclose(half, 0.5 * diagonal + 0.5 * full, atol=1e-12, rtol=0)


def test_covariance_shapes_degenerate_scores_return_identity(monkeypatch):
    evidence = toy(seed=31, environments=2)
    zeros = torch.zeros
    monkeypatch.setattr(evidence, 'restricted_score',
                        lambda indices: zeros((len(indices), evidence.environments),
                                              dtype=evidence.dtype))
    identity, diagonal, half, full = covariance_shapes(evidence, torch.tensor([0, 3]))
    identity_expected = torch.eye(2, dtype=torch.float64)
    for matrix in (identity, diagonal, half, full):
        assert torch.equal(matrix, identity_expected)


class SingleScoreEvidence:
    """Minimal stand-in whose single candidate scores only in environment 0."""

    def __init__(self, environments=2, materials=4, markers=1):
        self.environments = self.markers = self.materials = None
        self.environments, self.materials, self.markers = environments, materials, markers
        self.dtype, self.device = torch.float64, torch.device('cpu')
        self.a = torch.zeros(materials, markers, dtype=self.dtype)
        self.y = torch.zeros(materials, environments, dtype=self.dtype)
        self._ridge_b = torch.zeros(markers, environments, dtype=self.dtype)
        self._factor = torch.eye(materials, dtype=self.dtype)

    def phi(self, indices):
        return torch.zeros(self.materials, len(indices), dtype=self.dtype)

    def statistics(self, indices):
        return {'H': torch.ones((1, 1), dtype=self.dtype),
                'S': torch.tensor([[1.0, 0.0]], dtype=self.dtype)}


def test_single_candidate_can_borrow_an_absent_environment():
    stub = SingleScoreEvidence()
    d = torch.tensor([1.0], dtype=torch.float64)
    identity = torch.eye(2, dtype=torch.float64)
    _, c_isotropic = readout(stub, torch.tensor([0]), d, identity)
    assert torch.allclose(c_isotropic, torch.tensor([[0.5, 0.0]], dtype=torch.float64))
    borrowed = torch.tensor([[1.0, 0.8], [0.8, 1.0]], dtype=torch.float64)
    _, c_borrowed = readout(stub, torch.tensor([0]), d, borrowed)
    assert abs(float(c_borrowed[0, 0])) > 1e-6
    assert abs(float(c_borrowed[0, 1])) > 1e-3, \
        'a correlated prior must borrow the absent environment'


def test_residual_shapes_are_psd_and_carry_cross_environment_structure():
    generator = torch.Generator().manual_seed(43)
    materials, markers, environments = 40, 5, 2
    z = torch.randn(materials, markers, generator=generator, dtype=torch.float64)
    z = z - z.mean(dim=0)
    z = z / z.square().mean().sqrt()
    shared = torch.randn(materials, generator=generator, dtype=torch.float64)
    y = torch.stack([shared, shared + 0.1 * torch.randn(
        materials, generator=generator, dtype=torch.float64)], dim=1)
    evidence = Evidence(z, y, lambda_a=3.0)
    diagonal, half, full = residual_shapes(evidence)
    for matrix in (diagonal, half, full):
        assert matrix.shape == (environments, environments)
        assert torch.allclose(matrix, matrix.T, atol=1e-12, rtol=0)
        eigenvalues = torch.linalg.eigvalsh((matrix + matrix.T) / 2)
        assert float(eigenvalues.min()) >= -1e-10 * max(1.0, float(eigenvalues.abs().max()))
    assert abs(float(torch.trace(full)) - environments) < 1e-9
    off = float(full[0, 1]) / math.sqrt(float(full[0, 0]) * float(full[1, 1]))
    assert off > 0.8, off
    assert torch.allclose(diagonal, torch.diag(full.diagonal()), atol=1e-12, rtol=0)
    assert torch.allclose(half, 0.5 * diagonal + 0.5 * full, atol=1e-12, rtol=0)


def test_residual_shapes_zero_energy_returns_identity():
    materials, markers = 10, 3
    evidence = Evidence(torch.zeros(materials, markers, dtype=torch.float64),
                        torch.zeros(materials, 2, dtype=torch.float64), lambda_a=3.0)
    diagonal, half, full = residual_shapes(evidence)
    expected = torch.eye(2, dtype=torch.float64)
    for matrix in (diagonal, half, full):
        assert torch.equal(matrix, expected)
