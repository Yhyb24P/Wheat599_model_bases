import numpy as np
import pytest
import torch

from wheat599 import relation_regularizer as regularizer
from wheat599.kernel_baselines import kernels


def data(n=8, markers=6, environments=4, seed=0, dtype=torch.float64):
    rng = np.random.default_rng(seed)
    x = (rng.random((n, markers)) < 0.5).astype(np.float64)
    y = rng.normal(size=(n, environments))
    return torch.tensor(x, dtype=dtype), torch.tensor(y, dtype=dtype)


def psd_kernel(n=8, seed=1, dtype=torch.float64):
    generator = torch.Generator().manual_seed(seed)
    matrix = torch.randn(n, n, generator=generator, dtype=dtype)
    return matrix @ matrix.T / n + 0.2 * torch.eye(n, dtype=dtype)


def direct_formula(predictions, kernel, y_fit, kappa=0.1):
    if predictions.ndim > 2:
        m = predictions.mean(dim=tuple(range(2, predictions.ndim)))
    else:
        m = predictions
    n, environments = m.shape
    identity = torch.eye(n, dtype=m.dtype, device=m.device)
    centering = identity - torch.ones_like(identity) / n
    mean = y_fit.mean(dim=0)
    scale = y_fit.std(dim=0, unbiased=False)
    normalized = (y_fit - mean) / scale
    unit = normalized / torch.linalg.vector_norm(normalized, dim=0)
    b_matrix = 0.5 * torch.eye(environments, dtype=m.dtype, device=m.device) \
        + 0.5 * (unit.T @ unit)
    hm = centering @ m
    system = centering @ kernel @ centering + kappa * identity
    middle = hm.T @ torch.linalg.solve(system, hm)
    value = torch.trace(torch.linalg.solve(b_matrix, middle))
    return value / (n * environments)


def test_exact_formula_with_psd_kernel():
    _, y = data()
    kernel = psd_kernel()
    predictions = torch.randn(8, 4, dtype=torch.float64)
    value = regularizer.relation_penalty(predictions, kernel, y, kappa=0.1)
    expected = direct_formula(predictions, kernel, y)
    torch.testing.assert_close(value, expected, atol=1e-12, rtol=1e-12)


def test_kernel_construction_matches_existing_kernels():
    x, _ = data()
    additive, _, rbf, _ = kernels(x, torch.arange(8), torch.arange(8),
                                  1.3, 'cpu')
    expected = 0.35 * additive + 0.65 * rbf
    actual = regularizer.build_kernel(x, bandwidth=1.3, fraction=0.35)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)


def test_environment_intercept_does_not_change_penalty():
    _, y = data()
    kernel = psd_kernel()
    predictions = torch.randn(8, 4, 2, 5, dtype=torch.float64)
    offset = torch.tensor([1.5, -2.0, 0.3, 4.0], dtype=torch.float64)
    shifted = predictions + offset[None, :, None, None]
    first = regularizer.relation_penalty(predictions, kernel, y)
    second = regularizer.relation_penalty(shifted, kernel, y)
    torch.testing.assert_close(first, second, atol=1e-12, rtol=1e-12)


def test_each_quantile_receives_the_same_gradient():
    _, y = data()
    kernel = psd_kernel()
    predictions = torch.randn(8, 4, 2, 5, dtype=torch.float64,
                              requires_grad=True)
    value = regularizer.relation_penalty(predictions, kernel, y)
    value.backward()
    gradient = predictions.grad
    for member in range(2):
        reference = gradient[:, :, member, 0]
        for quantile in range(1, 5):
            torch.testing.assert_close(gradient[:, :, member, quantile], reference,
                                       atol=1e-12, rtol=1e-12)


def test_gradient_matches_central_finite_difference():
    _, y = data(n=6)
    kernel = psd_kernel(n=6)
    predictions = torch.randn(6, 4, dtype=torch.float64, requires_grad=True)
    value = regularizer.relation_penalty(predictions, kernel, y, kappa=0.3)
    value.backward()
    analytic = predictions.grad.clone()
    epsilon = 1e-6
    for row, column in ((0, 0), (2, 3), (5, 1)):
        plus = predictions.detach().clone()
        minus = predictions.detach().clone()
        plus[row, column] += epsilon
        minus[row, column] -= epsilon
        numeric = (regularizer.relation_penalty(plus, kernel, y, kappa=0.3)
                   - regularizer.relation_penalty(minus, kernel, y, kappa=0.3)) / (2 * epsilon)
        torch.testing.assert_close(analytic[row, column], numeric,
                                   atol=1e-7, rtol=1e-7)


def test_constant_material_predictions_have_zero_penalty():
    _, y = data()
    kernel = psd_kernel()
    offset = torch.tensor([1.0, -2.0, 3.0, 0.5], dtype=torch.float64)
    predictions = offset[None, :].expand(8, 4)
    value = regularizer.relation_penalty(predictions, kernel, y)
    assert float(value) == pytest.approx(0.0, abs=1e-14)


@pytest.mark.parametrize('kappa', [1e-6, 0.1, 1e3])
def test_extreme_positive_kappa_stays_finite(kappa):
    _, y = data()
    kernel = psd_kernel()
    predictions = torch.randn(8, 4, dtype=torch.float64)
    value = regularizer.relation_penalty(predictions, kernel, y, kappa=kappa)
    assert torch.isfinite(value)
    assert float(value) >= 0


def test_zero_kernel_with_positive_kappa_is_finite():
    _, y = data()
    predictions = torch.randn(8, 4, dtype=torch.float64)
    value = regularizer.relation_penalty(
        predictions, torch.zeros(8, 8, dtype=torch.float64), y, kappa=0.1)
    expected = direct_formula(predictions, torch.zeros(8, 8, dtype=torch.float64), y)
    torch.testing.assert_close(value, expected, atol=1e-12, rtol=1e-12)


def test_convenience_wrapper_matches_explicit_kernel():
    x, y = data()
    predictions = torch.randn(8, 4, dtype=torch.float64)
    kernel = regularizer.build_kernel(x, bandwidth=1.3, fraction=0.35)
    first = regularizer.relation_penalty(predictions, kernel, y)
    second = regularizer.relation_regularizer(
        x, y, predictions, bandwidth=1.3, fraction=0.35)
    torch.testing.assert_close(first, second, atol=1e-12, rtol=1e-12)


def test_float32_inputs_return_float32_finite_gradient():
    _, y = data()
    kernel = psd_kernel()
    predictions = torch.randn(8, 4, dtype=torch.float32, requires_grad=True)
    value = regularizer.relation_penalty(predictions, kernel, y)
    assert value.dtype == torch.float32 and torch.isfinite(value)
    value.backward()
    assert predictions.grad is not None
    assert predictions.grad.dtype == torch.float32
    assert torch.isfinite(predictions.grad).all()


def test_make_relation_matches_penalty_and_caches_factors(monkeypatch):
    _, y = data()
    kernel = psd_kernel()
    relation = regularizer.make_relation(kernel, y, eta=0.5, kappa=0.1)
    m = torch.randn(8, 4, dtype=torch.float64, requires_grad=True)
    value = relation(m)
    expected = regularizer.relation_penalty(m, kernel, y, eta=0.5, kappa=0.1)
    torch.testing.assert_close(value, expected, atol=1e-12, rtol=1e-12)
    assert relation.materials == 8 and relation.environments == 4
    value.backward()
    assert m.grad is not None and torch.isfinite(m.grad).all()

    def forbidden(*args, **kwargs):
        raise AssertionError('relation closure refactorized a matrix')

    monkeypatch.setattr(torch.linalg, 'cholesky', forbidden)
    assert torch.isfinite(relation(m.detach()))


def test_make_relation_supports_eta_and_rejects_bad_m():
    _, y = data()
    kernel = psd_kernel()
    m = torch.randn(8, 4, dtype=torch.float64)
    for eta in (0.0, 0.2, 0.5, 1.0):
        relation = regularizer.make_relation(kernel, y, eta=eta, kappa=0.1)
        expected = regularizer.relation_penalty(m, kernel, y, eta=eta, kappa=0.1)
        torch.testing.assert_close(relation(m), expected, atol=1e-12, rtol=1e-12)
    relation = regularizer.make_relation(kernel, y)
    with pytest.raises(ValueError, match='shape'):
        relation(torch.randn(8, 3, dtype=torch.float64))
    with pytest.raises(ValueError, match='finite'):
        relation(torch.full((8, 4), float('nan'), dtype=torch.float64))


def test_invalid_inputs_are_rejected():
    _, y = data()
    kernel = psd_kernel()
    predictions = torch.randn(8, 4, dtype=torch.float64)
    with pytest.raises(ValueError):
        regularizer.relation_penalty(predictions, kernel, y, kappa=0)
    with pytest.raises(ValueError):
        regularizer.relation_penalty(predictions, kernel, y, kappa=-1)
    asymmetric = kernel.clone()
    asymmetric[0, 1] += 1.0
    with pytest.raises(ValueError):
        regularizer.relation_penalty(predictions, asymmetric, y)
    with pytest.raises(ValueError):
        regularizer.relation_penalty(predictions, torch.eye(7, dtype=torch.float64), y)
    with pytest.raises(ValueError):
        regularizer.relation_penalty(predictions, kernel, torch.ones(8, 4,
                                                                    dtype=torch.float64))
    with pytest.raises(ValueError):
        regularizer.build_kernel(torch.zeros(8, 6), bandwidth=0.0, fraction=0.5)
    with pytest.raises(ValueError):
        regularizer.build_kernel(torch.zeros(8, 6), bandwidth=1.0, fraction=1.1)
