"""Gaussian-limit, operator-consistency and fit-scope checks for the posterior candidate.

Only the checks that could change a conclusion are kept:

1. With rho=0 the candidate's assembled predictor must equal the retained ICM
   path in ``multienv_kernel.fit_predict`` bit-for-bit in intent, including the
   fraction=0 branch and a singular task covariance.
2. The exact enumerated posterior must equal the analytic Gaussian posterior at
   rho=0, and the second-moment coupling identity must hold by Monte Carlo.
3. The dense Hessian and the per-marker diagonal blocks must agree, because the
   network consumes the blocks while the Gibbs reference consumes the dense form.
4. Every fitted quantity must derive from context rows only.
"""

import numpy as np
import pytest
import torch

from wheat599.multienv_kernel import fit_predict, task_correlation
from wheat599.posterior_exact import posterior_mean
from wheat599.posterior_operators import ConditionalLikelihood
from wheat599.posterior_synthetic import prepare_genotypes, psd_root, sample_task

Q = 4
TOLERANCE = 1e-11


def correlation_task(rng, singular=False):
    if singular:
        loadings = rng.normal(size=(Q, 2))
    else:
        loadings = rng.normal(size=(Q, Q + 2))
    gram = loadings @ loadings.T
    scale = np.sqrt(np.diag(gram))
    return (gram / np.outer(scale, scale)) * 0.85 + 0.15 * np.eye(Q)


def design(rng, n_fit, n_target, p, bandwidth=1.0):
    markers = rng.integers(0, 2, (n_fit + n_target, p)).astype(np.float64)
    return prepare_genotypes(torch.tensor(markers[:n_fit]), bandwidth, target=torch.tensor(markers[n_fit:]))


def standardized(y):
    mean = y.mean(0)
    scale = y.std(0, ddof=0)
    return (y - mean) / scale, mean, scale


def icm_reference(z, rbf, z_target, rbf_cross, y, task, fraction, ridge):
    """Retained ICM path: fit_predict on the same kernels, task matrix and ridge."""
    kernels = [z @ z.T, rbf]
    cross = [z_target @ z.T, rbf_cross]
    tasks = [fraction * task, (1 - fraction) * task]
    y, mean, scale = standardized(y)
    dual = fit_predict([kernel.detach().clone() for kernel in kernels],
                       [item.detach().clone() for item in cross],
                       y, tasks, np.array([ridge] * Q))
    likelihood = ConditionalLikelihood(z, rbf, torch.tensor(task), fraction, ridge)
    response = torch.tensor(y)
    effects = likelihood.gaussian_posterior_mean(response)
    analytic = likelihood.predict_given_weights(response, effects, z_target, rbf_cross).numpy()
    return np.abs(dual * scale + mean - (analytic * scale + mean)).max()


@pytest.mark.parametrize('fraction', [0.0, 0.25, 0.5, 1.0])
@pytest.mark.parametrize('singular', [False, True])
def test_gaussian_limit_matches_retained_icm(fraction, singular):
    rng = np.random.default_rng(7 if singular else 5)
    z, rbf, z_target, rbf_cross = design(rng, 34, 9, 8)
    task = torch.tensor(correlation_task(rng, singular))
    y = rng.normal(size=(z.shape[0], Q))
    gap = icm_reference(z, rbf, z_target, rbf_cross, y, task.numpy(), fraction, 0.2)
    assert gap < TOLERANCE, f'ICM equivalence gap {gap}'


def test_degenerate_fraction_leaves_no_correction_path():
    rng = np.random.default_rng(11)
    z, rbf, z_target, rbf_cross = design(rng, 30, 7, 6)
    task = torch.tensor(correlation_task(rng))
    y = torch.tensor(rng.normal(size=(z.shape[0], Q)))
    likelihood = ConditionalLikelihood(z, rbf, task, 0.0, 0.3)
    score = likelihood.sufficient_score(y)
    effects = likelihood.gaussian_posterior_mean(y)
    blocks = likelihood.hessian_diagonal_blocks(batch_size=3)
    assert torch.count_nonzero(score) == 0
    assert torch.count_nonzero(effects) == 0
    assert torch.count_nonzero(blocks) == 0
    base = likelihood.predict_given_weights(y, torch.zeros_like(effects), z_target, rbf_cross)
    perturbed = likelihood.predict_given_weights(y, torch.randn_like(effects), z_target, rbf_cross)
    assert (base - perturbed).abs().max() == 0


def test_rho_zero_enumeration_matches_analytic_posterior():
    rng = np.random.default_rng(13)
    p = 3
    z = torch.tensor(rng.normal(size=(14, p)))
    z = (z - z.mean(0)) / (p * z.square().sum() / (z.shape[0] * p)).sqrt()
    rbf = torch.eye(z.shape[0])
    task = torch.tensor(correlation_task(rng))
    y = torch.tensor(rng.normal(size=(z.shape[0], Q)))
    for fraction, ridge in [(0.5, 0.3), (0.25, 1.0)]:
        enumerated = posterior_mean(z, y, rbf, task, fraction, ridge, torch.tensor(0.0), torch.tensor(0.3))
        likelihood = ConditionalLikelihood(z, rbf, task, fraction, ridge)
        analytic = likelihood.gaussian_posterior_mean(y)
        assert (enumerated - analytic).abs().max() < 1e-9


def test_shared_scale_second_moment_coupling():
    """Cov(w_je^2, w_jf^2) = (E[s^2]-1) B_ee B_ff + 2 E[s^2] B_ef^2."""
    rng = np.random.default_rng(17)
    task = np.array([[1.0, 0.0, 0.2, 0.1],
                     [0.0, 1.0, 0.05, 0.3],
                     [0.2, 0.05, 1.0, 0.0],
                     [0.1, 0.3, 0.0, 1.0]])
    root = psd_root(torch.tensor(task)).numpy()
    for rho, inclusion in [(0.4, 0.2), (1.0, 0.05), (0.8, 0.34)]:
        active = rng.random(2000000) < inclusion
        scales = 1 - rho + rho * active / inclusion
        values = (np.sqrt(scales)[:, None] * rng.normal(size=(scales.size, Q))) @ root.T
        second_moment = 1 + rho ** 2 * (1 / inclusion - 1)
        for e, f in [(0, 1), (0, 3), (1, 2), (2, 3)]:
            empirical = np.cov(values[:, e] ** 2, values[:, f] ** 2, bias=True)[0, 1]
            closed = ((second_moment - 1) * task[e, e] * task[f, f]
                      + 2 * second_moment * task[e, f] ** 2)
            assert abs(empirical - closed) / max(abs(closed), 1e-9) < 0.03


def test_dense_hessian_agrees_with_blocks_and_operator():
    rng = np.random.default_rng(19)
    z, rbf, z_target, rbf_cross = design(rng, 26, 5, 7)
    task = torch.tensor(correlation_task(rng))
    likelihood = ConditionalLikelihood(z, rbf, task, 0.4, 0.25)
    dense = likelihood.dense_hessian(batch_size=4)
    blocks = likelihood.hessian_diagonal_blocks(batch_size=4)
    effects = torch.randn(z.shape[1], Q, dtype=z.dtype)
    through_operator = likelihood.hessian_apply(effects)
    through_dense = torch.einsum('iajb,jb->ia', dense, effects)
    assert (dense.diagonal(dim1=0, dim2=2) - blocks.permute(1, 2, 0)).abs().max() < 1e-12
    assert (through_operator - through_dense).abs().max() < 1e-12
    symmetric = dense.reshape(z.shape[1] * Q, z.shape[1] * Q)
    assert (symmetric - symmetric.T).abs().max() < 1e-12
    assert torch.linalg.eigvalsh(symmetric).min() > -1e-10


def test_batched_gaussian_mean_matches_single_call():
    rng = np.random.default_rng(23)
    z, rbf, z_target, rbf_cross = design(rng, 22, 4, 6)
    task = torch.tensor(correlation_task(rng))
    likelihood = ConditionalLikelihood(z, rbf, task, 0.5, 0.2)
    responses = torch.randn(5, z.shape[0], Q, dtype=z.dtype)
    batched = likelihood.gaussian_posterior_mean_batch(responses)
    single = torch.stack([likelihood.gaussian_posterior_mean(item) for item in responses])
    assert (batched - single).abs().max() < 1e-12


def test_fitted_quantities_use_context_rows_only():
    """Fit-side objects must not depend on target rows, in count or in labels."""
    rng = np.random.default_rng(29)
    n_fit, p = 24, 6
    markers = rng.integers(0, 2, (n_fit + 8, p)).astype(np.float64)
    response = torch.tensor(rng.normal(size=(n_fit, Q)))
    task = torch.tensor(task_correlation(response.numpy()))
    z, rbf, z_target, rbf_cross = prepare_genotypes(torch.tensor(markers[:n_fit]), 1.0,
                                                    target=torch.tensor(markers[n_fit:]))
    shorter, rbf_shorter, z_fewer, rbf_fewer = prepare_genotypes(
        torch.tensor(markers[:n_fit]), 1.0, target=torch.tensor(markers[n_fit + 3:]))
    assert torch.equal(z, shorter) and torch.equal(rbf, rbf_shorter)
    assert z.shape[0] == n_fit and rbf.shape == (n_fit, n_fit)
    assert z_fewer.shape == (z_target.shape[0] - 3, p)
    assert rbf_cross.shape[1] == rbf_fewer.shape[1] == n_fit
    first = ConditionalLikelihood(z, rbf, task, 0.5, 0.2)
    second = ConditionalLikelihood(shorter, rbf_shorter,
                                   torch.tensor(task_correlation(response.numpy())), 0.5, 0.2)
    gap = (first.gaussian_posterior_mean(response) - second.gaussian_posterior_mean(response)).abs().max()
    assert gap < 1e-15


def test_task_correlation_matches_icm_task_matrix():
    rng = np.random.default_rng(31)
    response = rng.normal(size=(40, Q))
    eta = 0.6
    task = (1 - eta) * np.eye(Q) + eta * task_correlation(response)
    eigenvalues = np.linalg.eigvalsh(task)
    assert eigenvalues.min() > 0
    assert abs(np.trace(task) / Q - (1 - eta) - eta * 1.0) < 1e-12
