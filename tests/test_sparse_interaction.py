"""Independent checks for the exact active-set group-lasso interaction solver.

Expectations come from explicit all-j<k feature columns and plain float64 linear algebra at
tiny sizes; the solver's own centering, gradient or KKT helpers are never used to build the
reference values.
"""
import itertools
import math

import numpy as np
import torch
from sklearn.linear_model import MultiTaskElasticNet

from wheat599.sparse_interaction import SparseInteraction

LAMBDA_A, LAMBDA_E = 0.7, 0.05


def _pairs(markers):
    return np.array(list(itertools.combinations(range(markers), 2)), dtype=np.int64)


def _design(start=0, materials=7, markers=4, environments=2):
    """A tiny non-degenerate problem: every interaction column varies on the context."""
    pairs = _pairs(markers)
    for seed in range(start, start + 40):
        generator = np.random.default_rng(seed)
        z = (generator.random((materials, markers)) < .65).astype(np.float64)
        raw = z[:, pairs[:, 0]] * z[:, pairs[:, 1]]
        scale = np.sqrt(((raw - raw.mean(axis=0)) ** 2).mean(axis=0))
        if (scale > 1e-6).all():
            return z, generator.normal(size=(materials, environments)), \
                (generator.random((3, markers)) < .5).astype(np.float64)
    raise AssertionError('no non-degenerate tiny design found')


def _explicit_phi(z, query, pairs):
    """Context-centered, context-column-RMS-scaled interaction design, built from scratch."""
    raw = z[:, pairs[:, 0]] * z[:, pairs[:, 1]]
    mean = raw.mean(axis=0)
    centered = raw - mean
    scale = np.sqrt((centered ** 2).mean(axis=0))
    assert (scale > 1e-9).all(), 'the tiny design must have no constant interaction column'
    context = centered / scale
    cross = (query[:, pairs[:, 0]] * query[:, pairs[:, 1]] - mean) / scale
    return context, cross


def _operator(z, lambda_a=LAMBDA_A):
    a = z / math.sqrt(z.shape[1])
    operator = lambda_a * np.linalg.inv(a @ a.T + lambda_a * np.eye(len(z)))
    return a, operator, np.linalg.cholesky(operator).T


def _expand(pairs, active, coefficients, environments):
    full = np.zeros((len(pairs), environments))
    position = {pair: index for index, pair in enumerate(map(tuple, pairs))}
    for row, pair in enumerate(np.asarray(active)):
        full[position[tuple(pair)]] = np.asarray(coefficients)[row]
    return full


def _solver(z, y, **options):
    settings = {'lambda_a': LAMBDA_A, 'lambda_e': LAMBDA_E, 'device': 'cpu',
                'dtype': torch.float64}
    settings.update(options)
    return SparseInteraction(z, y, **settings)


def test_profiled_objective_matches_full_primal_and_vector_kkt():
    z, y, _ = _design()
    pairs = _pairs(z.shape[1])
    phi, _ = _explicit_phi(z, z, pairs)
    a, operator, _ = _operator(z)
    ratio = 0.4
    candidate = _solver(z, y).fit([ratio])[0]
    assert candidate.status == 'optimal', candidate.as_dict()
    lambda_g = ratio * candidate.lambda_max

    coefficients = _expand(pairs, candidate.active, candidate.coefficients, y.shape[1])
    residual = y - phi @ coefficients
    quadratic = float((residual * (operator @ residual)).sum()) / (2 * len(z))
    group = lambda_g * float(np.linalg.norm(coefficients, axis=1).sum())
    ridge = LAMBDA_E / 2 * float((coefficients ** 2).sum())
    profiled = quadratic + group + ridge

    additive = a.T @ np.linalg.solve(a @ a.T + LAMBDA_A * np.eye(len(z)), residual)
    difference = residual - a @ additive
    primal = (float((difference ** 2).sum()) + LAMBDA_A * float((additive ** 2).sum())) \
        / (2 * len(z)) + group + ridge
    assert math.isclose(primal, profiled, rel_tol=1e-10, abs_tol=1e-12)
    assert math.isclose(candidate.objective, profiled, rel_tol=1e-9, abs_tol=1e-12)
    assert math.isclose(candidate.objective, primal, rel_tol=1e-9, abs_tol=1e-12)

    squared = float((residual * (operator @ (operator @ residual))).sum()) / (2 * len(z)) \
        + group + ridge
    assert not math.isclose(squared, candidate.objective, rel_tol=1e-3), \
        'the M squared convention must not reproduce the objective'

    assert np.allclose(np.asarray(candidate.additive), additive, rtol=1e-8, atol=1e-10)
    assert np.allclose(np.asarray(candidate.prediction), a @ additive + phi @ coefficients,
                       rtol=1e-8, atol=1e-10)

    gradient = phi.T @ operator @ residual / len(z)
    norms = np.linalg.norm(coefficients, axis=1)
    active_rows = norms > 1e-12
    stationarity = gradient - LAMBDA_E * coefficients
    stationarity[active_rows] -= lambda_g * coefficients[active_rows] / norms[active_rows, None]
    assert active_rows.sum() and np.abs(stationarity[active_rows]).max() < 1e-6
    assert np.linalg.norm(gradient[~active_rows], axis=1).max() <= lambda_g + 1e-6
    assert np.abs(gradient[active_rows] - LAMBDA_E * coefficients[active_rows]).min() > 1e-6, \
        'stationarity must involve the group direction, not only its norm'


def test_solver_matches_multitask_elasticnet_on_the_profiled_problem():
    z, y, query = _design(2)
    pairs = _pairs(z.shape[1])
    phi, cross = _explicit_phi(z, query, pairs)
    a, _, root = _operator(z)
    ratio = 0.35
    candidate = _solver(z, y, kkt_tolerance=1e-10).fit([ratio], query=query)[0]
    assert candidate.status == 'optimal', candidate.as_dict()
    lambda_g = ratio * candidate.lambda_max

    estimator = MultiTaskElasticNet(alpha=lambda_g + LAMBDA_E,
                                    l1_ratio=lambda_g / (lambda_g + LAMBDA_E),
                                    fit_intercept=False, tol=1e-12, max_iter=200000)
    estimator.fit(root @ phi, root @ y)
    reference = estimator.coef_.T
    mine = _expand(pairs, candidate.active, candidate.coefficients, y.shape[1])
    nonzero = np.linalg.norm(mine, axis=1) > 1e-8
    assert nonzero.sum() and np.abs(reference[~nonzero]).max() <= 1e-12
    assert np.linalg.norm(reference - mine, ord='fro') < 1e-8

    # The reference prediction is built only from the reference coefficients: its own
    # eliminated additive solution, then its own interaction term.
    reference_residual = y - phi @ reference
    reference_additive = a.T @ np.linalg.solve(a @ a.T + LAMBDA_A * np.eye(len(z)),
                                               reference_residual)
    expected = (query / math.sqrt(z.shape[1])) @ reference_additive + cross @ reference
    assert np.allclose(np.asarray(candidate.prediction), expected, rtol=1e-7, atol=1e-9)

    def objective(coefficients):
        residual = y - phi @ coefficients
        return float((residual * (root.T @ (root @ residual))).sum()) / (2 * len(z)) \
            + lambda_g * float(np.linalg.norm(coefficients, axis=1).sum()) \
            + LAMBDA_E / 2 * float((coefficients ** 2).sum())

    assert objective(mine) <= objective(reference) + 1e-12
    assert math.isclose(candidate.objective, objective(mine), rel_tol=1e-9, abs_tol=1e-12)

    gradient = phi.T @ (root.T @ (root @ (y - phi @ mine))) / len(z)
    stationarity = gradient - LAMBDA_E * mine
    stationarity[nonzero] -= lambda_g * mine[nonzero] / np.linalg.norm(
        mine[nonzero], axis=1)[:, None]
    assert np.abs(stationarity[nonzero]).max() < 1e-9
    assert candidate.kkt_absolute < 1e-9
    assert np.linalg.norm(gradient[~nonzero], axis=1).max() <= lambda_g


def test_query_independence_permutation_flip_invariance_and_zero_at_ratio_one():
    z, y, query = _design(4)
    solver = _solver(z, y)
    ratio = 0.3
    mine = solver.fit([ratio], query=query)[0]
    assert mine.status == 'optimal', mine.as_dict()
    other = np.array([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]])
    changed = solver.fit([ratio], query=other)[0]
    assert np.array_equal(np.asarray(mine.active), np.asarray(changed.active))
    assert np.allclose(np.asarray(mine.coefficients), np.asarray(changed.coefficients),
                       rtol=0, atol=1e-14)
    assert not np.allclose(np.asarray(mine.prediction), np.asarray(changed.prediction))

    order = np.array([3, 0, 5, 1, 6, 2, 4])
    moved = _solver(z[order], y[order]).fit([ratio], query=query)[0]
    assert np.array_equal(np.asarray(mine.active), np.asarray(moved.active))
    assert np.allclose(np.asarray(mine.prediction), np.asarray(moved.prediction),
                       rtol=1e-7, atol=1e-9)

    markers = np.array([2, 1, 3, 0])
    shifted = _solver(z[:, markers], y).fit([ratio], query=query[:, markers])[0]
    assert np.allclose(np.asarray(mine.prediction), np.asarray(shifted.prediction),
                       rtol=1e-7, atol=1e-9)

    flip = np.array([-1.0, 1.0, 1.0, -1.0])
    mirrored = _solver(z * flip, y).fit([ratio], query=query * flip)[0]
    assert np.allclose(np.asarray(mine.prediction), np.asarray(mirrored.prediction),
                       rtol=1e-7, atol=1e-9)

    zero = solver.fit([1.0], query=query)[0]
    assert zero.status == 'optimal', zero.as_dict()
    assert len(np.asarray(zero.active)) == 0
    assert np.asarray(zero.coefficients).size == 0 \
        or np.abs(np.asarray(zero.coefficients)).max() < 1e-14
    linear, weights = solver.ridge(query=query)
    assert np.allclose(np.asarray(zero.prediction), linear, rtol=1e-8, atol=1e-10)

    phi, _ = _explicit_phi(z, z, _pairs(z.shape[1]))
    _, operator, _ = _operator(z)
    gradient = phi.T @ operator @ y / len(z)
    assert math.isclose(zero.lambda_max, float(np.linalg.norm(gradient, axis=1).max()),
                        rel_tol=1e-9, abs_tol=1e-11)
