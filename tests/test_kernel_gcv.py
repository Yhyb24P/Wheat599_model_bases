import json
import numpy as np

from wheat599.kernel_baselines import gcv, gcv_summary


def test_gcv_matches_dense_hat_matrix_with_unpenalized_intercept():
    generator = np.random.default_rng(42)
    x = generator.normal(size=(9, 4))
    kernel = x @ x.T
    y = generator.normal(size=9) + 12
    eigenvalues, vectors = np.linalg.eigh(kernel)
    grid = [.001, .1, 1., 10., 1000.]
    observed = gcv(eigenvalues, vectors, y, grid)
    expected = []
    for ridge in grid:
        inverse = np.linalg.inv(kernel + ridge * np.eye(9))
        u = inverse @ np.ones(9)
        projection = inverse - np.outer(u, u) / u.sum()
        hat = np.eye(9) - ridge * projection
        error = y - hat @ y
        expected.append(float(np.mean(error**2) / (1 - np.trace(hat) / 9)**2))
    np.testing.assert_allclose(observed, expected, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(gcv(eigenvalues, vectors, y - 12, grid), observed, atol=1e-6, rtol=1e-6)


def test_gcv_summary_serializes_factor_ten_counts():
    choices = [dict(model=m, ridge=.1, pcc_selected_ridge=1.) for m in ['gaussian', 'additive_rbf']]
    rows = [dict(model=m, environment='1', rule=r, pcc=.5, rmse=1., mae=.8)
            for m in ['gaussian', 'additive_rbf'] for r in ['gcv', 'inner_pcc']]
    oof = [dict(model=m, environment='1', y_true=y, gcv=y/2, inner_pcc=y/2)
           for m in ['gaussian', 'additive_rbf'] for y in [-1., 1.]]
    summary = gcv_summary(choices, rows, oof, ['1'])
    assert summary['comparison']['gaussian']['within_factor_ten'] == 1
    json.dumps(summary, allow_nan=False)
