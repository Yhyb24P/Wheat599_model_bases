import numpy as np

from genopred.evaluation.polygenic_simulation import conditional_mean, draw
from genopred.models.kernels.wheat2000_gblup import predict_grid


def test_simulation_reproducibility_conditional_mean_and_query_label_boundary():
    x = np.array([[-1., 0.], [1., 0.], [0., -1.], [0., 1.], [1., 1.]])
    k = x @ x.T
    k /= np.diag(k).mean()
    values, vectors = np.linalg.eigh(k)
    g, y = draw(values, vectors, .6, [42, 43], 0)
    again_g, again_y = draw(values, vectors, .6, [42, 43], 0)
    np.testing.assert_array_equal(g, again_g)
    np.testing.assert_array_equal(y, again_y)
    fit, query = np.array([0, 1, 2]), np.array([3, 4])
    delta = .4 / .6
    expected = k[np.ix_(query, fit)] @ np.linalg.solve(k[np.ix_(fit, fit)] + delta * np.eye(3), y[fit])
    altered = y.copy()
    altered[query] = np.nan
    np.testing.assert_allclose(conditional_mean(k, altered, fit, query, delta), expected, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(predict_grid(k, altered, fit, query, [delta]),
                               predict_grid(k, y, fit, query, [delta]), atol=1e-6, rtol=1e-6)
