import numpy as np

from genopred.evaluation.extrapolation_diagnostic import aggregate, quartiles, spectral_summary


def test_spectral_trace_matches_dense_smoother_and_bins_preserve_ties():
    k = np.diag([0., 1., 3.])
    signal = np.array([[2., .4], [.4, 1.]])
    residual = np.array([[1., .2], [.2, .8]])
    full = np.kron(k, signal)
    noise = np.kron(np.eye(3), residual)
    expected = np.trace(np.linalg.solve(full + noise, full))
    result = spectral_summary(np.diag(k), signal, residual)
    np.testing.assert_allclose(result['effective_df'], expected, atol=1e-6, rtol=1e-6)
    bins = quartiles(np.array([.1, .2, .2, .3, .4]))
    assert bins[1] == bins[2] and bins[0] <= bins[-1]
    rows = [dict(truth_standardized=y, prediction_standardized=y / 2, nearest_hamming_fraction=.2)
            for y in [-1., 1.]]
    stats = aggregate(rows)
    assert stats['amplitude_ratio'] == .5 and stats['rmse'] == .5
