import numpy as np
import pytest

from wheat599.robust_icm import features, fit_predict
from wheat599.kernel_baselines import kernels


def test_injected_original_kernel_and_saved_thresholds_reproduce_prediction():
    rng = np.random.default_rng(41)
    x = rng.integers(0, 2, (26, 18)).astype(float)
    y = rng.normal(size=(26, 4))
    fit, query = np.arange(20), np.arange(20, 26)
    y[query] = np.nan
    base = {'bandwidth': 1., 'fraction': .25}
    original = features(x, y, fit, query, base)
    a, ac, r, rc = kernels(x, fit, query, 1., 'cpu')
    injected = features(x, y, fit, query, base,
                        material_kernels=(.25*a+.75*r, .25*ac+.75*rc))
    cfg = dict(max_iterations=300, function_tolerance=1e-11,
               gradient_tolerance=1e-7, accepted_gradient=1e-5)
    expected, diag = fit_predict(original, .3, .6, 1.345, cfg)
    actual, fixed = fit_predict(injected, .3, .6, 1.345, cfg,
                               fixed_thresholds=diag['thresholds'])
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)
    assert fixed['thresholds_fixed']
    with pytest.raises(ValueError, match='thresholds'):
        fit_predict(injected, .3, .6, 1.345, cfg, fixed_thresholds=[0]*4)
    with pytest.raises(ValueError, match='shapes'):
        features(x, y, fit, query, base, material_kernels=(np.eye(2), np.zeros((6, 2))))
