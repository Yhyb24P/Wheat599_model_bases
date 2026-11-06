import numpy as np
import pytest

pytest.importorskip('tabm')

from wheat599.joint_tabm_icm import train_unit
from wheat599.robust_icm import features, fit_predict


def test_zero_network_matches_fixed_baseline_and_joint_runs():
    rng = np.random.default_rng(70)
    x = rng.integers(0, 2, (32, 18)).astype(float)
    y = rng.normal(size=(32, 4))
    fit, val = np.arange(24), np.arange(24, 32)
    prepared = features(x, y, fit, val, {'bandwidth': 1., 'fraction': .25})
    configs = [dict(n_blocks=2, d_block=16, dropout=.1, k=k,
                    learning_rate=.001, weight_decay=.001) for k in (2, 3, 2, 3)]
    settings = dict(x_fit=x[fit], y_fit=y[fit], x_val=x[val], y_val=y[val],
                    prepared=prepared, ridge=.5, eta=.3, thresholds=[1.]*4,
                    configurations=configs, seed=42, epochs=1, patience=2)
    fixed, a = train_unit(**settings, joint=False)
    joint, b = train_unit(**settings, joint=True)
    assert a[0]['validation_pcc'] == pytest.approx(b[0]['validation_pcc'], abs=1e-7)
    assert fixed['prediction'].shape == joint['prediction'].shape == (8, 4)
    assert len(a) == len(b) == 2
    zero, _ = train_unit(**{**settings, 'epochs': 0}, joint=True)
    expected, _ = fit_predict(prepared, .5, .3, 1.,
        dict(max_iterations=300, function_tolerance=1e-11, gradient_tolerance=1e-7,
             accepted_gradient=1e-5), fixed_thresholds=[1.]*4)
    np.testing.assert_allclose(zero['prediction'], expected, atol=1e-7, rtol=0)
