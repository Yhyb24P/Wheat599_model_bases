"""Classical baseline grid order, seed policy and estimator wiring."""
import numpy as np
import pytest

from genopred.models.classical.classical_baselines import candidates, fit_predict, model_seeds
from genopred.evaluation.metrics import pcc


CFG = {
    'seeds': [42, 2026],
    'search': {
        'svr': {'C': [1.0, 10.0], 'gamma': ['scale'], 'epsilon': [0.1]},
        'rf': {'n_estimators': [100], 'max_features': ['sqrt', 0.5], 'min_samples_leaf': [1, 3]},
        'xgb': {'n_estimators': [300], 'max_depth': [2], 'learning_rate': [0.2],
                'subsample': [0.8], 'colsample_bytree': [0.5]},
        'lgbm': {'n_estimators': [300], 'num_leaves': [15], 'learning_rate': [0.2],
                 'feature_fraction': [0.5], 'bagging_fraction': [0.8]},
    },
}


def test_candidate_order_is_fixed_product_order():
    grid = candidates(CFG, 'rf')
    assert grid == [
        {'n_estimators': 100, 'max_features': 'sqrt', 'min_samples_leaf': 1},
        {'n_estimators': 100, 'max_features': 'sqrt', 'min_samples_leaf': 3},
        {'n_estimators': 100, 'max_features': 0.5, 'min_samples_leaf': 1},
        {'n_estimators': 100, 'max_features': 0.5, 'min_samples_leaf': 3},
    ]
    assert np.argmax([0.2, 0.5, 0.5]) == 1


def test_deterministic_model_uses_single_seed():
    assert model_seeds(CFG, 'svr') == (42,)
    assert model_seeds(CFG, 'rf') == (42, 2026)


@pytest.mark.parametrize('model', ['svr', 'rf', 'xgb', 'lgbm'])
def test_each_model_fits_tiny_problem(model):
    rng = np.random.default_rng(0)
    X = rng.integers(0, 2, size=(60, 40)).astype(float)
    y = X[:, :5].mean(1) + rng.normal(0, 0.05, size=60)
    train, target = np.arange(45), np.arange(45, 60)
    spec = candidates(CFG, model)[-1]
    pred = fit_predict(model, spec, X, y, train, target, 42)
    assert pred.shape == (15,)
    assert np.all(np.isfinite(pred))
    assert pcc(y[target], pred) > 0.5
