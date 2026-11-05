import inspect
import json

import numpy as np
import pytest
import torch

from wheat599 import grouped_weights as weights
from wheat599 import robust_icm


def synthetic(rows=30, markers=8, tasks=4, seed=0):
    rng = np.random.default_rng(seed)
    x = (rng.random((rows, markers)) < 0.5).astype(np.float64)
    y = rng.normal(size=(rows, tasks))
    x_val = (rng.random((9, markers)) < 0.5).astype(np.float64)
    return x, y, x_val


BASE = {'bandwidth': 1.0, 'fraction': 0.5, 'ridge': [0.1, 0.1, 0.1, 0.1]}
CHOICE = {'eta': 0.5, 'delta': 2.5, 'multiplier': 1.0}
THRESHOLDS = np.array([1.0, 1.0, 1.0, 1.0])
SOLVER = {'max_iterations': 300, 'function_tolerance': 1e-11,
          'gradient_tolerance': 1e-7, 'accepted_gradient': 1e-5}


@pytest.fixture(scope='module')
def unit():
    x, y, x_val = synthetic()
    return weights.run_unit(x, y, x_val, BASE, CHOICE, THRESHOLDS, SOLVER)


def test_api_has_no_validation_phenotypes_and_returns_all_arms(unit):
    parameters = inspect.signature(weights.run_unit).parameters
    assert 'y_val' not in parameters and 'Y_val' not in parameters
    assert 'y_val' not in inspect.signature(weights.run_unit).parameters
    assert set(unit['arms']) == {'identity', 'scalar', 'network'}
    for arm in unit['arms'].values():
        assert arm['prediction'].shape == (9, 4)
        assert np.isfinite(arm['prediction']).all()
        payload = {key: value for key, value in arm.items() if key != 'prediction'}
        json.dumps(payload)
    json.dumps(unit['metadata'])


def test_identity_scalar_and_network_weights_are_bounded(unit):
    for name, arm in unit['arms'].items():
        weights_table = np.asarray(arm['weights']['w'], dtype=np.float64)
        d = np.asarray(arm['weights']['D'], dtype=np.float64)
        omega = np.asarray(arm['weights']['omega'], dtype=np.float64)
        positive = omega > 0
        assert np.allclose(weights_table[positive].sum(axis=0), 1.0, atol=1e-10)
        assert np.all(weights_table[~positive] == 0)
        assert np.all(d[~positive] == 0)
        assert np.all(d[positive] >= np.exp(-1) - 1e-12)
        assert np.all(d[positive] <= np.exp(1) + 1e-12)
        if name == 'identity':
            np.testing.assert_allclose(d[positive], 1.0, atol=1e-12)


def test_training_uses_100_steps_and_scalar_beta_is_projected(unit):
    scalar = unit['arms']['scalar']['training']
    network = unit['arms']['network']['training']
    assert scalar['steps'] == network['steps'] == 100
    assert len(scalar['losses']) == len(network['losses']) == 100
    assert 0.0 <= scalar['beta'] <= 2.0
    assert np.isfinite(scalar['losses']).all()
    assert np.isfinite(network['losses']).all()
    assert network['parameters'] < 200 and network['seed'] == 42
    assert unit['arms']['identity']['training']['steps'] == 0


def test_zero_initialization_is_identity_for_scalar_and_network():
    x, y, _ = synthetic()
    context = weights.Context(x, BASE['bandwidth'])
    _, z = context.evidence(y)
    log_omega = torch.log(torch.tensor(context.weights, dtype=torch.float64))
    positive = torch.tensor(context.positive, dtype=torch.bool)
    beta = torch.tensor(0.0, dtype=torch.float64)
    scalar = weights._scalar_weights(
        beta, torch.tensor(z, dtype=torch.float64), log_omega, positive).numpy()
    np.testing.assert_allclose(scalar[context.positive],
                               np.repeat(context.weights[context.positive, None], 4, axis=1), atol=1e-12)
    torch.manual_seed(42)
    network = weights.SharedWeightNet(4)
    with torch.no_grad():
        initial = network(
            torch.tensor(z, dtype=torch.float64), log_omega,
            torch.eye(4, dtype=torch.float64), positive).numpy()
    np.testing.assert_allclose(initial[context.positive],
                               np.repeat(context.weights[context.positive, None], 4, axis=1), atol=1e-12)


def test_weights_and_training_ignore_validation_markers_but_predictions_change():
    x, y, x_val = synthetic()
    x_other = x_val.copy()
    x_other[0, 0] = 1.0 - x_other[0, 0]
    first = weights.run_unit(x, y, x_val, BASE, CHOICE, THRESHOLDS, SOLVER)
    second = weights.run_unit(x, y, x_other, BASE, CHOICE, THRESHOLDS, SOLVER)
    for name in ('identity', 'scalar', 'network'):
        assert first['arms'][name]['weights'] == second['arms'][name]['weights']
        assert first['arms'][name]['training'] == second['arms'][name]['training']
    assert not np.allclose(first['arms']['identity']['prediction'],
                           second['arms']['identity']['prediction'])


def test_identity_generic_solver_matches_the_current_separable_solver():
    x, y, x_val = synthetic()
    context = weights.Context(x, BASE['bandwidth'])
    components = context.components(x_val)
    material = BASE['fraction'] * components['additive'] \
        + (1 - BASE['fraction']) * (1 - weights.TAU) * components['rbf']
    cross = BASE['fraction'] * components['additive_cross'] \
        + (1 - BASE['fraction']) * (1 - weights.TAU) * components['rbf_cross']
    for position in range(context.groups):
        if context.positive[position]:
            material += (1 - BASE['fraction']) * weights.TAU \
                * context.weights[position] * components['group_fit'][position]
            cross += (1 - BASE['fraction']) * weights.TAU \
                * context.weights[position] * components['group_cross'][position]
    prepared = robust_icm.features(
        x, y, np.arange(len(x)), np.arange(len(x), len(x) + len(x_val)),
        BASE, material_kernels=(material, cross))
    reference, _ = robust_icm.fit_predict(
        prepared, BASE['ridge'][0] * CHOICE['multiplier'], CHOICE['eta'],
        CHOICE['delta'], SOLVER, fixed_thresholds=THRESHOLDS)
    identity_weights = np.repeat(context.weights[:, None], 4, axis=1)
    prediction, diagnostics = weights._solve_arm(
        context, x_val, identity_weights, y, BASE['ridge'][0] * CHOICE['multiplier'],
        THRESHOLDS, CHOICE['eta'], BASE['fraction'], SOLVER)
    np.testing.assert_allclose(prediction, reference, atol=1e-10, rtol=0)
    assert diagnostics['thresholds_fixed'] is True
    assert diagnostics['gradient_inf'] <= SOLVER['accepted_gradient']


def test_full_kernel_is_psd_and_diagnostics_are_finite(unit):
    for arm in unit['arms'].values():
        diagnostics = arm['diagnostics']
        assert diagnostics['full_kernel_rows'] == 30 * 4
        assert np.isfinite(diagnostics['objective'])
        assert np.isfinite(diagnostics['gradient_inf'])
        assert diagnostics['gradient_inf'] <= SOLVER['accepted_gradient']
        assert all(value > 0 for value in diagnostics['thresholds'])
    metadata = unit['metadata']
    assert metadata['groups'] >= 1
    assert metadata['tau'] == weights.TAU
    assert metadata['identity_d_in_range'] is True
