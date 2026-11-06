import numpy as np

from wheat599.profiled_huber import solve, profiled_neural_loss
from wheat599.robust_icm import features, fit_predict


def test_members_are_optimized_separately_and_zero_network_recovers_solution():
    rng = np.random.default_rng(7)
    design = rng.normal(size=(25, 4))
    factor = np.array([[1., 0.], [.3, .9]])
    y = rng.normal(size=(25, 2))
    zero = np.zeros((25, 3, 2))
    w, c, pred, diagnostics = solve(design, factor, y, zero, .4, np.ones(2))
    w1, c1, pred1, _ = solve(design, factor, y, zero[:, :1], .4, np.ones(2))
    np.testing.assert_allclose(pred, pred1, atol=1e-9)
    np.testing.assert_allclose(pred, design @ w @ factor.T + c)
    assert diagnostics['gradient_inf'] < 1e-5
    neural = zero.copy()
    neural[:, 0, 0] = 5
    neural[:, 1, 0] = -5
    _, _, pred_all, _ = solve(design, factor, y, neural, .4, np.ones(2))
    _, _, pred_mean, _ = solve(design, factor, y, neural.mean(axis=1, keepdims=True), .4, np.ones(2))
    assert np.max(np.abs(pred_all - pred_mean)) > 1e-3


def test_zero_network_recovers_existing_huber_predictions():
    rng = np.random.default_rng(31)
    x = rng.integers(0, 2, (38, 18)).astype(float)
    y = rng.normal(size=(38, 4))
    fit, val = np.arange(28), np.arange(28, 38)
    prepared = features(x, y, fit, val, {'bandwidth': 1., 'fraction': .25})
    settings = dict(max_iterations=300, function_tolerance=1e-11,
                    gradient_tolerance=1e-7, accepted_gradient=1e-5)
    thresholds = np.array([.5, .7, .9, 1.1])
    expected, _ = fit_predict(prepared, .3, .6, 1.345, settings,
                              fixed_thresholds=thresholds)
    signal = .4 * np.eye(4) + .6 * prepared['C_Y']
    eig, vec = np.linalg.eigh(signal)
    factor = vec * np.sqrt(eig)
    w, c, _, _ = solve(prepared['train_features'], factor, prepared['Y'].numpy(),
                       np.zeros((len(fit), 3, 4)), .3, thresholds)
    actual = (prepared['target_features'] @ w @ factor.T + c) * prepared['scale'] + prepared['mean']
    np.testing.assert_allclose(actual, expected, atol=2e-4, rtol=0)


def test_envelope_gradient_matches_reoptimized_objective():
    import torch
    rng = np.random.default_rng(83)
    design = rng.normal(size=(18, 3))
    factor = np.array([[1., 0.], [.2, .8]])
    target = rng.normal(size=(18, 2))
    neural = rng.normal(scale=.1, size=(18, 2, 2))
    delta = np.array([.7, .9])
    kwargs = dict(maxiter=1000, ftol=1e-14, gtol=1e-10, accepted_gradient=1e-7)
    _, _, fitted, _ = solve(design, factor, target, neural, .5, delta, **kwargs)
    f = torch.tensor(neural, dtype=torch.float64, requires_grad=True)
    profiled_neural_loss(f, torch.tensor(fitted), torch.tensor(target), torch.tensor(delta)).backward()
    eps = 1e-4
    a, b = neural.copy(), neural.copy()
    a[0, 0, 0] += eps
    b[0, 0, 0] -= eps
    plus = solve(design, factor, target, a, .5, delta, **kwargs)[3]['objective']
    minus = solve(design, factor, target, b, .5, delta, **kwargs)[3]['objective']
    np.testing.assert_allclose(f.grad[0, 0, 0], (plus-minus)/(2*eps), atol=2e-6)


def test_unequal_environment_member_counts_ignore_padding():
    import torch
    rng = np.random.default_rng(12)
    design = rng.normal(size=(16, 3))
    target = rng.normal(size=(16, 2))
    neural = rng.normal(size=(16, 4, 2))
    changed = neural.copy()
    changed[:, 2:, 0] = 1000
    args = (design, np.eye(2), target)
    a = solve(*args, neural, .5, np.ones(2), member_counts=[2, 4])
    b = solve(*args, changed, .5, np.ones(2), member_counts=[2, 4])
    np.testing.assert_allclose(a[2], b[2], atol=1e-12)
    f = torch.tensor(changed, requires_grad=True)
    profiled_neural_loss(f, torch.tensor(a[2]), torch.tensor(target), torch.ones(2), [2, 4]).backward()
    assert torch.count_nonzero(f.grad[:, 2:, 0]) == 0
