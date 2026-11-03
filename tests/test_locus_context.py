"""Required checks for the locus-preserving context model.

Covers exactly the agreed validation set: exact self-exclusion at the response input,
Gram consistency and positive semidefiniteness of the normalised feature kernel,
equivalence of the gamma = 0 branch with the retained GLS solution, gradients through
the head and the normalisation statistics, and the train/predict data boundary.
"""

import math

import numpy as np
import pytest
import torch

from wheat599.kernel_baselines import kernels as base_kernels
from wheat599.locus_context import (LocusContextFeatures, TASK_TOLERANCE, VARIANTS,
                                    feature_statistics, icm_prediction, train_context,
                                    predict_context, _combine, _episode)
from wheat599.multienv_kernel import fit_predict, task_correlation

torch.set_num_threads(1)


def toy(seed=3, materials=40, markers=25, environments=4):
    generator = np.random.default_rng(seed)
    X = (generator.random((materials, markers)) < 0.5).astype(np.float64)
    Y = generator.normal(size=(materials, environments))
    base = {'bandwidth': 1.0, 'fraction': 0.5, 'ridge': 0.1, 'eta': 0.8}
    cfg = {'rank': 4, 'learning_rate': 1e-3, 'training_steps': 3, 'checkpoints': [0, 1, 3],
           'support_fraction': 2.0 / 3.0, 'device': 'cpu', 'gpu_memory_fraction': 0.2}
    return X, Y, base, cfg


def test_context_is_exact_and_self_excluding():
    torch.manual_seed(0)
    model = LocusContextFeatures(25, rank=4, variant='full').eval()
    z = torch.randn(12, 25)
    focus = 7
    shifted = z.clone()
    shifted[:, focus] += 1.7
    with torch.no_grad():
        first = model.contexts(z)
        second = model.contexts(shifted)
    assert first.shape == (12, 25, 4)
    assert torch.allclose(first[:, focus], second[:, focus], atol=1e-5), \
        'focus context must not depend on the focus marker'
    assert not torch.allclose(first[:, 0], second[:, 0], atol=1e-5)
    linear = LocusContextFeatures(25, rank=4, variant='linear').eval()
    with torch.no_grad():
        linear.contexts(z)
    background = LocusContextFeatures(25, rank=4, variant='full_background').eval()
    with torch.no_grad():
        changed = background.contexts(shifted)
        base = background.contexts(z)
    assert changed.shape[1] == 1
    assert not torch.allclose(changed[:, 0], base[:, 0], atol=1e-6), \
        'the full-background control must read the focus marker'


def test_feature_kernel_is_consistent_and_psd():
    torch.manual_seed(1)
    model = LocusContextFeatures(25, rank=4, variant='full')
    z = torch.randn(30, 25)
    features = model(z)
    center, scale = feature_statistics(features)
    normalised = (features - center) / scale
    gram = normalised @ normalised.T
    assert torch.allclose(gram, gram.T, atol=1e-6)
    eigenvalues = torch.linalg.eigvalsh(gram.double())
    assert float(eigenvalues.min()) >= -1e-8 * float(eigenvalues.max())
    assert abs(float(torch.diagonal(gram).mean()) - 1.0) < 1e-4
    base = torch.eye(30, dtype=torch.float64)
    cross = torch.eye(30, dtype=torch.float64)
    mixed, _ = _combine(base, cross, normalised, normalised, 1.0)
    assert torch.allclose(mixed, (base + gram) / 2, atol=1e-12)
    assert abs(float(torch.diagonal(mixed).mean()) - 1.0) < 1e-4


def test_gamma_zero_matches_retained_gls():
    X, Y, base, cfg = toy(seed=5, materials=36)
    additive, additive_cross, rbf, rbf_cross = base_kernels(
        X, np.arange(len(X)), np.arange(len(X)), base['bandwidth'], torch.device('cpu'))
    kernel = base['fraction'] * additive + (1 - base['fraction']) * rbf
    task = (1 - base['eta']) * np.eye(4) + base['eta'] * task_correlation(Y)
    retained = fit_predict([kernel], [kernel], Y, [task], np.full(4, base['ridge']))
    candidate = predict_context(X, Y, X, base, 'full', 0.0, None, 5, cfg)
    assert candidate.shape == retained.shape
    difference = float(np.abs(candidate - retained).max())
    assert difference < 1e-8, f'gamma = 0 branch differs from the retained solve by {difference}'


@pytest.mark.parametrize('variant', VARIANTS)
def test_gradients_reach_parameters_and_statistics(variant):
    torch.manual_seed(2)
    model = LocusContextFeatures(25, rank=4, variant=variant)
    z = torch.randn(16, 25)
    features = model(z)
    center, scale = feature_statistics(features)
    assert center.requires_grad and scale.requires_grad, \
        'normalisation statistics must keep their gradient'
    weights = torch.randn_like(features)
    loss = (((features - center) / scale) * weights).sum()
    targets = [p for p in model.parameters()] + [center, scale]
    gradients = torch.autograd.grad(loss, targets, allow_unused=True)
    assert all(g is not None and torch.isfinite(g).all() for g in gradients)
    assert all(float(g.abs().sum()) > 0 for g in gradients)
    assert gradients[-2] is not None and float(gradients[-2].abs().sum()) > 0


def test_normalised_icm_gradient_matches_finite_difference():
    torch.manual_seed(21)
    support = torch.randn(6, 4, dtype=torch.float64, requires_grad=True)
    target = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
    labels = torch.randn(6, 2, dtype=torch.float64)
    truth = torch.randn(3, 2, dtype=torch.float64)
    task = torch.tensor([[1., .4], [.4, 1.]], dtype=torch.float64)
    base = torch.eye(6, dtype=torch.float64)
    cross = torch.zeros(3, 6, dtype=torch.float64)

    def objective(fit_features, target_features):
        center, scale = feature_statistics(fit_features)
        kernel, target_kernel = _combine(base, cross, (fit_features-center)/scale,
                                          (target_features-center)/scale, .5)
        prediction = icm_prediction(kernel, target_kernel, labels, task, .2)
        return (prediction-truth).square().mean()

    assert torch.autograd.gradcheck(objective, (support, target), eps=1e-6,
                                    atol=1e-5, rtol=1e-4)


def test_training_interface_and_checkpoints():
    X, Y, base, cfg = toy(seed=7)
    states, record = train_context(X, Y, base, 'full', 0.5, 1e-4, 11, cfg)
    assert sorted(states) == cfg['checkpoints']
    assert all(isinstance(key, int) for key in states)
    assert all(isinstance(value, torch.Tensor) and value.device.type == 'cpu'
               for state in states.values() for value in state.values())
    assert record['features_trained'] is True
    assert [entry['step'] for entry in record['curve']] == cfg['checkpoints']
    assert record['curve'][0]['loss'] is None
    assert all(entry['loss'] is not None and math.isfinite(entry['loss'])
               for entry in record['curve'][1:])
    assert record['cost']['steps'] == cfg['training_steps']
    prediction = predict_context(X, Y, X[:7], base, 'full', 0.5, states[3], 11, cfg)
    assert prediction.shape == (7, 4) and np.isfinite(prediction).all()
    with pytest.raises(ValueError):
        predict_context(X, Y, X[:7], base, 'full', 0.5, None, 11, cfg)
    with pytest.raises(ValueError):
        predict_context(X, Y, X[:7], base, 'full', 0.0, states[3], 11, cfg)


def test_episode_stream_is_shared_across_variants():
    X, Y, base, cfg = toy(seed=9)
    first, _ = train_context(X, Y, base, 'full', 0.5, 1e-4, 13, cfg)
    second, record = train_context(X, Y, base, 'global', 1.0, 1e-2, 13, cfg)
    other, other_record = train_context(X, Y, base, 'full', 0.5, 1e-4, 14, cfg)
    assert record['episode_digest'] == train_context(
        X, Y, base, 'full', 0.5, 1e-4, 13, cfg)[1]['episode_digest']
    assert other_record['episode_digest'] != record['episode_digest']
    assert sorted(first) == cfg['checkpoints'] and sorted(second) == cfg['checkpoints']


def test_episode_partitions_rows_and_labels_stay_inside():
    generator = torch.Generator().manual_seed(0)
    support, query = _episode(21, 2.0 / 3.0, generator)
    assert set(support) & set(query) == set()
    assert set(support) | set(query) == set(range(21))
    assert len(support) + len(query) == 21
    X, Y, base, cfg = toy(seed=8)
    with pytest.raises(ValueError):
        train_context(X, Y[:-1], base, 'full', 0.5, 1e-4, 0, cfg)
    with pytest.raises(ValueError):
        train_context(X, Y, base, 'nope', 0.5, 1e-4, 0, cfg)
    with pytest.raises(ValueError):
        train_context(X, Y, base, 'full', 0.5, 1e-4, 0, {**cfg, 'checkpoints': [9]})


def test_icm_prediction_rejects_indefinite_task():
    kernel = torch.eye(4, dtype=torch.float64)
    labels = torch.zeros(4, 2, dtype=torch.float64)
    bad = torch.tensor([[1.0, 2.0], [2.0, 1.0]], dtype=torch.float64)
    assert float(torch.linalg.eigvalsh(bad).min()) < -TASK_TOLERANCE
    with pytest.raises(ValueError):
        icm_prediction(kernel, kernel, labels, bad, 0.1)
