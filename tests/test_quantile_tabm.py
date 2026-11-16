import numpy as np
import pytest
import torch

pytest.importorskip('tabm')
from genopred.models.tabm.quantile import mixture_scores, quantile_loss, train


def test_quantile_loss_sums_quantiles_and_trains_members_separately():
    out = torch.tensor([[[0., 1.], [1., 3.]]], requires_grad=True)
    loss = quantile_loss(out, torch.tensor([[2.]]), torch.tensor([.25, .75]))
    assert loss.item() == pytest.approx((.5 + .75 + .25 + .25) / 2)
    loss.backward()
    assert torch.count_nonzero(out.grad) == 4


def test_real_tabm_outputs_finite_points_from_all_quantiles():
    rng = np.random.default_rng(31)
    x, y = rng.normal(size=(32, 8)), rng.normal(size=32)
    config = dict(n_blocks=2, d_block=16, dropout=.1, k=2,
                  learning_rate=.001, weight_decay=.001)
    settings = dict(n_quantiles=5, quantile_min=.001, quantile_max=.999,
        training=dict(max_epochs=2, patience=2, batch_size=8,
                      gradient_clip=5., tie_tolerance=1e-12))
    pred, info = train(x[:24], y[:24], x[24:], y[24:], config, 42, 0, settings, 'cpu')
    assert pred.shape == (8,) and np.isfinite(pred).all()
    assert 0 <= info['best_epoch'] <= 2


def test_mixture_scores_use_all_seed_and_member_atoms():
    distributions = [np.array([[[0., 2.], [1., 3.]], [[1., 3.], [2., 4.]]]),
                     np.array([[[4., 6.], [5., 7.]], [[5., 7.], [6., 8.]]])]
    truth = np.array([2.5, 10.])
    result = mixture_scores(distributions, truth)
    for i in range(len(truth)):
        atoms = np.concatenate([d[i].ravel() for d in distributions])
        expected = np.abs(atoms - truth[i]).mean() - np.abs(atoms[:, None] - atoms).mean() / 2
        lo, hi = np.quantile(atoms, [.05, .95], method='inverted_cdf')
        np.testing.assert_allclose(result[i], [expected, lo <= truth[i] <= hi, hi-lo])


@pytest.mark.parametrize('variant', ['midpoint', 'free', 'shared', 'shrink'])
def test_shape_training_produces_member_distributions(variant):
    rng = np.random.default_rng(34)
    x, y = rng.normal(size=(32, 8)), rng.normal(size=32)
    config = dict(n_blocks=2, d_block=16, dropout=.1, k=2,
                  learning_rate=.001, weight_decay=.001, shape_penalty=.1)
    settings = dict(n_quantiles=5, quantile_min=.001, quantile_max=.999,
                    optimizer_variant='shape_'+variant,
                    training=dict(objective='midpoint' if variant == 'midpoint' else 'shape',
                                  max_epochs=2, patience=2, batch_size=8,
                                  gradient_clip=5., tie_tolerance=1e-12))
    pred, info = train(x[:24], y[:24], x[24:], y[24:], config, 42, 0, settings, 'cpu')
    distribution = info['_distribution']
    assert distribution.shape == (8, 2, 5) and np.isfinite(distribution).all()
    np.testing.assert_allclose(pred, distribution.mean(axis=(1, 2)), atol=1e-6)
    if variant != 'midpoint':
        assert np.all(np.diff(distribution, axis=-1) >= 0)


def test_parallel_midpoint_preserves_predictions_and_distributions():
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    from experiments.wheat599.quantile_tabm_dev import _fit_worker, _init_fit_worker
    rng = np.random.default_rng(35)
    x, y = rng.normal(size=(32, 8)), rng.normal(size=(32, 4))
    folds = [dict(fold=0, inner=[dict(train_idx=list(range(24)), val_idx=list(range(24,32)))])]
    config = dict(n_blocks=2, d_block=16, dropout=.1, k=2, learning_rate=.001, weight_decay=.001)
    settings = dict(device='cpu', cpu_threads=1, optimizer_variant='shape_midpoint', n_quantiles=5,
                    training=dict(objective='midpoint', max_epochs=2, patience=2, batch_size=8,
                                  gradient_clip=5., tie_tolerance=1e-12))
    tasks = [(0, 0, 0, 0, seed) for seed in (42, 2026)]
    _init_fit_worker(x, y, folds, [config], settings, 1)
    serial = [_fit_worker(task) for task in tasks]
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context('spawn'),
                             initializer=_init_fit_worker, initargs=(x, y, folds, [config], settings, 2)) as pool:
        parallel = list(pool.map(_fit_worker, tasks))
    for a, b in zip(serial, parallel):
        np.testing.assert_array_equal(a[3], b[3])
        np.testing.assert_array_equal(a[4]['_distribution'], b[4]['_distribution'])
        assert a[4]['best_epoch'] == b[4]['best_epoch']


def test_fixed_epoch_refit_never_scores_query_labels(monkeypatch):
    from genopred.models.tabm import quantile
    rng = np.random.default_rng(36)
    x, y = rng.normal(size=(32, 8)), rng.normal(size=32)
    config = dict(n_blocks=2, d_block=16, dropout=.1, k=2, learning_rate=.001, weight_decay=.001)
    settings = dict(n_quantiles=5, optimizer_variant='shape_midpoint', fixed_epochs=2,
                    training=dict(objective='midpoint', max_epochs=10, patience=3, batch_size=8,
                                  gradient_clip=5., tie_tolerance=1e-12))
    def fail(*_):
        raise AssertionError('Outer labels must not enter training or selection')
    monkeypatch.setattr(quantile, 'pcc', fail)
    pred, info = train(x[:24], y[:24], x[24:], None, config, 42, 0, settings, 'cpu')
    assert pred.shape == (8,) and info['best_epoch'] == info['epochs_run'] == 2
    assert info['best_pcc'] is None and np.isfinite(info['_distribution']).all()
