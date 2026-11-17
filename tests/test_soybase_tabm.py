"""SoyBase neural classification: shapes, probability rows, learning, label guards."""
import numpy as np
import torch

from genopred.experiments.soybase.classification import fit_neural_proba

torch.set_num_threads(1)

SPEC = {'n_blocks': 1, 'd_block': 16, 'dropout': 0.0, 'k': 4,
        'learning_rate': 0.01, 'weight_decay': 0.0}
TRAIN = {'max_epochs': 30, 'patience': 30, 'batch_size': 32,
         'eval_batch_size': 64, 'gradient_clip': 5.0}


def separable(materials=160, markers=12, n_classes=3, seed=7):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_classes, markers)) * 3.0
    y = rng.integers(0, n_classes, size=materials)
    X = (centers[y] + rng.normal(size=(materials, markers))).astype(np.float32)
    train, target = np.arange(0, 120), np.arange(120, materials)
    return X, y, train, target


def fit_tabm(spec, train_cfg, X, y, train, target, seed, select_best=True):
    return fit_neural_proba('tabm', spec, train_cfg, X, y, train, target, seed, 3,
                            torch.device('cpu'), select_best)


def test_proba_shape_rows_and_learning():
    X, y, train, target = separable()
    proba, model, info = fit_tabm(SPEC, TRAIN, X, y, train, target, 42)
    assert proba.shape == (len(target), 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    assert (proba >= 0).all()
    assert model.backbone.output.out_features == 3
    assert 1 <= info['best_epoch'] <= info['epochs_run'] <= 30
    accuracy = (proba.argmax(axis=1) == y[target]).mean()
    assert accuracy > 0.8, accuracy


def test_seed_determinism():
    X, y, train, target = separable()
    first, _, _ = fit_tabm(SPEC, TRAIN, X, y, train, target, 42)
    second, _, _ = fit_tabm(SPEC, TRAIN, X, y, train, target, 42)
    np.testing.assert_array_equal(first, second)


def test_outer_refit_never_reads_target_labels():
    X, y, train, target = separable()
    train_cfg = dict(TRAIN, max_epochs=5)
    poisoned = y.copy()
    poisoned[target] = -12345
    proba_a, _, info_a = fit_tabm(SPEC, train_cfg, X, y, train, target, 42, select_best=False)
    proba_b, _, info_b = fit_tabm(SPEC, train_cfg, X, poisoned, train, target, 42, select_best=False)
    np.testing.assert_array_equal(proba_a, proba_b)
    assert info_a['epochs_run'] == 5
    assert info_a['best_epoch'] == 5
    assert np.isnan(info_a['best_val_ba'])


ISING_SPEC = {'spins': 4, 'learning_rate': 0.01, 'weight_decay': 0.0}


def test_ising_classification_shape_learning_and_label_guard():
    X, y, train, target = separable()
    proba, model, info = fit_neural_proba('ising', ISING_SPEC, TRAIN, X, y, train, target,
                                          42, 3, torch.device('cpu'))
    assert proba.shape == (len(target), 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    assert model.readout.shape == (4, 3)
    assert (proba.argmax(axis=1) == y[target]).mean() > 0.8
    train_cfg = dict(TRAIN, max_epochs=5)
    poisoned = y.copy()
    poisoned[target] = -12345
    a, _, _ = fit_neural_proba('ising', ISING_SPEC, train_cfg, X, y, train, target, 42, 3,
                               torch.device('cpu'), select_best=False)
    b, _, _ = fit_neural_proba('ising', ISING_SPEC, train_cfg, X, poisoned, train, target, 42, 3,
                               torch.device('cpu'), select_best=False)
    np.testing.assert_array_equal(a, b)


def test_chr_resmlp_groups_shape_learning_and_label_guard():
    X, y, train, target = separable(markers=12)
    chrom = np.asarray(['Gm01'] * 5 + ['Gm02'] * 4 + ['scaffold'] * 3)
    spec = {'d_group': 4, 'd_fuse': 8, 'learning_rate': 0.01, 'weight_decay': 0.0}
    proba, model, info = fit_neural_proba('chr_resmlp', spec, TRAIN, X, y, train, target,
                                          42, 3, torch.device('cpu'), chrom=chrom)
    assert proba.shape == (len(target), 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    assert model.groups == ['Gm01', 'Gm02', 'scaffold']
    assert (proba.argmax(axis=1) == y[target]).mean() > 0.8
    # main-effect path must see every marker, local path only its group
    assert model.main.in_features == 12
    assert [layer.in_features for layer in model.local] == [5, 4, 3]
    train_cfg = dict(TRAIN, max_epochs=5)
    poisoned = y.copy()
    poisoned[target] = -12345
    a, _, _ = fit_neural_proba('chr_resmlp', spec, train_cfg, X, y, train, target, 42, 3,
                               torch.device('cpu'), select_best=False, chrom=chrom)
    b, _, _ = fit_neural_proba('chr_resmlp', spec, train_cfg, X, poisoned, train, target, 42, 3,
                               torch.device('cpu'), select_best=False, chrom=chrom)
    np.testing.assert_array_equal(a, b)
