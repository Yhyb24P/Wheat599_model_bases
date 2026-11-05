import inspect
import json

import numpy as np
import pytest

from wheat599 import mulphe


def synthetic(rows=60, markers=50, tasks=4, seed=0):
    rng = np.random.default_rng(seed)
    x = (rng.random((rows, markers)) < 0.5).astype(np.float32)
    signal = x[:, :4].mean(axis=1)
    y = np.stack([signal + 0.3 * rng.normal(size=rows) for _ in range(tasks)],
                 axis=1).astype(np.float32)
    order = rng.permutation(rows)
    return x, y, order[:36], order[36:48], order[48:]


CONFIG = {
    'hidden_dim': 8, 'conv_channels': 4, 'kernel_size': 3, 'stride': 2,
    'batch_size': 8, 'lr': 0.01, 'direct_epochs': 2, 'pretrain_epochs': 2,
    'finetune_epochs': 2, 'fusion_epochs': 2, 'direct_head_hidden': 8,
    'fusion_hidden': 8, 'dropout': 0.0, 'weight_decay': 0.0, 'patience': 1,
    'attn_entropy_lambda': 0.0, 'grad_clip': 5.0, 'seed': 42,
}


def run(x_train, y_all_train, x_val, y_all_val, x_test, config=CONFIG):
    return mulphe.run_arms(
        x_train, y_all_train[:, 0], x_val, y_all_val[:, 0], x_test,
        target_name='e0', source_names=['e1', 'e2', 'e3'],
        source_train=y_all_train[:, 1:], source_val=y_all_val[:, 1:],
        config=config, seed=42)


@pytest.fixture(scope='module')
def suite():
    x, y, train, val, test = synthetic()
    return run(x[train], y[train], x[val], y[val], x[test]), \
        (x, y, train, val, test)


def test_validation_selection_uses_mse_plus_pearson_not_mse_only():
    truth = np.array([0.0, 1.0, 2.0, 3.0])
    low_mse_bad_corr = 1.5 - 0.01 * (truth - 1.5)
    higher_mse_good_corr = truth + 1.13
    mse_a = mulphe.selection_loss(low_mse_bad_corr, truth, combined=False)
    mse_b = mulphe.selection_loss(higher_mse_good_corr, truth, combined=False)
    comb_a = mulphe.selection_loss(low_mse_bad_corr, truth, combined=True)
    comb_b = mulphe.selection_loss(higher_mse_good_corr, truth, combined=True)
    assert mse_a < mse_b
    assert comb_a > comb_b


def test_api_never_accepts_test_phenotypes(suite):
    parameters = inspect.signature(mulphe.run_arms).parameters
    assert 'y_test' not in parameters and 'test_y' not in parameters
    assert 'y_test' not in inspect.signature(mulphe.run_arm).parameters
    result, _ = suite
    for arm in mulphe.ARMS:
        assert arm in result['arms']
        entry = result['arms'][arm]
        assert entry['prediction'].shape == (12,)
        assert np.isfinite(entry['prediction']).all()
        assert entry['val_prediction'].shape == (12,)
        for value in entry['val_metrics'].values():
            assert np.isfinite(value)
        assert len(entry['state_sha256']) == 64
        assert entry['parameters'] > 0 and entry['total_parameters'] >= entry['parameters']
        assert entry['epochs_run'] >= 0
        assert entry['train_seconds'] >= 0
        assert entry['amp'] is False
        json.dumps({key: value for key, value in entry.items()
                    if key not in ('prediction', 'val_prediction', 'source_scalars')})


def test_arms_differ_in_fusion_input_and_scaling(suite):
    result, _ = suite
    assert result['arms']['direct']['scaling']['mode'] == 'minmax'
    assert result['arms']['target_pretrain']['scaling']['mode'] == 'none'
    assert result['arms']['frozen_sources']['fusion_input'] == 'mean_embedding'
    assert result['arms']['scalar_prediction']['fusion_input'] == 'source_prediction'
    assert result['arms']['linear_prediction']['fusion_input'] == 'source_prediction_linear'
    assert result['arms']['embedding']['fusion_input'] == 'embedding'
    assert result['source_names'] == ['e1', 'e2', 'e3']
    assert result['source_pretraining_seconds'] >= 0
    assert result['impute_scope'] == 'train'
    assert result['official']['license'] == 'MIT'


def test_direct_scaler_is_fit_from_training_rows_only():
    x, y, train, val, test = synthetic()
    result = run(x[train], y[train], x[val], y[val], x[test])
    scaler = result['arms']['direct']['scaling']
    assert scaler['scope'] == 'train'
    assert scaler['low'] == float(y[train, 0].min())
    assert scaler['high'] == float(y[train, 0].max())


def test_pretrained_source_models_can_be_reused_across_targets():
    x, y, train, val, test = synthetic()
    models = mulphe.train_source_encoders(
        x[train], y[train], x[val], y[val], ['e0', 'e1', 'e2', 'e3'],
        {**CONFIG, 'pretrain_epochs': 1}, 1042, 'cpu')
    result = mulphe.run_arms(
        x[train], y[train, 0], x[val], y[val, 0], x[test],
        target_name='e0', source_names=['e1', 'e2', 'e3'],
        source_train=y[train, 1:], source_val=y[val, 1:],
        config=CONFIG, seed=42, arms=('frozen_sources',),
        source_models=models)
    assert result['source_pretraining_seconds'] == 0.0
    assert result['arms']['frozen_sources']['source_names'] == ['e1', 'e2', 'e3']
    assert np.isfinite(result['arms']['frozen_sources']['prediction']).all()


def _fixed_epochs():
    return {arm: 2 for arm in mulphe.ARMS if arm != 'linear_prediction'}


def test_full_dev_fixed_refit_has_no_validation_and_obeys_epochs():
    x, y, train, val, test = synthetic()
    dev = np.concatenate([train, val])
    result = mulphe.run_arms(
        x[dev], y[dev, 0], None, None, x[test],
        target_name='e0', source_names=['e1', 'e2', 'e3'],
        source_train=y[dev, 1:], source_val=None,
        config=CONFIG, seed=42, fixed_epochs=_fixed_epochs(),
        source_fixed_epochs={'e1': 2, 'e2': 2, 'e3': 2})
    assert result['mode'] == 'fixed_full_dev'
    for arm in mulphe.ARMS:
        entry = result['arms'][arm]
        assert entry['mode'] == 'fixed_full_dev'
        assert entry['val_prediction'] is None and entry['val_metrics'] is None
        assert entry['prediction'].shape == (12,) and np.isfinite(entry['prediction']).all()
        expected = 0 if arm == 'linear_prediction' else 2
        assert entry['epochs_run'] == expected
        assert entry['best_epoch'] == expected
    assert [entry['best_epoch'] for entry in result['source_pretraining']] == [2, 2, 2]


def test_fixed_refit_does_not_depend_on_test_rows_and_source_models_reuse():
    x, y, train, val, test = synthetic()
    dev = np.concatenate([train, val])
    changed = x.copy()
    changed[test] += 0.25
    fixed = _fixed_epochs()
    first = mulphe.run_arms(
        x[dev], y[dev, 0], None, None, x[test],
        target_name='e0', source_names=['e1', 'e2', 'e3'],
        source_train=y[dev, 1:], source_val=None, config=CONFIG, seed=42,
        fixed_epochs=fixed, source_fixed_epochs={'e1': 2, 'e2': 2, 'e3': 2})
    models = mulphe.train_source_encoders(
        x[dev], y[dev, 1:], None, None, ['e1', 'e2', 'e3'],
        {**CONFIG, 'pretrain_epochs': 2}, 1042, 'cpu', fixed_epochs={'e1': 2, 'e2': 2, 'e3': 2})
    second = mulphe.run_arms(
        x[dev], y[dev, 0], None, None, changed[test],
        target_name='e0', source_names=['e1', 'e2', 'e3'],
        source_train=y[dev, 1:], source_val=None, config=CONFIG, seed=42,
        fixed_epochs=fixed, source_fixed_epochs={'e1': 2, 'e2': 2, 'e3': 2},
        source_models=models)
    assert second['source_pretraining_seconds'] == 0.0
    for arm in mulphe.ARMS:
        assert first['arms'][arm]['history'] == second['arms'][arm]['history']
        assert first['arms'][arm]['state_sha256'] == second['arms'][arm]['state_sha256']
        assert not np.array_equal(first['arms'][arm]['prediction'],
                                  second['arms'][arm]['prediction'])


def test_fixed_mode_requires_fixed_epochs_and_forbids_validation_data():
    x, y, train, val, test = synthetic()
    dev = np.concatenate([train, val])
    with pytest.raises(ValueError, match='fixed_epochs'):
        mulphe.run_arms(
            x[dev], y[dev, 0], None, None, x[test],
            target_name='e0', source_names=['e1', 'e2', 'e3'],
            source_train=y[dev, 1:], source_val=None, config=CONFIG, seed=42)
    with pytest.raises(ValueError, match='fixed epochs'):
        mulphe.run_arms(
            x[dev], y[dev, 0], x[val], y[val, 0], x[test],
            target_name='e0', source_names=['e1', 'e2', 'e3'],
            source_train=y[dev, 1:], source_val=y[val, 1:], config=CONFIG, seed=42,
            fixed_epochs=_fixed_epochs())


def test_target_in_sources_is_rejected():
    x, y, train, val, test = synthetic()
    with pytest.raises(ValueError, match='Target environment'):
        mulphe.run_arms(
            x[train], y[train, 0], x[val], y[val, 0], x[test],
            target_name='e0', source_names=['e0', 'e2'],
            source_train=y[train][:, [0, 2]], source_val=y[val][:, [0, 2]],
            config=CONFIG, seed=42)


def test_test_rows_do_not_affect_fit_or_validation(suite):
    first, (x, y, train, val, test) = suite
    changed = x.copy()
    changed[test] += 0.25
    second = run(x[train], y[train], x[val], y[val], changed[test])
    for arm in mulphe.ARMS:
        a, b = first['arms'][arm], second['arms'][arm]
        np.testing.assert_array_equal(a['val_prediction'], b['val_prediction'])
        assert a['state_sha256'] == b['state_sha256']
        assert a['history'] == b['history']
        assert not np.array_equal(a['prediction'], b['prediction'])


def test_missing_values_are_imputed_without_nonfinite_predictions():
    x, y, train, val, test = synthetic()
    x_train = x[train].copy()
    x_val = x[val].copy()
    x_test = x[test].copy()
    x_train[0, 0] = np.nan
    x_val[0, 1] = np.inf
    x_test[0, 2] = np.nan
    y_train = y[train, 0].copy()
    y_train[0] = np.nan
    result = mulphe.run_arms(
        x_train, y_train, x_val, y[val, 0], x_test,
        target_name='e0', source_names=['e1', 'e2', 'e3'],
        source_train=y[train, 1:], source_val=y[val, 1:],
        config=CONFIG, seed=42, arms=('direct',))
    assert np.isfinite(result['arms']['direct']['prediction']).all()
    assert result['arms']['direct']['scaling']['scope'] == 'train'


def test_scalar_prediction_uses_source_head_predictions(suite):
    result, _ = suite
    entry = result['arms']['scalar_prediction']
    assert entry['fusion_input'] == 'source_prediction'
    source_metadata = entry['source_pretraining']
    assert [entry['name'] for entry in source_metadata] == ['e1', 'e2', 'e3']
    assert entry['source_scalars']['train'].shape == (36, 3)
    assert entry['source_scalars']['val'].shape == (12, 3)
    assert entry['source_scalars']['test'].shape == (12, 3)
    linear = result['arms']['linear_prediction']
    assert linear['fusion_input'] == 'source_prediction_linear'
    assert len(linear['coefficients']) == 4 and linear['epochs_run'] == 0
    assert linear['prediction'].shape == (12,)
