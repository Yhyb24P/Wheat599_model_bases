"""Unit tests for the complete Mul-PheG2P unified candidate (no formal experiment)."""
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from wheat599 import mulphe, mulphe_unified

RNG = np.random.default_rng(0)
MARKERS = 12


def _tiny_config(**over):
    config = {**mulphe.DEFAULT_CONFIG,
              'hidden_dim': 8, 'conv_channels': 4, 'kernel_size': 3, 'stride': 2,
              'batch_size': 8, 'pretrain_epochs': 2, 'finetune_epochs': 2,
              'fusion_epochs': 2, 'direct_epochs': 2, 'patience': 50, 'lr': 1e-3}
    config.update(over)
    return config


def _data(n_train=24, n_val=8, n_predict=8, n_sources=3):
    x_train = RNG.normal(size=(n_train, MARKERS))
    x_val = RNG.normal(size=(n_val, MARKERS))
    x_predict = RNG.normal(size=(n_predict, MARKERS))
    y_train = RNG.normal(size=n_train)
    y_val = RNG.normal(size=n_val)
    source_train = RNG.normal(size=(n_train, n_sources))
    source_val = RNG.normal(size=(n_val, n_sources))
    return dict(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val,
                x_predict=x_predict, source_train=source_train, source_val=source_val,
                target_name='e0', source_names=[f'e{i + 1}' for i in range(n_sources)])


def _fit(device='cpu', fixed_epochs=None, **over):
    d = _data()
    if fixed_epochs is not None:
        d['x_val'] = None
        d['y_val'] = None
        d['source_val'] = None
    return mulphe_unified.fit_candidate(
        d['x_train'], d['y_train'], d['x_val'], d['y_val'], d['x_predict'],
        target_name=d['target_name'], source_names=d['source_names'],
        source_train=d['source_train'], source_val=d['source_val'],
        config=_tiny_config(**over), seed=42, device=device, fixed_epochs=fixed_epochs)


def test_no_test_phenotype_in_signature():
    params = set(inspect.signature(mulphe_unified.fit_candidate).parameters)
    assert not params & {'y_test', 'y_predict', 'test_phenotype'}


def test_validation_mode_arms_and_target_included():
    result = _fit()
    assert set(result['arms']) == set(mulphe_unified.ARMS)
    assert result['mode'] == 'validation'
    scalar = result['arms']['scalar_with_target']
    # three sources plus the target encoder => target self-attention is possible
    assert scalar['num_traits'] == 4
    assert scalar['fusion_input'] == 'source_predictions_with_target_mse'
    assert result['arms']['hidden_with_target']['fusion_input'] == 'embedding_with_target'
    for arm in mulphe_unified.ARMS:
        entry = result['arms'][arm]
        assert entry['val_prediction'].shape == (8,)
        assert np.isfinite(entry['val_prediction']).all()
        assert entry['prediction'].shape == (8,)
    stages = result['stage_epochs']
    # source pretrain plus the independent target pretrain, then adaptation
    assert set(stages['pretrain']) == {'e1', 'e2', 'e3', 'e0__pretrain'}
    assert isinstance(stages['finetune'], int)
    assert set(stages['fusion']) == set(mulphe_unified.ARMS)


def test_fixed_full_dev_refit_mode():
    fixed = {'pretrain': {'e1': 2, 'e2': 2, 'e3': 2, 'e0__pretrain': 2}, 'finetune': 2, 'fusion': 2}
    result = _fit(fixed_epochs=fixed)
    assert result['mode'] == 'fixed_full_dev'
    for arm in mulphe_unified.ARMS:
        entry = result['arms'][arm]
        assert entry['val_prediction'] is None
        assert entry['prediction'].shape == (8,)
        assert np.isfinite(entry['prediction']).all()
        assert entry['best_epoch'] == 2


def test_deterministic_seed():
    fixed = {'pretrain': {'e1': 2, 'e2': 2, 'e3': 2, 'e0__pretrain': 2}, 'finetune': 2, 'fusion': 2}
    d = _data()
    d['x_val'] = d['y_val'] = d['source_val'] = None

    def once():
        return mulphe_unified.fit_candidate(
            d['x_train'], d['y_train'], None, None, d['x_predict'],
            target_name=d['target_name'], source_names=d['source_names'],
            source_train=d['source_train'], source_val=None,
            config=_tiny_config(), seed=42, device='cpu', fixed_epochs=fixed
        )['arms']['scalar_with_target']['prediction']

    assert np.allclose(once(), once())


def test_target_cannot_be_source():
    d = _data()
    with pytest.raises(ValueError):
        mulphe_unified.fit_candidate(
            d['x_train'], d['y_train'], d['x_val'], d['y_val'], d['x_predict'],
            target_name='e0', source_names=['e0', 'e1'],
            source_train=d['source_train'][:, :2], source_val=d['source_val'][:, :2],
            config=_tiny_config(), seed=42, device='cpu')


def test_median_stage_epochs():
    inner = [{'pretrain': {'e1': a, 'e2': b, 'e0__pretrain': c}, 'finetune': d,
              'fusion': {'scalar_with_target': e, 'hidden_with_target': e}}
             for a, b, c, d, e in [(1, 2, 3, 4, 5), (3, 4, 5, 6, 7), (5, 6, 7, 8, 9)]]
    fixed = mulphe_unified._median_stage_epochs(inner, 'scalar_with_target')
    assert fixed == {'pretrain': {'e1': 3, 'e2': 4, 'e0__pretrain': 5},
                     'finetune': 6, 'fusion': 7}


def test_fit_calibration():
    pred = np.array([0.0, 1.0, 2.0, 3.0])
    truth = 2.0 * pred + 1.0
    a, b = mulphe_unified._fit_calibration(pred, truth)
    assert np.isclose(a, 2.0) and np.isclose(b, 1.0)
    # degenerate constant prediction falls back to identity
    a, b = mulphe_unified._fit_calibration(np.ones(4), np.array([1.0, 2.0, 3.0, 4.0]))
    assert (a, b) == (1.0, 0.0)


def test_check_split_rejects_overlap():
    fold = {'dev_idx': [0, 1, 2, 3], 'test_idx': [4, 5]}
    mulphe_unified._check_split(fold, {'train_idx': [0, 1], 'val_idx': [2, 3]})
    with pytest.raises(ValueError):
        mulphe_unified._check_split(fold, {'train_idx': [0, 1], 'val_idx': [1, 2, 3]})
    with pytest.raises(ValueError):
        mulphe_unified._check_split(fold, {'train_idx': [0, 1, 4], 'val_idx': [2, 3]})


def _fake_fit(x_train, y_train, x_val, y_val, x_predict, *, target_name, source_names,
              source_train, source_val, config, seed, device, fixed_epochs=None):
    base = np.asarray(x_predict, dtype=np.float64).mean(axis=1)
    val = None if x_val is None else np.asarray(x_val, dtype=np.float64).mean(axis=1)
    arms = {}
    for arm in mulphe_unified.ARMS:
        arms[arm] = {'prediction': base.copy(),
                     'val_prediction': None if val is None else val.copy(),
                     'val_metrics': None if val is None else {'pcc': 0.0, 'rmse': 1.0, 'mae': 1.0},
                     'best_epoch': 2, 'epochs_run': 2, 'history': [], 'state_sha256': 'x',
                     'parameters': 1, 'total_parameters': 1, 'train_seconds': 0.0}
    return {'arms': arms,
            'stage_epochs': {'pretrain': {**{n: 2 for n in source_names},
                                          f'{target_name}__pretrain': 2},
                             'finetune': 2, 'fusion': {a: 2 for a in mulphe_unified.SYSTEMS}},
            'mode': 'validation' if x_val is not None else 'fixed_full_dev'}


def test_run_wiring(tmp_path, monkeypatch):
    n, n_env = 20, 4
    x = RNG.normal(size=(n, MARKERS))
    y = RNG.normal(size=(n, n_env))
    data = SimpleNamespace(X=x, Y=y, n=n, n_env=n_env, environments=['1', '2', '4', '5'])
    monkeypatch.setattr(mulphe_unified, 'load', lambda *a, **k: data)
    monkeypatch.setattr(mulphe_unified, 'fit_candidate', _fake_fit)

    folds = []
    for f in range(10):
        test = [2 * f, 2 * f + 1]
        dev = [i for i in range(n) if i not in test]
        inner = []
        for j in range(3):
            val = dev[j::3]
            train = [i for i in dev if i not in val]
            inner.append({'train_idx': train, 'val_idx': val})
        folds.append({'fold': f, 'dev_idx': dev, 'test_idx': test, 'inner': inner})
    splits = tmp_path / 'splits.json'
    splits.write_text(json.dumps({'folds': folds}))
    data_file = tmp_path / 'data.bin'
    data_file.write_bytes(b'synthetic')
    config_file = tmp_path / 'config.json'
    cfg = {'data': str(data_file), 'data_sha256': 'unused', 'splits': str(splits),
           'seeds': [42], 'device': 'cpu', 'cpu_threads': 1, 'gpu_memory_fraction': 0.5,
           'author_default_index': 2, 'model_config': _tiny_config(),
           'configurations': [{'hidden_dim': 8, 'lr': lr} for lr in
                              (0.0005, 0.001, 0.001, 0.0005, 0.002, 0.002, 0.003, 0.003)]}
    config_file.write_text(json.dumps(cfg))
    out = tmp_path / 'out'
    args = SimpleNamespace(out=out, config=config_file, selection=None)

    summary = mulphe_unified.run(cfg, args, Path(__file__))
    assert set(summary) == {f'{v}/{a}' for v in ('selected', 'author_default')
                            for a in mulphe_unified.SYSTEMS}
    for entry in summary.values():
        assert np.isfinite(entry['per_fold_macro']['pcc'])
        assert len(entry['grid']) == 40
    assert json.loads((out / 'execution_status.json').read_text())['status'] == 'complete'
    assert json.loads((out / 'failures.json').read_text()) == []
    assert len(json.loads((out / 'selection.json').read_text())) == 10
    assert (out / 'predictions.csv').is_file()
    assert json.loads((out / 'refit.json').read_text())
    assert (out / 'artifact_manifest.json').is_file()