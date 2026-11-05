"""Tiny two-phase tests for the unified DeepGS/BiMSGP runner; no real network or data."""
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from wheat599 import wheat2000_neural_run as runner


def test_bimsgp_univariate_sequence_input_contract():
    generator = np.random.default_rng(67)
    data = SimpleNamespace(X=generator.normal(size=(9, 5)).astype(np.float32),
                           Y=generator.normal(size=(9, 1)))
    tensors = runner.prepare_tensors(
        data, np.arange(6), np.arange(6, 9), 0,
        {'genotype_preprocessing': 'standardize'}, torch.float32,
        torch.device('cpu'), need_query_y=False, sequence_input=True)
    assert tensors.x_fit.shape == (6, 5, 1)
    assert tensors.x_query.shape == (3, 5, 1)
    # Same last-dimension contract as the official Linear(1, d_model) embedding.
    class SequenceEmbedding(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Linear(1, 4)

        def forward(self, x):
            return self.embedding(x).mean(dim=(1, 2))
    prediction = runner.predict(SequenceEmbedding(), tensors.x_query, 2)
    assert prediction.shape == (3,) and np.isfinite(prediction).all()


TRAITS = ('TKW', 'TW', 'GL', 'GW', 'GH', 'GP', 'SDS', 'PHT')


class StubModel(nn.Module):
    def __init__(self, n_markers, sequence_input=False):
        super().__init__()
        self.sequence_input = sequence_input
        self.head = nn.Linear(n_markers, 1)

    def forward(self, x):
        if self.sequence_input:
            assert x.ndim == 3 and x.shape[-1] == 1
            x = x.squeeze(-1)
        else:
            assert x.ndim == 2
        return self.head(x).reshape(-1)

    def parameter_count(self):
        return int(sum(parameter.numel() for parameter in self.parameters()))


def stub_build(config, data, seed, architecture, device):
    torch.manual_seed(int(seed))
    return StubModel(data.X.shape[1], sequence_input=
                     config['model'] == 'wheat2000_bimsgp').to(device)


def make_folds(n=48, n_folds=3):
    ids = np.arange(n)
    folds = []
    for fold, test in enumerate(np.array_split(ids, n_folds)):
        dev = np.setdiff1d(ids, test)
        inner = []
        for chunk in np.array_split(dev, 3):
            inner.append({'train_idx': np.setdiff1d(dev, chunk).tolist(),
                          'val_idx': chunk.tolist()})
        folds.append({'fold': int(fold), 'dev_idx': dev.tolist(),
                      'test_idx': test.tolist(), 'inner': inner})
    return folds


def toy_data(n=48, markers=6, traits=8, seed=0):
    generator = np.random.default_rng(seed)
    x = (generator.random((n, markers)) < 0.6).astype(np.float32)
    y = generator.normal(size=(n, traits)).astype(np.float64)
    return SimpleNamespace(X=x, Y=y, trait_names=TRAITS[:traits],
                           sample_ids=np.arange(n), genotype_ids=np.arange(n) + 100,
                           phenotype_ids=np.arange(n) + 200)


def make_config(tmp_path, folds, model='wheat2000_deepgs', configurations=2,
                seeds=(7, 11), traits=8, with_test_nan=False):
    splits_path = tmp_path / 'splits.json'
    splits_path.write_text(json.dumps({
        'schema_version': 1, 'data_sha256': '0' * 64, 'cache_sha256': '1' * 64,
        'seed': 7, 'outer_folds': len(folds), 'inner_folds': 3,
        'trait_names': list(TRAITS)[:traits],
        'valid_material_ids_all_traits': list(range(48)), 'folds': folds,
    }))
    config = {
        'model': model,
        'data': str(tmp_path / 'raw.RData'), 'data_sha256': '0' * 64,
        'cache': str(tmp_path / 'cache.npz'), 'cache_sha256': '1' * 64,
        'splits': str(splits_path), 'splits_sha256': runner.sha256_of(splits_path),
        'traits': list(TRAITS)[:traits], 'expected_shape': [48, 6, traits],
        'device': 'cpu', 'dtype': 'float32', 'cpu_threads': 1,
        'gpu_memory_fraction': 0.5, 'seeds': list(seeds),
        'expected_development_fits': len(folds) * traits * configurations
        * len(seeds) * 3,
        'expected_final_fits': len(folds) * traits * len(seeds),
        'save_weights': True, 'require_committed_selection': False,
        'source_path': str(tmp_path / 'bimsgp'), 'source_commit': '0' * 40,
        'configurations': [
            {'architecture': {'stub': 1},
             'optimizer': {'name': 'sgd', 'lr': 0.01, 'weight_decay': 0.0,
                           'momentum': 0.9}},
            {'architecture': {'stub': 2},
             'optimizer': {'name': 'adamw', 'lr': 0.02, 'weight_decay': 0.0}},
        ][:configurations],
        'training': {'max_epochs': 3, 'early_stopping_patience': 2,
                     'batch_size': 8, 'eval_batch_size': 16, 'gradient_clip': 1.0,
                     'amp': False, 'genotype_preprocessing': 'standardize',
                     'scheduler': 'plateau', 'lr_patience': 2, 'lr_factor': 0.5,
                     'loss_scale': 1.0, 'initialization': 'uniform',
                     'initialization_scale': 0.02},
    }
    config_path = tmp_path / 'config.json'
    config_path.write_text(json.dumps(config))
    return config, config_path


def patch_runner(monkeypatch, data):
    monkeypatch.setattr(runner, 'load_protocol', lambda config: (
        data, FOLDS, [str(value) for value in data.trait_names]))
    monkeypatch.setattr(runner, 'model_source_info',
                        lambda config, data: {'model': 'stub', 'sha256': 'stub'})
    monkeypatch.setattr(runner, 'build_network', stub_build)
    monkeypatch.setattr(runner, 'git_head', lambda: 'stub')


FOLDS = []


def run_development(tmp_path, monkeypatch, config, config_path, data, folds):
    global FOLDS
    FOLDS = folds
    patch_runner(monkeypatch, data)
    args = SimpleNamespace(out=tmp_path / 'development', selection=None,
                           config=str(config_path))
    runner.development(config, args, None)
    return args.out


def test_frozen_lr_uses_three_sequences_and_pads_with_the_last_value():
    assert runner.frozen_lr([[0.1, 0.05], [0.1], [0.01]], 4) == \
        [0.1, 0.05, 0.05, 0.05]
    assert runner.frozen_lr([[0.1, 0.05, 0.01], [0.1, 0.05], [0.1]], 2) == \
        [0.1, 0.05]
    with pytest.raises(ValueError):
        runner.frozen_lr([[0.1], [0.1]], 2)


def test_nonfinite_metrics_are_rejected():
    with pytest.raises(FloatingPointError):
        runner.metric_dict(np.zeros(4), np.zeros(4))
    data = toy_data()
    data.Y[np.arange(10), 0] = np.nan
    with pytest.raises(ValueError, match='Nonfinite fit phenotypes'):
        runner.prepare_tensors(data, np.arange(10), np.arange(10, 14), 0,
                               {'genotype_preprocessing': 'raw'}, torch.float32,
                               torch.device('cpu'), need_query_y=False)


def test_prepare_tensors_scales_fit_rows_only():
    data = toy_data()
    training = {'genotype_preprocessing': 'standardize'}
    fit, query_one, query_two = np.arange(20), np.arange(20, 24), np.arange(24, 28)
    first = runner.prepare_tensors(data, fit, query_one, 0, training, torch.float32,
                                   torch.device('cpu'), need_query_y=False)
    second = runner.prepare_tensors(data, fit, query_two, 0, training, torch.float32,
                                    torch.device('cpu'), need_query_y=False)
    np.testing.assert_array_equal(first.scaler['mean'], second.scaler['mean'])
    np.testing.assert_array_equal(first.scaler['std'], second.scaler['std'])
    assert torch.equal(first.x_fit, second.x_fit)
    assert first.y_mean == second.y_mean and first.y_std == second.y_std


@pytest.mark.parametrize('model', ['wheat2000_deepgs', 'wheat2000_bimsgp'])
def test_development_never_reads_outer_test_phenotypes(tmp_path, monkeypatch, model):
    folds = make_folds(n_folds=2)[:1]
    config, config_path = make_config(tmp_path, folds, model=model,
                                      configurations=1, seeds=(7,), traits=8)
    data = toy_data()
    data.Y[np.asarray(folds[0]['test_idx'])] = np.nan
    out = run_development(tmp_path, monkeypatch, config, config_path, data, folds)
    selection = json.loads((out / 'selection.json').read_text())
    assert len(selection['selected']) == 8
    selected = selection['selected'][0]
    assert set(selected['seeds']) == {'7'}
    block = selected['seeds']['7']
    assert block['median_epoch'] == sorted(block['per_inner_best_epochs'])[1]
    assert len(block['lr']) == block['median_epoch']
    dev = set(folds[0]['dev_idx'])
    with (out / 'predictions.csv').open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    assert {int(row['material_index']) for row in rows} == dev
    assert json.loads((out / 'execution_status.json').read_text())['status'] == 'complete'
    assert json.loads((out / 'failures.json').read_text()) == []


def test_final_refits_all_seeds_and_scores_only_after_all_fits(tmp_path, monkeypatch):
    folds = make_folds()
    config, config_path = make_config(tmp_path, folds)
    data = toy_data()
    out = run_development(tmp_path, monkeypatch, config, config_path, data, folds)
    selection_path = out / 'selection.json'
    selection = json.loads(selection_path.read_text())
    chosen = {(row['fold'], row['trait_index']): row for row in selection['selected']}
    truth = data.Y.copy()
    data.Y = GuardedTruth(truth, [fold['test_idx'] for fold in folds])
    calls = []
    original = runner.train_fixed

    def wrapped(*arguments, **keywords):
        result = original(*arguments, **keywords)
        calls.append(1)
        if len(calls) == config['expected_final_fits']:
            data.Y.locked = False
        return result

    monkeypatch.setattr(runner, 'train_fixed', wrapped)
    args = SimpleNamespace(out=tmp_path / 'final', selection=str(selection_path),
                           config=str(config_path))
    runner.final(config, args, None)
    assert len(calls) == config['expected_final_fits'] == 48
    with (args.out / 'seed_predictions.csv').open(newline='') as handle:
        seed_rows = list(csv.DictReader(handle))
    with (args.out / 'oof_predictions.csv').open(newline='') as handle:
        oof_rows = list(csv.DictReader(handle))
    assert len(seed_rows) == 8 * 2 * 48
    assert len(oof_rows) == 8 * 48
    assert 'y_true' not in seed_rows[0]
    summary = json.loads((args.out / 'summary.json').read_text())
    assert summary['counts'] == {'fits': 48, 'seed_prediction_rows': 768,
                                 'ensemble_oof_rows': 384,
                                 'per_fold_trait_rows': 24,
                                 'weights_saved': 48}
    assert set(summary['seed_readings']) == {'7', '11'}
    for seed in ('7', '11'):
        assert len(summary['seed_readings'][seed]) == 8
        for entry in summary['seed_readings'][seed]:
            assert set(entry) == {'trait', 'macro', 'pooled'}
            assert set(entry['macro']) == {'pcc', 'rmse', 'mae'}
            assert set(entry['pooled']) == {'pcc', 'rmse', 'mae'}
    assert len(summary['per_trait']) == 8
    assert len(json.loads((args.out / 'weights_manifest.json').read_text())) == 48
    status = json.loads((args.out / 'execution_status.json').read_text())
    assert status['status'] == 'complete' and status['completed_fits'] == 48
    for row in chosen.values():
        for seed in ('7', '11'):
            assert len(row['seeds'][seed]['lr']) == row['seeds'][seed]['median_epoch']


class GuardedTruth:
    """Blocks reads that are exactly one outer test set until every fit has finished."""

    def __init__(self, y, test_sets):
        self.y = np.asarray(y)
        self.shape = self.y.shape
        self.dtype = self.y.dtype
        self.test_sets = [set(map(int, values)) for values in test_sets]
        self.locked = True

    def __getitem__(self, key):
        rows = key[0] if isinstance(key, tuple) else key
        ids = set(np.asarray(rows).reshape(-1).tolist())
        if self.locked and any(ids == test for test in self.test_sets):
            raise AssertionError('outer test truth was read during fitting')
        return self.y[key]

    def __array__(self, *args, **kwargs):
        raise AssertionError('whole phenotype matrix was read')


def test_train_fixed_without_weights_marks_not_persisted(tmp_path, monkeypatch):
    folds = make_folds()
    config, _ = make_config(tmp_path, folds, configurations=1, seeds=(7,))
    config['save_weights'] = False
    data = toy_data()
    monkeypatch.setattr(runner, 'build_network', stub_build)
    result = runner.train_fixed(
        config, data, 0, 0, 0, 7, list(range(32)), list(range(32, 48)), 2,
        [0.01, 0.005], torch.device('cpu'), tmp_path / 'weights', False)
    assert result['weights_saved'] is False
    assert result['weight_path'] is None and result['weight_sha256'] is None
    assert result['state_note'] == 'state not persisted'
    assert result['trained_lrs'] == [0.01, 0.005]
    assert not (tmp_path / 'weights').exists()
    assert np.isfinite(result['prediction']).all()


def test_selection_rejects_source_model_and_config_tampering(tmp_path, monkeypatch):
    folds = make_folds()
    config, config_path = make_config(tmp_path, folds, configurations=1, seeds=(7,))
    data = toy_data()
    out = run_development(tmp_path, monkeypatch, config, config_path, data, folds)
    selection = json.loads((out / 'selection.json').read_text())
    chosen = runner.validate_selection(config, selection, config_path, None, folds,
                                       list(data.trait_names), data)
    assert len(chosen) == 24
    bad = json.loads((out / 'selection.json').read_text())
    bad['source_sha256'] = dict(bad['source_sha256'])
    bad['source_sha256'][str(config_path)] = 'f' * 64
    with pytest.raises(ValueError, match='source sha256'):
        runner.validate_selection(config, bad, config_path, None, folds,
                                  list(data.trait_names), data)
    bad_model = json.loads((out / 'selection.json').read_text())
    bad_model['model_source'] = {'model': 'other'}
    with pytest.raises(ValueError, match='model source'):
        runner.validate_selection(config, bad_model, config_path, None, folds,
                                  list(data.trait_names), data)
    bad_config = json.loads((out / 'selection.json').read_text())
    bad_config['config'] = dict(bad_config['config'])
    bad_config['config']['seeds'] = [1]
    with pytest.raises(ValueError, match='config does not match'):
        runner.validate_selection(config, bad_config, config_path, None, folds,
                                  list(data.trait_names), data)


def test_final_failure_is_recorded_and_never_scored(tmp_path, monkeypatch):
    folds = make_folds()
    config, config_path = make_config(tmp_path, folds, configurations=1, seeds=(7,))
    data = toy_data()
    out = run_development(tmp_path, monkeypatch, config, config_path, data, folds)
    selection_path = out / 'selection.json'
    calls = []

    def failing(*arguments, **keywords):
        calls.append(1)
        raise RuntimeError('synthetic neural failure')

    monkeypatch.setattr(runner, 'train_fixed', failing)
    args = SimpleNamespace(out=tmp_path / 'final', selection=str(selection_path),
                           config=str(config_path))
    with pytest.raises(RuntimeError, match='synthetic neural failure'):
        runner.final(config, args, None)
    failures = json.loads((args.out / 'failures.json').read_text())
    assert failures and 'synthetic neural failure' in failures[0]['reason']
    status = json.loads((args.out / 'execution_status.json').read_text())
    assert status['status'] == 'failed'
    assert not (args.out / 'oof_predictions.csv').exists()
    assert not (args.out / 'per_trait.csv').exists()
