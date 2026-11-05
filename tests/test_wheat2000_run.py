"""Tiny two-phase tests for the Wheat2000 TabM runner. A stub model replaces TabM."""
import copy
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from wheat599 import environment_run
from wheat599 import wheat2000_run as runner


def test_nonfinite_metric_fails_instead_of_publishing_nan():
    with pytest.raises(FloatingPointError, match='Nonfinite'):
        runner.metric_dict(np.arange(5.), np.ones(5))


class StubModel(nn.Module):
    """Single-target stand-in for the original TabM: fit-row buffers and (n, k, 1) output."""

    def __init__(self, train_x, train_y, *, k=3, dtype=torch.float32, **kwargs):
        super().__init__()
        x = torch.as_tensor(train_x, dtype=dtype)
        y = torch.as_tensor(train_y, dtype=dtype)
        self.k = int(k)
        std = x.std(dim=0, unbiased=False)
        self.register_buffer('x_mean', x.mean(dim=0))
        self.register_buffer('x_std', torch.where(std > 0, std, torch.ones_like(std)))
        self.register_buffer('y_mean', y.mean(dim=0))
        self.register_buffer('y_std', y.std(dim=0, unbiased=False).clamp_min(1e-12))
        self.head = nn.Linear(x.shape[1], y.shape[1], bias=True, dtype=dtype)
        with torch.no_grad():
            self.head.weight.normal_(0.0, 0.02)
            self.head.bias.zero_()

    def standardize_y(self, y):
        return (torch.as_tensor(y, dtype=self.y_mean.dtype) - self.y_mean) / self.y_std

    def inverse_y(self, prediction):
        return prediction * self.y_std + self.y_mean

    def forward(self, x):
        x = torch.as_tensor(x, dtype=self.x_mean.dtype)
        base = self.head((x - self.x_mean) / self.x_std)
        offsets = (torch.arange(self.k, dtype=base.dtype) - 0.5 * (self.k - 1)) * 0.001
        return base.unsqueeze(1) + offsets.reshape(1, -1, 1)

    def parameter_count(self):
        return int(sum(parameter.numel() for parameter in self.parameters()))


def stub_build(config, variant, seed, train_x, train_y, configuration, device):
    torch.manual_seed(int(seed))
    return StubModel(train_x, train_y, k=int(configuration['k'])).to(device)


def make_folds(n=60, n_folds=2):
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


def toy_data(n=60, markers=6, traits=2, seed=0):
    generator = np.random.default_rng(seed)
    x = (generator.random((n, markers)) < 0.6).astype(np.float32)
    y = generator.normal(size=(n, traits)).astype(np.float64)
    return SimpleNamespace(X=x, Y=y, trait_names=tuple(['TKW', 'TW'])[:traits],
                           sample_ids=np.arange(n), genotype_ids=np.arange(n) + 100,
                           phenotype_ids=np.arange(n) + 200)


def make_config(tmp_path, folds, n=60, markers=6, traits=2, seeds=(7, 11),
                configurations=2):
    splits_path = tmp_path / 'splits.json'
    document = {
        'schema_version': 1, 'data_sha256': '0' * 64, 'cache_sha256': '1' * 64,
        'seed': 7, 'outer_folds': len(folds), 'inner_folds': 3,
        'trait_names': list(['TKW', 'TW'])[:traits],
        'valid_material_ids_all_traits': list(range(n)), 'folds': folds,
    }
    splits_path.write_text(json.dumps(document))
    config = {
        'model': 'wheat2000_tabm',
        'data': str(tmp_path / 'raw.RData'), 'data_sha256': '0' * 64,
        'cache': str(tmp_path / 'cache.npz'), 'cache_sha256': '1' * 64,
        'splits': str(splits_path), 'splits_sha256': runner.sha256_of(splits_path),
        'traits': list(['TKW', 'TW'])[:traits], 'expected_shape': [n, markers, traits],
        'device': 'cpu', 'dtype': 'float32', 'cpu_threads': 1, 'gpu_memory_fraction': 0.5,
        'seeds': list(seeds),
        'expected_development_fits': len(folds) * traits * configurations
        * len(seeds) * 3,
        'expected_final_fits': len(folds) * traits * len(seeds),
        'save_weights': True,
        'configurations': [
            {'n_blocks': 1, 'd_block': 4, 'dropout': 0.0, 'k': 3,
             'learning_rate': 0.01, 'weight_decay': 0.0},
            {'n_blocks': 1, 'd_block': 4, 'dropout': 0.0, 'k': 2,
             'learning_rate': 0.02, 'weight_decay': 0.0},
        ][:configurations],
        'training': {'max_epochs': 2, 'patience': 1, 'batch_size': 8,
                     'eval_batch_size': 16, 'gradient_clip': 1.0,
                     'tie_tolerance': 1e-12},
        'efficiency': {'batch_size': 8, 'warmup': 1, 'repeats': 1},
        'versions': {},
        'require_committed_selection': False,
    }
    config_path = tmp_path / 'config.json'
    config_path.write_text(json.dumps(config))
    return config, config_path


def patch_runner(monkeypatch, data):
    monkeypatch.setattr(runner, 'load_data', lambda config: data)
    monkeypatch.setattr(runner, 'verify_versions', lambda config: {})
    monkeypatch.setattr(runner, 'model_source_sha', lambda: {'stub': 'wheat2000'})
    monkeypatch.setattr(runner, 'environment_provenance',
                        lambda device: {'device': str(device)})
    monkeypatch.setattr(runner, 'git_value', lambda *arguments: 'stub')
    monkeypatch.setattr(environment_run, 'build_environment_model', stub_build)


def run_development(tmp_path, monkeypatch, data, config, config_path):
    patch_runner(monkeypatch, data)
    args = SimpleNamespace(out=tmp_path / 'development', selection=None,
                           config=str(config_path))
    runner.development(config, args, None)
    return args.out


def test_development_uses_inner_rows_only_and_writes_expected_selection(tmp_path,
                                                                        monkeypatch):
    folds = make_folds(n_folds=2)[:1]
    config, config_path = make_config(tmp_path, folds)
    data = toy_data()
    data.Y[np.asarray(folds[0]['test_idx'])] = np.nan
    out = run_development(tmp_path, monkeypatch, data, config, config_path)

    selection = json.loads((out / 'selection.json').read_text())
    assert len(selection['configs']) == 2
    assert len(selection['epochs']) == 4
    for row in selection['configs']:
        assert row['config_index'] in (0, 1)
        assert 0 <= row['trait_index'] < 2
        assert np.isfinite(row['seed_ensemble_pcc'])
    for row in selection['epochs']:
        values = row['per_inner_best_epochs']
        assert len(values) == 3
        assert row['median_epoch'] == sorted(values)[1]
        assert 0 <= row['median_epoch'] <= 2
    assert selection['data_sha256'] == config['data_sha256']
    assert selection['splits_sha256'] == config['splits_sha256']

    dev = set(folds[0]['dev_idx'])
    with (out / 'predictions.csv').open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1 * 2 * 2 * 2 * 3 * 10
    assert {int(row['material_index']) for row in rows} == dev
    status = json.loads((out / 'execution_status.json').read_text())
    assert status['status'] == 'complete'
    assert status['completed_fits'] == 24
    assert json.loads((out / 'failures.json').read_text()) == []
    summary = json.loads((out / 'summary.json').read_text())
    assert len(summary['per_trait']) == 2
    assert summary['counts'] == {'fits': 24, 'selection_rows': 2,
                                 'median_epoch_rows': 4}
    assert (out / 'per_fold.csv').is_file() and (out / 'per_trait.csv').is_file()
    assert (out / 'trajectories.json').is_file()


def test_selection_rejects_tampering_and_accepts_zero_median_epoch(tmp_path,
                                                                   monkeypatch):
    folds = make_folds()
    config, config_path = make_config(tmp_path, folds)
    data = toy_data()
    out = run_development(tmp_path, monkeypatch, data, config, config_path)
    selection = json.loads((out / 'selection.json').read_text())

    chosen, epochs = runner.validate_selection(
        config, selection, config_path, None, folds, list(data.trait_names))
    assert len(chosen) == 4 and len(epochs) == 8

    zero = copy.deepcopy(selection)
    zero['epochs'][0]['per_inner_best_epochs'] = [0, 0, 2]
    zero['epochs'][0]['median_epoch'] = 0
    _, zero_epochs = runner.validate_selection(
        config, zero, config_path, None, folds, list(data.trait_names))
    assert zero_epochs[(0, 0, 7)] == 0

    bad_median = copy.deepcopy(selection)
    bad_median['epochs'][0]['median_epoch'] = \
        (bad_median['epochs'][0]['median_epoch'] + 1) % 3
    with pytest.raises(ValueError, match='Median epoch'):
        runner.validate_selection(config, bad_median, config_path, None, folds,
                                  list(data.trait_names))

    bad_data = copy.deepcopy(selection)
    bad_data['data_sha256'] = 'f' * 64
    with pytest.raises(ValueError, match='data sha256'):
        runner.validate_selection(config, bad_data, config_path, None, folds,
                                  list(data.trait_names))

    bad_source = copy.deepcopy(selection)
    bad_source['source_sha256'] = dict(bad_source['source_sha256'])
    bad_source['source_sha256'][str(config_path)] = 'f' * 64
    with pytest.raises(ValueError, match='source sha256'):
        runner.validate_selection(config, bad_source, config_path, None, folds,
                                  list(data.trait_names))


def test_frozen_selection_must_live_in_the_repository(tmp_path):
    path = tmp_path / 'selection.json'
    path.write_text('{}')
    with pytest.raises(ValueError, match='inside the repository'):
        runner.ensure_frozen_selection(path)


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


def test_final_fit_failure_is_recorded_and_never_scored(tmp_path, monkeypatch):
    folds = make_folds()[:1]
    config, config_path = make_config(tmp_path, folds)
    data = toy_data()
    out = run_development(tmp_path, monkeypatch, data, config, config_path)
    selection_path = out / 'selection.json'
    calls = []

    def failing_fit_one(row, local, fit_config, device, weights_dir, save_weights=True):
        calls.append(dict(row))
        if len(calls) == 2:
            raise RuntimeError('synthetic fit failure')
        return {'prediction': np.zeros(len(row['test_ids'])), 'test_ids': list(row['test_ids']),
                'weight_path': None, 'weight_sha256': None, 'weight_bytes': 0,
                'parameters': 10, 'epochs': int(row['epochs']),
                'preprocess_seconds': 0.0, 'train_seconds': 0.0,
                'predict_seconds': 0.0, 'memory': None}

    monkeypatch.setattr(runner, 'fit_one', failing_fit_one)
    args = SimpleNamespace(out=tmp_path / 'final', selection=str(selection_path),
                           config=str(config_path))
    with pytest.raises(RuntimeError, match='synthetic fit failure'):
        runner.final(config, args, None)
    failures = json.loads((args.out / 'failures.json').read_text())
    assert failures and 'synthetic fit failure' in failures[0]['reason']
    status = json.loads((args.out / 'execution_status.json').read_text())
    assert status['status'] == 'failed' and status['completed_fits'] == 1
    assert not (args.out / 'oof_predictions.csv').exists()
    assert not (args.out / 'per_trait.csv').exists()


def test_final_refits_every_seed_and_scores_only_after_all_fits(tmp_path, monkeypatch):
    folds = make_folds(n_folds=10)
    config, config_path = make_config(tmp_path, folds)
    data = toy_data()
    out = run_development(tmp_path, monkeypatch, data, config, config_path)
    selection_path = out / 'selection.json'
    selection = json.loads(selection_path.read_text())
    chosen = {(row['fold'], row['trait_index']): row['config_index']
              for row in selection['configs']}
    epochs = {(row['fold'], row['trait_index'], row['seed']): row['median_epoch']
              for row in selection['epochs']}

    truth = data.Y.copy()
    data.Y = GuardedTruth(truth, [fold['test_idx'] for fold in folds])
    calls = []

    def fake_fit_one(row, local, fit_config, device, weights_dir, save_weights=True):
        assert len(local.Y) == len(row['fit_ids'])
        assert len(local.X) == len(row['fit_ids']) + len(row['test_ids'])
        assert np.isfinite(local.Y).all()
        calls.append(dict(row))
        weights_dir.mkdir(parents=True, exist_ok=True)
        path = weights_dir / (f"f{row['fold']}_e{row['environment_index']}_"
                              f"s{row['seed']}_c{row['config_index']}.pt")
        torch.save({'epochs': row['epochs']}, path)
        prediction = (np.arange(len(row['test_ids']), dtype=np.float64) * 0.01
                      + float(row['seed']) + row['environment_index'])
        if len(calls) == config['expected_final_fits']:
            data.Y.locked = False
        return {'prediction': prediction, 'test_ids': list(row['test_ids']),
                'weight_path': str(path), 'weight_sha256': runner.sha256_of(path),
                'weight_bytes': path.stat().st_size, 'parameters': 10,
                'epochs': int(row['epochs']), 'preprocess_seconds': 0.0,
                'train_seconds': 0.0, 'predict_seconds': 0.0, 'memory': None}

    monkeypatch.setattr(runner, 'fit_one', fake_fit_one)
    args = SimpleNamespace(out=tmp_path / 'final', selection=str(selection_path),
                           config=str(config_path))
    runner.final(config, args, None)

    assert len(calls) == config['expected_final_fits'] == 40
    assert all(call['epochs'] == epochs[(call['fold'], call['environment_index'],
                                         call['seed'])] for call in calls)
    assert all(call['config_index'] == chosen[(call['fold'], call['environment_index'])]
               for call in calls)
    with (args.out / 'seed_predictions.csv').open(newline='') as handle:
        seed_rows = list(csv.DictReader(handle))
    with (args.out / 'oof_predictions.csv').open(newline='') as handle:
        oof_rows = list(csv.DictReader(handle))
    assert len(seed_rows) == 10 * 2 * 2 * 6
    assert len(oof_rows) == 10 * 2 * 6
    assert len(seed_rows) == 240 and len(oof_rows) == 120
    assert 'y_true' not in seed_rows[0]
    status = json.loads((args.out / 'execution_status.json').read_text())
    assert status['status'] == 'complete'
    assert status['completed_fits'] == 40
    summary = json.loads((args.out / 'summary.json').read_text())
    assert summary['counts']['fits'] == 40
    assert summary['counts']['seed_prediction_rows'] == len(seed_rows)
    assert summary['counts']['ensemble_oof_rows'] == len(oof_rows)
    assert len(summary['per_trait']) == 2
    assert (args.out / 'weights_manifest.json').is_file()
    assert len(json.loads((args.out / 'weights_manifest.json').read_text())) == 40
