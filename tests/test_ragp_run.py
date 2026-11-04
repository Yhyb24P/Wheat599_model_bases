"""Synthetic CPU checks for the RAGP development runner."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from wheat599.ragp import RAGPGenomic
from wheat599 import ragp_run as runner

torch.set_num_threads(1)


def toy_data(rows=40, markers=8, environments=2, seed=0, poison_test=False):
    generator = np.random.default_rng(seed)
    x = generator.normal(size=(rows, markers))
    y = generator.normal(size=(rows, environments))
    if poison_test:
        y[30:] = np.nan
    return SimpleNamespace(X=x, Y=y, n=rows, environments=['1', '2'])


def toy_fold(rows=40, development=30):
    parts = np.array_split(np.arange(development), 3)
    return {'fold': 0, 'dev_idx': np.arange(development).tolist(),
            'test_idx': np.arange(development, rows).tolist(),
            'inner': [{'train_idx': np.setdiff1d(np.arange(development), part).tolist(),
                       'val_idx': part.tolist()} for part in parts]}


def toy_config(tmp_path):
    return {
        'model': 'ragp', 'data': 'unused', 'data_sha256': '0' * 64,
        'splits': str(tmp_path / 'splits.json'), 'device': 'cpu', 'dtype': 'float32',
        'cpu_threads': 1, 'gpu_memory_fraction': 0.5, 'seeds': [7], 'outer_folds': [0],
        'architecture': {
            'hidden_dims': [4, 3], 'n_blocks': 1, 'dropout': 0.5, 'pca_variance': 0.95,
            'neighbors': 2, 'fusion_alpha': 0.4, 'label_scale': 0.01,
            'loss_alpha': [0.9, 0.8], 'triplet_anchors': 4, 'triplet_positives': 1,
            'triplet_negatives': 1, 'triplet_margin': 1.0,
        },
        'configurations': [{'learning_rate': 0.001, 'weight_decay': 0.0,
                            'max_epochs': 2, 'dropout': 0.0}],
        'training': {'batch_size': 8, 'eval_batch_size': 16, 'patience': 30,
                     'scheduler_gamma': 0.95, 'tie_tolerance': 1e-12},
        'selection_metric': 'seed_prediction_ensemble_macro_pcc',
        'checkpoint_storage': 'hash_only',
    }


def trajectory(fold=0, inner=0, seed=7, index=0, configuration=None):
    if configuration is None:
        configuration = {'learning_rate': 0.001, 'weight_decay': 0.0,
                         'max_epochs': 2, 'dropout': 0.0}
    return runner.Trajectory(fold, inner, seed, index, configuration)


def test_make_batches_merges_a_singleton_tail():
    assert [len(batch) for batch in runner.make_batches(torch.arange(9), 4)] == [4, 5]
    assert [len(batch) for batch in runner.make_batches(torch.arange(8), 4)] == [4, 4]
    assert [len(batch) for batch in runner.make_batches(torch.arange(3), 2)] == [3]
    assert [len(batch) for batch in runner.make_batches(torch.arange(5), 4)] == [5]
    with pytest.raises(ValueError, match='At least two'):
        runner.make_batches(torch.arange(1), 4)
    with pytest.raises(ValueError, match='batch_size'):
        runner.make_batches(torch.arange(4), 1)


def test_state_hash_is_deterministic_and_sensitive():
    first = {'b': torch.arange(4), 'a': torch.tensor([1.0, 2.0])}
    second = {'a': torch.tensor([1.0, 2.0]), 'b': torch.arange(4)}
    digest = runner.state_sha256(first)
    assert digest == runner.state_sha256(second)
    assert len(digest) == 64
    changed = {'a': torch.tensor([1.0, 3.0]), 'b': torch.arange(4)}
    assert runner.state_sha256(changed) != digest


def test_configuration_resolution_uses_entry_overrides():
    config = toy_config(Path('/tmp'))
    entry = config['configurations'][0]
    model = runner.model_settings(config, entry)
    assert model['dropout'] == 0.0
    assert model['hidden_dims'] == [4, 3]
    training = runner.training_settings(config, entry)
    assert training['learning_rate'] == 0.001
    assert training['max_epochs'] == 2
    assert training['patience'] == 30
    assert training['scheduler_gamma'] == 0.95
    assert training['batch_size'] == 8


def test_build_model_sets_seed_before_construction():
    data = toy_data()
    config = toy_config(Path('/tmp'))
    configuration = config['configurations'][0]
    x = torch.as_tensor(data.X, dtype=torch.float64)
    y = torch.as_tensor(data.Y, dtype=torch.float64)
    first = runner.build_model(config, 7, x, y, configuration, torch.device('cpu'))
    second = runner.build_model(config, 7, x, y, configuration, torch.device('cpu'))
    assert all(torch.equal(first.state_dict()[key], second.state_dict()[key])
               for key in first.state_dict())
    other = runner.build_model(config, 8, x, y, configuration, torch.device('cpu'))
    assert any(not torch.equal(first.state_dict()[key], other.state_dict()[key])
               for key in first.state_dict())


def test_tiny_trajectory_is_fit_only_and_reload_is_consistent():
    data = toy_data(poison_test=True)
    fold = toy_fold()
    config = toy_config(Path('/tmp'))
    record = runner.train_trajectory(data, fold, 0, trajectory(), config,
                                     torch.device('cpu'), data.environments)
    assert record['prediction'].shape == (10, 2)
    assert np.isfinite(record['prediction']).all()
    assert record['material_index'] == fold['inner'][0]['val_idx']
    assert record['reload_max_abs_difference'] <= 1e-10
    assert len(record['checkpoint_sha256']) == 64
    assert record['checkpoint_storage'] == 'hash_only'
    assert record['best_epoch'] >= 0
    assert 1 <= record['epochs_run'] <= 2
    assert len(record['curve']) == record['epochs_run']
    assert record['curve'][0]['epoch'] == 0
    assert record['model']['pca_components'] > 0
    assert record['memory'] is None
    assert record['inference_seconds'] >= 0
    assert set(record['per_environment']) == {'1', '2'}


def test_validation_never_runs_before_the_first_library_update(monkeypatch):
    original = RAGPGenomic.forward

    def guarded(self, x, query_ids=None, return_embeddings=False):
        if not self.training:
            assert all(environment.library_ready for environment in self.environments)
        return original(self, x, query_ids=query_ids,
                        return_embeddings=return_embeddings)

    monkeypatch.setattr(RAGPGenomic, 'forward', guarded)
    data = toy_data()
    config = toy_config(Path('/tmp'))
    record = runner.train_trajectory(data, toy_fold(), 0, trajectory(), config,
                                     torch.device('cpu'), data.environments)
    assert record['best_epoch'] >= 0


def test_best_reload_guard_detects_a_changed_prediction(monkeypatch):
    original = RAGPGenomic.load_state_dict

    def broken(self, state_dict, strict=True, assign=False):
        result = original(self, state_dict, strict=strict, assign=assign)
        with torch.no_grad():
            self.environments[0].encoder[0].weight.add_(0.5)
        return result

    monkeypatch.setattr(RAGPGenomic, 'load_state_dict', broken)
    data = toy_data()
    config = toy_config(Path('/tmp'))
    with pytest.raises(FloatingPointError, match='reload'):
        runner.train_trajectory(data, toy_fold(), 0, trajectory(), config,
                                torch.device('cpu'), data.environments)


def selection_config():
    return {
        'seeds': [42, 2026],
        'configurations': [{'learning_rate': 0.01}, {'learning_rate': 0.02}],
        'selection_metric': 'seed_prediction_ensemble_macro_pcc',
    }


def test_selection_uses_the_seed_ensemble_metric_and_reports_both_readings():
    truth = {inner: np.array([[0.0], [1.0]]) for inner in range(3)}
    records = []
    for inner in range(3):
        for seed, prediction, value in (
                (42, np.array([[-1.0], [1.0]]), 1.0),
                (2026, np.array([[-2.0], [2.0]]), -1.0)):
            records.append({'trajectory': {'fold': 0, 'inner': inner, 'seed': seed,
                                           'config_index': 0},
                            'prediction': prediction,
                            'metrics': {'pcc': value, 'rmse': 0.0, 'mae': 0.0}})
        for seed in (42, 2026):
            records.append({'trajectory': {'fold': 0, 'inner': inner, 'seed': seed,
                                           'config_index': 1},
                            'prediction': np.array([[1.0], [-1.0]]),
                            'metrics': {'pcc': 0.5, 'rmse': 0.1, 'mae': 0.1}})
    chosen = runner.select_common(records, selection_config(), truth)[0]
    assert chosen['config_index'] == 0
    assert chosen['seed_ensemble_pcc'] > 0.99
    assert chosen['seed_mean_pcc'] == pytest.approx(0.0)
    assert set(chosen['seed_mean_metrics']) == set(runner.METRICS)


def test_selection_ties_prefer_the_earlier_configuration():
    truth = {inner: np.array([[0.0], [1.0]]) for inner in range(3)}
    records = []
    for index in range(2):
        for inner in range(3):
            for seed in (42, 2026):
                records.append({'trajectory': {'fold': 0, 'inner': inner, 'seed': seed,
                                               'config_index': index},
                                'prediction': np.array([[0.0], [1.0]]),
                                'metrics': {'pcc': 1.0, 'rmse': 0.0, 'mae': 0.0}})
    assert runner.select_common(records, selection_config(), truth)[0]['config_index'] == 0


def test_tiny_run_writes_artifacts_and_no_weight_files(tmp_path, monkeypatch):
    config = toy_config(tmp_path)
    Path(config['splits']).write_text(json.dumps({'folds': [toy_fold()]}))
    monkeypatch.setattr(runner, 'load', lambda path, sha: toy_data(poison_test=True))
    args = SimpleNamespace(out=tmp_path / 'out', selection=None,
                           config=Path(config['splits']))
    runner.run(config, args, Path(__file__))
    produced = sorted(item.name for item in args.out.iterdir())
    for expected in ('provenance.json', 'trajectories.json', 'predictions.csv',
                     'selection.json', 'execution_status.json', 'timings.json',
                     'failures.json'):
        assert expected in produced, produced
    assert not any(name.endswith(('.pt', '.pth', '.ckpt')) for name in produced)
    status = json.loads((args.out / 'execution_status.json').read_text())
    assert status['status'] == 'complete'
    assert status['completed_trajectories'] == status['expected'] == 3
    selection = json.loads((args.out / 'selection.json').read_text())
    assert selection['0']['config_index'] == 0
    rows = (args.out / 'predictions.csv').read_text().strip().splitlines()
    assert len(rows) == 1 + 3 * 10
    trajectories = json.loads((args.out / 'trajectories.json').read_text())
    assert len(trajectories) == 3
    assert all('prediction' not in row for row in trajectories)
    assert all(len(row['checkpoint_sha256']) == 64 for row in trajectories)
    provenance = json.loads((args.out / 'provenance.json').read_text())
    assert provenance['budget'] == 3
    assert provenance['checkpoint_policy'].startswith('in-memory best state')
    assert provenance['upstream']['commit'].startswith('85675b1e')
    assert any(key.endswith('ragp_run.py') for key in provenance['source_sha256'])
    assert any(key.endswith('ragp.py') for key in provenance['source_sha256'])
    assert provenance['environment']['torch_version'] == torch.__version__
    assert provenance['environment']['python_executable']
    timings = json.loads((args.out / 'timings.json').read_text())
    assert timings['trajectories'] == 3
    assert timings['epochs'] == 6
    assert timings['total_seconds'] > 0
    assert json.loads((args.out / 'failures.json').read_text()) == []


def test_a_failed_trajectory_propagates_and_is_not_certified(tmp_path, monkeypatch):
    config = toy_config(tmp_path)
    Path(config['splits']).write_text(json.dumps({'folds': [toy_fold()]}))
    monkeypatch.setattr(runner, 'load', lambda path, sha: toy_data())
    real = runner.train_trajectory
    calls = {'count': 0}

    def failing(*arguments, **keywords):
        calls['count'] += 1
        if calls['count'] == 2:
            raise FloatingPointError('synthetic RAGP failure')
        return real(*arguments, **keywords)

    monkeypatch.setattr(runner, 'train_trajectory', failing)
    args = SimpleNamespace(out=tmp_path / 'out', selection=None,
                           config=Path(config['splits']))
    with pytest.raises(FloatingPointError, match='synthetic RAGP failure'):
        runner.run(config, args, Path(__file__))
    status = json.loads((args.out / 'execution_status.json').read_text())
    assert status['status'] == 'failed'
    assert status['completed_trajectories'] == 1
    assert json.loads((args.out / 'selection.json').read_text()) == {}
