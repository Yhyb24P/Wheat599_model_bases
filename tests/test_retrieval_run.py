"""Tiny tests for the retrieval-development runner. A stub model stands in for cn's model."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from wheat599 import retrieval_run as runner


class StubModel(nn.Module):
    """Mirrors RetrievalTabM: fit-row buffers, (n, k, q) output, encodable reference library."""

    def __init__(self, train_x, train_y, *, arm='raw', dtype=torch.float32, k=3, **kwargs):
        super().__init__()
        x = torch.as_tensor(train_x, dtype=dtype)
        y = torch.as_tensor(train_y, dtype=dtype)
        self.arm, self.k = arm, int(k)
        std = x.std(dim=0, unbiased=False)
        self.register_buffer('x_mean', x.mean(dim=0))
        self.register_buffer('x_std', torch.where(std > 0, std, torch.ones_like(std)))
        self.register_buffer('y_mean', y.mean(dim=0))
        self.register_buffer('y_std', y.std(dim=0, unbiased=False).clamp_min(1e-12))
        self.head = nn.Linear(x.shape[1], y.shape[1], bias=True, dtype=dtype)
        with torch.no_grad():
            self.head.weight.normal_(0.0, 0.02)
            self.head.bias.zero_()
        self.encoded = 0

    def standardize_y(self, y):
        return (torch.as_tensor(y, dtype=self.y_mean.dtype) - self.y_mean) / self.y_std

    def inverse_y(self, prediction):
        return prediction * self.y_std + self.y_mean

    def encode_reference(self):
        self.encoded += 1

    def forward(self, x, query_ids=None):
        if self.training and query_ids is None:
            raise ValueError('training forwards must pass the local query ids')
        if query_ids is not None and int(query_ids.numel()) != int(x.shape[0]):
            raise ValueError('query ids must match the batch rows')
        x = torch.as_tensor(x, dtype=self.x_mean.dtype)
        base = self.head((x - self.x_mean) / self.x_std)
        offsets = (torch.arange(self.k, dtype=base.dtype) - 0.5 * (self.k - 1)) * 0.01
        return base.unsqueeze(1) + offsets.reshape(1, -1, 1)

    def parameter_count(self):
        return int(sum(parameter.numel() for parameter in self.parameters()))


CONFIG = {
    'device': 'cpu', 'cpu_threads': 1, 'gpu_memory_fraction': 0.5, 'dtype': 'float32',
    'seeds': [7], 'variants': list(runner.VARIANTS), 'outer_folds': [0],
    'configurations': [{'n_blocks': 1, 'd_block': 4, 'dropout': 0.0, 'k': 3,
                        'context_size': 8, 'learning_rate': 0.01, 'weight_decay': 0.0}],
    'training': {'max_epochs': 2, 'patience': 1, 'batch_size': 8, 'eval_batch_size': 32,
                 'gradient_clip': 1.0, 'tie_tolerance': 1e-12},
    'efficiency': {'batch_size': 8, 'warmup': 1, 'repeats': 2},
    'selection_metric': 'seed_prediction_ensemble_macro_pcc',
    'data': 'unused', 'data_sha256': '0' * 64, 'splits': 'unused',
}


def toy_data(rows=36, markers=8, environments=2, seed=0):
    generator = np.random.default_rng(seed)
    x = (generator.random((rows, markers)) < .6).astype(np.float64)
    y = generator.normal(size=(rows, environments))
    return SimpleNamespace(X=x, Y=y, n=rows, environments=['1', '2'])


def toy_fold(rows=36):
    thirds = np.array_split(np.arange(rows), 3)
    return {'fold': 0, 'dev_idx': np.arange(rows).tolist(), 'test_idx': [],
            'inner': [{'train_idx': np.setdiff1d(np.arange(rows), part).tolist(),
                       'val_idx': part.tolist()} for part in thirds]}


def stub_build(config, variant, seed, train_x, train_y, configuration, device):
    torch.manual_seed(int(seed))
    model = StubModel(train_x, train_y, arm=variant, dtype=torch.float32,
                      **runner.architecture_for(configuration, variant))
    return model.to(device)


def test_architecture_overrides_only_change_the_single_member_variant():
    entry = CONFIG['configurations'][0]
    assert runner.architecture_for(entry, 'raw')['k'] == 3
    assert runner.architecture_for(entry, 'retrieval')['k'] == 3
    assert runner.architecture_for(entry, 'single_retrieval')['k'] == 1
    assert set(runner.architecture_for(entry, 'raw')) == set(runner.ARCHITECTURE_KEYS)


def test_trajectory_uses_fit_rows_only_and_records_reference_encoding(monkeypatch):
    data = toy_data()
    data.Y[30:] = np.nan  # a test row read would poison the loss
    fold = {'fold': 0, 'dev_idx': list(range(30)), 'test_idx': list(range(30, 36)),
            'inner': [{'train_idx': list(range(24)), 'val_idx': list(range(24, 30))}]}
    monkeypatch.setattr(runner, 'build_model', stub_build)
    trajectory = runner.Trajectory(0, 0, 7, 'retrieval', 0, CONFIG['configurations'][0])
    record = runner.train_trajectory(data, fold, 0, trajectory, CONFIG, torch.device('cpu'),
                                     data.environments)
    assert np.isfinite(record['metrics']['pcc'])
    assert record['prediction'].shape == (6, 2)
    assert record['material_index'] == list(range(24, 30))
    assert record['curve'][0]['epoch'] == 0
    assert record['reference_encode_seconds'] is not None and record['reference_encode_seconds'] >= 0
    assert set(record['per_environment']) == {'1', '2'}


def test_member_mean_predictions_are_written_with_material_ids(tmp_path):
    records = [{'trajectory': {'fold': 0, 'inner': 0, 'variant': 'raw', 'seed': 7,
                              'config_index': 0},
                'material_index': [10, 11], 'prediction': np.array([[1.0, 2.0], [3.0, 4.0]])}]
    path = tmp_path / 'predictions.csv'
    runner.write_predictions(path, records)
    lines = path.read_text().strip().splitlines()
    assert lines[0] == 'fold,inner,variant,seed,config_index,material_index,y_pred_0,y_pred_1'
    assert lines[1].split(',')[:6] == ['0', '0', 'raw', '7', '0', '10']
    assert len(lines) == 3


def test_selection_uses_the_seed_ensemble_metric_and_reports_both_readings():
    truth = {inner: np.array([[0.0], [1.0]]) for inner in range(3)}
    records = []
    for inner in range(3):
        for seed, (first, second) in zip((42, 2026),
                                         (((-1.0,), (1.0,)), ((-2.0,), (2.0,)))):
            prediction = np.array([first, second])
            records.append({'trajectory': {'fold': 0, 'inner': inner, 'variant': 'raw',
                                           'seed': seed, 'config_index': 0},
                            'prediction': prediction,
                            'metrics': {'pcc': 1.0 if seed == 42 else -1.0, 'rmse': 0.0,
                                        'mae': 0.0}})
        for seed, (first, second) in zip((42, 2026),
                                         (((1.0,), (-1.0,)), ((1.0,), (-1.0,)))):
            prediction = np.array([first, second])
            records.append({'trajectory': {'fold': 0, 'inner': inner, 'variant': 'raw', 'seed': seed,
                                           'config_index': 1},
                            'prediction': prediction,
                            'metrics': {'pcc': 0.5, 'rmse': 0.1, 'mae': 0.1}})
    config = json.loads(json.dumps(CONFIG))
    config['variants'] = ['raw']
    config['seeds'] = [42, 2026]
    base = CONFIG['configurations'][0]
    config['configurations'] = [{**base, 'learning_rate': 0.01},
                                {**base, 'learning_rate': 0.02}]
    chosen = runner.select_arm(records, config, truth)[0]['raw']
    # Config 0 averages to the exact truth across seeds while its per-seed PCCs are 1 and -1;
    # config 1 is constant and has a positive per-seed mean, so only the ensemble metric separates.
    assert chosen['config_index'] == 0
    assert chosen['seed_ensemble_pcc'] > 0.99
    assert chosen['seed_mean_pcc'] == pytest.approx(0.0)
    assert set(chosen['seed_mean_metrics']) == set(runner.METRICS)


def test_selection_ties_prefer_the_earlier_configuration():
    truth = {inner: np.array([[0.0], [1.0]]) for inner in range(3)}
    config = json.loads(json.dumps(CONFIG))
    config['variants'] = ['raw']
    config['seeds'] = [42, 2026]
    base = CONFIG['configurations'][0]
    config['configurations'] = [{**base, 'learning_rate': 0.01},
                                {**base, 'learning_rate': 0.02}]
    records = []
    for index in range(2):
        for inner in range(3):
            for seed in (42, 2026):
                records.append({'trajectory': {'fold': 0, 'inner': inner, 'variant': 'raw',
                                               'seed': seed, 'config_index': index},
                                'prediction': np.array([[0.0], [1.0]]),
                                'metrics': {'pcc': 1.0, 'rmse': 0.0, 'mae': 0.0}})
    assert runner.select_arm(records, config, truth)[0]['raw']['config_index'] == 0


def test_tiny_run_writes_all_artifacts_for_the_three_arms(tmp_path, monkeypatch):
    config = json.loads(json.dumps(CONFIG))
    config['splits'] = str(tmp_path / 'splits.json')
    Path(config['splits']).write_text(json.dumps({'folds': [toy_fold()]}))
    monkeypatch.setattr(runner, 'load', lambda path, sha: toy_data())
    monkeypatch.setattr(runner, 'build_model', stub_build)
    args = SimpleNamespace(out=tmp_path / 'out', selection=None, config=Path(config['splits']))
    runner.run(config, args, Path('scripts/run_background_nested.py'))
    produced = sorted(item.name for item in args.out.iterdir())
    for expected in ('provenance.json', 'trajectories.json', 'predictions.csv',
                     'selection.json', 'execution_status.json', 'timings.json', 'failures.json'):
        assert expected in produced, produced
    status = json.loads((args.out / 'execution_status.json').read_text())
    assert status['status'] == 'complete'
    assert status['completed_trajectories'] == status['expected'] == 9  # 3 variants x 3 inner x 1 seed
    selection = json.loads((args.out / 'selection.json').read_text())
    assert set(selection['0']) == set(runner.VARIANTS)
    rows = (args.out / 'predictions.csv').read_text().strip().splitlines()
    assert len(rows) == 1 + 9 * 12  # one header plus every trajectory's twelve validation rows
    assert json.loads((args.out / 'failures.json').read_text()) == []


def test_a_failed_trajectory_propagates_and_is_not_certified(tmp_path, monkeypatch):
    config = json.loads(json.dumps(CONFIG))
    config['splits'] = str(tmp_path / 'splits.json')
    Path(config['splits']).write_text(json.dumps({'folds': [toy_fold()]}))
    monkeypatch.setattr(runner, 'load', lambda path, sha: toy_data())
    monkeypatch.setattr(runner, 'build_model', stub_build)
    real = runner.train_trajectory
    calls = {'count': 0}

    def failing(*arguments, **keywords):
        calls['count'] += 1
        if calls['count'] == 2:
            raise FloatingPointError('synthetic retrieval failure')
        return real(*arguments, **keywords)

    monkeypatch.setattr(runner, 'train_trajectory', failing)
    args = SimpleNamespace(out=tmp_path / 'out', selection=None, config=Path(config['splits']))
    with pytest.raises(FloatingPointError, match='synthetic retrieval failure'):
        runner.run(config, args, Path('scripts/run_background_nested.py'))
    status = json.loads((args.out / 'execution_status.json').read_text())
    assert status['status'] == 'failed' and status['completed_trajectories'] == 1
    assert json.loads((args.out / 'selection.json').read_text()) == {}


def test_training_requires_query_ids_while_validation_does_not():
    data = toy_data()
    x = torch.tensor(data.X[:8], dtype=torch.float32)
    model = StubModel(x, torch.tensor(data.Y[:8], dtype=torch.float32), arm='retrieval', k=2)
    model.train()
    with pytest.raises(ValueError, match='query ids'):
        model(x)
    model(x, query_ids=torch.arange(8))
    model.eval()
    with torch.no_grad():
        model(x)  # validation queries are outside the library
