"""Tiny tests for the environment-sharing runner. A stub module stands in for cn's model."""
import csv
import json
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from wheat599 import environment_run as runner


class StubBase(nn.Module):
    def __init__(self, train_x, train_y, *, dtype=torch.float32, k=3, n_blocks=2,
                 d_block=4, dropout=0.0):
        super().__init__()
        x = torch.as_tensor(train_x, dtype=dtype)
        y = torch.as_tensor(train_y, dtype=dtype)
        self.k, self.environments = int(k), int(y.shape[1])
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
        offsets = (torch.arange(self.k, dtype=base.dtype) - 0.5 * (self.k - 1)) * 0.01
        return base.unsqueeze(1) + offsets.reshape(1, -1, 1)

    def parameter_count(self):
        return int(sum(parameter.numel() for parameter in self.parameters()))


class StubShared(StubBase):
    pass


class StubEnvironment(StubBase):
    def __init__(self, *args, environment_penalty=0.001, **kwargs):
        super().__init__(*args, **kwargs)
        self.environment_penalty_coefficient = float(environment_penalty)
        self.scale = nn.Parameter(torch.zeros(self.environments))

    def environment_penalty(self):
        return 1e-3 * self.scale.square().sum()

    def parameter_groups(self, weight_decay):
        decay = [parameter for name, parameter in self.named_parameters() if name != 'scale']
        return [{'params': decay, 'weight_decay': float(weight_decay)},
                {'params': [self.scale], 'weight_decay': 0.0}]


def stub_module():
    module = types.ModuleType('wheat599.environment_tabm')
    module.SharedTabM = StubShared
    module.EnvironmentTabM = StubEnvironment
    return module


CONFIG = {
    'device': 'cpu', 'cpu_threads': 1, 'gpu_memory_fraction': 0.5, 'dtype': 'float32',
    'seeds': [7], 'outer_folds': [0],
    'variants': ['shared', 'independent', 'conditioned'],
    'configurations': [{'n_blocks': 2, 'd_block': 4, 'dropout': 0.0, 'k': 3,
                        'learning_rate': 0.01, 'weight_decay': 0.0}],
    'training': {'max_epochs': 2, 'patience': 1, 'batch_size': 8, 'eval_batch_size': 32,
                 'gradient_clip': 1.0, 'tie_tolerance': 1e-12},
    'efficiency': {'batch_size': 8, 'warmup': 1, 'repeats': 2},
    'selection_metric': 'seed_prediction_ensemble_macro_pcc',
    'data': 'unused', 'data_sha256': '0' * 64, 'splits': 'unused',
    'shared_source': 'unused', 'fusion_source': 'unused', 'fixed_fusion_weight': 0.5,
}


def toy_data(rows=36, markers=8, environments=4, seed=0):
    generator = np.random.default_rng(seed)
    x = (generator.random((rows, markers)) < .6).astype(np.float64)
    y = generator.normal(size=(rows, environments))
    return SimpleNamespace(X=x, Y=y, n=rows, environments=['1', '2', '4', '5'])


def toy_fold(rows=36):
    thirds = np.array_split(np.arange(rows), 3)
    return {'fold': 0, 'dev_idx': np.arange(rows).tolist(), 'test_idx': [],
            'inner': [{'train_idx': np.setdiff1d(np.arange(rows), part).tolist(),
                       'val_idx': part.tolist()} for part in thirds]}


def make_source(tmp_path, overrides=None):
    """Synthetic frozen raw arm: provenance, status, selection, failures, predictions."""
    source = tmp_path / 'shared'
    source.mkdir(parents=True)
    import hashlib
    genome = Path(runner.__file__).with_name('genomic_tabm.py')
    provenance = {'data_sha256': CONFIG['data_sha256'],
                  'splits_sha256': hashlib.sha256(b'{}').hexdigest(),
                  'config': {'seeds': CONFIG['seeds'],
                             'configurations': [dict(entry) for entry in CONFIG['configurations']],
                             'variants': ['raw']},
                  'source_sha256': {str(genome): hashlib.sha256(genome.read_bytes()).hexdigest()}}
    status = {'status': 'complete', 'completed_trajectories': 480, 'expected': 480,
              'folds_complete': list(range(10))}
    selection = {str(fold): {'raw': {'config_index': 0}} for fold in range(10)}
    (source / 'provenance.json').write_text(json.dumps(provenance))
    (source / 'execution_status.json').write_text(json.dumps(status))
    (source / 'selection.json').write_text(json.dumps(selection))
    (source / 'failures.json').write_text('[]')
    for key, value in (overrides or {}).items():
        (source / key).write_text(json.dumps(value))
    return source


def test_reuse_report_accepts_a_consistent_source_and_lists_failures(tmp_path):
    config = json.loads(json.dumps(CONFIG))
    config['shared_source'] = str(make_source(tmp_path))
    config['splits'] = str(tmp_path / 'splits.json')
    Path(config['splits']).write_text('{}')
    import hashlib
    provenance = json.loads((Path(config['shared_source']) / 'provenance.json').read_text())
    provenance['splits_sha256'] = hashlib.sha256(Path(config['splits']).read_bytes()).hexdigest()
    (Path(config['shared_source']) / 'provenance.json').write_text(json.dumps(provenance))
    runner.reuse_report(config)

    broken = make_source(tmp_path / 'broken',
                         overrides={'selection.json': {'0': {'selection_error': 'boom'}},
                                    'execution_status.json': {'status': 'failed',
                                                              'completed_trajectories': 1,
                                                              'expected': 480,
                                                              'folds_complete': [0]}})
    config['shared_source'] = str(broken)
    with pytest.raises(RuntimeError, match='Shared arm cannot be reused'):
        runner.reuse_report(config)


def test_compare_saved_uses_the_serialisation_tolerance():
    values = np.array([[1.0, 2.0], [3.0, 4.0]])
    runner.compare_saved(values, values.copy(), [0, 1])
    runner.compare_saved(values + 5e-7, values, [0, 1])
    with pytest.raises(RuntimeError, match='Saved prediction mismatch'):
        runner.compare_saved(values + 1e-2, values, [0, 1])


def test_conditioned_uses_environment_groups_and_penalty(monkeypatch):
    monkeypatch.setattr(runner, 'environment_module', stub_module)
    data = toy_data()
    fold = toy_fold()
    trajectory = runner.Trajectory(0, 0, 7, 'conditioned', 0, CONFIG['configurations'][0])
    record = runner.train_trajectory(data, fold, 0, trajectory, CONFIG, torch.device('cpu'))
    assert record['prediction'].shape == (12, 4)
    assert np.isfinite(record['metrics']['pcc'])
    captured = {}
    real = torch.optim.AdamW

    def spy(parameters, **kwargs):
        captured['groups'] = parameters
        return real(parameters, **kwargs)

    monkeypatch.setattr(torch.optim, 'AdamW', spy)
    runner.train_trajectory(data, fold, 0, trajectory, CONFIG, torch.device('cpu'))
    groups = captured['groups']
    assert len(groups) == 2 and groups[1]['weight_decay'] == 0.0, 'environment scale has no decay'


def test_independent_trains_one_environment_at_a_time(monkeypatch):
    monkeypatch.setattr(runner, 'environment_module', stub_module)
    data = toy_data()
    fold = toy_fold()
    trajectory = runner.Trajectory(0, 0, 7, 'independent', 0, CONFIG['configurations'][0],
                                   environment=2)
    record = runner.train_trajectory(data, fold, 0, trajectory, CONFIG, torch.device('cpu'))
    assert record['prediction'].shape == (12, 1)
    assert set(record['per_environment']) == {'4'}, 'environment index 2 is labelled "4"'


def test_selection_chooses_per_environment_and_prefers_earlier_configs():
    truth = {inner: np.array([[0.0, 0.5, 1.0, 1.5], [1.0, 1.5, 2.0, 2.5]]) for inner in range(3)}
    config = json.loads(json.dumps(CONFIG))
    config['variants'] = ['independent']
    config['seeds'] = [42, 2026]
    base = CONFIG['configurations'][0]
    config['configurations'] = [{**base, 'learning_rate': 0.01}, {**base, 'learning_rate': 0.02}]
    records = []
    for index in range(2):
        for inner in range(3):
            for environment in range(4):
                for seed in (42, 2026):
                    records.append({'trajectory': {'fold': 0, 'inner': inner, 'variant': 'independent',
                                                   'environment': environment, 'seed': seed,
                                                   'config_index': index},
                                    'prediction': np.array([[0.0], [1.0]]),
                                    'metrics': {'pcc': 1.0, 'rmse': 0.0, 'mae': 0.0}})
    for inner in range(3):
        runner._truth_cache[(0, inner)] = truth[inner]
    chosen = runner.select_common(records, config)[0]['independent']
    assert set(chosen) == set(range(4))
    for environment, entry in chosen.items():
        assert entry['config_index'] == 0, 'ties prefer the earlier configuration'


def test_tiny_run_writes_artifacts_and_aborts_on_failure(tmp_path, monkeypatch):
    config = json.loads(json.dumps(CONFIG))
    config['splits'] = str(tmp_path / 'splits.json')
    Path(config['splits']).write_text(json.dumps({'folds': [toy_fold()]}))
    monkeypatch.setattr(runner, 'load', lambda path, sha: toy_data())
    monkeypatch.setattr(runner, 'environment_module', stub_module)
    monkeypatch.setattr(runner, 'reuse_report', lambda cfg: ({}, {}))
    monkeypatch.setattr(runner, 'load_shared', lambda cfg: {})
    monkeypatch.setattr(runner, 'verify_checkpoint_reuse', lambda cfg, saved, device: {'ok': True})
    monkeypatch.setattr(runner, 'summarise', lambda *arguments: {'synthetic': True})
    monkeypatch.setattr(runner, 'verify_versions', lambda cfg: {})
    shared = tmp_path / 'shared'
    shared.mkdir()
    (shared / 'predictions.csv').write_text('variant,fold\n')
    config['shared_source'] = str(shared)
    args = SimpleNamespace(out=tmp_path / 'out', selection=None, config=Path(config['splits']))
    runner.run(config, args, Path('scripts/run_background_nested.py'))
    produced = sorted(item.name for item in args.out.iterdir())
    for expected in ('provenance.json', 'trajectories.json', 'predictions.csv', 'selection.json',
                     'execution_status.json', 'timings.json', 'failures.json'):
        assert expected in produced, produced
    status = json.loads((args.out / 'execution_status.json').read_text())
    assert status['status'] == 'complete'
    assert status['completed_trajectories'] == status['expected'] == 15  # 1 fold x 3 inner x 5 env fits
    rows = list(csv.DictReader((args.out / 'predictions.csv').open()))
    assert set(rows[0]) == {'variant', 'fold', 'inner', 'environment', 'seed', 'config_index',
                            'material_index', 'y_pred'}
    assert len({len(row) for row in rows}) == 1
    independent_rows = [row for row in rows if row['variant'] == 'independent']
    assert {row['environment'] for row in independent_rows} == {'1', '2', '4', '5'}
    conditioned_rows = [row for row in rows if row['variant'] == 'conditioned']
    assert len(conditioned_rows) == len(independent_rows), 'both variants emit four rows per material'
    assert json.loads((args.out / 'failures.json').read_text()) == []
    provenance = json.loads((args.out / 'provenance.json').read_text())
    assert 'shared_reuse' in provenance and provenance['shared_reuse']['checkpoint_reload_check']

    real = runner.train_trajectory
    calls = {'count': 0}

    def failing(*arguments, **keywords):
        calls['count'] += 1
        if calls['count'] == 2:
            raise FloatingPointError('synthetic environment failure')
        return real(*arguments, **keywords)

    monkeypatch.setattr(runner, 'train_trajectory', failing)
    args2 = SimpleNamespace(out=tmp_path / 'out2', selection=None, config=Path(config['splits']))
    with pytest.raises(FloatingPointError, match='synthetic environment failure'):
        runner.run(config, args2, Path('scripts/run_background_nested.py'))
    status = json.loads((args2.out / 'execution_status.json').read_text())
    assert status['status'] == 'failed' and status['completed_trajectories'] == 1


def test_summarise_reports_ensemble_seed_mean_and_wins(tmp_path):
    truth = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    indices = [10, 11, 12]
    runner._truth_cache[(0, 0)] = truth
    config = json.loads(json.dumps(CONFIG))
    fold = {'fold': 0, 'inner': [{'train_idx': [], 'val_idx': indices}]}

    def record(variant, environment, seed, prediction):
        return {'trajectory': {'fold': 0, 'inner': 0, 'variant': variant,
                               'environment': environment, 'seed': seed, 'config_index': 0},
                'prediction': prediction,
                'metrics': {'pcc': 1.0, 'rmse': 0.0, 'mae': 0.0}}

    exact = np.asarray(truth, dtype=np.float64)
    records = [record('conditioned', None, 7, exact),
               record('independent', 0, 7, exact[:, :1]),
               record('independent', 1, 7, exact[:, 1:])]
    perturbed = exact[[0, 2, 1]]  # non-monotone rows: the shared arm's PCC drops below one
    saved = {(0, 0, 0, 7, material): perturbed[position] for position, material in enumerate(indices)}
    selection = {'0': {'raw': {'config_index': 0}}}
    common = {'0': {'conditioned': {None: {'config_index': 0}},
                    'independent': {environment: {'config_index': 0} for environment in range(2)}}}
    config['fusion_source'] = str(tmp_path / 'fusion')
    fusion = Path(config['fusion_source'])
    fusion.mkdir(parents=True)
    (fusion / 'predictions.json').write_text(json.dumps([
        {'fold': 0, 'inner': 0, 'material_index': material,
         'tabm': exact[position].tolist(), 'icm': exact[position].tolist()}
        for position, material in enumerate(indices)]))
    report = runner.summarise(config, records, saved, selection, common, [fold])
    assert set(report) == {'shared', 'fixed_fusion', 'conditioned', 'independent'}
    for name in ('fixed_fusion', 'conditioned', 'independent'):
        assert report[name]['seed_ensemble']['pcc'] == pytest.approx(1.0)
    assert report['shared']['seed_ensemble']['pcc'] < 1.0
    assert report['conditioned']['wins_vs_shared_of_30'] == 1
    assert report['independent']['per_environment_improvement_vs_shared']['1']['pcc'] == \
        pytest.approx(0.5)


def test_incomplete_grid_fails_the_run_and_is_recorded(tmp_path, monkeypatch):
    config = json.loads(json.dumps(CONFIG))
    config['splits'] = str(tmp_path / 'splits.json')
    Path(config['splits']).write_text(json.dumps({'folds': [toy_fold()]}))
    monkeypatch.setattr(runner, 'load', lambda path, sha: toy_data())
    monkeypatch.setattr(runner, 'environment_module', stub_module)
    monkeypatch.setattr(runner, 'reuse_report', lambda cfg: ({}, {}))
    monkeypatch.setattr(runner, 'load_shared', lambda cfg: {})
    monkeypatch.setattr(runner, 'verify_checkpoint_reuse', lambda cfg, saved, device: {'ok': True})
    monkeypatch.setattr(runner, 'summarise', lambda *arguments: {})
    monkeypatch.setattr(runner, 'verify_versions', lambda cfg: {})
    shared = tmp_path / 'shared'
    shared.mkdir()
    (shared / 'predictions.csv').write_text('variant,fold\n')
    config['shared_source'] = str(shared)

    def broken_select(*arguments, **keywords):
        raise ValueError('synthetic incomplete grid')

    monkeypatch.setattr(runner, 'select_common', broken_select)
    args = SimpleNamespace(out=tmp_path / 'out', selection=None, config=Path(config['splits']))
    with pytest.raises(ValueError, match='synthetic incomplete grid'):
        runner.run(config, args, Path('scripts/run_background_nested.py'))
    status = json.loads((args.out / 'execution_status.json').read_text())
    assert status['status'] == 'failed' and status['folds_complete'] == []
    failures = json.loads((args.out / 'failures.json').read_text())
    assert failures and failures[0]['stage'] == 'selection'


def test_summary_uses_the_common_configuration_not_per_inner_optima(tmp_path):
    """Per-inner winners differ; the summary must follow the frozen common selection."""
    truth = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    indices = [10, 11, 12]
    for inner in range(3):
        runner._truth_cache[(0, inner)] = truth
    permuted = truth[[0, 2, 1]]
    reversed_ = truth[::-1]
    config = json.loads(json.dumps(CONFIG))
    config['variants'] = ['conditioned']
    config['seeds'] = [7]
    base = CONFIG['configurations'][0]
    config['configurations'] = [{**base, 'learning_rate': 0.01}, {**base, 'learning_rate': 0.02}]

    def record(inner, index, prediction, pcc_value):
        return {'trajectory': {'fold': 0, 'inner': inner, 'variant': 'conditioned',
                               'environment': None, 'seed': 7, 'config_index': index},
                'prediction': prediction,
                'metrics': {'pcc': pcc_value, 'rmse': 0.0, 'mae': 0.0}}

    records = []
    # config 0 is mediocre but consistent; config 1 is perfect on inners 0-1 and wrong on inner 2.
    for inner in range(3):
        records.append(record(inner, 0, permuted, 0.5))
        records.append(record(inner, 1, truth if inner < 2 else reversed_,
                              1.0 if inner < 2 else -1.0))
    chosen = runner.select_common(records, config)[0]['conditioned'][None]
    assert chosen['config_index'] == 0, 'the common mean must beat the per-inner favourites'
    selection = {'0': {'raw': {'config_index': 0}}}
    saved = {(0, inner, 0, 7, material): truth[position]
             for inner in range(3) for position, material in enumerate(indices)}
    config['fusion_source'] = str(tmp_path / 'fusion')
    Path(config['fusion_source']).mkdir(parents=True)
    payload = [{'fold': 0, 'inner': inner, 'material_index': material,
                'tabm': truth[position].tolist(), 'icm': truth[position].tolist()}
               for inner in range(3) for position, material in enumerate(indices)]
    (Path(config['fusion_source']) / 'predictions.json').write_text(json.dumps(payload))
    report = runner.summarise(config, records, saved, selection,
                              {'0': {'conditioned': {None: chosen}}},
                              [{'fold': 0, 'inner': [{'train_idx': [], 'val_idx': indices}] * 3}])
    # config 0 (PCC 0.5 on every inner), not the per-inner optimum (0.833 on average).
    assert report['conditioned']['seed_ensemble']['pcc'] == pytest.approx(0.5, abs=1e-9)


def test_predictions_csv_reconstructs_independent_per_environment_rows(tmp_path):
    records = [{'trajectory': {'variant': 'independent', 'fold': 0, 'inner': 0,
                               'environment': 2, 'seed': 7, 'config_index': 0},
                'material_index': [10, 11], 'prediction': np.array([[1.5], [2.5]])},
               {'trajectory': {'variant': 'conditioned', 'fold': 0, 'inner': 0,
                               'environment': None, 'seed': 7, 'config_index': 0},
                'material_index': [10, 11], 'prediction': np.array([[1.0, 2.0, 3.0, 4.0],
                                                                   [5.0, 6.0, 7.0, 8.0]])}]
    path = tmp_path / 'predictions.csv'
    runner.write_predictions(path, records, ['1', '2', '4', '5'])
    rows = list(csv.DictReader(path.open()))
    assert len({len(row) for row in rows}) == 1
    independent = {(row['material_index'], row['environment']): float(row['y_pred'])
                   for row in rows if row['variant'] == 'independent'}
    assert independent == {('10', '4'): 1.5, ('11', '4'): 2.5}
    conditioned = {(row['material_index'], row['environment']): float(row['y_pred'])
                   for row in rows if row['variant'] == 'conditioned'}
    assert conditioned[('10', '5')] == 4.0 and conditioned[('11', '1')] == 5.0


def test_selection_is_stored_per_fold_not_nested_by_fold(tmp_path, monkeypatch):
    config = json.loads(json.dumps(CONFIG))
    config['splits'] = str(tmp_path / 'splits.json')
    Path(config['splits']).write_text(json.dumps({'folds': [toy_fold()]}))
    monkeypatch.setattr(runner, 'load', lambda path, sha: toy_data())
    monkeypatch.setattr(runner, 'environment_module', stub_module)
    monkeypatch.setattr(runner, 'reuse_report', lambda cfg: ({}, {}))
    monkeypatch.setattr(runner, 'load_shared', lambda cfg: {})
    monkeypatch.setattr(runner, 'verify_checkpoint_reuse', lambda cfg, saved, device: {'ok': True})
    monkeypatch.setattr(runner, 'verify_versions', lambda cfg: {})
    seen = {}

    def fake_summary(config_, records_, saved_, source_, selections_, folds_):
        seen['selections'] = selections_
        return {'synthetic': True}

    monkeypatch.setattr(runner, 'summarise', fake_summary)
    shared = tmp_path / 'shared'
    shared.mkdir()
    (shared / 'predictions.csv').write_text('variant,fold\n')
    config['shared_source'] = str(shared)
    args = SimpleNamespace(out=tmp_path / 'out', selection=None, config=Path(config['splits']))
    runner.run(config, args, Path('scripts/run_background_nested.py'))
    assert set(seen['selections']) == {'0'}, 'selection is keyed by the string fold'
    assert 'independent' in seen['selections']['0'] and 'conditioned' in seen['selections']['0']
    assert 0 not in seen['selections']['0'], 'no nested integer-fold key'


def test_a_summary_failure_revokes_completion(tmp_path, monkeypatch):
    config = json.loads(json.dumps(CONFIG))
    config['splits'] = str(tmp_path / 'splits.json')
    Path(config['splits']).write_text(json.dumps({'folds': [toy_fold()]}))
    monkeypatch.setattr(runner, 'load', lambda path, sha: toy_data())
    monkeypatch.setattr(runner, 'environment_module', stub_module)
    monkeypatch.setattr(runner, 'reuse_report', lambda cfg: ({}, {}))
    monkeypatch.setattr(runner, 'load_shared', lambda cfg: {})
    monkeypatch.setattr(runner, 'verify_checkpoint_reuse', lambda cfg, saved, device: {'ok': True})
    monkeypatch.setattr(runner, 'verify_versions', lambda cfg: {})
    monkeypatch.setattr(runner, 'summarise', lambda *arguments: (_ for _ in ()).throw(
        RuntimeError('synthetic summary failure')))
    shared = tmp_path / 'shared'
    shared.mkdir()
    (shared / 'predictions.csv').write_text('variant,fold\n')
    config['shared_source'] = str(shared)
    args = SimpleNamespace(out=tmp_path / 'out', selection=None, config=Path(config['splits']))
    runner.run(config, args, Path('scripts/run_background_nested.py'))
    status = json.loads((args.out / 'execution_status.json').read_text())
    assert status['status'] == 'failed', 'a summary failure must not leave the run complete'
    failures = json.loads((args.out / 'failures.json').read_text())
    assert failures and failures[-1]['stage'] == 'summary'
    assert (args.out / 'trajectories.json').exists(), 'training artifacts are kept'
