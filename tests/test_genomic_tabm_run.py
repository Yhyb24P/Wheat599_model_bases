"""Tiny tests for the TabM development runner. A stub model stands in for cn's model."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from wheat599 import genomic_tabm_run as runner


class StubModel(nn.Module):
    """Mirrors GenomicTabM's interface: fit-row buffers, (n, members, q) output, ridge penalty."""

    def __init__(self, train_x, train_y, *, variant='raw', beta=None, bias=None, beta_ridge=0.0,
                 dtype=torch.float32, n_blocks=1, d_block=4, dropout=0.0, k=3, groups=2, rank=1):
        super().__init__()
        x = torch.as_tensor(train_x, dtype=dtype)
        y = torch.as_tensor(train_y, dtype=dtype)
        self.variant, self.k, self.members = variant, k, k
        std = x.std(dim=0, unbiased=False)
        self.register_buffer('x_mean', x.mean(dim=0))
        self.register_buffer('x_std', torch.where(std > 0, std, torch.ones_like(std)))
        self.register_buffer('x_constant', std <= 0)
        self.register_buffer('y_mean', y.mean(dim=0))
        self.register_buffer('y_std', y.std(dim=0, unbiased=False).clamp_min(1e-12))
        self.head = nn.Linear(x.shape[1], y.shape[1], bias=True, dtype=dtype)
        with torch.no_grad():
            if variant == 'grouped_additive':
                # Mirrors the official zero-initialised output layer: the ridge branch alone.
                self.head.weight.zero_()
            else:
                self.head.weight.normal_(0.0, 0.01)
            self.head.bias.zero_()
        self.beta_ridge = float(beta_ridge)
        if variant == 'grouped_additive':
            if beta is None:
                raise ValueError('grouped_additive requires beta')
            self.beta = nn.Parameter(torch.as_tensor(beta, dtype=dtype).clone())
            self.bias = nn.Parameter(torch.zeros(y.shape[1], dtype=dtype)
                                    if bias is None else torch.as_tensor(bias, dtype=dtype).clone())

    def standardize_x(self, x):
        x = torch.as_tensor(x, dtype=self.x_mean.dtype)
        block = (x - self.x_mean) / self.x_std
        return torch.where(self.x_constant.unsqueeze(0), torch.zeros_like(block), block)

    def standardize_y(self, y):
        return (torch.as_tensor(y, dtype=self.y_mean.dtype) - self.y_mean) / self.y_std

    def inverse_y(self, prediction):
        return prediction * self.y_std + self.y_mean

    def forward(self, x):
        z = self.standardize_x(x)
        base = self.head(z)
        if self.variant == 'grouped_additive':
            base = base + z @ self.beta + self.bias
        offsets = (torch.arange(self.members, dtype=z.dtype) - 0.5 * (self.members - 1)) * 0.01
        return base.unsqueeze(1) + offsets.reshape(1, -1, 1)

    def parameter_count(self):
        return int(sum(parameter.numel() for parameter in self.parameters()))

    def ridge_penalty(self):
        return torch.zeros((), dtype=self.x_mean.dtype) if self.variant == 'raw' \
            else self.beta_ridge * self.beta.square().sum()

    def group_metadata(self):
        return {'variant': self.variant, 'groups_actual': 1, 'markers': self.head.in_features}


CONFIG = {
    'device': 'cpu', 'cpu_threads': 1, 'gpu_memory_fraction': 0.5,
    'seeds': [7], 'variants': ['raw', 'grouped_additive'], 'outer_folds': [0],
    'architecture': {'n_blocks': 1, 'd_block': 4, 'dropout': 0.0, 'k': 3, 'groups': 2, 'rank': 1},
    'training': {'max_epochs': 2, 'patience': 1, 'batch_size': 8, 'eval_batch_size': 32,
                 'learning_rates': [0.01], 'weight_decays': [0.0], 'gradient_clip': 1.0,
                 'ridge_grid': [1.0, 10.0], 'ridge_folds': 3, 'ridge_seed': 719,
                 'tie_tolerance': 1e-12},
    'efficiency': {'batch_size': 8, 'warmup': 1, 'repeats': 2, 'include_preprocessing': True},
    'data': 'unused', 'data_sha256': '0' * 64, 'splits': 'unused',
    'reference': 'results/robust_icm_development_20261103/development_comparison.csv',
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


def stub_build(config, variant, seed, train_x, train_y, beta=None, bias=None, device='cpu'):
    torch.manual_seed(int(seed))
    model = StubModel(train_x, train_y, variant=variant,
                      beta=None if beta is None else beta['beta'], bias=bias,
                      beta_ridge=0.0 if beta is None else float(beta['ridge_lambda'])
                      / (len(train_x) * train_y.shape[1]),
                      dtype=torch.float64, **config['architecture'])
    return model.double().to(device)


def test_standardisation_uses_fit_rows_only():
    data = toy_data()
    x = torch.as_tensor(data.X)
    fit = np.arange(24)
    first = runner.standardise(x[fit], x)
    disturbed = x.clone()
    disturbed[24:] = 99.0
    second = runner.standardise(disturbed[fit], disturbed)
    assert torch.equal(first[fit], second[fit])
    assert not torch.equal(first[24:], second[24:])
    assert torch.allclose(first[fit].mean(0), torch.zeros(8, dtype=torch.float64), atol=1e-12)
    constant = torch.ones(36, 1, dtype=torch.float64)
    assert torch.equal(runner.standardise(constant[:24], constant), torch.zeros_like(constant))


def test_ridge_cv_is_deterministic_inside_the_grid_and_scale_free():
    data = toy_data()
    x = torch.as_tensor(data.X, dtype=torch.float64)
    y = torch.as_tensor(data.Y, dtype=torch.float64)
    folds = runner.ridge_folds(24, 3, 719)
    value, totals = runner.choose_ridge(x[:24], y[:24], CONFIG['training']['ridge_grid'], folds)
    again, _ = runner.choose_ridge(x[:24], y[:24], CONFIG['training']['ridge_grid'], folds)
    assert value in CONFIG['training']['ridge_grid'] and again == value
    assert set(totals) == {1.0, 10.0} and all(totals[k] >= 0 for k in totals)
    scaled = y.clone()
    scaled[:24] *= 1000.0  # each fold re-standardises its own phenotypes
    value_scaled, totals_scaled = runner.choose_ridge(x[:24], scaled[:24],
                                                      CONFIG['training']['ridge_grid'], folds)
    assert value_scaled == value
    assert totals_scaled[value_scaled] == pytest.approx(totals[value], rel=1e-9)


def test_trajectory_never_reads_outer_test_rows(monkeypatch):
    data = toy_data()
    data.Y[30:] = np.nan  # reading a test row would poison the loss
    fold = {'fold': 0, 'dev_idx': list(range(30)), 'test_idx': list(range(30, 36)),
            'inner': [{'train_idx': list(range(24)), 'val_idx': list(range(24, 30))}]}
    monkeypatch.setattr(runner, 'build_model', stub_build)
    trajectory = runner.Trajectory(0, 0, 7, 'raw', 0, 0.01, 0.0)
    record = runner.train_trajectory(data, fold, 0, trajectory, CONFIG, torch.device('cpu'), {})
    assert np.isfinite(record['metrics']['pcc'])
    assert record['epochs_run'] <= CONFIG['training']['max_epochs']
    assert record['parameters'] > 0 and record['groups']['variant'] == 'raw'
    assert record['inference']['includes_preprocessing'] is True


def test_grouped_variant_initialises_beta_and_excludes_it_from_weight_decay(monkeypatch):
    data = toy_data()
    fold = toy_fold()
    monkeypatch.setattr(runner, 'build_model', stub_build)
    captured = {}
    real_optimizer = torch.optim.AdamW

    def spy(parameters, **kwargs):
        captured['groups'] = parameters
        return real_optimizer(parameters, **kwargs)

    monkeypatch.setattr(torch.optim, 'AdamW', spy)
    cache = {}
    trajectory = runner.Trajectory(0, 0, 7, 'grouped_additive', 0, 0.01, 0.5)
    record = runner.train_trajectory(data, fold, 0, trajectory, CONFIG, torch.device('cpu'), cache)
    groups = captured['groups']
    assert len(groups) == 2 and groups[0]['weight_decay'] == 0.5
    assert groups[1]['weight_decay'] == 0.0 and len(groups[1]['params']) == 1
    beta_parameter = groups[1]['params'][0]
    assert isinstance(beta_parameter, nn.Parameter) and beta_parameter.shape == (8, 2)
    assert (0, 0) in cache and record['ridge_lambda'] in CONFIG['training']['ridge_grid']
    assert record['groups']['variant'] == 'grouped_additive'
    assert record['state_sha256'] if 'state_sha256' in record else 'state' in record


def test_select_common_breaks_ties_by_configuration_order():
    config = json.loads(json.dumps(CONFIG))
    config['training']['learning_rates'] = [0.01, 0.02]
    config['seeds'] = [7, 8]
    config['variants'] = ['raw']
    records = []
    for index in range(2):
        for inner in range(3):
            for seed in (7, 8):
                records.append({'trajectory': {'fold': 0, 'inner': inner, 'seed': seed,
                                               'variant': 'raw', 'config_index': index},
                                'metrics': {'pcc': 0.5, 'rmse': 1.0, 'mae': 0.8}})
    chosen = runner.select_common(records, config)[0]['raw']
    assert chosen['config_index'] == 0
    del records[-1]
    with pytest.raises(ValueError, match='Incomplete trajectory grid'):
        runner.select_common(records, config)


def test_tiny_run_writes_records_and_guards_its_inputs(tmp_path, monkeypatch):
    config = json.loads(json.dumps(CONFIG))
    config['splits'] = str(tmp_path / 'splits.json')
    Path(config['splits']).write_text(json.dumps({'folds': [toy_fold()]}))
    monkeypatch.setattr(runner, 'load', lambda path, sha: toy_data())
    monkeypatch.setattr(runner, 'build_model', stub_build)
    args = SimpleNamespace(out=tmp_path / 'out', selection=None, config=Path(config['splits']))
    runner.run(config, args, Path('scripts/run_background_nested.py'))
    produced = sorted(item.name for item in args.out.iterdir())
    assert 'provenance.json' in produced and 'trajectories.json' in produced
    assert 'selection.json' in produced and 'timings.json' in produced
    records = json.loads((args.out / 'trajectories.json').read_text())
    assert len(records) == 2 * 3, records  # both variants x three inner splits x one seed/config
    assert json.loads((args.out / 'failures.json').read_text()) == []
    selection = json.loads((args.out / 'selection.json').read_text())
    assert set(selection['0']) == {'raw', 'grouped_additive'}
    provenance = json.loads((args.out / 'provenance.json').read_text())
    assert provenance['budget'] == 6 and 'reference' in provenance

    missing = json.loads(json.dumps(config))
    missing['outer_folds'] = [0, 1]
    with pytest.raises(ValueError, match='missing from the split manifest'):
        runner.run(missing, SimpleNamespace(out=tmp_path / 'bad', selection=None,
                                            config=Path(config['splits'])),
                   Path('scripts/run_background_nested.py'))
    with pytest.raises(ValueError, match='no selection-input'):
        runner.run(config, SimpleNamespace(out=tmp_path / 'bad2', selection=Path('x.json'),
                                           config=Path(config['splits'])),
                   Path('scripts/run_background_nested.py'))


def test_epoch_zero_is_recorded_as_the_initialisation(monkeypatch):
    data = toy_data()
    fold = toy_fold()
    config = json.loads(json.dumps(CONFIG))
    config['training']['max_epochs'] = 0  # evaluation only; the initialisation is the checkpoint
    monkeypatch.setattr(runner, 'build_model', stub_build)
    trajectory = runner.Trajectory(0, 0, 7, 'grouped_additive', 0, 0.01, 0.0)
    record = runner.train_trajectory(data, fold, 0, trajectory, config, torch.device('cpu'), {})
    assert record['epochs_run'] == 0 and record['best_epoch'] == 0
    assert record['curve'][0]['epoch'] == 0 and record['curve'][0]['train_mse'] is None
    assert np.isfinite(record['curve'][0]['validation_pcc']) + 1


def test_grouped_initial_prediction_equals_the_ridge_solution_in_standardised_space(monkeypatch):
    data = toy_data()
    fold = toy_fold()
    monkeypatch.setattr(runner, 'build_model', stub_build)
    trajectory = runner.Trajectory(0, 0, 7, 'grouped_additive', 0, 0.01, 0.0)
    fit = np.asarray(fold['inner'][0]['train_idx'])
    x = torch.as_tensor(data.X, dtype=torch.float64)
    y = torch.as_tensor(data.Y, dtype=torch.float64)
    value, _ = runner.choose_ridge(x[fit], y[fit], CONFIG['training']['ridge_grid'],
                                   runner.ridge_folds(len(fit), 3, 719))
    beta, bias = runner.ridge_init(x[fit], y[fit], value)
    mean, spread = y[fit].mean(0), y[fit].std(0, unbiased=False).clamp_min(1e-12)
    scaled = (y[fit] - mean) / spread
    z = runner.standardise(x[fit], x[fit])
    expected = z @ beta + bias
    model = runner.build_model(CONFIG, 'grouped_additive', 7, x[fit], y[fit],
                               beta={'beta': beta, 'ridge_lambda': value}, bias=bias)
    with torch.no_grad():
        initial = runner.ensemble(model(x[fit])).mean(1)
    # The linear branch reproduces the ridge fit; the zero-initialised backbone adds nothing.
    assert torch.allclose(initial, expected, rtol=0, atol=1e-10)


def test_dual_ridge_matches_the_primal_solution_with_an_unpenalised_intercept():
    generator = torch.Generator().manual_seed(3)
    for rows, markers in ((30, 6), (6, 30)):  # both sides of p > n
        x = torch.randn(rows, markers, generator=generator, dtype=torch.float64)
        y = torch.randn(rows, 2, generator=generator, dtype=torch.float64)
        ridge = 2.5
        beta, bias = runner.ridge_solution(x, y, ridge)
        design = torch.cat((torch.ones(rows, 1, dtype=torch.float64), x), dim=1)
        penalty = ridge * torch.eye(markers + 1, dtype=torch.float64)
        penalty[0, 0] = 0.0
        primal = torch.linalg.solve(design.T @ design + penalty, design.T @ y)
        assert torch.allclose(beta, primal[1:], rtol=1e-10, atol=1e-12)
        assert torch.allclose(bias, primal[0], rtol=1e-10, atol=1e-12)
        prediction = x @ beta + bias
        assert torch.allclose(prediction, design @ primal, rtol=1e-10, atol=1e-12)


def test_float32_configuration_reaches_the_model_while_the_ridge_stays_float64(monkeypatch):
    data = toy_data()
    fold = toy_fold()
    config = json.loads(json.dumps(CONFIG))
    config['dtype'] = 'float32'
    captured = {}

    def capture(config, variant, seed, train_x, train_y, beta=None, bias=None, device='cpu'):
        captured['train_x_dtype'] = train_x.dtype
        captured['beta'] = None if beta is None else beta['beta']
        return stub_build(config, variant, seed, train_x, train_y, beta=beta, bias=bias,
                          device=device)

    monkeypatch.setattr(runner, 'build_model', capture)
    cache = {}
    trajectory = runner.Trajectory(0, 0, 7, 'grouped_additive', 0, 0.01, 0.0)
    record = runner.train_trajectory(data, fold, 0, trajectory, config, torch.device('cpu'), cache)
    assert captured['train_x_dtype'] == torch.float32
    assert cache[(0, 0)]['beta'].dtype == torch.float64, 'the ridge solve stays float64 on CPU'
    assert record['ridge_cache_hit'] is False and record['ridge_cv_seconds'] > 0
    second = runner.train_trajectory(data, fold, 0,
                                     runner.Trajectory(0, 0, 8, 'grouped_additive', 0, 0.01, 0.0),
                                     config, torch.device('cpu'), cache)
    assert second['ridge_cache_hit'] is True and second['ridge_cv_seconds'] == record['ridge_cv_seconds']


def test_per_environment_metrics_and_execution_status_are_written(tmp_path, monkeypatch):
    config = json.loads(json.dumps(CONFIG))
    config['splits'] = str(tmp_path / 'splits.json')
    Path(config['splits']).write_text(json.dumps({'folds': [toy_fold()]}))
    monkeypatch.setattr(runner, 'load', lambda path, sha: toy_data())
    monkeypatch.setattr(runner, 'build_model', stub_build)
    args = SimpleNamespace(out=tmp_path / 'out', selection=None, config=Path(config['splits']))
    runner.run(config, args, Path('scripts/run_background_nested.py'))
    records = json.loads((args.out / 'trajectories.json').read_text())
    for record in records:
        assert set(record['per_environment']) == {'1', '2'}
        for values in record['per_environment'].values():
            assert set(values) == set(runner.METRICS)
    status = json.loads((args.out / 'execution_status.json').read_text())
    assert status['status'] == 'complete'
    assert status['completed_trajectories'] == status['expected'] == len(records) == 6
    assert status['folds_complete'] == [0]
    provenance = json.loads((args.out / 'provenance.json').read_text())
    assert 'mean_bke' in provenance['loss']
    assert set(provenance['model_source']) == {'tabm', 'rtdl_num_embeddings'}


def test_a_failed_trajectory_keeps_its_error_and_does_not_certify_completion(tmp_path, monkeypatch):
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
            raise FloatingPointError('synthetic trajectory failure')
        return real(*arguments, **keywords)

    monkeypatch.setattr(runner, 'train_trajectory', failing)
    args = SimpleNamespace(out=tmp_path / 'out', selection=None, config=Path(config['splits']))
    with pytest.raises(FloatingPointError, match='synthetic trajectory failure'):
        runner.run(config, args, Path('scripts/run_background_nested.py'))
    status = json.loads((args.out / 'execution_status.json').read_text())
    assert status['status'] == 'failed' and status['completed_trajectories'] == 1
    assert status['folds_complete'] == []
    failures = json.loads((args.out / 'failures.json').read_text())
    assert failures and 'synthetic trajectory failure' in failures[0]['reason']
    selection = json.loads((args.out / 'selection.json').read_text())
    assert selection == {}, 'an incomplete fold must not be certified'


def test_pinned_versions_are_verified_before_any_output(tmp_path, monkeypatch):
    config = json.loads(json.dumps(CONFIG))
    config['versions'] = {'definitely_absent_package_xyz': '1.0'}
    config['splits'] = str(tmp_path / 'splits.json')
    Path(config['splits']).write_text(json.dumps({'folds': [toy_fold()]}))
    monkeypatch.setattr(runner, 'load', lambda path, sha: toy_data())
    args = SimpleNamespace(out=tmp_path / 'out', selection=None, config=Path(config['splits']))
    with pytest.raises(RuntimeError, match='Pinned package versions'):
        runner.run(config, args, Path('scripts/run_background_nested.py'))
    assert not (tmp_path / 'out').exists()
