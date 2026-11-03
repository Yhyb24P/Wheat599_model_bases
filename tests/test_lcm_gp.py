"""Checks for the defining locus-complement calculation and ridge limit."""

import torch
import json
from types import SimpleNamespace

from wheat599.lcm_comparators import AdditiveMLP, ContextModulation
from wheat599.lcm_gp import (LCMNetwork, VARIANTS, configurations, fit_model, prepare,
                             ridge_reference, run, selected_records)


def test_parameter_budget_and_comparator_shapes():
    model = LCMNetwork(1279, 4, rank=8, variant='full')
    assert sum(p.numel() for p in model.parameters()) == 26160
    z = torch.randn(2, 1279)
    for cls in (AdditiveMLP, ContextModulation):
        comparator = cls(1279)
        assert 25000 <= sum(p.numel() for p in comparator.parameters()) <= 27000
        output = comparator(z)
        assert output.shape == (2, 4)
        output.square().mean().backward()


def test_variants_predict_and_differentiate():
    torch.manual_seed(1)
    z = torch.randn(3, 7, dtype=torch.float64)
    for variant in VARIANTS:
        model = LCMNetwork(7, 2, rank=3, variant=variant).double().initialize()
        output = model(z)
        assert output.shape == (3, 2)
        assert torch.isfinite(output).all()
        output.square().mean().backward()


def test_complement_excludes_own_marker():
    torch.manual_seed(2)
    model = LCMNetwork(7, 2, rank=3).double().initialize()
    z = torch.randn(1, 7, dtype=torch.float64, requires_grad=True)
    aggregate = (z @ model.background) * model.scale
    j = 4
    context = aggregate - z[:, j:j + 1] * model.background[j] * model.scale
    gradient = torch.autograd.grad(model.network(context).sum(), z)[0]
    assert gradient[0, j].abs() < 1e-12
    assert gradient[0, :j].abs().max() > 1e-8


def test_linear_response_matches_distinct_pair_formula():
    torch.manual_seed(3)
    p, rank, q = 6, 3, 2
    model = LCMNetwork(p, q, rank=rank, variant='linear_background').double().initialize()
    z = torch.randn(5, p, dtype=torch.float64)
    expected = z @ model.coefficients + model.bias
    pair_sum = torch.zeros_like(expected)
    for j in range(p):
        for k in range(j + 1, p):
            uj = model.modulation[j] * model.readout
            uk = model.modulation[k] * model.readout
            coefficient = (uj @ model.linear_map.weight @ model.background[k]
                           + uk @ model.linear_map.weight @ model.background[j])
            pair_sum = pair_sum + z[:, j:j + 1] * z[:, k:k + 1] * coefficient / p
    torch.testing.assert_close(model(z), expected + pair_sum, rtol=1e-10, atol=1e-10)


def test_additive_ridge_reference_minimizes_same_objective():
    torch.manual_seed(4)
    z = torch.randn(18, 7, dtype=torch.float64)
    z = z - z.mean(0)
    y = torch.randn(18, 3, dtype=torch.float64)
    lam = 0.07
    beta = ridge_reference(z, y, lam).detach().requires_grad_()
    intercept = y.mean(0).detach().requires_grad_()
    loss = (z @ beta + intercept - y).square().mean() + lam * beta.square().sum()
    for grad in torch.autograd.grad(loss, (beta, intercept)):
        assert grad.abs().max() < 1e-11


def test_additive_training_reaches_same_ridge_objective():
    torch.manual_seed(11)
    x = torch.randn(24, 7, dtype=torch.float64).numpy()
    y = torch.randn(24, 3, dtype=torch.float64).numpy()
    fit = list(range(20))
    prepared = prepare(x, y, fit, list(range(20, 24)), torch.device('cpu'), dtype=torch.float64)
    parameters = {'rank': 3, 'lambda_A': 0.05, 'lambda_S': 0., 'lambda_N': 0.}
    cfg = {'environments': 3, 'learning_rate': 0.05, 'max_epochs': 600}
    model, _ = fit_model(prepared, cfg, 'additive_only', parameters, seed=9)
    z, target = prepared['z_fit'], prepared['y_fit']
    optimal = ridge_reference(z, target, parameters['lambda_A'])
    objective = lambda beta, pred: (pred - target).square().mean() + parameters['lambda_A'] * beta.square().sum()
    observed = objective(model.coefficients, model(z))
    reference = objective(optimal, z @ optimal + target.mean(0))
    torch.testing.assert_close(observed, reference, rtol=1e-6, atol=1e-7)


def test_modulation_is_optimized():
    torch.manual_seed(5)
    model = LCMNetwork(7, 2, rank=3).double().initialize()
    optimizer = torch.optim.Adam(model.parameter_groups(0.01, 0.01), lr=0.01)
    before = model.modulation.detach().clone()
    prediction = model(torch.randn(8, 7, dtype=torch.float64))
    (prediction - 1).square().mean().backward()
    optimizer.step()
    assert not torch.equal(before, model.modulation.detach())


def test_group_proximal_produces_exact_zero_rows():
    model = LCMNetwork(3, 2, rank=2).double()
    with torch.no_grad():
        model.modulation.copy_(torch.tensor([[0.03, 0.04], [0.3, 0.4], [0., 0.]], dtype=torch.float64))
    model.proximal(0.1)
    torch.testing.assert_close(model.modulation[0], torch.zeros(2, dtype=torch.float64))
    torch.testing.assert_close(model.modulation[1], torch.tensor([0.24, 0.32], dtype=torch.float64))
    assert model.zero_rows() == 2


def test_preprocessing_uses_only_training_materials():
    torch.manual_seed(7)
    x = torch.randn(9, 5, dtype=torch.float64).numpy()
    y = torch.randn(9, 2, dtype=torch.float64).numpy()
    fit, target = [0, 1, 2, 3, 4, 5], [6, 7, 8]
    first = prepare(x, y, fit, target, torch.device('cpu'))
    x[target] += 100
    y[target] -= 100
    second = prepare(x, y, fit, target, torch.device('cpu'))
    for key in ('z_fit', 'y_fit', 'mean', 'scale'):
        torch.testing.assert_close(first[key], second[key])


def test_selection_freezes_zero_epoch_and_includes_comparators():
    cfg = {'variants': ['full'], 'comparators': ['additive_mlp'], 'tie_tolerance': 1e-12,
           'max_epochs': 300}
    grid = [{'variant': 'full'}, {'variant': 'additive_mlp'}]
    chosen = selected_records({0: [0.4, 0.5], 1: [0.3, 0.4]},
                              {0: [0, 0, 20], 1: [10, 20, 30]}, cfg, grid)
    assert chosen['full']['epochs'] == 0
    assert chosen['additive_mlp']['epochs'] == 20


def test_comparator_grid_includes_network_regularization():
    cfg = {'variants': ['full'], 'comparators': ['additive_mlp', 'context_modulation'],
           'ranks': [8], 'lambda_additive': [0.001], 'lambda_sparse': [0., 1.],
           'lambda_network': [0., 0.01]}
    grid = configurations(cfg)
    for name in cfg['comparators']:
        entries = [item for item in grid if item['variant'] == name]
        assert {item['lambda_N'] for item in entries} == {0., 0.01}
        assert {item['lambda_S'] for item in entries} == {0.}


def test_comparators_use_shared_ridge_initialization_and_zero_epoch():
    torch.manual_seed(13)
    x = torch.randn(14, 7, dtype=torch.float64).numpy()
    y = torch.randn(14, 2, dtype=torch.float64).numpy()
    prepared = prepare(x, y, list(range(10)), list(range(10, 14)), torch.device('cpu'),
                       dtype=torch.float64)
    cfg = {'environments': 2, 'learning_rate': 0.05, 'max_epochs': 5}
    pars = {'rank': 3, 'lambda_A': 0.05, 'lambda_S': 0., 'lambda_N': 0.01}
    reference = ridge_reference(prepared['z_fit'], prepared['y_fit'], pars['lambda_A'])
    for name in ('additive_mlp', 'context_modulation'):
        model, record = fit_model(prepared, cfg, name, pars, seed=2, epochs=0)
        assert record['epochs'] == 0
        torch.testing.assert_close(model(prepared['z_fit']), prepared['z_fit'] @ reference,
                                   atol=1e-11, rtol=1e-10)


def test_dense_and_unpenalized_full_are_identical():
    torch.manual_seed(19)
    x = torch.randn(14, 7, dtype=torch.float64).numpy()
    y = torch.randn(14, 2, dtype=torch.float64).numpy()
    prepared = prepare(x, y, list(range(10)), list(range(10, 14)), torch.device('cpu'),
                       dtype=torch.float64)
    cfg = {'environments': 2, 'learning_rate': 0.01, 'max_epochs': 3}
    pars = {'rank': 3, 'lambda_A': 0.05, 'lambda_S': 0., 'lambda_N': 0.01}
    full, _ = fit_model(prepared, cfg, 'full', pars, seed=4)
    dense, _ = fit_model(prepared, cfg, 'dense_modulation', pars, seed=4)
    torch.testing.assert_close(full(prepared['z_target']), dense(prepared['z_target']), rtol=0, atol=0)


def test_nested_selection_then_refit_uses_frozen_epochs(tmp_path, monkeypatch):
    torch.manual_seed(17)
    data = SimpleNamespace(X=torch.randn(24, 7).numpy(), Y=torch.randn(24, 4).numpy(),
                           environments=('1', '2', '4', '5'), n=24)
    folds = [{'fold': 0, 'dev_idx': list(range(16)), 'test_idx': list(range(16, 24)),
              'inner': [{'train_idx': [i for i in range(16) if i not in range(k, k + 4)],
                         'val_idx': list(range(k, k + 4))} for k in (0, 4, 8)]}]
    splits = tmp_path / 'splits.json'
    splits.write_text(json.dumps({'folds': folds}))
    cfg = {'data': 'unused', 'data_sha256': 'unused', 'splits': str(splits), 'seed': 42,
           'environments': 4, 'gpu_memory_fraction': 0.2, 'tie_tolerance': 1e-12,
           'learning_rate': 0.003, 'max_epochs': 1, 'evaluate_every': 1,
           'min_improvement': 1e-6, 'patience': 50, 'ranks': [3], 'lambda_additive': [0.1],
           'lambda_sparse': [0.], 'lambda_network': [0., 0.01],
           'variants': ['full', 'additive_only'], 'comparators': ['additive_mlp', 'context_modulation']}
    monkeypatch.setattr('wheat599.lcm_gp.load', lambda *args: data)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    selection_dir, final_dir = tmp_path / 'selection', tmp_path / 'final'
    runner = tmp_path / 'runner.py'
    runner.write_text('# runner\n')
    run(cfg, SimpleNamespace(out=selection_dir, selection=None), runner)
    selected = json.loads((selection_dir / 'selection.json').read_text())
    assert set(selected['selected'][0]['chosen']) == set(cfg['variants'] + cfg['comparators'])
    run(cfg, SimpleNamespace(out=final_dir, selection=selection_dir / 'selection.json'), runner)
    summary = json.loads((final_dir / 'summary.json').read_text())
    assert set(summary['variants']) == set(cfg['variants'] + cfg['comparators'])
    assert summary['n_failures'] == 0
