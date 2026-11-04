"""Tiny pass-through and boundary tests for the joint round-two pilot entry.

Sizes are deliberately far below the frozen budget; these tests check structure and
information boundaries, not performance.
"""
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from wheat599 import joint_training as protocol
from wheat599.mixed_tasks import sample_mixed_task

TINY = {
    'model': 'joint_inference', 'device': 'cpu', 'cpu_threads': 1, 'gpu_memory_fraction': 0.5,
    'seeds': [42, 2026], 'variants': ['joint', 'independent'],
    'lambda_a_grid': [0.3, 3.0], 'context_cv_folds': 3, 'lambda_draw_seed': 0,
    'network': {'environments': 2, 'width': 8, 'rounds': 1, 'add_per_round': 2,
                'max_candidates': 4, 'max_variance': 100.0},
    'tasks': {'context_sizes': [12], 'marker_counts': [6], 'query_size': 4, 'environments': 2,
              'designs': ['independent'], 'backgrounds': ['dense', 'sparse'],
              'gammas': [0.0, 1.0], 'additive_scales': [1.0], 'noise_scales': [1.0],
              'pair_counts': [1, 2], 'active_counts': [2], 'block_size': 3},
    'training': {'steps': 2, 'learning_rate': 0.01, 'weight_decay': 0.0001, 'gradient_clip': 1.0,
                 'checkpoints': [0, 1, 2], 'role_base': 1_000_000_000_000},
    'selection': {'role_base': 2_000_000_000_000, 'bases_per_background': 1, 'query_size': 4},
    'gate': {'train_base': 3_000_000_000_000, 'train_tasks': 4, 'selection_base': 4_000_000_000_000,
             'selection_bases_per_background': 1, 'query_size': 4, 'width': 4, 'epochs': 2, 'seed': 4242,
             'learning_rate': 0.01, 'checkpoints': [0, 2]},
    'confirmation': {'role_base': 5_000_000_000_000, 'bases_per_background': 1, 'query_size': 4,
                     'context_sizes': [12], 'marker_counts': [6]},
    'statistical': {'ratios': [1.0, 0.5], 'lambda_e': 0.001, 'relaxed_variance': 100.0,
                    'same_candidate_variance': 1.0,
                    'solver': {'block_size': 8, 'cache_phi': True, 'max_active': 8,
                               'max_iter': 200, 'max_outer': 8, 'kkt_tolerance': 1e-6,
                               'dual_gap_tolerance': 1e-5}},
}


def test_role_ids_cannot_collide_across_panels_and_training():
    sets = {role: (protocol.panel_ids(TINY, role) if role != 'training'
                   else protocol.training_ids(TINY)) for role in protocol.ROLES}
    assert all(values for values in sets.values())
    names = list(sets)
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            assert not sets[first] & sets[second], f'{first} overlaps {second}'
    # A model seed must never move the stream: ids depend on the enumerated run index only.
    changed = {**TINY, 'seeds': [4990, 8990]}
    assert protocol.training_ids(changed) == protocol.training_ids(TINY)
    assert len(protocol.build_panel(TINY, 'selection')) == 2 * 1 * 2  # backgrounds x bases x gammas
    assert len(protocol.build_panel(TINY, 'gate_train')) == TINY['gate']['train_tasks']
    assert len(protocol.build_panel(TINY, 'confirmation')) == 2 * 1 * 2


def test_additive_cv_reads_only_noisy_context_and_never_the_query_rows():
    task = sample_mixed_task(TINY['tasks'], protocol.task_seed(TINY, 'selection', 0),
                            background='dense', gamma=1.0)
    value, losses = protocol.additive_cv(task, TINY)
    blinded = replace(task, query_x=torch.zeros_like(task.query_x),
                      query_signal=torch.zeros_like(task.query_signal),
                      query_y=torch.zeros_like(task.query_y))
    assert protocol.additive_cv(blinded, TINY) == (value, losses)
    assert value in TINY['lambda_a_grid']
    assert len(losses) == len(TINY['lambda_a_grid']) and all(np.isfinite(list(losses.values())))
    rows = len(task.context_x)
    assert [len(fold) for fold in protocol.folds_of(rows, TINY['context_cv_folds'])] == [4, 4, 4]


def test_gate_candidates_include_fixed_policies_and_scale_only_from_gate_train():
    train = [{'run': 'joint_42', 'features': torch.tensor([1.0, 2.0]), 'joint_mse': 0.9,
              'additive_mse': 0.4},
             {'run': 'joint_42', 'features': torch.tensor([3.0, 1.0]), 'joint_mse': 1.1,
              'additive_mse': 0.5}]
    selection = [{'run': 'joint_42', 'features': torch.tensor([2.0, 3.0]), 'joint_mse': 1.0,
                  'additive_mse': 0.3}]
    gate, choice = protocol.fit_gate(TINY, train, selection)
    kinds = [item['kind'] for item in choice['candidates']]
    assert 'always_off' in kinds and 'always_on' in kinds
    assert any(kind.startswith('learned_') for kind in kinds)
    assert np.allclose(gate['center'], np.mean([[1.0, 2.0], [3.0, 1.0]], axis=0))
    moved = [dict(row, joint_mse=row['joint_mse'] + 10.0, additive_mse=row['additive_mse'] + 10.0)
             for row in selection]
    same_gate, _ = protocol.fit_gate(TINY, train, moved)
    assert torch.equal(same_gate['center'], gate['center']) and torch.equal(same_gate['scale'],
                                                                           gate['scale'])
    always = protocol.apply_gate(gate, {'kind': 'always_off', 'epoch': None},
                                {'features': selection[0]['features'], 'joint_mse': 1.0,
                                 'additive_mse': 0.3})
    assert always == (0.3, 0.0)


def test_ties_prefer_always_off_and_the_reference_stops_on_a_failed_certificate():
    train = [{'run': 'joint_42', 'features': torch.tensor([0.0]), 'joint_mse': 1.0,
              'additive_mse': 1.0}]
    selection = [{'run': 'joint_42', 'features': torch.tensor([0.0]), 'joint_mse': 1.0,
                  'additive_mse': 1.0}]
    _, choice = protocol.fit_gate(TINY, train, selection)
    assert choice['kind'] == 'always_off' and choice['epoch'] is None

    task = sample_mixed_task(TINY['tasks'], protocol.task_seed(TINY, 'confirmation', 0),
                             background='dense', gamma=1.0)
    data, _ = protocol.context(task)
    evidence = protocol.Evidence(data.x, data.y, lambda_a=1.0)
    outcome = protocol.solver_result(TINY, evidence, data.x, data.y, data.query, 1.0,
                                     TINY['statistical']['ratios'][-1], torch.device('cpu'))
    assert np.isfinite(outcome['original']).all() and outcome['pairs'] >= 0
    broken = json.loads(json.dumps(TINY, default=str))
    broken['statistical']['solver']['kkt_tolerance'] = 1e-30
    try:
        protocol.solver_result(broken, evidence, data.x, data.y, data.query, 1.0,
                               broken['statistical']['ratios'][-1], torch.device('cpu'))
        raise AssertionError('a non-optimal reference must raise, not be filtered')
    except RuntimeError as error:
        assert 'certificate failed' in str(error)


def test_tiny_pass_through_writes_selections_before_confirmation(tmp_path):
    config = json.loads(json.dumps(TINY, default=str))
    config['gate']['checkpoints'] = [0, 2]
    path = tmp_path / 'joint_inference.yaml'
    path.write_text(json.dumps(config))
    args = SimpleNamespace(out=tmp_path / 'run', selection=None, config=path)
    protocol.run(config, args, Path('scripts/run_background_nested.py'))
    out = args.out
    produced = sorted(item.name for item in out.iterdir())
    for expected in ('provenance.json', 'training.json', 'gate.json', 'gate.pt',
                     'confirmation_panel.json', 'confirmation.json', 'summary.json', 'timings.json',
                     'failures.json'):
        assert expected in produced, produced
    assert json.loads((out / 'failures.json').read_text()) == []

    runs = json.loads((out / 'training.json').read_text())
    assert len(runs) == len(config['variants']) * len(config['seeds'])
    for run in runs:
        assert 'state' not in run, 'weights belong only in the .pt files'
        steps = [record['step'] for record in run['history']]
        assert steps == config['training']['checkpoints']
        best = min(run['history'], key=lambda record: (round(record['selection_mse'], 12),
                                                      record['step']))
        assert run['selected_step'] == best['step']
        draws = run['training_lambda_draws']
        assert {float(key) for key in draws['counts']} == {float(value) for value in config['lambda_a_grid']}
        assert sum(draws['counts'].values()) == config['training']['steps']
        assert len(run['checkpoint_sha256']) == 64 and Path(out / f"{run['variant']}_{run['model_seed']}.pt").exists()
    # Both variants share one stream: the same run index must draw the same tasks and lambda_A.
    paired = {(run['variant'], run['run_index']): run for run in runs}
    for seed_index in range(len(config['seeds'])):
        joint = paired[('joint', seed_index)]
        independent = paired[('independent', seed_index)]
        assert joint['training_lambda_draws'] == independent['training_lambda_draws']
        assert [record['drawn_background'] for record in joint['history']] == \
            [record['drawn_background'] for record in independent['history']]

    gate = json.loads((out / 'gate.json').read_text())
    assert {'always_off', 'always_on'} <= {item['kind'] for item in gate['selected']['candidates']}
    assert gate['seed'] == 4242 and len(gate['gate_sha256']) == 64
    gate_state = torch.load(out / 'gate.pt', map_location='cpu', weights_only=True)
    assert set(gate_state) == {'center', 'scale', 'states', 'selected', 'features', 'hidden', 'seed'}

    records = json.loads((out / 'confirmation.json').read_text())
    assert len(records) == len(config['tasks']['backgrounds']) * \
        config['confirmation']['bases_per_background'] * len(config['tasks']['gammas'])
    for record in records:
        assert set(record['neural']) == set(record['gated']) == set(record['fixed_readout'])
        assert record['reference']['chosen']['ratio'] in config['statistical']['ratios']
        assert len(record['reference']['grid']) == len(config['statistical']['ratios']) * 2
        assert record['metadata']['context_signal_variance']
        assert record['lambda_a'] in config['lambda_a_grid']  # evaluation uses the shared context CV
    summary = json.loads((out / 'summary.json').read_text())
    assert summary and all('reference' in value for value in summary.values())


def test_pair_index_formula_and_completed_confirmation_records_survive_an_interruption(tmp_path, monkeypatch):
    task = sample_mixed_task(TINY['tasks'], protocol.task_seed(TINY, 'selection', 0),
                            background='dense', gamma=1.0)
    data, _ = protocol.context(task)
    evidence = protocol.Evidence(data.x, data.y, lambda_a=1.0)
    requested = np.array([[0, 1], [1, 2], [2, 5]])
    positions = protocol.pair_indices(evidence, requested)
    assert [tuple(evidence.pairs[int(index)].tolist()) for index in positions] == \
        [tuple(pair) for pair in requested]

    config = json.loads(json.dumps(TINY))
    path = tmp_path / 'joint_inference.yaml'
    path.write_text(json.dumps(config))
    args = SimpleNamespace(out=tmp_path / 'interrupted', selection=None, config=path)
    counter = {'calls': 0}
    real = protocol.reference_cv

    def failing(config_, task_, lambda_a_, device_):
        counter['calls'] += 1
        if counter['calls'] == 2:
            raise RuntimeError('synthetic interruption')
        return real(config_, task_, lambda_a_, device_)

    monkeypatch.setattr(protocol, 'reference_cv', failing)
    with pytest.raises(RuntimeError):
        protocol.run(config, args, Path('scripts/run_background_nested.py'))
    kept = json.loads((args.out / 'confirmation.json').read_text())
    assert len(kept) == 1 and kept[0]['reference']['chosen']['ratio'] in config['statistical']['ratios']
    failures = json.loads((args.out / 'failures.json').read_text())
    assert failures and 'synthetic interruption' in failures[0]['reason']


def test_reference_predictions_live_on_the_pipeline_scale():
    """The interaction reference must score in the same units as the additive fit and the target."""
    task = sample_mixed_task(TINY['tasks'], protocol.task_seed(TINY, 'confirmation', 0),
                            background='dense', gamma=1.0)
    data, target = protocol.context(task)
    lambda_a = 3.0
    evidence = protocol.Evidence(data.x, data.y, lambda_a=lambda_a)
    empty = protocol.solver_result(TINY, evidence, data.x, data.y, data.query, lambda_a, 1.0,
                                   torch.device('cpu'))
    assert empty['pairs'] == 0, 'ratio 1 must leave the support empty'
    ridge = evidence.ridge(data.query)
    assert torch.allclose(empty['original'], ridge, rtol=1e-10, atol=1e-12)
    assert torch.allclose(empty['relaxed'], ridge, rtol=1e-10, atol=1e-12)
    assert abs(protocol.mse(empty['original'], target) - protocol.mse(ridge, target)) < 1e-12

    loose = TINY['statistical']['ratios'][-1]
    small = protocol.solver_result(TINY, evidence, data.x, data.y, data.query, lambda_a, loose,
                                   torch.device('cpu'))
    if small['pairs']:
        precision = torch.full((small['pairs'],), TINY['statistical']['relaxed_variance'],
                               dtype=torch.float64)
        additive, coefficients = evidence.readout(small['indices'], precision)
        expected = evidence.predict(data.query, small['indices'], additive, coefficients)
        assert torch.allclose(small['relaxed'], expected, rtol=1e-10, atol=1e-12)
        assert not torch.allclose(small['relaxed'], small['original']), \
            'a non-empty support must change the relaxed readout'

    report = protocol.reference_cv(TINY, task, lambda_a, torch.device('cpu'))
    assert 0.0 <= report['mse'] < 1e2, 'a raw-unit score would be off by orders of magnitude'
    assert report['mse'] == report['kind_mse'][report['chosen']['kind']]
    assert report['chosen']['kind'] in ('original', 'relaxed')


def test_built_networks_are_double_and_reload_reproduces_outputs_exactly():
    config = json.loads(json.dumps(TINY))
    device = torch.device('cpu')
    task = sample_mixed_task(config['tasks'], protocol.task_seed(config, 'selection', 0),
                            background='dense', gamma=1.0)
    data, _ = protocol.context(task)
    evidence = protocol.Evidence(data.x, data.y, lambda_a=1.0)
    network = protocol.build_network(config, 'joint', 42).to(device)
    assert next(network.parameters()).dtype == torch.float64, \
        'dtype must be fixed before optimizer construction'
    with torch.no_grad():
        first = network(evidence, data.query)
    state = {key: value.detach().clone() for key, value in network.state_dict().items()}
    assert all(value.dtype == torch.float64 for value in state.values())
    reloaded = protocol.network_at(config, 'joint', state, device)
    with torch.no_grad():
        second = reloaded(evidence, data.query)
    assert torch.equal(first['prediction'], second['prediction'])
    assert torch.equal(first['gate_features'], second['gate_features'])
    assert torch.equal(first['additive_prediction'], second['additive_prediction'])
