"""Tiny correctness tests for the readout diagnostic. No training, CPU only."""
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from wheat599 import vector_diagnostic as diagnostic
from wheat599.joint_training import build_network, context, folds_of, mse
from wheat599.mixed_tasks import sample_mixed_task

TINY = {
    'device': 'cpu', 'cpu_threads': 1, 'gpu_memory_fraction': 0.5,
    'lambda_a_grid': [0.3, 3.0], 'context_cv_folds': 3,
    'network': {'environments': 2, 'width': 8, 'rounds': 1, 'add_per_round': 2,
                'max_candidates': 4, 'max_variance': 100.0},
    'tasks': {'context_sizes': [12], 'marker_counts': [6], 'query_size': 4, 'environments': 2,
              'designs': ['independent'], 'backgrounds': ['dense', 'sparse'],
              'gammas': [0.0, 1.0], 'additive_scales': [1.0], 'noise_scales': [1.0],
              'pair_counts': [1, 2], 'active_counts': [2], 'block_size': 3},
    'statistical': {'ratios': [1.0, 0.5], 'lambda_e': 0.001, 'relaxed_variance': 100.0,
                    'same_candidate_variance': 1.0,
                    'solver': {'block_size': 8, 'cache_phi': True, 'max_active': 8,
                               'max_iter': 200, 'max_outer': 8, 'kkt_tolerance': 1e-6,
                               'dual_gap_tolerance': 1e-5}},
    'confirmation': {'role_base': 6_000_000_000_000, 'bases_per_background': 1, 'query_size': 4,
                     'context_sizes': [12], 'marker_counts': [6]},
    'readout': {'scales': [0.1, 1.0, 10.0], 'shape_order': ['identity', 'diagonal', 'half', 'full']},
    'decision': {'strong_relative_gain': 0.02, 'direction_relative_gain': 0.005,
                 'paired_win_fraction': 0.6, 'zero_relative_degradation_max': 0.01},
}


def _task(gamma=1.0, background='dense'):
    return sample_mixed_task(TINY['tasks'], 6000000000000, background=background, gamma=gamma,
                             return_support=True)


def test_options_are_the_shape_scale_product_and_families_are_nested():
    items = diagnostic.options(TINY)
    assert [item['name'] for item in items] == [
        'zero', 'identity_0.1', 'identity_1', 'identity_10', 'diagonal_0.1', 'diagonal_1',
        'diagonal_10', 'half_0.1', 'half_1', 'half_10', 'full_0.1', 'full_1', 'full_10']
    scalar = set(diagnostic.family_options(TINY, 'scalar'))
    diagonal = set(diagnostic.family_options(TINY, 'diagonal'))
    full = set(diagnostic.family_options(TINY, 'full'))
    assert scalar < diagonal < full, 'families must be nested'
    assert scalar == {'zero', 'identity_0.1', 'identity_1', 'identity_10'}
    assert diagonal == scalar | {'diagonal_0.1', 'diagonal_1', 'diagonal_10'}
    assert full == {item['name'] for item in items}


def test_nested_lambda_ignores_the_held_out_rows_of_its_own_fold():
    task, _ = _task()
    rows = len(task.context_x)
    lambdas = diagnostic.fold_lambdas(task, rows, TINY)
    hold = folds_of(rows, TINY['context_cv_folds'])[0]
    disturbed = task.context_y.clone()
    disturbed[hold] = disturbed[hold] + 5.0
    changed = replace(task, context_y=disturbed)
    again = diagnostic.fold_lambdas(changed, rows, TINY)
    assert again[tuple(hold.tolist())] == lambdas[tuple(hold.tolist())], \
        'the fold lambda must depend only on that fold fit rows'

    network = build_network(TINY, 'joint', 42)
    x, y, query, truth = diagnostic.inner_context(task, hold, rows)
    x2, y2, query2, truth2 = diagnostic.inner_context(changed, hold, rows)
    evidence = diagnostic.Evidence(x, y, lambda_a=lambdas[tuple(hold.tolist())])
    evidence2 = diagnostic.Evidence(x2, y2, lambda_a=lambdas[tuple(hold.tolist())])
    indices, precision = diagnostic.extracted(network, evidence, query)
    indices2, precision2 = diagnostic.extracted(network, evidence2, query2)
    assert torch.equal(indices, indices2) and torch.equal(precision, precision2)
    mapping = diagnostic.shape_map(evidence, indices)
    mapping2 = diagnostic.shape_map(evidence2, indices2)
    assert set(mapping) == set(mapping2)
    for name in mapping:
        if mapping[name] is None:
            assert mapping2[name] is None
        else:
            assert torch.equal(mapping[name], mapping2[name])
    assert not torch.equal(truth, truth2)


def test_blinding_the_query_does_not_change_the_fold_selection():
    task, _ = _task()
    rows = len(task.context_x)
    lambdas = diagnostic.fold_lambdas(task, rows, TINY)
    network = build_network(TINY, 'joint', 42)
    totals = diagnostic.fold_totals(TINY, network, task, lambdas, torch.device('cpu'))
    blinded = replace(task, query_x=torch.zeros_like(task.query_x),
                      query_signal=torch.zeros_like(task.query_signal),
                      query_y=torch.zeros_like(task.query_y))
    again = diagnostic.fold_totals(TINY, network, blinded, lambdas, torch.device('cpu'))
    assert totals == again, 'query rows must not enter any selection quantity'


def test_identity_candidates_are_exact_and_empty_support_is_additive():
    task, _ = _task()
    data, target = context(task)
    evidence = diagnostic.Evidence(data.x, data.y, lambda_a=3.0)
    indices, precision = diagnostic.extracted(build_network(TINY, 'joint', 42), evidence, data.query)
    mapping = diagnostic.shape_map(evidence, indices)
    one = diagnostic.readout(evidence, indices, precision,
                             {'shape': 'identity', 'scale': 1.0}, mapping)
    reference = evidence.readout(indices, precision)
    assert torch.equal(one[0], reference[0]) and torch.equal(one[1], reference[1])
    scaled = diagnostic.readout(evidence, indices, precision,
                                {'shape': 'identity', 'scale': 10.0}, mapping)
    reference_scaled = evidence.readout(indices, precision * 10.0)
    assert torch.equal(scaled[0], reference_scaled[0])
    empty = torch.zeros(0, dtype=torch.long)
    additive, coefficients = diagnostic.readout(evidence, empty, torch.zeros(0),
                                                {'shape': 'full', 'scale': 1.0}, {})
    assert coefficients.numel() == 0
    assert torch.equal(evidence.predict(data.query, empty, additive, coefficients),
                       evidence.ridge(data.query))


def test_shapes_come_from_the_real_vector_readout_operators():
    task, _ = _task()
    data, _ = context(task)
    evidence = diagnostic.Evidence(data.x, data.y, lambda_a=3.0)
    indices, _ = diagnostic.extracted(build_network(TINY, 'joint', 42), evidence, data.query)
    mapping = diagnostic.shape_map(evidence, indices)
    assert {'identity', 'diagonal', 'half', 'full'} <= set(mapping)
    assert mapping['identity'] is None
    for name in ('diagonal', 'half', 'full'):
        assert mapping[name].shape == (2, 2) and torch.isfinite(mapping[name]).all()
    assert torch.allclose(mapping['diagonal'], torch.diag(torch.diagonal(mapping['diagonal'])))
    assert diagnostic.shape_map(evidence, torch.zeros(0, dtype=torch.long)) == {}


def test_pick_ties_prefer_zero_then_the_original_precision_option():
    totals = {'zero': 1.0, 'identity_0.1': 1.0, 'identity_1': 1.0, 'diagonal_1': 1.0}
    assert diagnostic.pick(totals, ['identity_0.1', 'identity_1']) == 'identity_1'
    assert diagnostic.pick(totals, ['zero', 'identity_1']) == 'zero'
    assert diagnostic.pick({'identity_0.1': 0.5, 'identity_1': 1.0},
                           ['identity_0.1', 'identity_1']) == 'identity_0.1'


def test_support_indices_follow_the_enumerated_pair_order():
    task, _ = _task()
    data, _ = context(task)
    evidence = diagnostic.Evidence(data.x, data.y, lambda_a=3.0)
    pairs = np.array([[0, 1], [1, 4], [2, 5]])
    positions = diagnostic.support_indices(evidence, pairs)
    assert [tuple(evidence.pairs[int(index)].tolist()) for index in positions] == \
        [tuple(pair) for pair in pairs]
    assert int(diagnostic.support_indices(evidence, np.zeros((0, 2), dtype=np.int64)).numel()) == 0


def test_mse_pair_moves_both_operands_and_matches_the_manual_value():
    prediction = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
    truth = torch.tensor([[0.5, 1.0]], dtype=torch.float32)
    manual = float(((prediction.double() - truth.double()) ** 2).mean())
    assert diagnostic.mse_pair(prediction, truth) == pytest.approx(manual, rel=0, abs=1e-12)


def test_empty_candidate_reports_the_additive_error_as_the_original_precision():
    task, _ = _task()
    data, target = context(task)
    evidence = diagnostic.Evidence(data.x, data.y, lambda_a=3.0)
    report = diagnostic.family_report(TINY, {}, evidence, data.query, target,
                                      torch.zeros(0, dtype=torch.long), torch.zeros(0))
    assert report['original_precision']['query_mse'] == pytest.approx(
        diagnostic.mse_pair(evidence.ridge(data.query), target), rel=0, abs=1e-12)
    assert report['picks'] == {'scalar': None, 'diagonal': None, 'full': None}


def test_decision_uses_group_mean_ratios_and_the_full_family_floor():
    def record(background, gamma, additive, full, scalar, diagonal):
        return {'background': background, 'gamma': gamma, 'additive_mse': additive,
                'frozen': {'a.pt': {'family_query_mse': {'full': full, 'scalar': scalar,
                                                         'diagonal': diagonal}}}}
    records = [record('dense', 1.0, 1.0, 0.8, 1.0, 0.9),
               record('sparse', 0.5, 1.0, 0.9, 0.95, 0.9),
               record('dense', 0.0, 1.0, 1.005, 1.0, 1.0),
               record('hidden', 0.0, 1.0, 1.2, 1.0, 1.0)]
    report = diagnostic.decision(records, TINY)['a.pt']
    assert report['tasks_gamma_ge_half'] == 2
    assert report['group_mean_mse']['full'] == pytest.approx((0.8 + 0.9) / 2)
    assert report['relative_gain_vs_scalar'] == pytest.approx(
        1 - ((0.8 + 0.9) / 2) / ((1.0 + 0.95) / 2))
    assert report['relative_gain_vs_diagonal'] == pytest.approx(
        1 - ((0.8 + 0.9) / 2) / ((0.9 + 0.9) / 2))
    assert report['paired_win_fraction_vs_diagonal'] == pytest.approx(0.5)
    assert report['floor_relative_degradation'] == pytest.approx(0.005)
    assert report['thresholds'] == TINY['decision']

    skewed = [record('dense', 0.0, 1.0, 0.9, 1.0, 1.0), record('sparse', 0.0, 2.0, 2.2, 2.0, 2.0)]
    floor = diagnostic.decision(skewed, TINY)['a.pt']['floor_relative_degradation']
    assert floor == pytest.approx((0.9 + 2.2) / (1.0 + 2.0) - 1), 'group means, not mean ratios'

    missing = [record('dense', 0.0, 1.0, 1.0, 1.0, 1.0)]
    del missing[0]['frozen']['a.pt']['family_query_mse']['full']
    with pytest.raises(KeyError):
        diagnostic.decision(missing, TINY)


def test_tiny_end_to_end_uses_the_frozen_checkpoint_and_writes_records(tmp_path):
    import yaml

    config = json.loads(json.dumps(TINY))
    config['checkpoint_root'] = str(tmp_path)
    network = build_network(config, 'joint', 42)
    weight = tmp_path / 'joint_42.pt'
    torch.save({key: value.detach().clone() for key, value in network.state_dict().items()}, weight)
    digest = hashlib.sha256(weight.read_bytes()).hexdigest()
    config['checkpoints'] = {'joint_42.pt': digest}
    path = tmp_path / 'vector_diagnostic.yaml'
    path.write_text(yaml.safe_dump(config))
    args = SimpleNamespace(out=tmp_path / 'run', selection=None, config=path)
    diagnostic.run(config, args, Path('scripts/run_background_nested.py'))
    records = json.loads((args.out / 'vector_dimension.json').read_text())
    assert len(records) == 2 * 2, records
    for record in records:
        frozen = record['frozen']['joint_42.pt']
        assert set(frozen['fold_totals']) == {item['name'] for item in diagnostic.options(config)}
        assert set(frozen['picks']) == set(diagnostic.CV_FAMILIES)
        assert frozen['original_precision']['selected_by_cv'] is False
        assert len(record['fold_lambdas']) == config['context_cv_folds']
        assert record['fold_lambdas'] and record['full_lambda'] in config['lambda_a_grid']
        assert record['reference']['chosen']['ratio'] in config['statistical']['ratios']
        assert record['reference']['fold_lambdas'] == record['fold_lambdas']
        oracle = record['oracle_cv']['joint_42.pt']
        assert oracle['support_pairs'] == (0 if record['gamma'] == 0 else oracle['support_pairs'])
    assert json.loads((args.out / 'failures.json').read_text()) == []
    assert (args.out / 'decision.json').exists()
    provenance = json.loads((args.out / 'provenance.json').read_text())
    assert provenance['checkpoint_sha256'] == {'joint_42.pt': digest}
