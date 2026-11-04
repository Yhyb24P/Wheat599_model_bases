import torch
import pytest

from wheat599.matrix_training import task_panel, train_one


@pytest.mark.parametrize('variant', ['matrix', 'direct'])
def test_validation_does_not_change_training_task_stream(variant):
    config = {
        'network': {'width': 4, 'layers': 1, 'psi_hidden': 8, 'gate_hidden': 8,
                    'moment_scale': 'n', 'query_scale': 'one'},
        'tasks': {'context_sizes': [12], 'marker_counts': [8], 'query_size': 3,
                  'environments': 2, 'families': ['dense_additive'],
                  'designs': ['independent']},
        'training': {'steps': 2, 'evaluate_every': 1, 'learning_rate': .001,
                     'weight_decay': .01, 'gradient_clip': 1., 'task_seed': 1000},
    }
    first = task_panel(config['tasks'], ['dense_additive'], 70000, 2)
    second = task_panel(config['tasks'], ['dense_additive'], 90000, 2)
    _, a = train_one(config, variant, 17, first, torch.device('cpu'))
    torch.rand(117)  # Simulate unrelated caller RNG usage before reconstruction.
    _, b = train_one(config, variant, 17, second, torch.device('cpu'))
    assert [x['training_mse'] for x in a['history']] == [x['training_mse'] for x in b['history']]
    assert a['validation_tasks'] != b['validation_tasks']
    for record in (a, b):
        expected = min(record['history'], key=lambda x: x['validation_mse'])
        assert record['step'] == expected['step']
        assert record['validation_mse'] == expected['validation_mse']
