"""Contract tests for the independent synthetic task module (small CPU sizes only).

Covered: seed determinism without global-RNG coupling, finiteness and binary
genotypes for all four task families, context-only preprocessing that query values
cannot influence, encoding-flip and column-permutation behaviour of that
preprocessing. Hidden-cause exposure is also checked by reading the generator.
"""
import dataclasses
import math

import torch

from wheat599.matrix_tasks import FAMILIES, prepare_context, sample_task

CONTEXT, MARKERS, QUERY, ENVIRONMENTS = 24, 16, 5, 4
CONFIG = {'context_sizes': [CONTEXT], 'marker_counts': [MARKERS], 'query_size': QUERY,
          'environments': ENVIRONMENTS, 'block_size': 8, 'active_counts': [1, 4],
          'pair_counts': [1, 4]}


def _doubles(task):
    return (task.context_x.double(), task.context_y.double(), task.query_x.double())


def test_same_seed_reproduces_task_independently_of_global_rng():
    torch.manual_seed(1234)
    before = torch.get_rng_state().clone()
    first = sample_task(CONFIG, 777)
    assert torch.equal(before, torch.get_rng_state()), 'sample_task consumed the global RNG'

    torch.manual_seed(90210)
    torch.nn.Linear(MARKERS, ENVIRONMENTS)
    after_other_work = torch.get_rng_state().clone()
    second = sample_task(CONFIG, 777)
    assert torch.equal(after_other_work, torch.get_rng_state()), 'sample_task consumed the global RNG'

    for field in ('context_x', 'context_y', 'query_x', 'query_signal', 'query_y'):
        assert torch.equal(getattr(first, field), getattr(second, field))
    assert first.metadata == second.metadata
    third = sample_task(CONFIG, 778)
    assert not torch.equal(first.query_signal, third.query_signal)


def test_every_family_returns_finite_targets_and_binary_genotypes():
    for family in FAMILIES:
        for seed in (0, 1, 2):
            task = sample_task(CONFIG, seed, family=family)
            meta = task.metadata
            assert meta['family'] == family and meta['seed'] == seed
            assert (meta['context_size'], meta['markers'], meta['environments']) \
                == (CONTEXT, MARKERS, ENVIRONMENTS)
            assert task.context_x.shape == (CONTEXT, MARKERS)
            assert task.query_x.shape == (QUERY, MARKERS)
            assert task.context_y.shape == (CONTEXT, ENVIRONMENTS)
            assert task.query_signal.shape == (QUERY, ENVIRONMENTS)
            assert task.query_y.shape == (QUERY, ENVIRONMENTS)
            for name in ('context_x', 'query_x'):
                tensor = getattr(task, name)
                assert bool(((tensor == 0) | (tensor == 1)).all()), f'{family} {name} not binary'
                assert torch.unique(tensor).numel() > 1, f'{family} {name} is constant'
            for name in ('context_y', 'query_signal', 'query_y'):
                assert bool(torch.isfinite(getattr(task, name)).all()), f'{family} {name} not finite'


def test_context_preprocessing_and_target_ignore_query_values():
    task = sample_task(CONFIG, 5)
    context, response_y, query = _doubles(task)
    base = prepare_context(context, response_y, query)

    flipped = query.clone()
    flipped[:, :3] = 1 - flipped[:, :3]
    perturbed = prepare_context(context, response_y, flipped)
    assert torch.equal(base.x, perturbed.x)
    assert torch.equal(base.y_mean, perturbed.y_mean)
    assert torch.equal(base.y_scale, perturbed.y_scale)
    assert not torch.equal(base.query, perturbed.query)

    zeroed = prepare_context(context, response_y, torch.zeros_like(query))
    assert torch.equal(base.x, zeroed.x) and torch.equal(base.y_mean, zeroed.y_mean)

    altered = dataclasses.replace(task, query_y=torch.randn_like(task.query_y) * 100)
    prepared, target = task.prepare()
    assert torch.equal(target, (task.query_signal - prepared.y_mean) / prepared.y_scale)
    assert torch.equal(altered.prepare()[1], target), 'the supervised target read query phenotypes'
    assert torch.equal(altered.prepare()[0].x, prepared.x)
    assert torch.allclose(prepared.restore(target), task.query_signal, rtol=0, atol=1e-6)


def test_encoding_flip_and_column_permutation_pass_through_preprocessing():
    task = sample_task(CONFIG, 11)
    context, response_y, query = _doubles(task)
    original = prepare_context(context, response_y, query)

    flip = torch.tensor([0, 1, 4, 9])
    outside = torch.tensor([j for j in range(MARKERS) if int(j) not in set(flip.tolist())])
    flipped_context = context.clone()
    flipped_query = query.clone()
    flipped_context[:, flip] = 1 - flipped_context[:, flip]
    flipped_query[:, flip] = 1 - flipped_query[:, flip]
    flipped = prepare_context(flipped_context, response_y, flipped_query)

    assert torch.allclose(flipped.x[:, flip], -original.x[:, flip], rtol=1e-9, atol=1e-12)
    assert torch.allclose(flipped.x[:, outside], original.x[:, outside], rtol=1e-9, atol=1e-12)
    assert torch.allclose(flipped.query[:, flip], -original.query[:, flip], rtol=1e-9, atol=1e-12)
    assert torch.allclose(flipped.y_scale, original.y_scale, rtol=1e-9, atol=1e-12)
    assert torch.allclose(flipped.y_mean, original.y_mean, rtol=0, atol=0)

    permutation = torch.randperm(MARKERS, generator=torch.Generator().manual_seed(3))
    permuted = prepare_context(context[:, permutation], response_y, query[:, permutation])
    assert torch.allclose(permuted.x, original.x[:, permutation], rtol=1e-9, atol=1e-12)
    assert torch.allclose(permuted.query, original.query[:, permutation], rtol=1e-9, atol=1e-12)
    assert torch.allclose(permuted.y_scale, original.y_scale, rtol=1e-9, atol=1e-12)


def test_hidden_additive_forces_linked_design_with_latent_width_bound():
    blocks = math.ceil(MARKERS / CONFIG['block_size'])
    for seed in (0, 1, 5):
        task = sample_task(CONFIG, seed, family='hidden_additive', design='independent')
        assert task.metadata['design'] == 'linked'
        assert 1 <= task.metadata['effects'] <= blocks
        assert sample_task(CONFIG, seed, family='dense_additive',
                           design='independent').metadata['design'] == 'independent'
        assert sample_task(CONFIG, seed, family='pairwise',
                           design='linked').metadata['design'] == 'linked'


def test_metadata_records_the_drawn_prior():
    low, high = .1, .8
    task = sample_task({**CONFIG, 'heritability_range': (low, high),
                        'families': ('dense_additive',), 'designs': ('admixture',)}, 13)
    meta = task.metadata
    assert meta['family'] == 'dense_additive' and meta['design'] == 'admixture'
    assert len(meta['nominal_h2']) == ENVIRONMENTS
    assert all(low <= value <= high for value in meta['nominal_h2'])
    assert meta['query_size'] == QUERY and meta['effects'] >= 1
    correlation = torch.tensor(meta['effect_correlation'], dtype=torch.float64)
    assert correlation.shape == (ENVIRONMENTS, ENVIRONMENTS)
    assert bool(torch.isfinite(correlation).all())
    assert torch.allclose(correlation, correlation.T, rtol=0, atol=0)
    # The task is drawn in float32, so the unit diagonal carries single-precision error.
    assert torch.allclose(torch.diagonal(correlation),
                          torch.ones(ENVIRONMENTS, dtype=torch.float64), rtol=0, atol=1e-6)
    assert isinstance(meta['constant_context_signal_design'], bool)
