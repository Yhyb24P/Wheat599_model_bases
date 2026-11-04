"""Contract tests for the mixed additive/interaction generator. CPU only, tiny sizes.

Covered: seed determinism with an untouched global RNG, the pairing guarantee across gamma,
context-only normalization, metadata that never carries the drawn support, interaction
parents that need no main effect, forced endpoint sharing at small marker counts, and the
pair-space bijection.
"""
from dataclasses import replace

import torch

from wheat599.matrix_tasks import prepare_context
from wheat599.mixed_tasks import (BACKGROUNDS, decode_pairs, normalize_features, pair_space,
                                 sample_mixed_task)

CONFIG = {'context_sizes': [24], 'marker_counts': [9], 'query_size': 5, 'environments': 4,
          'designs': ['independent', 'linked'], 'backgrounds': list(BACKGROUNDS),
          'gammas': [0., .15, .5, 1.], 'additive_scales': [.5, 1., 2.],
          'noise_scales': [.5, 1.], 'pair_counts': [1, 4, 8], 'active_counts': [4, 16],
          'block_size': 4}
SMALL = {**CONFIG, 'context_sizes': [12], 'marker_counts': [6], 'query_size': 4,
         'backgrounds': ['sparse'], 'active_counts': [2], 'pair_counts': [8]}


def test_same_seed_reproduces_task_without_touching_global_rng():
    torch.manual_seed(1234)
    before = torch.get_rng_state().clone()
    first = sample_mixed_task(CONFIG, 31)
    assert torch.equal(before, torch.get_rng_state()), 'the generator consumed the global RNG'

    torch.manual_seed(777)
    torch.nn.Linear(9, 4)
    after_other_work = torch.get_rng_state().clone()
    second = sample_mixed_task(CONFIG, 31)
    assert torch.equal(after_other_work, torch.get_rng_state())
    for field in ('context_x', 'context_y', 'query_x', 'query_signal', 'query_y'):
        assert torch.equal(getattr(first, field), getattr(second, field))
    assert first.metadata == second.metadata
    assert not torch.equal(first.query_signal, sample_mixed_task(CONFIG, 32).query_signal)


def test_gamma_override_pairs_genotypes_interaction_direction_and_noise():
    zero = sample_mixed_task(CONFIG, 91, background='dense', gamma=0.)
    weak = sample_mixed_task(CONFIG, 91, background='dense', gamma=.15)
    strong = sample_mixed_task(CONFIG, 91, background='dense', gamma=1.)
    for other in (weak, strong):
        assert torch.equal(zero.context_x, other.context_x)
        assert torch.equal(zero.query_x, other.query_x)
        # query_y - query_signal is exactly the noise draw: equality pins the noise stream.
        recovered = (other.query_y - other.query_signal).double() \
            - (zero.query_y - zero.query_signal).double()
        noise = (zero.query_y - zero.query_signal).double().abs().max()
        assert recovered.abs().max() < 1e-5 * noise, \
            'the noise stream moved with gamma; only float32 rounding is admissible here'
        assert zero.metadata['additive_scale'] == other.metadata['additive_scale']
        assert zero.metadata['noise_scale'] == other.metadata['noise_scale']
        assert zero.metadata['effects'] == other.metadata['effects']
        assert zero.metadata['interaction_count'] == other.metadata['interaction_count']
    # The interaction block is the same direction for every gamma: (signal_g - signal_0)/g.
    assert torch.allclose((weak.query_signal - zero.query_signal) / .15,
                          strong.query_signal - zero.query_signal, rtol=1e-5, atol=1e-6)
    assert (strong.query_signal - zero.query_signal).abs().max() > 0, \
        'gamma=0 skipped the interaction draw, so the streams are not paired'


def test_normalization_is_context_only_and_constant_blocks_are_not_amplified():
    generator = torch.Generator().manual_seed(5)
    context = (torch.rand(10, 4, generator=generator, dtype=torch.float64) < .6).double()
    query = (torch.rand(6, 4, generator=generator, dtype=torch.float64) < .6).double()
    block = torch.cat((context, query), dim=0)
    normalized, scale = normalize_features(block, len(context))
    assert torch.allclose(normalized[:len(context)].square().mean(),
                          torch.ones((), dtype=torch.float64), rtol=1e-12, atol=1e-12)
    assert torch.allclose(normalized[:len(context)].mean(0),
                          torch.zeros(4, dtype=torch.float64), rtol=0, atol=1e-12)

    replaced = torch.cat((context, torch.zeros_like(query)), dim=0)
    other, other_scale = normalize_features(replaced, len(context))
    assert scale == other_scale, 'query rows entered the scale'
    assert torch.equal(other[:len(context)], normalized[:len(context)])

    all_constant = torch.ones(16, 3, dtype=torch.float64)
    untouched, constant_scale = normalize_features(all_constant, 10)
    assert constant_scale == 0.0
    assert torch.equal(untouched, all_constant - all_constant[:10].mean(0)), \
        'a constant block must keep its units instead of being divided by a tiny floor'

    task = sample_mixed_task(CONFIG, 7)
    clean = prepare_context(task.context_x, task.context_y, task.query_x)
    blinded = prepare_context(task.context_x, task.context_y, torch.zeros_like(task.query_x))
    assert torch.equal(clean.x, blinded.x) and torch.equal(clean.y_scale, blinded.y_scale)
    assert torch.equal(replace(task, query_y=torch.zeros_like(task.query_y)).prepare()[1],
                       task.prepare()[1]), 'the supervised target read query phenotypes'


def test_support_is_diagnostic_only_and_interaction_parents_need_no_main_effect():
    task, support = sample_mixed_task(CONFIG, 13, background='sparse', gamma=1.,
                                     return_support=True)
    assert 'additive_support' not in task.metadata and 'pairs' not in task.metadata
    assert 'support' not in str(task.metadata) and str(support['pairs'].tolist()) not in str(task.metadata)
    assert not hasattr(task, 'support')
    plain = sample_mixed_task(CONFIG, 13, background='sparse', gamma=1.)
    assert type(plain) is type(task)
    assert plain.metadata == task.metadata

    markers = task.metadata['markers']
    pairs = support['pairs']
    assert (pairs[:, 0] < pairs[:, 1]).all()
    assert len(set(map(tuple, pairs))) == len(pairs) == task.metadata['interaction_count']
    assert (pairs >= 0).all() and (pairs < markers).all()
    assert len(support['additive_support']) == task.metadata['effects']
    assert set(pairs.reshape(-1).tolist()) - set(support['additive_support'].tolist()), \
        'interaction endpoints were filtered to markers that already carry a main effect'


def test_small_marker_count_forces_shared_endpoints_and_every_background_is_drawable():
    task, support = sample_mixed_task(SMALL, 3, background='sparse', gamma=1.,
                                     return_support=True)
    pairs = support['pairs']
    assert len(pairs) == min(SMALL['pair_counts'][0], pair_space(SMALL['marker_counts'][0]))
    assert len(set(pairs.reshape(-1).tolist())) < 2 * len(pairs), \
        'shared endpoints across interaction pairs must be allowed'

    for background in BACKGROUNDS:
        drawn = sample_mixed_task(CONFIG, 21, background=background, gamma=.5)
        assert drawn.metadata['background'] == background
        assert drawn.metadata['effects'] >= 1 and drawn.metadata['interaction_count'] >= 1
        assert torch.isfinite(drawn.context_y).all() and torch.isfinite(drawn.query_signal).all()
        assert bool(((drawn.context_x == 0) | (drawn.context_x == 1)).all())
        assert bool(((drawn.query_x == 0) | (drawn.query_x == 1)).all())
        assert drawn.context_x.shape == (drawn.metadata['context_size'], drawn.metadata['markers'])
        assert drawn.query_signal.shape == (drawn.metadata['query_size'], drawn.metadata['environments'])

    forced = sample_mixed_task(CONFIG, 21, background='hidden', gamma=.5, design='independent')
    assert forced.metadata['design'] == 'linked' and forced.metadata['auto_linked']
    assert forced.metadata['gamma'] == .5, 'a hidden background must not force gamma to zero'
    quiet = sample_mixed_task(CONFIG, 21, background='hidden', gamma=0., design='independent')
    assert torch.equal(quiet.context_x, forced.context_x)


def test_decode_pairs_covers_the_full_pair_space_at_every_configured_size():
    for markers in (2, 3, 6, 9, 128, 512, 1279):
        total = pair_space(markers)
        left, right = decode_pairs(torch.arange(total), markers)
        assert bool((left >= 0).all()) and bool((left < right).all()) and bool((right < markers).all())
        assert int(torch.unique(left * markers + right).numel()) == total
        index = torch.randperm(total, generator=torch.Generator().manual_seed(markers))[:min(8, total)]
        sampled_left, sampled_right = decode_pairs(index, markers)
        assert int(torch.unique(sampled_left * markers + sampled_right).numel()) == len(index)
