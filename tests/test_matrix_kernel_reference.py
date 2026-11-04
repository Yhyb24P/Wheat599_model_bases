"""Tests for the finite-kernel reference: explicit-feature equivalence, context-only
statistics and invariances, and validation-only hyperparameter selection. Tiny CPU sizes."""
import dataclasses
import json
from types import SimpleNamespace

import torch

from wheat599.matrix_kernel_reference import (candidate_grid,
                                              normalized_kernels, predict, run, select)
from wheat599.matrix_tasks import prepare_context, sample_task

TINY = {'context_sizes': [12], 'marker_counts': [9], 'query_size': 4, 'environments': 2,
        'families': ('dense_additive', 'sparse_additive'), 'designs': ('independent',),
        'block_size': 4, 'active_counts': [1, 2], 'pair_counts': [1, 2],
        'heritability_range': (.4, .6)}
GRID = {'kernels': {'mixtures': [0.0, 0.5, 1.0], 'ridge': [0.1, 3.0]}}


def _raw(seed=0, n=6, markers=3, environments=2, query=2):
    generator = torch.Generator().manual_seed(seed)
    x = (torch.rand(n, markers, generator=generator) < .6).double()
    y = torch.randn(n, environments, generator=generator).double()
    q = (torch.rand(query, markers, generator=generator) < .4).double()
    return x, y, q


def _context(seed=0):
    return prepare_context(*_raw(seed))


def _explicit_features(data):
    """Features that must reproduce the reference kernels and cross blocks exactly."""
    markers = data.x.shape[1]
    # Build features independently of the kernel implementation. Both support
    # and queries use the same support mean, including quadratic monomials.
    monomials = (data.x[:, :, None] * data.x[:, None, :]).reshape(len(data.x), markers ** 2)
    query_monomials = (data.query[:, :, None] * data.query[:, None, :]).reshape(
        len(data.query), markers ** 2)
    features, query_features = {}, {}
    for name, support, query in (('additive', data.x, data.query),
                                  ('quadratic', monomials, query_monomials)):
        mean = support.mean(0, keepdim=True)
        design, future = support - mean, query - mean
        scale = design.square().sum(1).mean().sqrt()
        design, future = design / scale, future / scale
        features[name] = (design, design @ design.T, future @ design.T)
        query_features[name] = future
    return features, query_features


def test_kernel_prediction_matches_explicit_centered_and_scaled_features():
    data = _context()
    kernels = normalized_kernels(data)
    features, query_features = _explicit_features(data)
    for name in ('additive', 'quadratic'):
        design, kernel, cross = features[name]
        assert torch.allclose(design @ design.T, kernel, rtol=1e-9, atol=1e-11), name
        assert torch.allclose(query_features[name] @ design.T, cross, rtol=1e-9, atol=1e-11), name
        assert torch.allclose(kernel, kernels[name], rtol=1e-9, atol=1e-11)
        assert torch.allclose(cross, kernels[f'{name}_cross'], rtol=1e-9, atol=1e-11)

    fraction = 1.0
    ridge = 0.3
    design, kernel, cross = features['quadratic']
    weight = torch.linalg.solve(design.T @ design + ridge * torch.eye(design.shape[1], dtype=torch.float64),
                                design.T @ data.y)
    assert torch.allclose(predict(kernels, data.y, fraction, ridge),
                          query_features['quadratic'] @ weight, rtol=1e-8, atol=1e-10)

    fraction = 0.5
    mixed_design = torch.cat(((1 - fraction) ** .5 * features['additive'][0],
                              fraction ** .5 * features['quadratic'][0]), dim=1)
    mixed_query = torch.cat(((1 - fraction) ** .5 * query_features['additive'],
                             fraction ** .5 * query_features['quadratic']), dim=1)
    mixed_kernel = (1 - fraction) * kernels['additive'] + fraction * kernels['quadratic']
    assert torch.allclose(mixed_design @ mixed_design.T, mixed_kernel, rtol=1e-9, atol=1e-11)
    for ridge in (0.1, 3.0):
        weight = torch.linalg.solve(
            mixed_design.T @ mixed_design + ridge * torch.eye(mixed_design.shape[1], dtype=torch.float64),
            mixed_design.T @ data.y)
        assert torch.allclose(predict(kernels, data.y, fraction, ridge),
                              mixed_query @ weight, rtol=1e-8, atol=1e-10)


def test_kernels_and_predictions_use_context_only_and_are_invariant_to_redundant_changes():
    x, y, q = _raw(seed=3)
    data = prepare_context(x, y, q)
    kernels = normalized_kernels(data)
    candidates = candidate_grid(GRID)
    base = {c: predict(kernels, data.y, *c) for c in candidates}

    replacement = (torch.rand(q.shape, generator=torch.Generator().manual_seed(4)) < .5).double()
    alt_kernels = normalized_kernels(prepare_context(x, y, replacement))
    assert torch.allclose(alt_kernels['additive'], kernels['additive'], rtol=1e-12, atol=1e-14)
    assert torch.allclose(alt_kernels['quadratic'], kernels['quadratic'], rtol=1e-12, atol=1e-14)
    assert not torch.allclose(alt_kernels['additive_cross'], kernels['additive_cross'])

    order = torch.randperm(len(x), generator=torch.Generator().manual_seed(5))
    shuffled = prepare_context(x[order], y[order], q)
    shuffled_kernels = normalized_kernels(shuffled)
    for candidate in candidates:
        assert torch.allclose(predict(shuffled_kernels, shuffled.y, *candidate), base[candidate],
                              rtol=1e-8, atol=1e-10), 'material order changed the prediction'

    permutation = torch.tensor([2, 0, 1])
    sign = torch.tensor([1.0, -1.0, 1.0])
    moved = prepare_context(x[:, permutation] * sign, y, q[:, permutation] * sign)
    moved_kernels = normalized_kernels(moved)
    for candidate in candidates:
        assert torch.allclose(predict(moved_kernels, moved.y, *candidate), base[candidate],
                              rtol=1e-8, atol=1e-10), 'marker order or coding direction changed it'

    rows = torch.randperm(len(q), generator=torch.Generator().manual_seed(7))
    reordered = prepare_context(x, y, q[rows])
    for candidate in candidates:
        assert torch.allclose(predict(normalized_kernels(reordered), reordered.y, *candidate),
                              base[candidate][rows], rtol=1e-10, atol=1e-12)


def test_selection_uses_only_validation_and_breaks_ties_by_grid_order(tmp_path, monkeypatch):
    validation = [sample_task(TINY, seed) for seed in (11, 12)]
    held_out = [sample_task(TINY, seed + 50) for seed in (11, 12)]
    # Execute the actual runner twice, changing only the held-out target.
    config = {**GRID, 'device': 'cpu', 'cpu_threads': 2}
    config_path = tmp_path / 'config.json'
    config_path.write_text(json.dumps(config))
    blinded = [dataclasses.replace(task, query_signal=torch.zeros_like(task.query_signal),
                                   query_y=torch.zeros_like(task.query_y)) for task in held_out]
    for index, panel in enumerate((held_out, blinded)):
        monkeypatch.setattr('wheat599.matrix_kernel_reference.panels',
                            lambda cfg, panel=panel: (validation, panel))
        args = SimpleNamespace(selection=None, out=tmp_path / str(index), config=config_path)
        run(config, args, __file__)
    assert (tmp_path / '0' / 'selection.json').read_bytes() == (tmp_path / '1' / 'selection.json').read_bytes()
    assert (tmp_path / '0' / 'validation.json').read_bytes() == (tmp_path / '1' / 'validation.json').read_bytes()
    assert (tmp_path / '0' / 'held_out.json').read_bytes() != (tmp_path / '1' / 'held_out.json').read_bytes()

    tied_candidates = [(0.0, 0.1), (0.0, 3.0), (0.5, 1.0)]
    wrapped = [{'task': {}, 'mse': [1.0, 1.0, 2.0]}, {'task': {}, 'mse': [1.0, 1.0, 2.0]}]
    first = select(wrapped, tied_candidates)
    assert first['candidate_index'] == 0 and first['validation_mse'] == 1.0
    restricted = select(wrapped, tied_candidates, [1, 2])
    assert restricted['candidate_index'] == 1 and restricted['fraction'] == 0.0
    assert restricted['ridge'] == 3.0
