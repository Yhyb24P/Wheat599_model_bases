import numpy as np
import pytest
import torch

from wheat599 import grouped_kernel as grouped
from wheat599.kernel_baselines import kernels


def toy_markers(rows=40, markers=24, seed=3):
    rng = np.random.default_rng(seed)
    frequency = rng.uniform(0.2, 0.8, size=markers)
    return (rng.random((rows, markers)) < frequency).astype(np.float64)


def test_tau_zero_reproduces_the_global_rbf_and_one_group_is_identical():
    x = toy_markers()
    fit, query = np.arange(30), np.arange(30, 40)
    bandwidth, groups = 0.7, 6
    additive, additive_cross, global_train, global_cross = kernels(
        x, fit, query, bandwidth, torch.device('cpu'))
    zero = grouped.grouped_rbf_kernels(x, fit, query, bandwidth=bandwidth,
                                       tau=0.0, groups=groups)
    np.testing.assert_allclose(zero.train, global_train.numpy(), atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(zero.cross, global_cross.numpy(), atol=1e-12, rtol=1e-12)
    one = grouped.grouped_rbf_kernels(x, fit, query, bandwidth=bandwidth,
                                      tau=0.25, groups=1)
    np.testing.assert_allclose(one.train, zero.train, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(one.cross, zero.cross, atol=1e-12, rtol=1e-12)


def test_grouped_kernel_is_psd_and_metadata_is_consistent():
    x = toy_markers()
    fit, query = np.arange(30), np.arange(30, 40)
    result = grouped.grouped_rbf_kernels(x, fit, query, bandwidth=0.7,
                                        tau=0.25, groups=6)
    assert result.train.shape == (30, 30)
    assert result.cross.shape == (10, 30)
    np.testing.assert_allclose(result.train, result.train.T, atol=0, rtol=0)
    eigenvalues = np.linalg.eigvalsh(result.train)
    assert eigenvalues.min() >= -1e-10
    metadata = result.metadata
    count = metadata['groups']
    assert count == len(metadata['anchors']) == len(metadata['sizes'])
    assert len(metadata['weights']) == len(metadata['group_d0']) == count
    assert np.isclose(sum(metadata['weights']), 1.0, atol=1e-12)
    assert sum(metadata['sizes']) == int((x[fit].std(axis=0) > 0).sum())
    for weight, scale in zip(metadata['weights'], metadata['group_d0']):
        assert np.isclose(scale, metadata['d0'] * weight, atol=1e-12, rtol=0)
    np.testing.assert_allclose(np.diag(result.train), 1.0, atol=1e-12)
    assert metadata['dg'] == 'identity'
    assert metadata['query_phenotypes_used'] is False


def test_constant_markers_stay_in_group_zero_and_have_no_weight():
    x = toy_markers()
    x[:, 0] = 1.0
    fit, query = np.arange(30), np.arange(30, 40)
    result = grouped.grouped_rbf_kernels(x, fit, query, bandwidth=0.7,
                                        tau=0.25, groups=6)
    metadata = result.metadata
    assert 0 in metadata['constant_markers']
    assert metadata['assignment'][0] == 0
    assert 0 not in metadata['anchors']
    assert sum(metadata['sizes']) == x.shape[1] - 1
    assert np.isclose(sum(metadata['weights']), 1.0, atol=1e-12)


def test_zero_variance_group_gets_zero_scale_without_epsilon():
    x = toy_markers()
    x[:, 0] = 1.0
    fit, query = np.arange(30), np.arange(30, 40)
    assignment = np.zeros(x.shape[1], dtype=np.int64)
    assignment[0] = 1  # only the constant marker sits in group 1
    result = grouped.grouped_rbf_kernels(x, fit, query, bandwidth=0.7,
                                        tau=0.25, groups=2,
                                        assignment=assignment)
    metadata = result.metadata
    assert metadata['weights'][1] == 0.0
    assert metadata['group_d0'][1] == 0.0
    assert metadata['group_distance_status'][1] == 'zero_weight'
    global_kernel = grouped.grouped_rbf_kernels(x, fit, query, bandwidth=0.7,
                                                tau=0.0, groups=2,
                                                assignment=assignment)
    np.testing.assert_allclose(result.train, global_kernel.train, atol=1e-12)
    np.testing.assert_allclose(result.cross, global_kernel.cross, atol=1e-12)


def test_fit_constant_marker_cannot_enter_the_grouped_cross_kernel():
    x = toy_markers()
    x[:, 0] = 1.0
    fit, query = np.arange(30), np.arange(30, 40)
    changed = x.copy()
    changed[query, 0] = 0.0  # varies only at a marker that is constant on fit
    grouped_a = grouped.grouped_rbf_kernels(x, fit, query, bandwidth=0.7,
                                           tau=1.0, groups=6)
    grouped_b = grouped.grouped_rbf_kernels(changed, fit, query, bandwidth=0.7,
                                           tau=1.0, groups=6)
    np.testing.assert_array_equal(grouped_a.cross, grouped_b.cross)
    np.testing.assert_array_equal(grouped_a.train, grouped_b.train)
    global_a = grouped.grouped_rbf_kernels(x, fit, query, bandwidth=0.7,
                                          tau=0.0, groups=6)
    global_b = grouped.grouped_rbf_kernels(changed, fit, query, bandwidth=0.7,
                                          tau=0.0, groups=6)
    assert not np.allclose(global_a.cross, global_b.cross)
    np.testing.assert_array_equal(global_a.train, global_b.train)


def test_fit_metadata_and_train_kernel_do_not_depend_on_query_rows():
    x = toy_markers(rows=44)
    fit, query_one = np.arange(30), np.arange(30, 36)
    query_two = np.arange(30, 44)
    first = grouped.grouped_rbf_kernels(x, fit, query_one, bandwidth=0.7,
                                       tau=0.25, groups=6)
    second = grouped.grouped_rbf_kernels(x, fit, query_two, bandwidth=0.7,
                                        tau=0.25, groups=6)
    for key in ('anchors', 'assignment', 'sizes', 'weights', 'group_d0',
                'constant_markers', 'd0', 'groups'):
        assert first.metadata[key] == second.metadata[key]
    np.testing.assert_allclose(first.train, second.train, atol=0, rtol=0)
    np.testing.assert_allclose(first.cross, second.cross[:len(query_one)], atol=0, rtol=0)


def test_random_assignments_preserve_group_sizes_and_frequency_layers():
    x = toy_markers(rows=40, markers=30)
    x[:, 0] = 1.0
    fit = np.arange(30)
    grouping = grouped.fit_group_assignment(x[fit], groups=6)
    frequencies = grouped.marker_frequencies(x[fit])
    constant = np.zeros(x.shape[1], dtype=bool)
    constant[0] = True
    seeds = (1, 2, 3, 4, 5, 6)
    bins = 5
    outputs = grouped.stratified_random_assignments(
        grouping['assignment'], frequencies, seeds, bins=bins,
        constant_mask=constant)
    assert set(outputs) == set(seeds)
    original = np.asarray(grouping['assignment'])
    edges = np.linspace(0.0, 1.0, bins + 1)
    layer = np.clip(np.digitize(frequencies, edges[1:-1], right=False), 0, bins - 1)
    original_sizes = np.bincount(original[~constant], minlength=len(grouping['sizes']))
    for seed, assignment in outputs.items():
        assert assignment.shape == original.shape
        assert assignment[0] == 0
        np.testing.assert_array_equal(
            np.bincount(assignment[~constant], minlength=len(original_sizes)),
            original_sizes)
        for position in range(bins):
            members = (layer == position) & ~constant
            np.testing.assert_array_equal(
                np.bincount(original[members], minlength=len(original_sizes)),
                np.bincount(assignment[members], minlength=len(original_sizes)))
    assert any(not np.array_equal(outputs[seed], original) for seed in seeds)


def test_random_kernel_is_psd_and_invalid_inputs_are_rejected():
    x = toy_markers(rows=40, markers=30)
    fit, query = np.arange(30), np.arange(30, 40)
    grouping = grouped.fit_group_assignment(x[fit], groups=6)
    frequencies = grouped.marker_frequencies(x[fit])
    random_assignment = grouped.stratified_random_assignments(
        grouping['assignment'], frequencies, (1,), bins=5)[1]
    result = grouped.grouped_rbf_kernels(x, fit, query, bandwidth=0.7,
                                        tau=0.25, groups=6,
                                        assignment=random_assignment)
    assert np.linalg.eigvalsh(result.train).min() >= -1e-10
    with pytest.raises(ValueError):
        grouped.grouped_rbf_kernels(x, fit, query, bandwidth=0.0, groups=6)
    with pytest.raises(ValueError):
        grouped.grouped_rbf_kernels(x, fit, query, bandwidth=0.7, tau=1.1, groups=6)
    with pytest.raises(ValueError):
        grouped.grouped_rbf_kernels(x, np.arange(31), query, bandwidth=0.7,
                                    groups=6)
    with pytest.raises(ValueError):
        grouped.grouped_rbf_kernels(x, fit, np.arange(20, 30), bandwidth=0.7,
                                    groups=6)
    with pytest.raises(ValueError):
        grouped.stratified_random_assignments(
            grouping['assignment'], frequencies, (1,), bins=0)
