"""Fit-only grouped marker kernels for Wheat599 material similarity.

The kernel is the material-level RBF part of the multi-environment candidate:

    (1 - tau) * K_R + tau * sum_g omega_g * K_g     (D_g = I in stage 1)

All grouping, distance scales, and weights are estimated from fit rows only.
``K_g`` is an RBF kernel over the markers assigned to group ``g``, using the same
bandwidth as the global kernel. Its distance scale is ``d0_g = omega_g*d0``,
where ``d0`` is the global median off-diagonal squared distance and ``omega_g``
is the group's share of fit marker variance. This preserves the expected
distance scale even when a small group's own median Hamming distance is zero;
zero-variance groups get zero weight and contribute no kernel. No epsilon
inflates a degenerate group. Each diagonal is one, so the aggregated group
kernel's diagonal is one because the weights sum to one.

Groups follow ``GenomicTabM._group``: nonconstant fit markers are anchored by
raw fit variance first, then by minimum maximum absolute correlation to the
chosen anchors; every marker joins its highest-absolute-correlation anchor.
Constant markers are fixed to group 0, excluded from anchor and size counts,
and have zero variance weight. Group kernels also exclude fit-constant markers
from their member distances, so query variation at a fit-constant marker cannot
enter the grouped component; the global RBF keeps all markers unchanged. Query
phenotypes never enter this module; only query marker rows form the cross
kernel. Random controls permute group labels inside equal-width fit-frequency
strata; the six seeds are returned as separate assignments and must be fitted
as separate kernels, not averaged first.
"""
from typing import NamedTuple

import numpy as np

DEFAULT_RANDOM_SEEDS = (101, 202, 303, 404, 505, 606)
DEFAULT_FREQUENCY_BINS = 10


class GroupedKernel(NamedTuple):
    train: np.ndarray
    cross: np.ndarray
    metadata: dict


# -- validation and grouping ----------------------------------------------
def _validate_binary_matrix(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] < 1 or not np.isfinite(x).all():
        raise ValueError('Expected a finite material-by-marker matrix')
    return x


def _validate_ids(x, fit_ids, query_ids):
    fit = np.asarray(fit_ids, dtype=np.int64)
    query = np.asarray(query_ids, dtype=np.int64)
    if fit.ndim != 1 or query.ndim != 1 or len(fit) < 2 or len(query) < 1:
        raise ValueError('fit_ids need at least two rows and query_ids at least one')
    if len(np.unique(fit)) != len(fit) or len(np.unique(query)) != len(query):
        raise ValueError('Duplicate fit or query material IDs')
    if set(fit) & set(query):
        raise ValueError('Fit and query material IDs must be disjoint')
    if np.any(fit < 0) or np.any(query < 0) \
            or np.any(fit >= len(x)) or np.any(query >= len(x)):
        raise ValueError('Material ID out of range')
    return fit, query


def fit_group_assignment(x_fit, groups=64):
    """Correlation-anchor grouping on fit rows, replicating GenomicTabM._group."""
    x = _validate_binary_matrix(x_fit)
    groups = int(groups)
    if groups < 1:
        raise ValueError('groups must be positive')
    rows, markers = x.shape
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    constant = std <= 0
    safe = np.where(constant, 1.0, std)
    standardised = (x - mean) / safe
    standardised[:, constant] = 0.0
    candidates = np.flatnonzero(~constant)
    variance = np.square(std)
    result = {
        'groups_requested': groups, 'groups': 0,
        'anchors': [], 'assignment': np.zeros(markers, dtype=np.int64),
        'sizes': np.zeros(0, dtype=np.int64),
        'constant_markers': np.flatnonzero(constant).astype(np.int64).tolist(),
    }
    if candidates.size == 0:
        return result
    best = np.zeros(markers, dtype=np.float64)
    anchors = []
    remaining = candidates.copy()
    for _ in range(min(groups, candidates.size)):
        if not anchors:
            pick = int(remaining[np.argmax(variance[remaining])])
        else:
            pick = int(remaining[np.argmin(best[remaining])])
        anchors.append(pick)
        remaining = remaining[remaining != pick]
        correlation = (standardised.T @ standardised[:, pick]) / rows
        best = np.maximum(best, np.abs(correlation))
    anchors = np.asarray(sorted(set(anchors)), dtype=np.int64)
    anchor_correlation = (standardised.T @ standardised[:, anchors]) / rows
    assignment = np.abs(anchor_correlation).argmax(axis=1).astype(np.int64)
    assignment[anchors] = np.arange(len(anchors), dtype=np.int64)
    assignment[constant] = 0
    sizes = np.asarray([((assignment == position) & ~constant).sum()
                        for position in range(len(anchors))], dtype=np.int64)
    result.update({'groups': int(len(anchors)), 'anchors': anchors.tolist(),
                   'assignment': assignment, 'sizes': sizes})
    return result


def marker_frequencies(x_fit):
    return _validate_binary_matrix(x_fit).mean(axis=0)


def stratified_random_assignments(assignment, frequencies, seeds=DEFAULT_RANDOM_SEEDS,
                                  *, bins=DEFAULT_FREQUENCY_BINS, constant_mask=None):
    """Permute group labels inside marker-frequency strata.

    Nonconstant markers keep the original group labels within every frequency
    bin, so every random assignment preserves each group's total size and its
    count in every frequency layer. Constants stay in their original group.
    Each seed is returned separately; the caller fits all six kernels and must
    not average the kernels before fitting.
    """
    assignment = np.asarray(assignment, dtype=np.int64)
    frequencies = np.asarray(frequencies, dtype=np.float64)
    if assignment.ndim != 1 or frequencies.shape != assignment.shape:
        raise ValueError('assignment and frequencies must share one marker axis')
    if np.any(assignment < 0) or not np.isfinite(frequencies).all():
        raise ValueError('Invalid assignment or marker frequency')
    bins = int(bins)
    if bins < 1:
        raise ValueError('bins must be positive')
    if constant_mask is None:
        constant = np.zeros(len(assignment), dtype=bool)
    else:
        constant = np.asarray(constant_mask)
        if constant.dtype != bool:
            raise ValueError('constant_mask must be boolean')
        if constant.shape != assignment.shape:
            raise ValueError('constant_mask must match the marker axis')
    edges = np.linspace(0.0, 1.0, bins + 1)
    layer = np.clip(np.digitize(frequencies, edges[1:-1], right=False), 0, bins - 1)
    outputs = {}
    for seed in seeds:
        rng = np.random.default_rng(int(seed))
        shuffled = assignment.copy()
        for position in range(bins):
            indices = np.flatnonzero((layer == position) & ~constant)
            if indices.size > 1:
                shuffled[indices] = rng.permutation(assignment[indices])
        outputs[int(seed)] = shuffled
    return outputs


# -- distances and kernels -------------------------------------------------
def _squared_distances(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    distance = (np.square(left).sum(axis=1)[:, None]
                + np.square(right).sum(axis=1)[None, :]
                - 2.0 * left @ right.T)
    return np.maximum(distance, 0.0)


def _off_diagonal(distance):
    return distance[np.triu_indices(len(distance), 1)]


def _distance_scale(distance):
    values = _off_diagonal(distance)
    if values.size == 0:
        raise ValueError('At least two fit rows are required')
    scale = float(np.median(values))
    if scale <= 0:
        raise ValueError('Zero global distance median')
    return scale


def _group_weights(variance, assignment, groups, constant):
    weights = np.zeros(groups, dtype=np.float64)
    for position in range(groups):
        members = (assignment == position) & ~constant
        weights[position] = float(variance[members].sum())
    total = float(weights.sum())
    if total <= 0:
        raise ValueError('All grouped markers have zero fit variance')
    return weights / total


def _global_rbf(x_fit, x_query, bandwidth):
    distance = _squared_distances(x_fit, x_fit)
    scale = _distance_scale(distance)
    train = np.exp(-bandwidth * distance / scale)
    cross_distance = _squared_distances(x_query, x_fit)
    cross = np.exp(-bandwidth * cross_distance / scale)
    return train, cross, float(scale)


def _group_scales(global_scale, weights):
    return [float(global_scale * weight) for weight in weights]


def _grouped_rbf(x_fit, x_query, assignment, groups, bandwidth, weights, scales,
                 constant):
    rows, query_rows = len(x_fit), len(x_query)
    train = np.zeros((rows, rows), dtype=np.float64)
    cross = np.zeros((query_rows, rows), dtype=np.float64)
    for position in range(groups):
        if weights[position] == 0:
            continue
        members = np.flatnonzero((assignment == position) & ~constant)
        if members.size == 0:
            raise ValueError('Positive group weight without member markers')
        distance = _squared_distances(x_fit[:, members], x_fit[:, members])
        train += weights[position] * np.exp(
            -bandwidth * distance / scales[position])
        cross_distance = _squared_distances(x_query[:, members], x_fit[:, members])
        cross += weights[position] * np.exp(
            -bandwidth * cross_distance / scales[position])
    return train, cross


# -- public kernel ---------------------------------------------------------
def grouped_rbf_kernels(x, fit_ids, query_ids, *, bandwidth, tau=0.25, groups=64,
                        assignment=None):
    """Return the blended train/cross RBF kernels and fit-only group metadata.

    ``train`` is fit-by-fit, ``cross`` is query-by-fit (the orientation used by
    ``kernel_baselines.kernels``). ``tau=0`` reproduces the global RBF kernel
    exactly. A supplied ``assignment`` is used as-is (for random controls); the
    weights and distance scales are still re-estimated from the fit rows.
    """
    x = _validate_binary_matrix(x)
    fit, query = _validate_ids(x, fit_ids, query_ids)
    bandwidth = float(bandwidth)
    tau = float(tau)
    groups = int(groups)
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError('bandwidth must be finite and positive')
    if not 0 <= tau <= 1:
        raise ValueError('tau must be in [0, 1]')
    if groups < 1:
        raise ValueError('groups must be positive')
    x_fit = x[fit]
    x_query = x[query]
    global_train, global_cross, global_scale = _global_rbf(
        x_fit, x_query, bandwidth)
    if assignment is None:
        grouping = fit_group_assignment(x_fit, groups=groups)
        grouping_kind = 'correlation'
        anchors = grouping['anchors']
        assignment = np.asarray(grouping['assignment'], dtype=np.int64)
        group_count = grouping['groups']
    else:
        grouping_kind = 'provided'
        anchors = []
        assignment = np.asarray(assignment, dtype=np.int64)
        if assignment.ndim != 1 or len(assignment) != x.shape[1] \
                or np.any(assignment < 0):
            raise ValueError('assignment must hold one non-negative label per marker')
        group_count = int(assignment.max()) + 1 if len(assignment) else 0
    mean = x_fit.mean(axis=0)
    std = x_fit.std(axis=0)
    constant = std <= 0
    variance = np.square(std)
    weights = _group_weights(variance, assignment, group_count, constant) \
        if group_count else np.zeros(0, dtype=np.float64)
    sizes = np.asarray([((assignment == position) & ~constant).sum()
                        for position in range(group_count)], dtype=np.int64)
    if tau > 0 and group_count == 0:
        raise ValueError('tau > 0 requires at least one marker group')
    group_scales = _group_scales(global_scale, weights)
    group_train = np.zeros_like(global_train)
    group_cross = np.zeros_like(global_cross)
    if tau > 0 and group_count:
        group_train, group_cross = _grouped_rbf(
            x_fit, x_query, assignment, group_count, bandwidth, weights,
            group_scales, constant)
    train = (1.0 - tau) * global_train + tau * group_train
    cross = (1.0 - tau) * global_cross + tau * group_cross
    metadata = {
        'grouping': grouping_kind,
        'groups_requested': groups,
        'groups': int(group_count),
        'anchors': [int(value) for value in anchors],
        'assignment': [int(value) for value in assignment],
        'sizes': [int(value) for value in sizes],
        'weights': [float(value) for value in weights],
        'group_d0': group_scales,
        'group_distance_status': ['variance_scaled' if weight > 0 else 'zero_weight'
                                  for weight in weights],
        'constant_markers': np.flatnonzero(constant).astype(np.int64).tolist(),
        'd0': float(global_scale),
        'bandwidth': float(bandwidth),
        'tau': float(tau),
        'dg': 'identity',
        'weight_rule': 'sum fit marker variance in group / total fit marker variance',
        'distance_scale_rule': 'd0_g = d0 * group variance share; zero-weight groups '
                               'contribute no kernel and no epsilon is added',
        'group_members_exclude_fit_constants': True,
        'query_phenotypes_used': False,
    }
    return GroupedKernel(train=train, cross=cross, metadata=metadata)
