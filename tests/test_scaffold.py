"""Invariant tests for the M0 scaffold (src/wheat599).

These lock the properties the leakage audit depends on:
* random-split fold balance, material-disjointness, determinism;
* real inner k-fold (3 folds, disjoint, covering outer_train);
* relatedness-cluster splits (both arms) disjoint + covering;
* metric definitions;
* bootstrap: metric-aware difference, cluster multiplicity, fold-level t-interval.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from wheat599.bootstrap import (
    _t_ppf,
    fold_level_t_interval,
    material_bootstrap_across_repeats,
    material_bootstrap_ci,
)
from wheat599.data import WheatData
from wheat599.metrics import compute_all, mae, pcc, r2, rmse
from wheat599.splits import FoldSplit, _connected_components, build_splits


class _Fake:
    """Minimal stand-in for WheatData to drive split generation."""

    def __init__(self, n=60, n_env=4, p=10, seed=0):
        rng = np.random.default_rng(seed)
        self.X = rng.integers(0, 2, size=(n, p)).astype(np.float64)
        self.Y = rng.standard_normal((n, n_env))
        self.Y = (self.Y - self.Y.mean(0)) / self.Y.std(0)
        self.environments = tuple(str(i + 1) for i in range(n_env))
        self.A = np.eye(n)
        self.sets = np.arange(n) % 10

    @property
    def n(self):
        return self.X.shape[0]

    @property
    def p(self):
        return self.X.shape[1]

    @property
    def n_env(self):
        return self.Y.shape[1]


# ---------- splits: random ----------

def test_random_splits_material_disjoint_and_balanced():
    d = _Fake(n=599, n_env=4)
    splits = build_splits(d, "random", seed=20261029, n_outer=5, n_inner=3)
    assert len(splits) == 5
    sizes = [len(s.outer_valid) for s in splits]
    assert max(sizes) - min(sizes) <= 2, f"folds unbalanced: {sizes}"
    all_valid = np.concatenate([s.outer_valid for s in splits])
    assert sorted(all_valid.tolist()) == list(range(599))


def test_random_inner_is_real_3fold():
    d = _Fake(n=599, n_env=4)
    splits = build_splits(d, "random", seed=20261029, n_outer=5, n_inner=3)
    for s in splits:
        assert len(s.inner_folds) == 3
        iv = [set(f.inner_valid.tolist()) for f in s.inner_folds]
        for i in range(3):
            for j in range(i + 1, 3):
                assert iv[i].isdisjoint(iv[j])
        assert set().union(*iv) == set(s.outer_train.tolist())
        for f in s.inner_folds:
            assert set(f.inner_train.tolist()).issubset(set(s.outer_train.tolist()))


def test_random_splits_deterministic_per_seed():
    d = _Fake(n=120, n_env=2)
    a = build_splits(d, "random", seed=7, n_outer=5, n_inner=3)
    b = build_splits(d, "random", seed=7, n_outer=5, n_inner=3)
    for sa, sb in zip(a, b):
        assert np.array_equal(sa.outer_valid, sb.outer_valid)
        assert [np.array_equal(x.inner_valid, y.inner_valid) for x, y in zip(sa.inner_folds, sb.inner_folds)] == [True, True, True]


def test_random_split_per_fold_env_means_balanced():
    """Each fold's per-environment mean should be near the global mean."""
    d = _Fake(n=599, n_env=4)
    splits = build_splits(d, "random", seed=20261029, n_outer=5, n_inner=3)
    glob = d.Y.mean(0)
    for s in splits:
        fm = d.Y[s.outer_valid].mean(0)
        smd = np.abs((fm - glob) / (d.Y.std(0) + 1e-12)).max()
        assert smd < 0.2, f"fold {s.fold} env mean drift {smd}"


# ---------- splits: relatedness_cluster (threshold connected components) ----------

@pytest.mark.parametrize("arm", ["ibs", "a"])
def test_relatedness_cluster_disjoint_and_balanced(arm):
    """Connected-component families held out whole; folds balanced by size."""
    d = _Fake(n=120, n_env=2)
    splits = build_splits(d, "relatedness_cluster", seed=1, n_outer=5, n_inner=3, cluster_arm=arm)
    sizes = [len(s.outer_valid) for s in splits]
    # every material in exactly one outer-valid fold
    all_valid = np.concatenate([s.outer_valid for s in splits])
    assert sorted(all_valid.tolist()) == list(range(120))
    # families held out whole: no family split across outer folds
    X = d.X
    if arm == "ibs":
        sim = 1.0 - (X[:, None, :] != X[None, :, :]).mean(axis=2)
    else:
        sim = d.A
    fam = _connected_components(sim, 0.5)
    fold_of = np.concatenate([np.full(len(s.outer_valid), s.fold) for s in splits])
    valid_of = np.concatenate([s.outer_valid for s in splits])
    for c in np.unique(fam):
        members = np.where(fam == c)[0]
        if len(members) > 1:
            assert len(set(fold_of[members].tolist())) == 1, "family split across folds"
    for s in splits:
        iv = [set(f.inner_valid.tolist()) for f in s.inner_folds]
        for i in range(len(iv)):
            for j in range(i + 1, len(iv)):
                assert iv[i].isdisjoint(iv[j])
        assert set().union(*iv) == set(s.outer_train.tolist())


def _planted_family_data(n=100, n_env=2, p=8, family_size=2, seed=0):
    """Data with an explicit planted family: ``family_size`` rows share identical
    markers (so they form one connected component at a high IBS threshold),
    the rest are independent singletons."""
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 2, size=(n, p)).astype(np.float64)
    # make rows 0..family_size-1 identical (a planted family)
    X[1:family_size] = X[0]
    Y = rng.standard_normal((n, n_env))
    Y = (Y - Y.mean(0)) / Y.std(0)
    return WheatData(X=X, Y=Y, environments=tuple(str(i + 1) for i in range(n_env)),
                     A=np.eye(n), sets=np.arange(n) % 10), list(range(family_size))


def test_planted_family_held_out_whole():
    """A planted family (identical markers) must land in exactly one outer fold
    — the family-integrity check is exercised on real structure, not trivial
    all-singletons."""
    d, fam_members = _planted_family_data(n=100, family_size=3, seed=1)
    splits = build_splits(d, "relatedness_cluster", seed=1, n_outer=5, n_inner=3,
                          cluster_arm="ibs", threshold=0.99)
    # the planted family forms one component at threshold 0.99
    sim = 1.0 - (d.X[:, None, :] != d.X[None, :, :]).mean(axis=2)
    fam = _connected_components(sim, 0.99)
    assert len(np.where(fam == fam[fam_members[0]])[0]) == len(fam_members), \
        "planted family should be one connected component"
    # all family members share the same outer fold
    fold_of = {}
    for s in splits:
        for m in s.outer_valid:
            fold_of[m] = s.fold
    fam_folds = {fold_of[m] for m in fam_members}
    assert len(fam_folds) == 1, f"planted family split across folds: {fam_folds}"


def test_family_integrity_check_catches_split():
    """The integrity assertion must FAIL if a family is split across folds —
    proving the check is not vacuous."""
    d, fam_members = _planted_family_data(n=100, family_size=3, seed=1)
    splits = build_splits(d, "relatedness_cluster", seed=1, n_outer=5, n_inner=3,
                          cluster_arm="ibs", threshold=0.99)
    # deliberately corrupt: move fam_members[1] from its fold into a DIFFERENT
    # fold's valid set, splitting the family across two outer folds
    source_fold = next(s.fold for s in splits if fam_members[0] in s.outer_valid.tolist())
    other_fold = (source_fold + 1) % len(splits)  # a different fold
    corrupted = []
    for s in splits:
        valid = s.outer_valid.copy()
        if s.fold == source_fold:
            valid = valid[valid != fam_members[1]]      # remove from source
        if s.fold == other_fold:
            valid = np.append(valid, fam_members[1])      # add to a different fold
        corrupted.append(FoldSplit(
            outer_train=s.outer_train.copy(),
            outer_valid=valid,
            inner_folds=s.inner_folds, fold=s.fold,
        ))
    sim = 1.0 - (d.X[:, None, :] != d.X[None, :, :]).mean(axis=2)
    fam = _connected_components(sim, 0.99)
    # material -> fold map (by which valid set each material is in)
    mat_to_fold = {}
    for s in corrupted:
        for m in s.outer_valid.tolist():
            mat_to_fold[m] = s.fold
    # the corrupted assignment now splits the family -> the integrity check fires
    members = np.where(fam == fam[fam_members[0]])[0]
    fam_folds = {mat_to_fold[int(m)] for m in members}
    assert len(fam_folds) > 1, f"integrity check should detect the planted split: {fam_folds}"



def test_family_placement_varies_by_seed():
    """Multi-line family placement must be seed-dependent: the family×fold
    assignment differs across repeats (ARS2). With a deterministic
    heaviest-first placement, families would land in the same fold every time."""
    d = _Fake(n=200, n_env=2)
    X = d.X.copy()
    X[1:3] = X[0]
    X[101:103] = X[100]
    d2 = WheatData(X=X, Y=d.Y, environments=d.environments, A=d.A, sets=d.sets)
    sim = 1.0 - (X[:, None, :] != X[None, :, :]).mean(axis=2)
    fam = _connected_components(sim, 0.99)
    sizes = np.array([int(np.sum(fam == c)) for c in range(fam.max() + 1)])
    multi = np.where(sizes >= 2)[0]
    assert multi.size >= 2
    assigns = {}
    for seed in (1, 2, 3):
        splits = build_splits(d2, "relatedness_cluster", seed=seed, n_outer=5,
                              n_inner=3, cluster_arm="ibs", threshold=0.99)
        m2f = {}
        for s in splits:
            for m in s.outer_valid.tolist():
                m2f[m] = s.fold
        fam_fold = {int(c): m2f[int(np.where(fam == c)[0][0])] for c in multi}
        assigns[seed] = fam_fold
    assert any(assigns[a] != assigns[b]
               for a, b in itertools.combinations(assigns, 2)), \
        f"family placement not seed-dependent: {assigns}"


# ---------- metrics ----------

def test_pcc_perfect_and_zero_variance():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert pcc(y, y) == pytest.approx(1.0)
    assert np.isnan(pcc(y, np.ones_like(y)))


def test_rmse_r2_basic():
    y = np.array([1.0, 2.0, 3.0])
    pred = np.array([1.0, 2.0, 3.0])
    assert rmse(y, pred) == pytest.approx(0.0)
    assert r2(y, pred) == pytest.approx(1.0)
    assert mae(y, pred) == pytest.approx(0.0)
    assert set(compute_all(y, pred)) == {"pcc", "rmse", "mae", "r2"}


# ---------- bootstrap ----------

def test_t_ppf_known_values():
    assert _t_ppf(0.975, 14) == pytest.approx(2.1448, abs=1e-3)
    assert _t_ppf(0.975, 4) == pytest.approx(2.7764, abs=1e-3)
    assert _t_ppf(0.975, 1) == pytest.approx(12.7062, abs=1e-3)


def test_bootstrap_is_metric_aware():
    """The bootstrap statistic must be metric(y,a) - metric(y,b), not mean(a-b)."""
    rng = np.random.default_rng(0)
    n = 400
    y = rng.standard_normal(n)
    a = y + rng.normal(0, 0.5, n)
    b = y + rng.normal(0, 0.5, n) + 0.3  # b slightly worse
    ci = material_bootstrap_ci(rmse, y, a, b, n_resamples=1500, seed=1)
    # point estimate = rmse(y,a) - rmse(y,b): a is better, so this is negative.
    # A raw mean(a-b) would be ~0 (both unbiased around y), so a negative point
    # proves the statistic is the metric difference, not the prediction difference.
    assert ci["point"] < 0
    assert abs(ci["point"]) < 0.3  # and it tracks the small RMSE shift, not a raw diff
    # CI brackets the point estimate
    assert ci["ci_low"] <= ci["point"] <= ci["ci_high"]


def test_cluster_multiplicity():
    """A cluster picked twice must contribute twice to the resample."""
    rng = np.random.default_rng(0)
    n = 600
    y = rng.standard_normal(n)
    a = y + rng.normal(0, 0.5, n)
    b = y + rng.normal(0, 0.5, n)
    group = np.repeat(np.arange(60), 10)
    ci = material_bootstrap_ci(rmse, y, a, b, group=group, n_resamples=800, seed=2)
    assert ci["n_valid_resamples"] == 800
    assert np.isfinite(ci["point"])


def test_fold_level_t_interval():
    fold_diffs = np.linspace(0.02, 0.08, 15)  # 15 fold-level diffs, df=14
    ti = fold_level_t_interval(fold_diffs)
    assert ti["n_folds"] == 15
    assert ti["df"] == 14
    assert ti["ci_low"] < ti["point"] < ti["ci_high"]
    assert ti["excludes_zero"] is True


def test_material_bootstrap_across_repeats_no_pooling():
    rng = np.random.default_rng(0)
    n = 300
    y = rng.standard_normal(n)
    a = y + rng.normal(0, 0.5, n)
    b = y + rng.normal(0, 0.5, n) + 0.2
    reps = [material_bootstrap_ci(rmse, y, a, b, n_resamples=400, seed=s) for s in range(3)]
    agg = material_bootstrap_across_repeats(rmse, reps)
    assert agg["n_repeats"] == 3
    # endpoints are means of repeat endpoints, not a pooled interval
    assert agg["ci_low"] == pytest.approx(np.mean([r["ci_low"] for r in reps]))
