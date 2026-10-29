"""Cross-validation split generation.

Two protocols are defined in docs/RESEARCH_PLAN.md §4.1:

* ``random`` — material-level stratified 5-fold outer, 3-fold inner, repeated
  with fixed seeds. Implemented here.
* ``relatedness_cluster`` — threshold connected-components: a "family" is a
  connected component of materials whose pairwise similarity meets a threshold
  (IBS default 0.8, A default 1.5; threshold is a parameter). Multi-member
  families are held out whole; singletons are distributed by size balance.
  Cluster-level 5-fold outer / 3-fold inner. The earlier K=10 + bin-packing
  scheme was abandoned (it produced one giant family that could not be split
  into balanced folds). Implemented here (see ``_relatedness_split``).

Design notes (per the 2026-10-29 G0 review):
* The random-protocol stratification key is a *single* categorical: the
  5-quantile bin of the row-mean of the standardized yields. A high-cardinality
  key (per-environment quartile × marker-mean quartile) produced hundreds of
  single-sample strata that all collapsed into fold 0; one coarse key avoids
  that while still keeping the yield marginal roughly even across folds.
* Within each stratum, members are dealt to folds with a *random* per-stratum
  offset (``fold = (offset + i) % n_folds``) so single- or few-member strata do
  not all land in fold 0.
* Each ``FoldSplit`` carries the full list of inner folds (``inner_folds``), so
  model selection can loop over all inner folds within ``outer_train`` and
  average the validation metric — a real k-fold, not a single holdout.

Every split is material-disjoint across outer folds and across inner folds, and
the assignment is fully determined by ``(protocol, seed, n_folds, n_env)`` so
runs are reproducible and auditable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import WheatData


@dataclass(frozen=True)
class InnerFold:
    """One inner fold: a train/valid partition of an outer fold's train set."""

    inner_train: np.ndarray  # indices into the full data
    inner_valid: np.ndarray  # indices into the full data


@dataclass(frozen=True)
class FoldSplit:
    """One outer fold's train/valid materials plus the inner k-fold of train."""

    outer_train: np.ndarray  # indices into the full data
    outer_valid: np.ndarray
    inner_folds: list[InnerFold]
    fold: int

    # Back-compat accessors: the first inner fold's partition. Model selection
    # must use ``inner_folds`` (all of them), not these.
    @property
    def inner_train(self) -> np.ndarray:
        return self.inner_folds[0].inner_train

    @property
    def inner_valid(self) -> np.ndarray:
        return self.inner_folds[0].inner_valid


def _row_mean_bins(Y: np.ndarray, n_bins: int = 5) -> np.ndarray:
    """Single stratification key: 5-quantile bin of the per-material row-mean.

    ``Y`` is already standardized per environment (mean 0, sd 1), so the row
    mean is a comparable scalar across materials. Rank-based bins are robust to
    ties; ``np.searchsorted`` on the row-mean values gives quantile edges.
    """
    row_mean = Y.mean(axis=1)
    edges = np.quantile(row_mean, np.linspace(0, 1, n_bins + 1)[1:-1])
    return np.searchsorted(edges, row_mean, side="right").astype(np.int64)


def _stratified_assign(
    keys: np.ndarray, n_folds: int, rng: np.random.Generator
) -> np.ndarray:
    """Assign each material to a fold, stratified by ``keys``.

    Returns a ``fold_of`` array. Within each stratum, members are shuffled and
    dealt round-robin starting from a *random* offset, so few-member strata do
    not concentrate in fold 0.
    """
    n = keys.shape[0]
    fold_of = np.empty(n, dtype=np.int64)
    for stratum in np.unique(keys):
        members = np.where(keys == stratum)[0]
        rng.shuffle(members)
        offset = int(rng.integers(n_folds))
        for i, idx in enumerate(members):
            fold_of[idx] = (offset + i) % n_folds
    return fold_of


def _fold_indices_from_assignment(fold_of: np.ndarray, n_folds: int) -> list[np.ndarray]:
    return [np.where(fold_of == f)[0] for f in range(n_folds)]


def _random_split(
    data: WheatData,
    seed: int,
    n_outer: int,
    n_inner: int,
) -> list[FoldSplit]:
    keys = _row_mean_bins(data.Y, n_bins=5)

    rng_outer = np.random.default_rng(seed)
    outer_assign = _stratified_assign(keys, n_outer, rng_outer)
    outer_folds = _fold_indices_from_assignment(outer_assign, n_outer)

    splits: list[FoldSplit] = []
    for fold, valid in enumerate(outer_folds):
        train_mask = np.ones(data.n, dtype=bool)
        train_mask[valid] = False
        outer_train = np.where(train_mask)[0]

        # inner k-fold: stratified on the same key, restricted to outer_train
        inner_keys = keys[outer_train]
        rng_inner = np.random.default_rng(seed + 1_000_003 * fold)
        inner_assign = _stratified_assign(inner_keys, n_inner, rng_inner)
        inner_local = _fold_indices_from_assignment(inner_assign, n_inner)
        inner_folds = [
            InnerFold(
                inner_train=outer_train[np.sort(np.concatenate(inner_local[:k] + inner_local[k + 1 :]))],
                inner_valid=outer_train[inner_local[k]],
            )
            for k in range(n_inner)
        ]
        splits.append(
            FoldSplit(
                outer_train=outer_train,
                outer_valid=valid,
                inner_folds=inner_folds,
                fold=fold,
            )
        )
    return splits


# Default family thresholds (R2 skeleton, pending gb's ruling). A "family" is a
# connected component of materials whose pairwise similarity meets/exceeds the
# threshold. Multi-member families are held out whole; singletons are
# distributed by size balance. Clustering uses X/A only — never Y.
DEFAULT_FAMILY_THRESHOLD = {"ibs": 0.8, "a": 1.5}


def _relatedness_split(
    data: WheatData,
    seed: int,
    n_outer: int,
    n_inner: int,
    cluster_arm: str = "ibs",
    threshold: float | None = None,
) -> list[FoldSplit]:
    """Cluster-level splits via threshold connected components (R2 skeleton).

    ``cluster_arm`` in {"ibs", "a"}:
      * "ibs" — similarity = IBS on the binary markers (1 - Hamming/p).
      * "a"   — similarity = A (additive relationship matrix).
    ``threshold`` is the family-joining similarity (defaults: IBS 0.8, A 1.5);
    it is a parameter so gb can rule without a code change. Families (connected
    components at >= threshold) are held out whole; singletons are distributed to
    folds by size balance. The inner 3-fold splits whole families. A only
    constrains the split; it never enters a model.
    """
    X = data.X
    n, p = X.shape
    if cluster_arm == "ibs":
        sim = 1.0 - (X[:, None, :] != X[None, :, :]).mean(axis=2)
    elif cluster_arm == "a":
        sim = data.A
    else:
        raise ValueError(f"unknown cluster_arm: {cluster_arm!r}")
    if threshold is None:
        threshold = DEFAULT_FAMILY_THRESHOLD[cluster_arm]

    family_of = _connected_components(sim, threshold)
    n_families = int(family_of.max()) + 1
    sizes = np.array([int(np.sum(family_of == c)) for c in range(n_families)])

    rng = np.random.default_rng(seed)
    multi = np.where(sizes >= 2)[0]   # multi-line families
    single = np.where(sizes == 1)[0]  # singletons

    # Place multi-line families whole into the lightest fold. To keep the
    # assignment seed-dependent (so families do not land in the same fold
    # across repeats), shuffle the family order with the seed before the
    # heaviest-first pass, and break fold ties by a seeded random offset.
    multi_order = multi.copy()
    rng.shuffle(multi_order)
    # within the shuffled order, still place heavier families first (stable by
    # size descending), but the shuffle randomizes equal-size ordering
    multi_order = sorted(multi_order.tolist(), key=lambda c: -sizes[c])
    # re-shuffle within equal-size groups deterministically
    by_size: dict[int, list[int]] = {}
    for c in multi_order:
        by_size.setdefault(int(sizes[c]), []).append(c)
    multi_order = []
    for size in sorted(by_size, reverse=True):
        group = by_size[size]
        rng.shuffle(group)
        multi_order.extend(group)

    fold_of = np.empty(n, dtype=np.int64)
    fold_load = np.zeros(n_outer, dtype=np.int64)
    fold_families: list[list[int]] = [[] for _ in range(n_outer)]
    for c in multi_order:
        # lightest fold; tie-break by seeded random offset so equal-load folds
        # are chosen differently across seeds
        min_load = fold_load.min()
        candidates = np.where(fold_load == min_load)[0]
        offset = int(rng.integers(len(candidates)))
        f = int(candidates[offset])
        fold_of[np.where(family_of == c)[0]] = f
        fold_load[f] += sizes[c]
        fold_families[f].append(c)

    # Distribute singletons by size balance (round-robin over lightest folds,
    # with a random offset so the assignment is seed-dependent but balanced).
    if single.size:
        order = single.copy()
        rng.shuffle(order)
        for c in order:
            f = int(np.argmin(fold_load))
            fold_of[np.where(family_of == c)[0]] = f
            fold_load[f] += 1
            fold_families[f].append(int(c))

    # inner 3-fold: split whole families of the train set into 3 groups,
    # balancing by family size, never splitting a family across inner folds.
    splits: list[FoldSplit] = []
    for fold in range(n_outer):
        valid = np.where(fold_of == fold)[0]
        outer_train = np.where(fold_of != fold)[0]
        train_families = np.unique(family_of[outer_train])
        fam_sizes = np.array([int(np.sum(family_of == c)) for c in train_families])
        rng_inner = np.random.default_rng(seed + 2_000_033 * fold)
        inner_assign = _stratified_assign(fam_sizes, n_inner, rng_inner)
        fam_fold = {int(c): int(inner_assign[i]) for i, c in enumerate(train_families)}
        inner_local: list[list[int]] = [[] for _ in range(n_inner)]
        for c in train_families:
            members = np.where(family_of == c)[0]
            members = members[fold_of[members] != fold]  # exclude outer-valid
            inner_local[fam_fold[int(c)]].extend(members.tolist())
        outer_train_set = set(outer_train.tolist())
        inner_folds = []
        for k in range(n_inner):
            iv = np.array(inner_local[k], dtype=np.int64)
            inner_folds.append(
                InnerFold(
                    inner_train=np.array(sorted(outer_train_set - set(iv.tolist())), dtype=np.int64),
                    inner_valid=iv,
                )
            )
        splits.append(
            FoldSplit(outer_train=outer_train, outer_valid=valid,
                      inner_folds=inner_folds, fold=fold)
        )
    return splits


def _connected_components(sim: np.ndarray, threshold: float) -> np.ndarray:
    """Connected components of the graph with edges where sim >= threshold.

    Iterative DFS (no recursion limit issues for n=599). Returns a label array
    in 0..k-1 by first appearance.
    """
    n = sim.shape[0]
    adj = sim >= threshold
    np.fill_diagonal(adj, False)
    labels = np.full(n, -1, dtype=np.int64)
    next_label = 0
    for start in range(n):
        if labels[start] != -1:
            continue
        stack = [start]
        labels[start] = next_label
        while stack:
            u = stack.pop()
            for v in np.where(adj[u])[0]:
                if labels[v] == -1:
                    labels[v] = next_label
                    stack.append(int(v))
        next_label += 1
    return labels


def build_splits(
    data: WheatData,
    protocol: str,
    seed: int,
    n_outer: int = 5,
    n_inner: int = 3,
    cluster_arm: str = "ibs",
    threshold: float | None = None,
) -> list[FoldSplit]:
    """Build the outer/inner split list for one protocol and seed.

    ``cluster_arm`` selects the relatedness-cluster similarity ("ibs" or "a");
    ``threshold`` overrides the family-joining similarity (defaults: IBS 0.8,
    A 1.5). Both are ignored for the random protocol.
    """
    if protocol == "random":
        return _random_split(data, seed, n_outer, n_inner)
    if protocol == "relatedness_cluster":
        return _relatedness_split(
            data, seed, n_outer, n_inner, cluster_arm=cluster_arm, threshold=threshold
        )
    raise ValueError(f"unknown split protocol: {protocol!r}")
