"""Split-construction report (NB8 / G1 input).

Outputs the raw quantities gb's audit script needs to judge split quality: per-fold
material counts, the family-size distribution, the percentage of samples in
multi-line families, and the mean train×test IBS / A per fold. This module only
reports; it does not decide whether a split is acceptable (that is gb's audit).

The report covers the relatedness_cluster protocol, where families and
train×test relatedness are meaningful. For the random protocol there are no
families, so only fold sizes are reported.
"""

from __future__ import annotations

import numpy as np

from .data import WheatData
from .splits import FoldSplit, _connected_components


def _ibs_matrix(X: np.ndarray) -> np.ndarray:
    return 1.0 - (X[:, None, :] != X[None, :, :]).mean(axis=2)


def _report_fold_relatedness(
    split: FoldSplit, X: np.ndarray, A: np.ndarray, ibs: np.ndarray
) -> dict:
    tr, va = split.outer_train, split.outer_valid
    return {
        "fold": split.fold,
        "n_valid": int(len(va)),
        "n_train": int(len(tr)),
        "mean_ibs_train_test": float(ibs[np.ix_(tr, va)].mean()),
        "mean_A_train_test": float(A[np.ix_(tr, va)].mean()),
    }


def relatedness_report(
    data: WheatData,
    splits: list[FoldSplit],
    cluster_arm: str = "ibs",
    threshold: float | None = None,
) -> dict:
    """Family structure + per-fold relatedness for a relatedness_cluster run."""
    X, A = data.X, data.A
    if cluster_arm == "ibs":
        sim = _ibs_matrix(X)
    else:
        sim = A
    if threshold is None:
        from .splits import DEFAULT_FAMILY_THRESHOLD

        threshold = DEFAULT_FAMILY_THRESHOLD[cluster_arm]

    fam = _connected_components(sim, threshold)
    n_families = int(fam.max()) + 1
    sizes = np.array([int(np.sum(fam == c)) for c in range(n_families)])
    n_multi = int(np.sum(sizes >= 2))
    pct_multi = float(100.0 * sizes[sizes >= 2].sum() / data.n)

    ibs = _ibs_matrix(X)
    folds = [_report_fold_relatedness(s, X, A, ibs) for s in splits]

    # family×fold table: which families (by size) land in which fold
    fam_to_fold = {}
    for s in splits:
        for m in s.outer_valid.tolist():
            fam_to_fold[m] = s.fold
    family_fold = {}
    for c in range(n_families):
        members = np.where(fam == c)[0]
        family_fold[int(c)] = fam_to_fold[int(members[0])]
    # histogram of family sizes per fold
    fold_family_sizes: dict[int, list[int]] = {f: [] for f in range(len(splits))}
    for c, f in family_fold.items():
        fold_family_sizes[f].append(int(sizes[c]))
    family_x_fold = {
        "per_fold_family_size_histogram": {
            str(f): {int(k): int(v) for k, v in zip(*np.unique(fold_family_sizes[f], return_counts=True))}
            for f in range(len(splits))
        },
        "multi_family_folds": {
            int(c): int(family_fold[int(c)]) for c in np.where(sizes >= 2)[0]
        },
    }

    return {
        "protocol": "relatedness_cluster",
        "cluster_arm": cluster_arm,
        "threshold": threshold,
        "n_materials": int(data.n),
        "n_families": n_families,
        "n_singletons": int(np.sum(sizes == 1)),
        "n_multi_families": n_multi,
        "max_family_size": int(sizes.max()),
        "family_size_histogram": {
            int(k): int(v) for k, v in zip(*np.unique(sizes, return_counts=True))
        },
        "pct_samples_in_multi_families": round(pct_multi, 2),
        "family_x_fold": family_x_fold,
        "folds": folds,
    }


def random_report(data: WheatData, splits: list[FoldSplit]) -> dict:
    """Per-fold sizes for the random protocol (no family structure)."""
    return {
        "protocol": "random",
        "n_materials": int(data.n),
        "folds": [
            {"fold": s.fold, "n_valid": int(len(s.outer_valid)),
             "n_train": int(len(s.outer_train))}
            for s in splits
        ],
    }
