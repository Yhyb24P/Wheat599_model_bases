"""Uncertainty for paired method comparisons (docs/RESEARCH_PLAN.md §4.3).

Two interval families, per gb's 2026-10-29 R3 ruling:

* **Primary — fold-level paired t-interval.** For each (repeat, outer-fold) we
  compute the per-fold metric difference between two methods' OOF predictions
  (15 values for 3 repeats × 5 folds). The 95% interval is the t-interval of
  those fold-level differences (df = 14). This treats each fold's OOF metric as
  one paired observation and avoids manufacturing small standard errors from
  the material count.

* **Secondary — material-level paired bootstrap.** Within each repeat,
  resample materials (or clusters for the relatedness protocol) and recompute
  the *metric difference* ``metric(y, a) - metric(y, b)`` per resample. The
  three repeats' intervals are reported separately and their CI endpoints are
  averaged; repeats are never pooled into one resampling pool.

Resampling respects cluster multiplicity: when groups are resampled with
replacement, a cluster picked ``m`` times contributes its materials ``m`` times.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

from .metrics import pcc, rmse


def _paired_metric(
    metric: Callable[[np.ndarray, np.ndarray], float],
    y: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    """metric(y, a) - metric(y, b); NaN-safe (propagates NaN)."""
    return float(metric(y, a) - metric(y, b))


def _bootstrap_distribution(
    metric: Callable[[np.ndarray, np.ndarray], float],
    y: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    group: np.ndarray | None,
    n_resamples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Bootstrap distribution of the paired metric difference.

    Resampling unit is the material (``group is None``) or the cluster/group.
    Cluster multiplicity is handled by expanding each picked group by its pick
    count, so a cluster drawn twice contributes twice.
    """
    diffs = np.empty(n_resamples, dtype=np.float64)
    if group is None:
        n = y.shape[0]
        for i in range(n_resamples):
            idx = rng.integers(0, n, size=n)
            diffs[i] = _paired_metric(metric, y[idx], a[idx], b[idx])
    else:
        group = np.asarray(group).ravel()
        uniq = np.unique(group)
        # precompute member indices per group for fast expansion
        members = {g: np.where(group == g)[0] for g in uniq}
        for i in range(n_resamples):
            pick = rng.choice(uniq, size=len(uniq), replace=True)
            idx = np.concatenate([members[g] for g in pick])
            diffs[i] = _paired_metric(metric, y[idx], a[idx], b[idx])
    return diffs


def material_bootstrap_ci(
    metric: Callable[[np.ndarray, np.ndarray], float],
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    group: np.ndarray | None = None,
    n_resamples: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """One repeat's material-level (or cluster-level) paired bootstrap CI."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    pred_a = np.asarray(pred_a, dtype=np.float64).ravel()
    pred_b = np.asarray(pred_b, dtype=np.float64).ravel()
    if not (y_true.shape == pred_a.shape == pred_b.shape):
        raise ValueError("y_true/pred_a/pred_b must have the same shape")

    rng = np.random.default_rng(seed)
    dist = _bootstrap_distribution(
        metric, y_true, pred_a, pred_b, group, n_resamples, rng
    )
    dist = dist[~np.isnan(dist)]
    point = _paired_metric(metric, y_true, pred_a, pred_b)
    if dist.size == 0:
        return {
            "metric": getattr(metric, "__name__", "metric"),
            "point": point,
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "excludes_zero": False,
            "n_resamples": n_resamples,
            "n_valid_resamples": 0,
            "seed": seed,
        }
    lo = float(np.quantile(dist, alpha / 2))
    hi = float(np.quantile(dist, 1 - alpha / 2))
    return {
        "metric": getattr(metric, "__name__", "metric"),
        "point": point,
        "ci_low": lo,
        "ci_high": hi,
        "excludes_zero": bool(lo > 0 or hi < 0),
        "n_resamples": n_resamples,
        "n_valid_resamples": int(dist.size),
        "seed": seed,
    }


def material_bootstrap_across_repeats(
    metric: Callable[[np.ndarray, np.ndarray], float],
    repeats: list[dict],
    alpha: float = 0.05,
) -> dict:
    """Average CI endpoints across repeats without pooling them.

    ``repeats`` is a list of ``material_bootstrap_ci`` result dicts (one per
    repeat). The point estimate is the mean of repeat points; CI endpoints are
    the mean of repeat endpoints. Repeats are never merged into one resampling
    pool, so the material count does not artificially shrink the interval.
    """
    if not repeats:
        raise ValueError("no repeats supplied")
    points = np.array([r["point"] for r in repeats], dtype=np.float64)
    lows = np.array([r["ci_low"] for r in repeats], dtype=np.float64)
    highs = np.array([r["ci_high"] for r in repeats], dtype=np.float64)
    lo = float(np.nanmean(lows))
    hi = float(np.nanmean(highs))
    return {
        "metric": getattr(metric, "__name__", "metric"),
        "point": float(np.nanmean(points)),
        "ci_low": lo,
        "ci_high": hi,
        "excludes_zero": bool(lo > 0 or hi < 0),
        "n_repeats": len(repeats),
        "per_repeat": repeats,
    }


def fold_level_t_interval(
    fold_diffs: np.ndarray,
    alpha: float = 0.05,
) -> dict:
    """Primary paired t-interval over fold-level metric differences.

    ``fold_diffs`` is the array of per-fold metric differences (one per outer
    fold, concatenated across repeats): length ``n_repeats * n_outer`` (e.g.
    15). The interval is the one-sample t-interval of these differences about
    their mean, with ``df = len(fold_diffs) - 1``.
    """
    fold_diffs = np.asarray(fold_diffs, dtype=np.float64).ravel()
    fold_diffs = fold_diffs[~np.isnan(fold_diffs)]
    n = fold_diffs.size
    if n < 2:
        return {
            "point": float(np.nanmean(fold_diffs)) if n else float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "excludes_zero": False,
            "n_folds": int(n),
            "df": max(n - 1, 0),
        }
    mean = float(fold_diffs.mean())
    sd = float(fold_diffs.std(ddof=1))
    df = n - 1
    # Student's t critical value via inverse survival function (no scipy dep)
    t_crit = _t_ppf(1 - alpha / 2, df)
    half = t_crit * sd / math.sqrt(n)
    lo, hi = mean - half, mean + half
    return {
        "point": mean,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "excludes_zero": bool(lo > 0 or hi < 0),
        "n_folds": int(n),
        "df": int(df),
        "sd": sd,
    }


def _t_ppf(p: float, df: int) -> float:
    """Inverse Student-t CDF (two-tailed critical value) without scipy.

    Uses the incomplete-beta inverse via a bisection on the regularized
    incomplete beta function, which is accurate enough for CI construction.
    """
    # CDF of Student's t: for t>0, P(T<=t) = 1 - 0.5 * I_{x}(df/2, 1/2),
    # where x = df/(df+t^2). We invert by bisection on t.
    from math import lgamma

    def betacf(a: float, b: float, x: float) -> float:
        MAXIT, EPS, FPMIN = 200, 3e-12, 1e-300
        qab, qap, qam = a + b, a + 1.0, a - 1.0
        c, d = 1.0, 1.0 - qab * x / qap
        if abs(d) < FPMIN:
            d = FPMIN
        d = 1.0 / d
        h = d
        for m in range(1, MAXIT + 1):
            m2 = 2 * m
            aa = m * (b - m) * x / ((qam + m2) * (a + m2))
            d = 1.0 + aa * d
            if abs(d) < FPMIN:
                d = FPMIN
            c = 1.0 + aa / c
            if abs(c) < FPMIN:
                c = FPMIN
            d = 1.0 / d
            h *= d * c
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
            d = 1.0 + aa * d
            if abs(d) < FPMIN:
                d = FPMIN
            c = 1.0 + aa / c
            if abs(c) < FPMIN:
                c = FPMIN
            d = 1.0 / d
            delta = d * c
            h *= delta
            if abs(delta - 1.0) < EPS:
                break
        return h

    def betai(a: float, b: float, x: float) -> float:
        if x <= 0.0:
            return 0.0
        if x >= 1.0:
            return 1.0
        lbeta = lgamma(a + b) - lgamma(a) - lgamma(b)
        bt = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
        if x < (a + 1.0) / (a + b + 2.0):
            return bt * betacf(a, b, x) / a
        return 1.0 - bt * betacf(b, a, 1.0 - x) / b

    def t_cdf(t: float) -> float:
        x = df / (df + t * t)
        tail = 0.5 * betai(df / 2.0, 0.5, x)
        return 1.0 - tail if t > 0 else tail

    # bisection for t such that t_cdf(t) = p
    lo, hi = 0.0, 100.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if t_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0
