"""Stage-1 gate band tests (gb grayfix ruling, frozen pre-run).

Trigger band: >=3/4 environments with mean > 0 AND cross-environment mean
>= 1.2*delta. Gray band [0.8*delta, 1.2*delta): recorded, never triggers.
Negative logging: >=3/4 environments with mean <= -delta AND cross <= -delta.
OR rule across metrics.
"""

from __future__ import annotations

import numpy as np

from wheat599.gate import (
    GRAY_LO_FACTOR,
    TRIGGER_FACTOR,
    aggregate_signal,
    evaluate_comparison,
)

ENV4 = ["1", "2", "4", "5"]
DELTA = {"pcc": 0.03, "rmse": 0.05}


def _comp(env_means: dict[str, float], metric: str = "pcc"):
    """Fold diffs as a constant list per env so env mean == env_means[env]."""
    by_env = {e: {metric: [v, v, v]} for e, v in env_means.items()}
    return evaluate_comparison("T1:x-ref", by_env, DELTA)


def _agg(by_metric):
    return aggregate_signal(by_metric)


def test_trigger_band_requires_1p2_delta_cross_mean():
    # cross mean = 0.0475 >= 1.2*0.03 = 0.036, 4/4 positive -> TRIGGER
    res = _comp({e: 0.0475 for e in ENV4})
    agg = _agg(res)
    assert agg["trigger"] and agg["decision"] == "TRIGGER"
    assert agg["triggered_metrics"] == ["pcc"]


def test_gray_band_does_not_trigger_but_records():
    # 3/4 positive (0.04), one negative -> cross = (0.04*3-0.01)/4 = 0.0275
    # 0.8*0.03=0.024 <= 0.0275 < 0.036 -> gray trace, NO trigger
    res = _comp({"1": 0.04, "2": 0.04, "4": 0.04, "5": -0.01})
    agg = _agg(res)
    assert not agg["trigger"]
    assert agg["decision"] == "GRAY_TRACE"
    assert agg["gray_trace_metrics"] == ["pcc"]
    assert agg["per_metric"]["pcc"]["n_envs_positive"] == 3


def test_exactly_two_positive_envs_does_not_trigger():
    res = _comp({"1": 0.05, "2": 0.05, "4": -0.001, "5": -0.001})
    agg = _agg(res)
    assert not agg["trigger"]
    assert agg["per_metric"]["pcc"]["n_envs_positive"] == 2


def test_negative_logging():
    res = _comp({e: -0.05 for e in ENV4[:3]} | {"5": -0.02})
    agg = _agg(res)
    assert not agg["trigger"]
    assert agg["per_metric"]["pcc"]["negative_signal"]
    assert agg["decision"] == "NEGATIVE"


def test_or_rule_other_metric_can_trigger_alone():
    pcc_means = {e: 0.0475 for e in ENV4}          # would trigger on pcc too —
    # make a case where ONLY rmse passes: pcc cross 0.03 (<0.036 gray), rmse cross 0.08 (>=0.06)
    by_env_pcc = {e: {"pcc": [0.03, 0.03, 0.03]} for e in ENV4}
    by_env_rmse = {e: {"rmse": [0.08, 0.08, 0.08]} for e in ENV4}
    merged = {e: {"pcc": by_env_pcc[e]["pcc"], "rmse": by_env_rmse[e]["rmse"]}
              for e in ENV4}
    res = evaluate_comparison("T2:y-ref", merged, DELTA)
    agg = _agg(res)
    assert agg["trigger"]
    assert agg["triggered_metrics"] == ["rmse"]
    assert agg["per_metric"]["pcc"]["gray_trace"]  # 0.03 in [0.024, 0.036)
    assert pcc_means  # (kept explicit for readability)


def test_band_constants_match_ruling():
    assert TRIGGER_FACTOR == 1.2 and GRAY_LO_FACTOR == 0.8


# ---- fixed pre-run delta (takeover deltafix, 2026-10-29) ------------------

from wheat599.gate import resolve_deltas  # noqa: E402


def test_resolve_deltas_fixed_and_ignores_fold_diffs():
    """Mutating the fold diffs must not change the gate delta."""
    fixed = resolve_deltas({"pcc": 0.03, "rmse": 0.05})
    assert fixed["pcc"] == {"delta": 0.03, "source": "config"}
    assert fixed["rmse"] == {"delta": 0.05, "source": "config"}
    deltas = {m: v["delta"] for m, v in fixed.items()}
    r1 = evaluate_comparison("T1:x", {e: {"pcc": [0.001] * 15} for e in ENV4}, deltas)
    r2 = evaluate_comparison("T1:x", {e: {"pcc": [0.4] * 15} for e in ENV4}, deltas)
    assert all(g.delta == 0.03 for g in r1["pcc"])
    assert all(g.delta == 0.03 for g in r2["pcc"])


def test_resolve_deltas_fallback_source():
    out = resolve_deltas({"pcc": None, "rmse": 0.05})
    assert out["pcc"] == {"delta": 0.03, "source": "fallback"}
    assert out["rmse"] == {"delta": 0.05, "source": "config"}


def test_resolve_deltas_does_not_accept_fold_diffs():
    import inspect
    params = inspect.signature(resolve_deltas).parameters
    assert "fold_diffs" not in params and "pooled" not in params
