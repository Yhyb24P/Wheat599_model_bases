"""gb signal gate for Stage 1 (configs/stage1.yaml ``gate``).

Decides whether the observed paired differences between a primary model and the
Ridge reference constitute a "signal" worth escalating. The gate is deliberately
conservative:

* The unit is the fold-level paired difference (one per outer fold, per
  environment). For 3 repeats x 5 folds = 15 values per (comparison, env).
* The primary interval is the t-interval of those 15 values (df=14), reported
  with a fixed caveat that it does not adjust for material reuse / fold
  correlation (optimistic; magnitude only, not a stable-improvement claim).
* ``delta`` is the smoke-estimated detectability scale of the fold-level
  paired difference
  (delta = t(0.975,14) * SD / sqrt(15)), fallback PCC 0.03 / RMSE 0.05.
* Trigger band (gb grayfix ruling, frozen before the run): a comparison
  TRIGGERS if >=3/4 environments have mean > 0 AND the cross-environment mean
  >= 1.2*delta (upper edge of the gray band; "almost" must not become a
  narrative). Bands for per-env labels and logging: positive if
  mean >= 1.2*delta, gray/weak-trace if 0.8*delta <= mean < 1.2*delta
  (recorded, never triggering), negative if mean <= -delta.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .bootstrap import _t_ppf

TRIGGER_FACTOR = 1.2
GRAY_LO_FACTOR = 0.8


@dataclass
class GateResult:
    comparison: str  # e.g. "T1:tabpfn_v3_5-ridge"
    environment: str
    fold_diffs: list[float]  # the 15 (or n) fold-level paired differences
    mean: float
    min: float
    max: float
    df: int
    ci_low: float
    ci_high: float
    excludes_zero: bool
    delta: float
    crosses_delta: bool
    gray_zone: bool
    signal: str  # "positive" | "negative" | "gray" | "none"


def estimate_delta(fold_diffs: list[float], metric: str, fallback: float) -> float:
    """Estimate delta from smoke fold-level paired-difference SD (DIAGNOSTIC).

    delta = t(0.975, n-1) * SD / sqrt(n). Falls back to ``fallback`` when there
    are too few values or SD is zero. NOTE: the value returned here is a
    pre-run calibration aid only; the gate itself must use ``resolve_deltas``
    (fixed values) — never a delta estimated from the formal run's own folds.
    """
    arr = np.asarray([d for d in fold_diffs if not math.isnan(d)], dtype=np.float64)
    if arr.size < 3:
        return fallback
    sd = float(arr.std(ddof=1))
    if sd == 0:
        return fallback
    t_crit = _t_ppf(0.975, arr.size - 1)
    return round(t_crit * sd / math.sqrt(arr.size), 4)


def resolve_deltas(
    cfg_deltas: dict | None,
    fallbacks: dict | None = None,
) -> dict[str, dict]:
    """FIXED pre-run delta (gate rule 1: frozen before the formal run).

    ``cfg_deltas`` = {"pcc": value-or-None, "rmse": value-or-None} from the
    run config; missing metrics use the stated fallback. Returns
    metric -> {"delta": float, "source": "config" | "fallback"}.

    This function takes NO fold differences by design: the formal run's own
    fold-level differences must not influence the gate threshold (post-hoc
    self-calibration is the exact failure mode the pre-registration prevents).
    ``estimate_delta`` output may be recorded as a diagnostic, never fed here.
    """
    fallbacks = fallbacks or {"pcc": 0.03, "rmse": 0.05}
    out: dict[str, dict] = {}
    for metric, fb in fallbacks.items():
        v = (cfg_deltas or {}).get(metric)
        if v is None:
            out[metric] = {"delta": float(fb), "source": "fallback"}
        else:
            out[metric] = {"delta": float(v), "source": "config"}
    return out


def evaluate_comparison(
    comparison: str,
    fold_diffs_by_env: dict[str, dict[str, list[float]]],
    deltas: dict[str, float],
    trigger_factor: float = TRIGGER_FACTOR,
    gray_lo_factor: float = GRAY_LO_FACTOR,
) -> dict[str, list[GateResult]]:
    """Evaluate one comparison across environments and metrics.

    ``fold_diffs_by_env`` maps environment -> metric -> list of fold-level
    paired differences (e.g. {"1": {"pcc": [...], "rmse": [...]}}). Positive
    always means "model better than reference": for PCC that is
    PCC(model) - PCC(ref); for RMSE it is RMSE(ref) - RMSE(model).
    ``deltas`` maps metric -> delta (PCC: smoke-estimated, fallback 0.03;
    RMSE: 0.05). Per-env band labels (reporting only — the TRIGGER is decided
    on the cross-environment mean in ``aggregate_signal``):
    positive: mean >= trigger_factor*delta; gray: gray_lo_factor*delta <=
    mean < trigger_factor*delta; negative: mean <= -delta; else none.
    Returns metric -> list of GateResult.
    """
    by_metric: dict[str, list[GateResult]] = {}
    # fold_diffs_by_env is {env: {metric: diffs}}; transpose to {metric: {env: diffs}}
    envs = list(fold_diffs_by_env.keys())
    metrics: list[str] = []
    for env in envs:
        for m in fold_diffs_by_env[env]:
            if m not in metrics:
                metrics.append(m)
    for metric in metrics:
        delta = deltas.get(metric, 0.03)
        trigger_band = trigger_factor * delta
        gray_lo = gray_lo_factor * delta
        results = []
        for env in envs:
            diffs = fold_diffs_by_env.get(env, {}).get(metric, [])
            arr = np.asarray([d for d in diffs if not math.isnan(d)], dtype=np.float64)
            n = arr.size
            if n < 2:
                continue
            mean = float(arr.mean())
            df = n - 1
            sd = float(arr.std(ddof=1))
            t_crit = _t_ppf(0.975, df)
            half = t_crit * sd / math.sqrt(n) if n > 0 else float("nan")
            crosses = mean >= trigger_band
            gray = gray_lo <= mean < trigger_band
            if mean > 0 and crosses:
                signal = "positive"
            elif mean <= -delta:
                signal = "negative"
            elif gray:
                signal = "gray"
            else:
                signal = "none"
            results.append(GateResult(
                comparison=comparison, environment=env,
                fold_diffs=[float(x) for x in arr],
                mean=mean, min=float(arr.min()), max=float(arr.max()),
                df=int(df),
                ci_low=float(mean - half), ci_high=float(mean + half),
                excludes_zero=bool((mean - half) > 0 or (mean + half) < 0),
                delta=delta, crosses_delta=crosses, gray_zone=gray, signal=signal,
            ))
        by_metric[metric] = results
    return by_metric


def aggregate_signal(
    by_metric: dict[str, list[GateResult]],
    min_envs: int = 3,
    trigger_factor: float = TRIGGER_FACTOR,
    gray_lo_factor: float = GRAY_LO_FACTOR,
) -> dict:
    """Aggregate per-metric gate results into an OR-rule trigger decision.

    gb's rule (OR + grayfix): a comparison triggers on a metric if EITHER
    (a) >=min_envs environments have mean > 0 (direction), AND
    (b) the cross-environment mean >= trigger_factor * delta (1.2δ).
    [0.8δ, 1.2δ) is a gray/weak band: recorded with the gray_trace flag, never
    a trigger. Tension: a metric with >=min_envs environments at
    mean <= -delta while the other metric triggers.
    """
    per_metric_decision = {}
    triggered_metrics = []
    tension_metrics = []
    gray_trace_metrics = []
    for metric, results in by_metric.items():
        if not results:
            continue
        delta = results[0].delta
        pos = sum(1 for r in results if r.mean > 0)
        neg = sum(1 for r in results if r.mean <= -delta)
        cross_env_mean = float(np.mean([r.mean for r in results]))
        pos_trigger = pos >= min_envs and cross_env_mean >= trigger_factor * delta
        gray_trace = (not pos_trigger and pos >= min_envs
                      and gray_lo_factor * delta <= cross_env_mean
                      < trigger_factor * delta)
        neg_signal = neg >= min_envs and cross_env_mean <= -delta
        per_metric_decision[metric] = {
            "n_envs_positive": pos, "n_envs_negative": neg,
            "n_envs_total": len(results), "cross_env_mean": cross_env_mean,
            "delta": delta, "trigger_band": trigger_factor * delta,
            "positive_trigger": bool(pos_trigger),
            "gray_trace": bool(gray_trace),
            "negative_signal": bool(neg_signal),
        }
        if pos_trigger:
            triggered_metrics.append(metric)
        if gray_trace:
            gray_trace_metrics.append(metric)
        if neg_signal:
            tension_metrics.append(metric)
    trigger = len(triggered_metrics) > 0
    return {
        "trigger": bool(trigger),
        "triggered_metrics": triggered_metrics,
        "tension_metrics": tension_metrics,
        "gray_trace_metrics": gray_trace_metrics,
        "per_metric": per_metric_decision,
        "min_envs_required": min_envs,
        "decision": (
            "TRIGGER" if trigger else
            "NEGATIVE" if any(d["negative_signal"] for d in per_metric_decision.values()) else
            "GRAY_TRACE" if any(d["gray_trace"] for d in per_metric_decision.values()) else
            "NO_SIGNAL"
        ),
    }


def write_trigger_md(
    path: Path,
    run_dir: str,
    versions: dict,
    deltas: dict[str, float],
    sd_by_metric: dict[str, float],
    t1: dict,
    t2: dict,
    caveat: str,
) -> None:
    """Write trigger.md aligned with docs/TRIGGER_TEMPLATE.md.

    ``t1``/``t2`` are the per-comparison gate outputs: each is
    ``{"by_metric": {metric: [GateResult]}, "agg": aggregate_signal_dict}``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    envs = ["1", "2", "4", "5"]

    def _fmt(v, nd=4):
        return "—" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{nd}f}"

    lines = [
        "# Stage 1 Trigger Report",
        "",
        f"运行目录：`results/{run_dir}/trigger.md`",
        "",
        "## 1. 版本",
        "",
        "| 项 | 值 |",
        "|----|----|",
        f"| 数据 SHA-256 | {versions.get('data_sha256','')} |",
        f"| 代码提交 | {versions.get('code_commit','(uncommitted)')} |",
        "| 配置 | `configs/stage1.yaml` |",
        f"| 依赖锁 | {versions.get('lockfile','env/lock.txt')} |",
        f"| 运行目录 | `results/{run_dir}/` |",
        "",
        "## 2. δ 与出处",
        "",
        "| 项 | 值 |",
        "|----|----|",
        "| δ 估计方法 | 冒烟 fold 级配对差 SD；$\\delta=t_{0.975,14}\\times SD/\\sqrt{15}$ |",
        f"| SD（池化两对比的 fold 级差值） | PCC {sd_by_metric.get('pcc', float('nan')):.4f} / "
        f"RMSE {sd_by_metric.get('rmse', float('nan')):.4f} |",
        f"| δ_PCC（计算值 / 取整） | {deltas.get('pcc', float('nan')):.4f} |",
        f"| δ_RMSE（兜底 0.05） | {deltas.get('rmse', 0.05):.4f} |",
        f"| 是否使用兜底 | {versions.get('delta_fallback', False)} |",
        f"| δ 固定时间 | {versions.get('delta_fixed_at', '正式运行前')} |",
        "",
        "## 3. T1/T2 判定表",
        "",
        "T1 = TabPFN − Ridge；T2 = FT-Transformer − Ridge。改善方向："
        "$\\Delta$PCC = PCC(model) − PCC(Ridge)；$\\Delta$RMSE = RMSE(Ridge) − RMSE(model)。",
        "",
        "| 对比 | 指标 | 环境 1 均值 | 环境 2 均值 | 环境 4 均值 | 环境 5 均值 | "
        "跨环境均值 | ≥3/4 环境 >0 | 跨环境均值 ≥1.2δ | 结论 |",
        "|------|------|------------|------------|------------|------------|"
        "------------|--------------|----------------|------|",
    ]

    def _env_mean(by_metric, metric, env):
        for r in by_metric.get(metric, []):
            if r.environment == env:
                return r.mean
        return None

    for comp_name, comp in [("T1", t1), ("T2", t2)]:
        by_metric = comp["by_metric"]
        agg = comp["agg"]
        for metric in ("pcc", "rmse"):
            results = by_metric.get(metric, [])
            means = [_env_mean(by_metric, metric, e) for e in envs]
            cross = np.mean([m for m in means if m is not None]) if any(m is not None for m in means) else None
            pm = agg["per_metric"].get(metric, {})
            n_pos = pm.get("n_envs_positive", 0)
            ge3 = "是" if n_pos >= 3 else "否"
            ge_delta = "是" if pm.get("positive_trigger") else "否"
            concl = "触发" if pm.get("positive_trigger") else (
                "负向留痕" if pm.get("negative_signal") else
                "灰区留痕" if pm.get("gray_trace") else "不成立")
            lines.append(
                f"| {comp_name} | Δ{metric.upper()} | {_fmt(means[0])} | {_fmt(means[1])} | "
                f"{_fmt(means[2])} | {_fmt(means[3])} | {_fmt(cross)} | {ge3} | {ge_delta} | {concl} |"
            )

    lines += [
        "",
        f"**指标规则（gb 裁决：OR）：** 触发指标来源 = "
        f"{', '.join(t1['agg']['triggered_metrics'] + t2['agg']['triggered_metrics']) or '无'}。"
        f"对立形态（张力）：{', '.join(t1['agg']['tension_metrics'] + t2['agg']['tension_metrics']) or '无'}。",
        "",
        "每个环境另附 15 个 fold 级配对差逐值、均值 [min,max]、df=14 近似区间与固定 caveat。",
        "",
        "### Fold 级配对差逐值",
        "",
        "```json",
        json.dumps({
            comp: {m: {r.environment: r.fold_diffs for r in res}
                   for m, res in t["by_metric"].items()}
            for comp, t in [("T1", t1), ("T2", t2)]
        }, indent=2),
        "```",
        "",
        "## 4. 指标规则（gb 裁决：OR，任一满足即触发）",
        "",
        "- [x] **OR**：PCC 与 RMSE 各自判定；任一指标满足 ≥3/4 环境均值 >0 且跨环境均值 ≥1.2δ，即正向信号（触发带下界 = 1.2δ，gb grayfix 裁决）。",
        "- 每指标用自己的 δ：δ_PCC（冒烟估计，兜底 0.03）、δ_RMSE（0.05）；先验化，禁看结果后改。",
        f"- **触发指标来源**：{', '.join(t1['agg']['triggered_metrics'] + t2['agg']['triggered_metrics']) or '无'}（必填）。",
        f"- **对立形态**：{', '.join(t1['agg']['tension_metrics'] + t2['agg']['tension_metrics']) or '无'}。",
        f"- **灰区留痕**：跨环境均值落在 [0.8δ, 1.2δ) 且 ≥3/4 环境同向 → 留痕、未触发（T1: "
        f"{', '.join(t1['agg'].get('gray_trace_metrics', [])) or '无'}；T2: "
        f"{', '.join(t2['agg'].get('gray_trace_metrics', [])) or '无'}）。",
        "- 第一眼触发只建立“进入 Stage 2”，不建立正向声称；“稳定改进”保留给 Stage 2 判据。",
        "",
        "## 5. 动作映射",
        "",
        "| 情况 | 记录 | 动作 |",
        "|------|------|------|",
    ]
    for comp_name, comp in [("T1", t1), ("T2", t2)]:
        d = comp["agg"]["decision"]
        if d == "TRIGGER":
            action = "建议冻结 Stage 2；命名对应已注册判断"
        elif d == "NEGATIVE":
            action = "记录负向信号，供用户再裁"
        elif d == "GRAY_TRACE":
            action = "灰区留痕、未触发；是否投入 Stage 2 由用户决定"
        else:
            action = "记录负结果/不成立；是否投入 Stage 2 由用户决定"
        lines.append(f"| {comp_name} {d} | 见 §3 | {action} |")
    lines += [
        "",
        "## 6. 结论与签核",
        "",
        f"- 触发结论：T1={t1['agg']['decision']}，T2={t2['agg']['decision']}。",
        f"- 是否建议进入 Stage 2：{'是（任一对比触发）' if (t1['agg']['trigger'] or t2['agg']['trigger']) else '否'}。",
        "",
        "## Caveat (fixed)",
        "",
        caveat,
    ]
    path.write_text("\n".join(lines))
