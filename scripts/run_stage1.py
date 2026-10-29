#!/usr/bin/env python
"""Stage 1 runner — TabPFN-3.5 + FT-Transformer + Ridge, random splits only.

Per configs/stage1.yaml and the lead's 2026-10-29 scope:
* 3 repeats x 5-fold outer x 3-fold inner, random protocol only.
* 4 environments, PCC/RMSE (primary) + MAE/R2 (secondary).
* Inner selection metric: RMSE per environment.
* Models: linear (reference), ft_transformer (primary), tabpfn_v3_5 (primary).
* No relatedness / A / wheat.sets.
* gb signal gate: 15 fold-level paired diffs per (comparison, env) -> delta
  (smoke-estimated, fallback PCC 0.03 / RMSE 0.05) -> OR-rule trigger.md.
* 2 GPU-hours cap; CPU preferred. Torch-based models skip cleanly if deps absent.

The 15 fold-level paired differences are saved per value (gate requirement).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import yaml

from wheat599.bootstrap import fold_level_t_interval
from wheat599.contracts import check_adapter
from wheat599.data import load, sha256_of
from wheat599.gate import (
    aggregate_signal,
    estimate_delta,
    evaluate_comparison,
    resolve_deltas,
    write_trigger_md,
)
from wheat599.metrics import compute_all, rmse
from wheat599.models import available_adapters, get_adapter
from wheat599.oof import OOF
from wheat599.splits import build_splits

CAVEAT = ("material reuse and fold correlation not adjusted; optimistic; "
          "magnitude only; not a stable-improvement claim")


def load_config(path: Path) -> dict:
    with path.open() as h:
        return yaml.safe_load(h)


def device_info() -> dict:
    import os
    info = {"python": sys.version.split()[0], "numpy": np.__version__,
            "torch": None, "cuda_available": None, "device": "cpu"}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if info["cuda_available"]:
            info["device"] = "cuda"
            info["gpu_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    info["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    return info


def run_adapter_oof(adapter, data, splits, smoke, seed) -> tuple[np.ndarray, np.ndarray, dict]:
    """OOF via inner-fold RMSE selection + refit on full outer_train.

    Leakage contract (takeover fix, 2026-10-29): hyperparameters are chosen
    ONLY from inner folds inside outer_train (mean RMSE over the 3 inner
    folds); the final model refits on the full outer_train; the outer-valid
    slice is used for prediction and metrics, never for selection. FT-T's
    refit epoch count = median of its inner-fold best epochs (cap
    max_epochs). Single-config grids (TabPFN frozen ICL) skip the inner loop
    entirely — nothing to select.
    """
    import itertools

    preds = np.zeros((data.n, data.n_env))
    fold_of = np.full(data.n, -1, dtype=np.int64)
    sel = []
    grid = adapter.param_grid(smoke=smoke)
    keys = list(grid.keys())
    params_list = [dict(zip(keys, combo))
                   for combo in itertools.product(*[grid.grid[k] for k in keys])]
    supports_epochs = hasattr(type(adapter), "best_epoch")

    for split in splits:
        tr, va = split.outer_train, split.outer_valid
        Xva = data.X[va]
        for j in range(data.n_env):
            ycol = data.Y[:, j]
            rec: dict = {"fold": split.fold, "env": j, "n_configs": len(params_list)}
            refit_epochs = None
            if len(params_list) == 1:
                best = params_list[0]
                rec["inner_scores"] = None
            else:
                scores = np.full((len(split.inner_folds), len(params_list)), np.nan)
                epochs = np.zeros((len(split.inner_folds), len(params_list)))
                for fi, ifold in enumerate(split.inner_folds):
                    Xiv, yiv = data.X[ifold.inner_valid], ycol[ifold.inner_valid]
                    for pi, p in enumerate(params_list):
                        cand = type(adapter)()
                        cp = dict(p)
                        cp["_val"] = (Xiv, yiv)
                        cp["_seed"] = seed + 7 * split.fold + fi
                        cand.fit(data.X[ifold.inner_train], ycol[ifold.inner_train], cp)
                        scores[fi, pi] = rmse(yiv, cand.predict(Xiv))
                        if supports_epochs:
                            epochs[fi, pi] = cand.best_epoch()
                mean_scores = np.nanmean(scores, axis=0)
                best_pi = int(np.nanargmin(mean_scores))
                best = params_list[best_pi]
                rec["inner_scores"] = [float(s) for s in mean_scores]
                if supports_epochs:
                    refit_epochs = int(np.median(epochs[:, best_pi])) + 1
            rec["params"] = best
            rec["refit_epochs"] = refit_epochs
            final = type(adapter)()
            fp = dict(best)
            fp["_seed"] = seed + 7 * split.fold
            if refit_epochs is not None:
                fp["_refit_epochs"] = refit_epochs
            final.fit(data.X[tr], ycol[tr], fp)
            preds[va, j] = final.predict(Xva)
            sel.append(rec)
        fold_of[va] = split.fold
    if np.any(fold_of < 0):
        raise RuntimeError("material without outer-fold assignment")
    return preds, fold_of, sel


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/stage1.yaml"))
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true", help="SMOKE_ONLY: 1 fold, tiny grid")
    parser.add_argument("--models", nargs="+", default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    proto = cfg["protocol"]
    data_path = args.data or Path(proto["data"])
    expected_sha = proto["data_sha256"]
    envs = proto["environments"]

    run_name = args.out.name if args.out else (
        "stage1_smoke" if args.smoke else f"stage1_{time.strftime('%Y%m%d_%H%M%S')}")
    out = args.out or Path("results") / run_name
    out.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        (out / "SMOKE_ONLY").write_text("smoke run — not a benchmark result\n")

    n_outer = 2 if args.smoke else proto["outer_folds"]
    n_inner = 2 if args.smoke else proto["inner_folds"]
    seeds = proto["seeds"][:1] if args.smoke else proto["seeds"]

    models = args.models or ["linear", "ft_transformer", "tabpfn_v3_5"]
    avail = set(available_adapters())
    run_models = [m for m in models if m in avail]
    skipped = [m for m in models if m not in avail]

    data = load(data_path, expected_sha)
    dev = device_info()
    try:
        import subprocess
        code_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            cwd=str(Path(__file__).resolve().parent.parent)).stdout.strip()
        code_commit = code_commit or "uncommitted-worktree"
    except Exception:
        code_commit = "uncommitted-worktree"

    provenance = {
        "run": run_name, "config": str(args.config), "data": str(data_path),
        "data_sha256": sha256_of(data_path), "device": dev,
        "code_commit": code_commit,
        "smoke": bool(args.smoke), "n_outer": n_outer, "n_inner": n_inner,
        "seeds": seeds, "models": run_models, "skipped_models": skipped,
        "environments": envs, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2))
    print(f"[stage1] models={run_models} skipped={skipped} device={dev['device']} "
          f"folds={n_outer}x{n_inner} repeats={len(seeds)}")

    # OOF per model per env per repeat
    oof_store = {m: {} for m in run_models}  # model -> env -> repeat -> y_pred
    fold_of_by_seed: dict = {}               # seed -> fold_of (shared by env/model)
    fold_rmses = {m: {} for m in run_models}  # model -> env -> repeat -> [per-fold rmse]
    for m in run_models:
        adapter = get_adapter(m)
        for seed in seeds:
            splits = build_splits(data, "random", seed=seed,
                                  n_outer=n_outer, n_inner=n_inner)
            for s in splits:
                check_adapter(adapter.contract_adapter(), s, data)
            preds, fold_of, _ = run_adapter_oof(adapter, data, splits, args.smoke, seed)
            fold_of_by_seed[seed] = fold_of
            for j, env in enumerate(data.environments):
                oof_store[m].setdefault(env, {})[seed] = preds[:, j]
                fold_rmses[m].setdefault(env, {})[seed] = [
                    rmse(data.Y[s.outer_valid, j], preds[s.outer_valid, j])
                    for s in splits
                ]
            print(f"  [{m} seed={seed}] done")

    # save OOF + per-fold metrics
    for m in run_models:
        for env in envs:
            for seed, yp in oof_store[m][env].items():
                j = data.environments.index(env)
                oof = OOF(environment=env, y_true=data.Y[:, j], y_pred=yp,
                          repeat=seed, fold_of=fold_of_by_seed[seed],
                          provenance={"model": m, "protocol": "random"})
                oof.save(out / "oof" / f"{m}_{env}_{seed}.npz")

    # gate: T1 = tabpfn - ridge, T2 = ft - ridge; values are FOLD-level paired
    # differences (metric computed on each outer-valid slice, per repeat)
    comparisons = {}
    gate = {"deltas": {}, "sd_by_metric": {}, "delta_source": {}}
    if "linear" in run_models:
        fold_diffs_all: dict = {}
        for primary, tname in (("tabpfn_v3_5", "T1"), ("ft_transformer", "T2")):
            if primary not in run_models:
                continue
            for metric, sign in (("pcc", 1.0), ("rmse", -1.0)):
                fn = _pcc if metric == "pcc" else rmse
                by_env = {}
                for env in envs:
                    j = data.environments.index(env)
                    vals = []
                    for seed in seeds:
                        fo = fold_of_by_seed[seed]
                        for f in range(n_outer):
                            idx = np.where(fo == f)[0]
                            if idx.size < 2:
                                continue
                            ref = fn(data.Y[idx, j], oof_store["linear"][env][seed][idx])
                            prim = fn(data.Y[idx, j], oof_store[primary][env][seed][idx])
                            vals.append(sign * (prim - ref))
                    by_env[env] = vals
                fold_diffs_all[(tname, metric)] = by_env
        # FIXED pre-run delta (gate rule 1): taken from config/fallback; the
        # formal run's own fold diffs NEVER set the threshold (post-hoc
        # self-calibration is the failure mode pre-registration forbids).
        fixed = resolve_deltas({
            "pcc": cfg.get("gate", {}).get("delta_pcc"),
            "rmse": cfg.get("gate", {}).get("delta_rmse"),
        })
        gate["deltas"] = {m: v["delta"] for m, v in fixed.items()}
        gate["delta_source"] = {m: v["source"] for m, v in fixed.items()}
        gate["sd_by_metric"] = {m: float("nan") for m in fixed}
        # diagnostic ONLY: mini-estimate from this run's fold diffs, recorded
        # and explicitly not used for the gate (smoke design too small to fix δ)
        smoke_estimate = {}
        for metric in ("pcc", "rmse"):
            pooled = [d for (t, m), byenv in fold_diffs_all.items() if m == metric
                      for dlist in byenv.values() for d in dlist]
            smoke_estimate[metric] = estimate_delta(pooled, metric, gate["deltas"][metric])
            gate["sd_by_metric"][metric] = (float(np.std(pooled, ddof=1))
                                            if len(pooled) > 2 else float("nan"))
        gate["smoke_estimate"] = {
            "values": smoke_estimate, "used": False,
            "note": "mini estimate from this run's fold diffs; not used to set the gate delta",
        }
        provenance["delta_fixed_at"] = "pre-run fixed (config/fallback)"
        provenance["delta_fallback"] = any(
            s == "fallback" for s in gate["delta_source"].values())
        for tname, primary in (("T1", "tabpfn_v3_5"), ("T2", "ft_transformer")):
            if (tname, "pcc") not in fold_diffs_all:
                continue
            comp = {"by_metric": {}, "agg": None}
            for metric in ("pcc", "rmse"):
                comp["by_metric"][metric] = evaluate_comparison(
                    f"{tname}:{primary}-linear",
                    {e: {metric: fold_diffs_all[(tname, metric)][e]} for e in envs},
                    {metric: gate["deltas"][metric]})[metric]
            comp["agg"] = aggregate_signal(comp["by_metric"])
            comparisons[tname] = comp

    # write per-value fold-level diffs (gate requirement, per-value on disk)
    (out / "fold_level_diffs.json").write_text(json.dumps({
        tname: {m: {r.environment: r.fold_diffs for r in res}
                for m, res in comp["by_metric"].items()}
        for tname, comp in comparisons.items()
    }, indent=2))

    # trigger.md
    if comparisons:
        t1 = comparisons.get("T1", {"by_metric": {m: [] for m in ("pcc", "rmse")},
                                     "agg": aggregate_signal({"pcc": [], "rmse": []})})
        t2 = comparisons.get("T2", {"by_metric": {m: [] for m in ("pcc", "rmse")},
                                     "agg": aggregate_signal({"pcc": [], "rmse": []})})
        write_trigger_md(out / "trigger.md", run_name, provenance,
                         gate["deltas"], gate["sd_by_metric"], t1, t2, CAVEAT)
        print(f"[stage1] trigger.md written: T1={t1['agg']['decision']} T2={t2['agg']['decision']}")

    (out / "summary.json").write_text(json.dumps({
        "provenance": provenance,
        "fold_rmses": {m: {e: {str(s): v for s, v in fold_rmses[m][e].items()}
                           for e in envs} for m in run_models},
        "gate": gate,
        "comparisons": {t: c["agg"] for t, c in comparisons.items()},
    }, indent=2, default=str))
    print(f"done. wrote {out}/")
    return 0


def _pcc(y, p):
    from wheat599.metrics import pcc
    return pcc(y, p)


if __name__ == "__main__":
    raise SystemExit(main())
