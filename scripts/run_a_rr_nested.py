#!/usr/bin/env python
"""Fair nested protocol for A (FT-Transformer) and RR-BLUP.

Isomorphic with the MoEGP/BiMSGP nested runners: outer test folds come from the
fixed BiMSGP manifest (results/bimsgp_ftt_fixed_20261029/splits_manifest.json);
dev = the remaining 90%. Inside dev a fixed 3-fold CV (seed 42) selects a
candidate by mean inner-val PCC (A) or inner val RMSE (RR); refit
epochs = ceil(median best epochs of the winning candidate across its inner folds),
capped at the candidate max_epochs. Preprocessing is fitted on each inner
training fold and then refitted on the full dev before test prediction.

This removes the "A tuned, RR did not" asymmetry: both models get the same nested
selection budget. It does NOT re-search the frozen architectures.

Reuses run_bimsgp (imported as rb) for data loading, the FT-T builder/trainer,
the RR-BLUP pattern, the Scaler and the fold evaluator. No data logic is copied.

NOTE: filename run_a_rr_nested (not run_bimsgp_nested) to avoid collision with
the BiMSGP nested runner.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import run_bimsgp as rb  # noqa: E402


def load_outer_folds(manifest_path: Path) -> list[dict]:
    """Outer test folds from the fixed BiMSGP manifest (row-number indices)."""
    m = json.loads(Path(manifest_path).read_text())
    return [{"fold": int(f["fold"]),
             "test": np.asarray(f["test_idx"], dtype=int)} for f in m["folds"]]


def inner_splits(dev: np.ndarray, k: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Fixed k-fold on dev; returns (inner_train, inner_val) as POSITIONS into
    dev (0..len(dev)-1), so they index the dev-subset arrays directly."""
    n = len(dev)
    rng = np.random.default_rng(seed)
    parts = np.array_split(rng.permutation(n), k)
    out = []
    for i in range(k):
        ival = parts[i]
        itr = np.concatenate([parts[j] for j in range(k) if j != i])
        out.append((itr, ival))
    return out


def run_a_outer(env, fold, X, y, dev, test, inner, candidates, mcfg, tcfg, device, seed):
    """Nested A (FT-T): select candidate by mean inner-val PCC; refit on dev."""
    cand_rows = []
    for cand in candidates:
        lr, max_epochs = float(cand["lr"]), int(cand["max_epochs"])
        inner_pccs, inner_epochs, n_failed = [], [], 0
        for itr, iva in inner:
            train, val = dev[itr], dev[iva]
            sx = rb.Scaler().fit(X[train])
            sy = rb.Scaler().fit(y[train])
            tcfg_i = dict(tcfg)
            tcfg_i["lr"] = lr
            tcfg_i["epochs"] = max_epochs
            try:
                model = rb.build_model(X.shape[1], mcfg, seed=seed)
                _, _, fit = rb.train_one_fold(
                    model, sx.transform(X[train]), sy.transform(y[train]),
                    sx.transform(X[val]), sy.transform(y[val]),
                    tcfg_i, device, seed=seed)
                if not np.isfinite(fit["best_pcc"]):
                    n_failed += 1
                    continue
                inner_pccs.append(float(fit["best_pcc"]))
                inner_epochs.append(int(fit["best_pcc_epoch"]) + 1)
            except Exception:
                n_failed += 1
        mean_pcc = float(np.mean(inner_pccs)) if inner_pccs else float("nan")
        cand_rows.append({
            "candidate": cand["name"], "lr": lr, "max_epochs": max_epochs,
            "mean_inner_pcc": mean_pcc, "inner_pccs": inner_pccs,
            "inner_epochs": inner_epochs, "n_failed": n_failed,
        })

    finite = [c for c in cand_rows if c["n_failed"] == 0 and np.isfinite(c["mean_inner_pcc"])]
    if not finite:
        return {"environment": env, "fold": fold, "selected": None,
                "failed": True, "reason": "all candidates failed",
                "candidates": cand_rows}, None, None
    finite.sort(key=lambda c: (-c["mean_inner_pcc"], c["max_epochs"], c["lr"]))
    winner = finite[0]
    refit_epochs = int(min(math.ceil(np.median(winner["inner_epochs"])),
                           winner["max_epochs"]))

    tcfg_r = dict(tcfg)
    tcfg_r["lr"] = winner["lr"]
    tcfg_r["epochs"] = refit_epochs
    sx = rb.Scaler().fit(X[dev])
    sy = rb.Scaler().fit(y[dev])
    X_dev_s = sx.transform(X[dev])
    y_dev_s = sy.transform(y[dev])
    model = rb.build_model(X.shape[1], mcfg, seed=seed)
    # refit on full dev; val=dev so the checkpoint uses dev (test never enters)
    _, _, fit = rb.train_one_fold(
        model, X_dev_s, y_dev_s, X_dev_s, y_dev_s, tcfg_r, device, seed=seed)
    X_test_s = sx.transform(X[test])
    pred_s = rb.predict(model, X_test_s, device)
    pred = sy.inverse_transform(pred_s)
    row = {
        "environment": env, "fold": fold, "selected": winner["candidate"],
        "lr": winner["lr"], "refit_epochs": refit_epochs,
        "inner_pcc": winner["mean_inner_pcc"], "failed": False, "reason": "",
    }
    metrics = rb.evaluate_fold(env, fold, y[test], pred)
    row.update({k: metrics[k] for k in ("pcc", "rmse", "mae", "failed", "reason")})
    return row, cand_rows, pred


def run_rr_outer(env, fold, X, y, dev, test, inner, alphas):
    """Nested RR-BLUP: select alpha by mean inner val RMSE; refit on dev."""
    from sklearn.linear_model import Ridge
    def _fit_alpha(alpha, train, val):
        sx = rb.Scaler().fit(X[train])
        sy = rb.Scaler().fit(y[train])
        model = Ridge(alpha=float(alpha), fit_intercept=True).fit(
            sx.transform(X[train]), sy.transform(y[train]))
        pred = sy.inverse_transform(model.predict(sx.transform(X[val])))
        return float(np.sqrt(np.mean((y[val] - pred) ** 2)))

    cand_rows = []
    for alpha in alphas:
        inner_rmses = []
        for itr, iva in inner:
            try:
                rmse_v = _fit_alpha(alpha, dev[itr], dev[iva])
                if np.isfinite(rmse_v):
                    inner_rmses.append(rmse_v)
            except Exception:
                continue
        mean_rmse = float(np.mean(inner_rmses)) if inner_rmses else float("nan")
        cand_rows.append({"candidate": f"alpha={alpha}", "alpha": float(alpha),
                          "mean_inner_rmse": mean_rmse,
                          "n_failed": len(inner) - len(inner_rmses)})

    finite = [c for c in cand_rows if c["n_failed"] == 0 and np.isfinite(c["mean_inner_rmse"])]
    if not finite:
        return {"environment": env, "fold": fold, "selected": None,
                "failed": True, "reason": "all alphas failed"}, None, None
    finite.sort(key=lambda c: c["mean_inner_rmse"])  # lower RMSE better
    winner = finite[0]

    sx = rb.Scaler().fit(X[dev])
    sy = rb.Scaler().fit(y[dev])
    m = Ridge(alpha=winner["alpha"], fit_intercept=True).fit(
        sx.transform(X[dev]), sy.transform(y[dev]))
    pred_s = m.predict(sx.transform(X[test]))
    pred = sy.inverse_transform(pred_s)
    metrics = rb.evaluate_fold(env, fold, y[test], pred)
    row = {"environment": env, "fold": fold, "selected": winner["candidate"],
           "alpha": winner["alpha"], "refit_epochs": 0,
           "inner_rmse": winner["mean_inner_rmse"], "failed": False, "reason": ""}
    row.update({k: metrics[k] for k in ("pcc", "rmse", "mae", "failed", "reason")})
    return row, cand_rows, pred


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--models", nargs="+", default=["A", "RR"])
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--folds", nargs="+", type=int, default=None)
    ap.add_argument("--envs", nargs="+", default=None)
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text())
    data = rb.load(Path(cfg["data"]), cfg.get("data_sha256"))
    X, Y = data.X, data.Y
    envs = args.envs or cfg["environments"]
    seed = int(cfg["seed"])

    outer = load_outer_folds(Path(cfg["splits"]))
    if args.folds:
        outer = [f for f in outer if f["fold"] in args.folds]

    out = args.out or Path("results") / cfg.get("run_name", "a_rr_nested_20261102")
    out.mkdir(parents=True, exist_ok=True)
    provenance = {
        "run": out.name, "config": str(args.config),
        "data_sha256": cfg.get("data_sha256"), "splits": str(cfg["splits"]),
        "seed": seed, "models": args.models,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=1))

    per_fold, selection_rows, oof_rows, failures = [], [], [], []
    inner_folds = int(cfg["training"]["inner_folds"])
    inner_seed = int(cfg["training"]["inner_seed"])

    for model_name in args.models:
        is_a = model_name.upper() == "A"
        mcfg = cfg.get("model", {}) if is_a else {}
        tcfg = cfg.get("training", {}) if is_a else {}
        if is_a:
            tcfg = {**tcfg, "weight_decay": mcfg["weight_decay"]}
        candidates = cfg["candidates"]["A"] if is_a else cfg["candidates"]["RR"]["alphas"]
        for fold in outer:
            test = fold["test"]
            mask = np.ones(X.shape[0], dtype=bool)
            mask[test] = False
            dev = np.where(mask)[0]
            if len(set(dev.tolist()) & set(test.tolist())):
                raise ValueError("dev/test overlap")
            inner = inner_splits(dev, inner_folds, inner_seed)
            for env in envs:
                y = Y[:, cfg["environments"].index(env)]
                t0 = time.time()
                if is_a:
                    row, cand_rows, pred = run_a_outer(
                        env, fold["fold"], X, y, dev, test, inner,
                        candidates, mcfg, tcfg, "cpu", seed)
                else:
                    row, cand_rows, pred = run_rr_outer(
                        env, fold["fold"], X, y, dev, test, inner, candidates)
                row["model"] = model_name
                row["seconds"] = round(time.time() - t0, 1)
                for c in (cand_rows or []):
                    c.update({"model": model_name, "environment": env,
                              "fold": fold["fold"]})
                    selection_rows.append(c)
                if row.get("failed"):
                    failures.append({"model": model_name, "environment": env,
                                     "fold": fold["fold"], "reason": row.get("reason", "")})
                else:
                    for i, idx in enumerate(test):
                        oof_rows.append({"model": model_name, "environment": env,
                                         "fold": fold["fold"], "material_index": int(idx),
                                         "y_true": float(y[idx]), "y_pred": float(pred[i])})
                per_fold.append(row)
                pcc_s = (f"{row['pcc']:.4f}"
                         if row.get("pcc") is not None and np.isfinite(row.get("pcc", np.nan))
                         else "NA")
                print(f"[{model_name} {env} fold{fold['fold']}] sel={row.get('selected')} "
                      f"pcc={pcc_s} {row['seconds']}s"
                      f"{'' if not row.get('failed') else ' FAILED:' + str(row.get('reason',''))}")

    def _write(name, rows):
        if not rows:
            (out / name).write_text("")
            return
        keys = []
        for r in rows:
            for k in r.keys():
                if k not in keys:
                    keys.append(k)
        with (out / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)

    _write("per_fold.csv", per_fold)
    _write("selection.csv", selection_rows)
    _write("oof.csv", oof_rows)
    _write("failures.csv", failures)

    summary = {}
    for model_name in args.models:
        mf = [r for r in per_fold if r["model"] == model_name and not r.get("failed")]
        summary[model_name] = {
            "n_outer_units": len([r for r in per_fold if r["model"] == model_name]),
            "n_failed": len([r for r in per_fold if r["model"] == model_name and r.get("failed")]),
        }
        for metric in ("pcc", "rmse", "mae"):
            vals = [r[metric] for r in mf if np.isfinite(r.get(metric, np.nan))]
            summary[model_name][metric] = {
                "mean": float(np.mean(vals)) if vals else None,
                "sd_ddof1": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
                "n": len(vals),
            }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"done. wrote {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
