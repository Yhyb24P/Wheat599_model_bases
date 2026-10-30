#!/usr/bin/env python
"""BiMSGP-protocol diagnosis (user request, 2026-10-29).

Two questions, no tuning, no grid expansion, same seed-42 10-fold manifest
as the BiMSGP runner:
1. RR-BLUP (Ridge on X; X/y centered on train only, fit_intercept=True;
   alpha selected on val only; test evaluated once). Confirms that the X-Y
   row correspondence carries predictable signal.
2. FT-Transformer refit diagnostics: same model/training config/seed as
   configs/bimsgp.yaml, but additionally recording TRAIN and VAL metrics next
   to the (single) TEST evaluation, to separate underfitting (low train PCC)
   from generalization failure (high train, low test).

Outputs go to a separate run directory (never the main results):
per_fold_rrblup.csv, per_fold_ftt.csv, summary.json, diagnosis.md.
Read-only imports from scripts/run_bimsgp.py (folds, scalers, model, loop).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ / "src"))
sys.path.insert(0, str(PROJ / "scripts"))

import numpy as np
import yaml

import run_bimsgp as rb
from wheat599.data import load, sha256_of
from wheat599.metrics import mae, pcc, rmse

ALPHAS = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]  # Stage-1 alpha grid, no expansion


# ----------------------------------------------------------------- Ridge ----

def rrblup_one(env, fold, X_tr, y_tr, X_va, y_va, X_te, y_te) -> dict:
    """Train-only centering + Ridge; alpha chosen on val RMSE; test once."""
    from sklearn.linear_model import Ridge

    mu_x, mu_y = X_tr.mean(axis=0), float(y_tr.mean())
    Xtr_c, Xva_c, Xte_c = X_tr - mu_x, X_va - mu_x, X_te - mu_x
    best_alpha, best_rmse = None, np.inf
    for a in ALPHAS:
        m = Ridge(alpha=a, fit_intercept=True).fit(Xtr_c, y_tr - mu_y)
        s = rmse(y_va, m.predict(Xva_c) + mu_y)
        if s < best_rmse:
            best_rmse, best_alpha = s, a
    model = Ridge(alpha=best_alpha, fit_intercept=True).fit(Xtr_c, y_tr - mu_y)
    pred_te = model.predict(Xte_c) + mu_y
    row = {"environment": env, "fold": int(fold), "alpha": best_alpha,
           "val_rmse": float(best_rmse),
           "test_pcc": float(pcc(y_te, pred_te)),
           "test_rmse": float(rmse(y_te, pred_te)),
           "test_mae": float(mae(y_te, pred_te)),
           "failed": False, "reason": ""}
    if not np.isfinite(pred_te).all() or not np.isfinite(row["test_pcc"]):
        row.update(failed=True, reason="non_finite")
        row["test_pcc"] = row["test_rmse"] = row["test_mae"] = float("nan")
    return row


def _num(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _is_failed(row) -> bool:
    v = row.get("failed", False)
    return v is True or str(v).lower() in ("true", "1")


def summarize_metric(rows, metric):
    vals = np.asarray([_num(r.get(metric)) for r in rows
                       if not _is_failed(r) and np.isfinite(_num(r.get(metric)))],
                      dtype=np.float64)
    if vals.size == 0:
        return {"mean": float("nan"), "sd_ddof1": float("nan"), "n": 0}
    return {"mean": float(vals.mean()),
            "sd_ddof1": float(vals.std(ddof=1)) if vals.size > 1 else float("nan"),
            "n": int(vals.size)}


# ------------------------------------------------------------------- main ----

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=PROJ / "configs" / "bimsgp.yaml")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--parts", nargs="+", default=["rrblup", "ftt"],
                    choices=["rrblup", "ftt"])
    ap.add_argument("--folds", nargs="*", type=int, default=None,
                    help="subset of fold ids (top-up runs); default all")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text())
    proto, mcfg, tcfg = cfg["protocol"], cfg["model"], cfg["training"]
    envs = proto["environments"]
    out = args.out or (PROJ / "results" / time.strftime("bimsgp_diag_%Y%m%d_%H%M%S"))
    out.mkdir(parents=True, exist_ok=True)

    folds = rb.make_folds(n=599, seed=int(proto["fold_seed"]),
                          test_sizes=proto["test_sizes"],
                          val_size=int(proto["val_size"]))
    data = load(Path(proto["data"]), proto.get("data_sha256"))
    Y = rb.load_y(Path(proto["y_path"]), Path(proto["data"]),
                  proto.get("data_sha256"))
    X = data.X
    provenance = {
        "run": out.name, "config": str(args.config), "parts": args.parts,
        "data_sha256": sha256_of(Path(proto["data"])),
        "folds_seed": proto["fold_seed"],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=1))
    summary: dict = {"provenance": provenance}
    # merge with an earlier partial run so `--parts ftt` does not drop RR-BLUP
    if (out / "summary.json").exists():
        try:
            old = json.loads((out / "summary.json").read_text())
            for key in ("rrblup", "ftt"):
                if key in old:
                    summary[key] = old[key]
        except Exception:  # noqa: BLE001 - a corrupt partial summary is ignorable
            pass

    # ---- RR-BLUP ----
    if "rrblup" in args.parts:
        print("[diag] RR-BLUP ...", flush=True)
        rows = []
        for fold in folds:
            tr, va, te = fold["train"], fold["val"], fold["test"]
            for j, env in enumerate(envs):
                y = Y[:, j]
                rows.append(rrblup_one(env, fold["fold"], X[tr], y[tr],
                                       X[va], y[va], X[te], y[te]))
        with (out / "per_fold_rrblup.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        summary["rrblup"] = {
            env: {m: summarize_metric([r for r in rows if r["environment"] == env], f"test_{m}")
                  for m in ("pcc", "rmse", "mae")} | {
                "n_failed": int(sum(1 for r in rows
                                    if r["environment"] == env and r["failed"]))}
            for env in envs}
        for env in envs:
            e = summary["rrblup"][env]
            print(f"  [rrblup env {env}] test PCC {e['pcc']['mean']:.4f} "
                  f"± {e['pcc']['sd_ddof1']:.4f} (n={e['pcc']['n']})", flush=True)

    # ---- FT-T diagnostics ----
    if "ftt" in args.parts:
        import gc
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            # half of the shared card; exceed -> error instead of OOM-killing others
            torch.cuda.set_per_process_memory_fraction(0.5)
        print(f"[diag] FT-T refit on {device} (mem_fraction=0.5) ...", flush=True)
        rows = []
        for fold in folds:
            if args.folds and fold["fold"] not in set(args.folds):
                continue
            tr, va, te = fold["train"], fold["val"], fold["test"]
            for j, env in enumerate(envs):
                y = Y[:, j]
                t0 = time.time()
                sx = rb.Scaler().fit(X[tr]); sy = rb.Scaler().fit(y[tr])
                Xtr_s, Xva_s, Xte_s = sx.transform(X[tr]), sx.transform(X[va]), sx.transform(X[te])
                ytr_s, yva_s = sy.transform(y[tr]), sy.transform(y[va])
                model = rb.build_model(X.shape[1], mcfg)
                model, hist, best_epoch, best_pcc = rb.train_one_fold(
                    model, Xtr_s, ytr_s, Xva_s, yva_s, tcfg, device,
                    seed=int(tcfg["seed"]))
                pred = {k: sy.inverse_transform(rb.predict(model, Xz, device))
                        for k, Xz in (("train", Xtr_s), ("val", Xva_s), ("test", Xte_s))}
                row = {"environment": env, "fold": fold["fold"],
                       "best_epoch": int(best_epoch), "best_val_pcc": float(best_pcc),
                       "seconds": round(time.time() - t0, 1), "failed": False}
                for k, yz in (("train", y[tr]), ("val", y[va]), ("test", y[te])):
                    row[f"{k}_pcc"] = float(pcc(yz, pred[k]))
                    row[f"{k}_rmse"] = float(rmse(yz, pred[k]))
                for k in ("train", "val", "test"):
                    if not np.isfinite(row[f"{k}_pcc"]):
                        row.update(failed=True, reason=f"non_finite_{k}_pcc")
                row["source"] = "full"
                rows.append(row)
                print(f"  [ftt env {env} fold {fold['fold']}] "
                      f"train {row['train_pcc']:.3f} val {row['val_pcc']:.3f} "
                      f"test {row['test_pcc']:.3f} ep={best_epoch} "
                      f"{row['seconds']}s", flush=True)
                # per-run cleanup: a diag run must not grow VRAM across folds
                del model, hist, pred
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()
        csv_path = out / "per_fold_ftt.csv"
        merged: dict = {}
        if csv_path.exists():
            with csv_path.open() as f:
                for old in csv.DictReader(f):
                    merged[(old["environment"], int(old["fold"]))] = old
        for r in rows:
            merged[(r["environment"], int(r["fold"]))] = r
        rows_all = [merged[k] for k in sorted(merged, key=lambda k: (k[0], k[1]))]
        fieldnames = ["environment", "fold", "best_epoch", "best_val_pcc",
                      "seconds", "failed", "reason", "train_pcc", "val_pcc",
                      "test_pcc", "train_rmse", "val_rmse", "test_rmse", "source"]
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader(); w.writerows(rows_all)
        summary["ftt"] = {
            env: {f"{split}_{m}": summarize_metric(
                      [r for r in rows_all if r["environment"] == env],
                      f"{split}_{m}")
                  for m in ("pcc", "rmse") for split in ("train", "val", "test")}
            for env in envs}
        summary["ftt"]["epochs"] = {
            env: {"median_best_epoch": float(np.median(
                [_num(r.get("best_epoch")) for r in rows_all
                 if r["environment"] == env and np.isfinite(_num(r.get("best_epoch")))]))}
            for env in envs}
        summary["ftt"]["n_rows"] = len(rows_all)
        summary["ftt"]["sources"] = sorted({str(r.get("source", "")) for r in rows_all})

    (out / "summary.json").write_text(json.dumps(summary, indent=1))

    # ---- diagnosis.md ----
    lines = ["# BiMSGP FT-T 诊断（seed-42 10 折，同 config；不调参、不扩网格）", ""]
    if "rrblup" in summary:
        lines += ["## RR-BLUP（Ridge，train-only 中心化，alpha on val）", "",
                  "| env | test PCC (mean±SD, ddof=1) | test RMSE | test MAE |",
                  "|---|---|---|---|"]
        for env in envs:
            e = summary["rrblup"][env]
            lines.append(f"| {env} | {e['pcc']['mean']:.4f} ± {e['pcc']['sd_ddof1']:.4f} "
                         f"(n={e['pcc']['n']}) | {e['rmse']['mean']:.4f} | {e['mae']['mean']:.4f} |")
        lines.append("")
    if "ftt" in summary:
        lines += ["## FT-T（同 configs/bimsgp.yaml，seed 42）", "",
                  "| env | train PCC | val PCC | test PCC | train→test 差 | 判读 |",
                  "|---|---|---|---|---|---|"]
        for env in envs:
            t = summary["ftt"][env]
            tr = t["train_pcc"]["mean"]; va = t["val_pcc"]["mean"]; te = t["test_pcc"]["mean"]
            gap = tr - te
            verdict = ("欠拟合（train PCC 低）" if tr < 0.5 else
                       "泛化失败（train 高 test 低）" if gap > 0.3 else
                       "居中")
            lines.append(f"| {env} | {tr:.4f} | {va:.4f} | {te:.4f} | {gap:+.4f} | {verdict} |")
        lines.append("")
        lines.append("注：判读阈值为诊断性经验规则（train PCC<0.5→欠拟合；train−test>0.3→泛化失败），非正式判据。")
    (out / "diagnosis.md").write_text("\n".join(lines))
    print(f"[diag] done -> {out}/", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
