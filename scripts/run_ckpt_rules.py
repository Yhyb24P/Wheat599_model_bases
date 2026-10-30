#!/usr/bin/env python
"""Pre-specified checkpoint-rule comparison on one training trajectory.

Both rules are fixed before any test prediction is made:

* ``val_pcc``  — checkpoint with the highest validation PCC (ties -> earlier);
* ``val_rmse`` — checkpoint with the lowest validation RMSE (ties -> earlier).

One model is trained per (environment, fold); both checkpoints come from that
same run.  Test is predicted once per rule after training.  No rule is selected
per environment or fold, and test is never used to choose a rule or to stitch
the better environment-level result.

The comparison reports test PCC, RMSE, MAE, prediction SD, and prediction-SD /
truth-SD.  Same-fold RR-BLUP metrics are merged when the reference CSV exists.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ / "src"))
sys.path.insert(0, str(PROJ / "scripts"))

import run_bimsgp as rb  # noqa: E402
from wheat599.data import load  # noqa: E402

RULES = ("val_pcc", "val_rmse")
FIELDS = [
    "environment", "fold", "rule", "checkpoint_epoch",
    "val_pcc", "val_rmse", "test_pcc", "test_rmse", "test_mae",
    "pred_sd", "true_sd", "sd_ratio",
    "n_train", "n_val", "n_test", "seconds", "n_params", "failed", "reason",
]


def _summ(values) -> dict:
    vals = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if vals.size == 0:
        return {"mean": float("nan"), "sd_ddof1": float("nan"), "n": 0}
    return {
        "mean": float(vals.mean()),
        "sd_ddof1": float(vals.std(ddof=1)) if vals.size > 1 else float("nan"),
        "n": int(vals.size),
    }


def _sd_ratio(pred, truth) -> tuple[float, float, float]:
    ps = float(np.std(pred, ddof=1))
    ts = float(np.std(truth, ddof=1))
    ratio = ps / ts if ts > 0 else float("nan")
    return ps, ts, float(ratio)


def _load_rrblup(path: Path) -> dict:
    if not path.exists():
        return {}
    out = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            if str(row.get("failed", "")).lower() == "true":
                continue
            out[(str(row["environment"]), int(row["fold"]))] = {
                "pcc": float(row["test_pcc"]),
                "rmse": float(row["test_rmse"]),
                "mae": float(row["test_mae"]),
            }
    return out


def _macro(per_env: dict, envs: list[str], metric: str) -> dict:
    means = [per_env[e][metric]["mean"] for e in envs
             if np.isfinite(per_env[e][metric]["mean"])]
    return _summ(means)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path,
                    default=PROJ / "configs" / "bimsgp.yaml")
    ap.add_argument("--out", type=Path,
                    default=PROJ / "results" / "bimsgp_ckpt_rules_20261030")
    ap.add_argument("--folds", nargs="*", type=int, default=None,
                    help="subset of folds (pilot); default all 10")
    ap.add_argument("--rrblup", type=Path,
                    default=PROJ / "results" / "bimsgp_diag_fixed_20261029"
                    / "per_fold_rrblup.csv")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text())
    proto, mcfg, tcfg = cfg["protocol"], cfg["model"], cfg["training"]
    envs = [str(e) for e in proto["environments"]]
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    folds = rb.make_folds(n=599, seed=int(proto["fold_seed"]),
                          test_sizes=proto["test_sizes"],
                          val_size=int(proto["val_size"]))
    if args.folds:
        folds = [f for f in folds if f["fold"] in set(args.folds)]
    (out / "splits_manifest.json").write_text(json.dumps({
        "seed": proto["fold_seed"], "test_sizes": proto["test_sizes"],
        "val_size": proto["val_size"], "folds": rb.fold_table(folds),
    }, indent=1))

    data = load(Path(proto["data"]), proto.get("data_sha256"))
    Y = rb.load_y(Path(proto["y_path"]), Path(proto["data"]),
                  proto.get("data_sha256"))
    X = data.X
    provenance = {
        "run": out.name,
        "config": str(args.config),
        "data": str(proto["data"]),
        "data_sha256": rb.sha256_of(Path(proto["data"])),
        "folds": [f["fold"] for f in folds],
        "seed": int(tcfg["seed"]),
        "rules": list(RULES),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=1))

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        # The card is shared; stay well below the full device to avoid spill.
        torch.cuda.set_per_process_memory_fraction(0.45)
    print(f"[ckpt] device={device} torch={torch.__version__} "
          f"folds={[f['fold'] for f in folds]}", flush=True)

    rows: list[dict] = []
    oof_rows: list[dict] = []
    failures: list[dict] = []
    n_trainings = 0

    for fold in folds:
        k = fold["fold"]
        tr, va, te = fold["train"], fold["val"], fold["test"]
        for j, env in enumerate(envs):
            y = Y[:, j]
            t0 = time.time()
            model = None
            try:
                sx = rb.Scaler().fit(X[tr])
                sy = rb.Scaler().fit(y[tr])
                Xtr_s = sx.transform(X[tr])
                ytr_s = sy.transform(y[tr])
                Xva_s = sx.transform(X[va])
                yva_s = sy.transform(y[va])
                Xte_s = sx.transform(X[te])

                model = rb.build_model(X.shape[1], mcfg, seed=int(tcfg["seed"]))
                model, hist, fit = rb.train_one_fold(
                    model, Xtr_s, ytr_s, Xva_s, yva_s, tcfg, device,
                    seed=int(tcfg["seed"]))
                n_trainings += 1

                for rule in RULES:
                    if rule == "val_rmse":
                        if fit["best_rmse_state"] is None:
                            raise RuntimeError("no finite validation RMSE checkpoint")
                        model.load_state_dict(fit["best_rmse_state"])
                    model.eval()
                    pv = rb.predict(model, Xva_s, device)
                    pt = rb.predict(model, Xte_s, device)
                    pred = sy.inverse_transform(pt)
                    val_pred = sy.inverse_transform(pv)
                    vrow = rb.evaluate_fold(env, k, y[va], val_pred)
                    trow = rb.evaluate_fold(env, k, y[te], pred)
                    ps, ts, ratio = _sd_ratio(pred, y[te])
                    epoch = (fit["best_pcc_epoch"] if rule == "val_pcc"
                             else fit["best_rmse_epoch"])
                    row = {
                        "environment": env, "fold": k, "rule": rule,
                        "checkpoint_epoch": int(epoch),
                        "val_pcc": vrow["pcc"], "val_rmse": vrow["rmse"],
                        "test_pcc": trow["pcc"], "test_rmse": trow["rmse"],
                        "test_mae": trow["mae"],
                        "pred_sd": ps, "true_sd": ts, "sd_ratio": ratio,
                        "n_train": int(tr.size), "n_val": int(va.size),
                        "n_test": int(te.size),
                        "seconds": round(time.time() - t0, 1),
                        "n_params": int(sum(p.numel()
                                            for p in model.parameters())),
                        "failed": bool(vrow["failed"] or trow["failed"]),
                        "reason": ";".join(r for r in (vrow["reason"],
                                                       trow["reason"]) if r),
                    }
                    rows.append(row)
                    for i, idx in enumerate(te):
                        oof_rows.append({
                            "material_index": int(idx), "environment": env,
                            "fold": k, "rule": rule,
                            "y_true": float(y[idx]),
                            "y_pred": float(pred[i]),
                        })
                    print(f"  [env {env} fold {k} {rule}] "
                          f"test_pcc={row['test_pcc']:.4f} "
                          f"test_rmse={row['test_rmse']:.4f} "
                          f"pred_sd={row['pred_sd']:.4f} "
                          f"sd_ratio={row['sd_ratio']:.3f} "
                          f"ep={row['checkpoint_epoch']} {row['seconds']}s",
                          flush=True)
                    if row["failed"]:
                        failures.append({"environment": env, "fold": k,
                                         "rule": rule, "reason": row["reason"]})
            except Exception as exc:  # noqa: BLE001 - per-run isolation
                reason = f"{type(exc).__name__}: {exc}"
                for rule in RULES:
                    rows.append({
                        "environment": env, "fold": k, "rule": rule,
                        "checkpoint_epoch": -1,
                        "val_pcc": float("nan"), "val_rmse": float("nan"),
                        "test_pcc": float("nan"), "test_rmse": float("nan"),
                        "test_mae": float("nan"), "pred_sd": float("nan"),
                        "true_sd": float("nan"), "sd_ratio": float("nan"),
                        "n_train": int(tr.size), "n_val": int(va.size),
                        "n_test": int(te.size),
                        "seconds": round(time.time() - t0, 1),
                        "n_params": 0, "failed": True, "reason": reason,
                    })
                    failures.append({"environment": env, "fold": k,
                                     "rule": rule, "reason": reason})
                print(f"  [env {env} fold {k}] FAILED: {reason}", flush=True)
            finally:
                del model
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()

    with (out / "per_fold.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    with (out / "oof.csv").open("w", newline="") as f:
        fields = ["material_index", "environment", "fold", "rule",
                  "y_true", "y_pred"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(oof_rows)

    with (out / "failures.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["environment", "fold", "rule", "reason"])
        w.writeheader()
        w.writerows(failures)

    # ---- summaries ----
    summary: dict = {}
    for rule in RULES:
        per_env = {}
        for env in envs:
            rs = [r for r in rows if r["rule"] == rule
                  and r["environment"] == env and not r["failed"]]
            per_env[env] = {
                m: _summ([r[m] for r in rs]) for m in
                ("val_pcc", "val_rmse", "test_pcc", "test_rmse", "test_mae",
                 "pred_sd", "sd_ratio", "checkpoint_epoch")
            }
        per_env["macro"] = {
            m: _macro(per_env, envs, m) for m in
            ("test_pcc", "test_rmse", "test_mae", "pred_sd", "sd_ratio")
        }
        summary[rule] = per_env

    # Paired rule differences: val_rmse rule minus val_pcc rule.
    pairs = {}
    for r in rows:
        pairs.setdefault((r["environment"], r["fold"]), {})[r["rule"]] = r
    paired = {}
    for env in envs:
        diffs = {m: [] for m in ("test_pcc", "test_rmse", "test_mae",
                                 "pred_sd", "sd_ratio")}
        for (e, _fold), by_rule in pairs.items():
            if e != env or set(by_rule) != set(RULES):
                continue
            a, b = by_rule["val_rmse"], by_rule["val_pcc"]
            if a["failed"] or b["failed"]:
                continue
            for m in diffs:
                diffs[m].append(a[m] - b[m])
        paired[env] = {m: _summ(v) for m, v in diffs.items()}
        paired[env]["n_rmse_rule_better_pcc"] = int(sum(
            1 for (e, _f), by_rule in pairs.items()
            if e == env and set(by_rule) == set(RULES)
            and not by_rule["val_rmse"]["failed"]
            and not by_rule["val_pcc"]["failed"]
            and by_rule["val_rmse"]["test_pcc"] > by_rule["val_pcc"]["test_pcc"]))
        paired[env]["n_rmse_rule_lower_rmse"] = int(sum(
            1 for (e, _f), by_rule in pairs.items()
            if e == env and set(by_rule) == set(RULES)
            and not by_rule["val_rmse"]["failed"]
            and not by_rule["val_pcc"]["failed"]
            and by_rule["val_rmse"]["test_rmse"] < by_rule["val_pcc"]["test_rmse"]))
    paired["macro"] = {
        m: _summ([paired[e][m]["mean"] for e in envs
                  if np.isfinite(paired[e][m]["mean"])])
        for m in ("test_pcc", "test_rmse", "test_mae", "pred_sd", "sd_ratio")
    }
    paired["macro"]["total_rmse_rule_better_pcc"] = int(sum(
        paired[e]["n_rmse_rule_better_pcc"] for e in envs))
    paired["macro"]["total_rmse_rule_lower_rmse"] = int(sum(
        paired[e]["n_rmse_rule_lower_rmse"] for e in envs))

    rr = _load_rrblup(args.rrblup)
    vs_rr = {}
    if rr:
        for rule in RULES:
            per_env = {}
            for env in envs:
                diffs = {"pcc": [], "rmse": [], "mae": []}
                for r in rows:
                    if (r["rule"] != rule or r["environment"] != env
                            or r["failed"]):
                        continue
                    ref = rr.get((env, r["fold"]))
                    if ref is None:
                        continue
                    diffs["pcc"].append(r["test_pcc"] - ref["pcc"])
                    diffs["rmse"].append(r["test_rmse"] - ref["rmse"])
                    diffs["mae"].append(r["test_mae"] - ref["mae"])
                per_env[env] = {m: _summ(v) for m, v in diffs.items()}
            per_env["macro"] = {
                m: _summ([per_env[e][m]["mean"] for e in envs
                          if np.isfinite(per_env[e][m]["mean"])])
                for m in ("pcc", "rmse", "mae")
            }
            vs_rr[rule] = per_env
        vs_rr["reference"] = {
            "path": str(args.rrblup),
            "n_folds": len(rr),
            "macro": {
                m: _summ([np.mean([v[m] for (e, _f), v in rr.items()
                                   if e == env]) for env in envs])
                for m in ("pcc", "rmse", "mae")
            },
        }

    summary_out = {
        "provenance": provenance,
        "rules": list(RULES),
        "summary": summary,
        "paired_rmse_minus_pcc": paired,
        "vs_rrblup": vs_rr,
        "n_trainings": n_trainings,
        "n_rule_evaluations": len(rows),
        "n_failures": len(failures),
        "limits": [
            "one training run per (environment, fold); both checkpoints come "
            "from the same trajectory",
            "10 folds share materials -> not independent; descriptive only, "
            "no p-values/CIs",
            "single seed (42); initialization seed fixed before build_model",
            "test is not used to choose or combine rules",
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary_out, indent=1))
    print(f"[ckpt] done. trainings={n_trainings} evals={len(rows)} "
          f"failures={len(failures)} -> {out}/", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
