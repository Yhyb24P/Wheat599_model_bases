#!/usr/bin/env python
"""Fixed RR-BLUP baseline + FT-Transformer residual correction (first round).

For each (environment, fold):

* fit the current RR-BLUP on the train fold (train-only centering; alpha chosen
  on the validation fold by RMSE) and freeze it;
* standardize y with the train-fold scaler;
* train the current A FT-Transformer backbone on the train residual
  z - b^z, where b^z = (RR_train - mu_T) / s_T;
* begin from an exactly-zero correction by zeroing only the final linear
  output layer;
* use the combined prediction RR + s_T * g for validation, scheduler and
  checkpoint selection;
* compare the residual model's validation PCC with the RR-only candidate
  before touching test. Strictly greater selects residual; a tie or a lower
  value keeps RR-only. Test is evaluated once per selected branch per fold.

Candidate test metrics (RR-only and residual) are stored as diagnostic rows but
are never used to choose a branch.  "A" in "A backbone" is the architecture
label; this runner remains marker-only and does not use the pedigree wheat.A.
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
from wheat599.metrics import mae, pcc, rmse  # noqa: E402

NEAR_CONSTANT_RATIO = 0.1
EXPECTED_N_PARAMS = 247_425
FIELDS = [
    "environment", "fold", "alpha",
    "rr_val_pcc", "rr_val_rmse",
    "residual_val_pcc", "residual_val_rmse",
    "best_epoch", "best_val_pcc", "best_val_rmse",
    "selected", "selected_is_residual",
    "test_pcc", "test_rmse", "test_mae", "test_pred_sd", "test_true_sd",
    "test_sd_ratio", "near_constant",
    "rr_test_pcc", "rr_test_rmse", "rr_test_mae", "rr_test_pred_sd",
    "residual_test_pcc", "residual_test_rmse", "residual_test_mae",
    "residual_test_pred_sd",
    "correction_mean_val_std", "correction_sd_val_std",
    "correction_mean_test_std", "correction_sd_test_std",
    "correction_sd_test_orig", "correction_to_target_sd",
    "target_sd_train_std", "target_sd_val_std",
    "mse_selected", "mse_offset_selected", "mse_direction_selected",
    "mse_scale_selected", "r_selected", "k_selected", "b_selected",
    "mse_rr", "mse_offset_rr", "mse_direction_rr", "mse_scale_rr",
    "r_rr", "k_rr", "b_rr",
    "mse_residual", "mse_offset_residual", "mse_direction_residual",
    "mse_scale_residual", "r_residual", "k_residual", "b_residual",
    "n_train", "n_val", "n_test", "seconds", "n_params",
    "failed", "reason",
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


def _decompose(y, pred) -> dict:
    """Exact per-fold MSE decomposition, all terms in MSE units."""
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    sy = float(y.std(ddof=0))
    sp = float(pred.std(ddof=0))
    r = float(np.corrcoef(y, pred)[0, 1]) if sy > 0 and sp > 0 else float("nan")
    k = sp / sy if sy > 0 else float("nan")
    b = float(pred.mean() - y.mean())
    mse = float(np.mean((pred - y) ** 2))
    offset = b * b
    direction = sy * sy * (1.0 - r * r) if np.isfinite(r) else float("nan")
    scale = sy * sy * (k - r) ** 2 if np.isfinite(r) else float("nan")
    return {
        "mse": mse, "offset": offset, "direction": direction,
        "scale": scale, "r": r, "k": k, "b": b,
    }


def _load_rr_reference(path: Path) -> dict:
    if not path.exists():
        return {}
    out = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            if str(row.get("failed", "")).lower() == "true":
                continue
            out[(str(row["environment"]), int(row["fold"]))] = {
                "alpha": float(row["alpha"]),
                "val_rmse": float(row["val_rmse"]),
                "test_pcc": float(row["test_pcc"]),
                "test_rmse": float(row["test_rmse"]),
                "test_mae": float(row["test_mae"]),
            }
    return out


def _macro(per_env: dict, envs: list[str], metric: str) -> dict:
    vals = [per_env[e][metric]["mean"] for e in envs
            if np.isfinite(per_env[e][metric]["mean"])]
    return _summ(vals)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path,
                    default=PROJ / "configs" / "bimsgp_residual.yaml")
    ap.add_argument("--out", type=Path,
                    default=PROJ / "results" / "bimsgp_residual_20261030")
    ap.add_argument("--folds", nargs="*", type=int, default=None)
    ap.add_argument("--rr-reference", type=Path,
                    default=PROJ / "results" / "bimsgp_diag_fixed_20261029"
                    / "per_fold_rrblup.csv")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text())
    proto, mcfg, tcfg = cfg["protocol"], cfg["model"], cfg["training"]
    rcfg = cfg["residual"]
    envs = [str(e) for e in proto["environments"]]
    if rcfg.get("baseline") != "rrblup":
        raise ValueError(f"unsupported baseline: {rcfg.get('baseline')}")
    if not rcfg.get("zero_init_output", False):
        raise ValueError("residual runner requires zero_init_output: true")
    if rcfg.get("checkpoint") != "val_pcc":
        raise ValueError("residual runner expects checkpoint: val_pcc")

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
        "baseline": "rrblup",
        "correction": "standardized_residual",
        "zero_init_output": True,
        "selection": "val_pcc_strictly_greater_than_rr_only",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=1))

    rr_ref = _load_rr_reference(args.rr_reference)
    rr_check = {"path": str(args.rr_reference), "n": len(rr_ref),
                "max_alpha_diff": 0.0, "max_val_rmse_diff": 0.0,
                "max_test_pcc_diff": 0.0, "max_test_rmse_diff": 0.0,
                "max_test_mae_diff": 0.0}

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.45)
    print(f"[residual] device={device} torch={torch.__version__} "
          f"folds={[f['fold'] for f in folds]}", flush=True)

    rows: list[dict] = []
    oof_rows: list[dict] = []
    history_rows: list[dict] = []
    failures: list[dict] = []

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
                Xva_s = sx.transform(X[va])
                Xte_s = sx.transform(X[te])
                ytr_s, yva_s = sy.transform(y[tr]), sy.transform(y[va])

                rr = rb.fit_rrblup(X[tr], y[tr], X[va], y[va])
                rr_tr = rb.predict_rrblup(rr, X[tr])
                rr_va = rb.predict_rrblup(rr, X[va])
                rr_te = rb.predict_rrblup(rr, X[te])
                rr_tr_s, rr_va_s, rr_te_s = (sy.transform(a)
                                             for a in (rr_tr, rr_va, rr_te))
                target_tr = ytr_s - rr_tr_s
                target_va = yva_s - rr_va_s

                if (env, k) in rr_ref:
                    ref = rr_ref[(env, k)]
                    rr_check["max_alpha_diff"] = max(
                        rr_check["max_alpha_diff"],
                        abs(rr["alpha"] - ref["alpha"]))
                    rr_check["max_val_rmse_diff"] = max(
                        rr_check["max_val_rmse_diff"],
                        abs(rr["val_rmse"] - ref["val_rmse"]))
                    rr_check["max_test_pcc_diff"] = max(
                        rr_check["max_test_pcc_diff"],
                        abs(pcc(y[te], rr_te) - ref["test_pcc"]))
                    rr_check["max_test_rmse_diff"] = max(
                        rr_check["max_test_rmse_diff"],
                        abs(rmse(y[te], rr_te) - ref["test_rmse"]))
                    rr_check["max_test_mae_diff"] = max(
                        rr_check["max_test_mae_diff"],
                        abs(mae(y[te], rr_te) - ref["test_mae"]))

                model = rb.build_model(X.shape[1], mcfg,
                                       seed=int(tcfg["seed"]))
                rb.zero_init_output(model)
                model = model.to(device)
                model.eval()
                with torch.no_grad():
                    z = torch.as_tensor(Xtr_s[:2], dtype=torch.float32,
                                        device=device)
                    initial = model(x_cont=z, x_cat=None)
                if not torch.allclose(initial, torch.zeros_like(initial),
                                      atol=1e-6):
                    raise RuntimeError("zero-init output is not exactly zero")
                model, hist, fit = rb.train_one_fold(
                    model, Xtr_s, target_tr, Xva_s, target_va, tcfg, device,
                    seed=int(tcfg["seed"]), base_tr=rr_tr_s, base_val=rr_va_s)

                g_tr = rb.predict(model, Xtr_s, device)
                g_va = rb.predict(model, Xva_s, device)
                g_te = rb.predict(model, Xte_s, device)
                res_tr = rb.residual_prediction(rr_tr, g_tr, sy)
                res_va = rb.residual_prediction(rr_va, g_va, sy)
                res_te = rb.residual_prediction(rr_te, g_te, sy)

                rr_val_pcc = float(pcc(y[va], rr_va))
                rr_val_rmse = float(rmse(y[va], rr_va))
                residual_val_pcc = float(pcc(y[va], res_va))
                residual_val_rmse = float(rmse(y[va], res_va))
                if abs(residual_val_pcc - fit["best_pcc"]) > 1e-6:
                    raise RuntimeError(
                        "combined validation PCC does not match checkpoint "
                        f"{residual_val_pcc} vs {fit['best_pcc']}")
                # Pre-specified rule: strict improvement only; ties keep RR.
                selected_is_residual = bool(residual_val_pcc > rr_val_pcc)
                selected = "residual" if selected_is_residual else "rr_only"
                selected_te = res_te if selected_is_residual else rr_te

                rr_m = _decompose(y[te], rr_te)
                res_m = _decompose(y[te], res_te)
                sel_m = _decompose(y[te], selected_te)
                pred_sd = float(np.std(selected_te, ddof=1))
                true_sd = float(np.std(y[te], ddof=1))
                corr_va = float(np.std(g_va, ddof=1))
                corr_te = float(np.std(g_te, ddof=1))
                target_va_sd = float(np.std(target_va, ddof=1))
                target_tr_sd = float(np.std(target_tr, ddof=1))
                n_params = int(sum(p.numel() for p in model.parameters()))
                if n_params != EXPECTED_N_PARAMS:
                    raise RuntimeError(
                        f"unexpected parameter count {n_params}, "
                        f"expected {EXPECTED_N_PARAMS}")

                row = {
                    "environment": env, "fold": k, "alpha": rr["alpha"],
                    "rr_val_pcc": rr_val_pcc, "rr_val_rmse": rr_val_rmse,
                    "residual_val_pcc": residual_val_pcc,
                    "residual_val_rmse": residual_val_rmse,
                    "best_epoch": fit["best_pcc_epoch"],
                    "best_val_pcc": fit["best_pcc"],
                    "best_val_rmse": fit["best_rmse"],
                    "selected": selected,
                    "selected_is_residual": selected_is_residual,
                    "test_pcc": float(pcc(y[te], selected_te)),
                    "test_rmse": float(rmse(y[te], selected_te)),
                    "test_mae": float(mae(y[te], selected_te)),
                    "test_pred_sd": pred_sd,
                    "test_true_sd": true_sd,
                    "test_sd_ratio": pred_sd / true_sd if true_sd > 0 else float("nan"),
                    "near_constant": bool(pred_sd < NEAR_CONSTANT_RATIO * true_sd),
                    "rr_test_pcc": float(pcc(y[te], rr_te)),
                    "rr_test_rmse": float(rmse(y[te], rr_te)),
                    "rr_test_mae": float(mae(y[te], rr_te)),
                    "rr_test_pred_sd": float(np.std(rr_te, ddof=1)),
                    "residual_test_pcc": float(pcc(y[te], res_te)),
                    "residual_test_rmse": float(rmse(y[te], res_te)),
                    "residual_test_mae": float(mae(y[te], res_te)),
                    "residual_test_pred_sd": float(np.std(res_te, ddof=1)),
                    "correction_mean_val_std": float(np.mean(g_va)),
                    "correction_sd_val_std": corr_va,
                    "correction_mean_test_std": float(np.mean(g_te)),
                    "correction_sd_test_std": corr_te,
                    "correction_sd_test_orig": float(sy.std * corr_te),
                    "correction_to_target_sd": (corr_va / target_va_sd
                                                if target_va_sd > 0 else float("nan")),
                    "target_sd_train_std": target_tr_sd,
                    "target_sd_val_std": target_va_sd,
                    "mse_selected": sel_m["mse"],
                    "mse_offset_selected": sel_m["offset"],
                    "mse_direction_selected": sel_m["direction"],
                    "mse_scale_selected": sel_m["scale"],
                    "r_selected": sel_m["r"], "k_selected": sel_m["k"],
                    "b_selected": sel_m["b"],
                    "mse_rr": rr_m["mse"], "mse_offset_rr": rr_m["offset"],
                    "mse_direction_rr": rr_m["direction"],
                    "mse_scale_rr": rr_m["scale"],
                    "r_rr": rr_m["r"], "k_rr": rr_m["k"], "b_rr": rr_m["b"],
                    "mse_residual": res_m["mse"],
                    "mse_offset_residual": res_m["offset"],
                    "mse_direction_residual": res_m["direction"],
                    "mse_scale_residual": res_m["scale"],
                    "r_residual": res_m["r"], "k_residual": res_m["k"],
                    "b_residual": res_m["b"],
                    "n_train": int(tr.size), "n_val": int(va.size),
                    "n_test": int(te.size),
                    "seconds": round(time.time() - t0, 1),
                    "n_params": n_params,
                    "failed": False, "reason": "",
                }
                rows.append(row)
                for h in hist:
                    history_rows.append({"environment": env, "fold": k, **h})
                for candidate, pred in (("selected", selected_te),
                                        ("rr_only", rr_te),
                                        ("residual", res_te)):
                    for i, idx in enumerate(te):
                        oof_rows.append({
                            "material_index": int(idx), "environment": env,
                            "fold": k, "candidate": candidate,
                            "y_true": float(y[idx]),
                            "y_pred": float(pred[i]),
                        })
                print(f"  [env {env} fold {k}] selected={selected} "
                      f"rr_val={rr_val_pcc:.4f} res_val={residual_val_pcc:.4f} "
                      f"test_pcc={row['test_pcc']:.4f} "
                      f"test_rmse={row['test_rmse']:.4f} "
                      f"corr_sd={row['correction_sd_test_std']:.3f} "
                      f"target_val_sd={target_va_sd:.3f} "
                      f"ep={row['best_epoch']} {row['seconds']}s", flush=True)
            except Exception as exc:  # noqa: BLE001 - per-run isolation
                reason = f"{type(exc).__name__}: {exc}"
                row = {name: float("nan") for name in FIELDS}
                row.update({
                    "environment": env, "fold": k, "selected": "failed",
                    "selected_is_residual": False, "failed": True,
                    "reason": reason, "n_train": int(tr.size),
                    "n_val": int(va.size), "n_test": int(te.size),
                    "seconds": round(time.time() - t0, 1),
                })
                rows.append(row)
                failures.append({"environment": env, "fold": k,
                                 "reason": reason})
                print(f"  [env {env} fold {k}] FAILED: {reason}", flush=True)
            finally:
                del model
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()

    if rr_ref:
        max_diff = max(rr_check[k] for k in
                       ("max_alpha_diff", "max_val_rmse_diff",
                        "max_test_pcc_diff", "max_test_rmse_diff",
                        "max_test_mae_diff"))
        if max_diff > 1e-12:
            raise RuntimeError(f"RR-BLUP reference mismatch: {rr_check}")
        rr_check["passed"] = True
    else:
        rr_check["passed"] = False

    with (out / "per_fold.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    with (out / "oof.csv").open("w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["material_index", "environment", "fold",
                           "candidate", "y_true", "y_pred"],
            lineterminator="\n")
        w.writeheader()
        w.writerows(oof_rows)
    with (out / "failures.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["environment", "fold", "reason"],
                           lineterminator="\n")
        w.writeheader()
        w.writerows(failures)
    with (out / "history.csv").open("w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["environment", "fold", "epoch", "train_rmse",
                           "val_rmse", "val_pcc", "lr"],
            lineterminator="\n")
        w.writeheader()
        w.writerows(history_rows)

    def env_summary() -> dict:
        out_env = {}
        for env in envs:
            rs = [r for r in rows if r["environment"] == env and not r["failed"]]
            out_env[env] = {
                m: _summ([r[m] for r in rs]) for m in
                ("test_pcc", "test_rmse", "test_mae", "test_pred_sd",
                 "test_sd_ratio", "correction_sd_val_std",
                 "correction_sd_test_std", "correction_to_target_sd",
                 "target_sd_train_std", "target_sd_val_std",
                 "mse_offset_selected", "mse_direction_selected",
                 "mse_scale_selected", "mse_selected",
                 "r_selected", "k_selected", "b_selected")
            }
            out_env[env]["n_selected"] = len(rs)
            out_env[env]["n_near_constant"] = int(
                sum(1 for r in rs if r["near_constant"]))
        return out_env

    selected_env = env_summary()
    selected_env["macro"] = {
        m: _macro(selected_env, envs, m) for m in
        ("test_pcc", "test_rmse", "test_mae", "test_pred_sd",
         "test_sd_ratio", "correction_sd_val_std",
         "correction_sd_test_std", "correction_to_target_sd",
         "target_sd_train_std", "target_sd_val_std",
         "mse_offset_selected", "mse_direction_selected",
         "mse_scale_selected", "mse_selected", "r_selected", "k_selected",
         "b_selected")
    }
    selected_env["macro"]["n_selected"] = sum(
        selected_env[e]["n_selected"] for e in envs)
    selected_env["macro"]["n_near_constant"] = sum(
        selected_env[e]["n_near_constant"] for e in envs)

    candidates = {}
    for candidate in ("rr_only", "residual"):
        per_env = {}
        for env in envs:
            rs = [r for r in rows if r["environment"] == env and not r["failed"]]
            per_env[env] = {
                **{m: _summ([r[m] for r in rs]) for m in
                   ("test_pcc", "test_rmse", "test_mae", "test_pred_sd",
                    "rr_val_pcc", "rr_val_rmse")},
                **{f"rr_test_{m}": _summ([r[f"rr_test_{m}"] for r in rs])
                   for m in ("pcc", "rmse", "mae")},
                **{f"residual_test_{m}": _summ(
                    [r[f"residual_test_{m}"] for r in rs])
                   for m in ("pcc", "rmse", "mae")},
            }
        per_env["macro"] = {
            m: _macro(per_env, envs, m) for m in
            ("test_pcc", "test_rmse", "test_mae", "test_pred_sd",
             "rr_val_pcc", "rr_val_rmse",
             "rr_test_pcc", "rr_test_rmse", "rr_test_mae",
             "residual_test_pcc", "residual_test_rmse",
             "residual_test_mae")
        }
        candidates[candidate] = per_env

    selection = {}
    for env in envs:
        rs = [r for r in rows if r["environment"] == env and not r["failed"]]
        selection[env] = {
            "n": len(rs),
            "rr_only": int(sum(r["selected"] == "rr_only" for r in rs)),
            "residual": int(sum(r["selected"] == "residual" for r in rs)),
            "rate_residual": (float(sum(r["selected"] == "residual"
                                        for r in rs)) / len(rs)
                              if rs else float("nan")),
            "median_best_epoch": float(np.median(
                [r["best_epoch"] for r in rs])) if rs else float("nan"),
        }
    selection["macro"] = {
        "n": sum(selection[e]["n"] for e in envs),
        "rr_only": sum(selection[e]["rr_only"] for e in envs),
        "residual": sum(selection[e]["residual"] for e in envs),
    }
    selection["macro"]["rate_residual"] = (
        selection["macro"]["residual"] / selection["macro"]["n"]
        if selection["macro"]["n"] else float("nan"))

    vs_rr = {}
    for env in envs:
        rs = [r for r in rows if r["environment"] == env and not r["failed"]]
        vs_rr[env] = {
            "d_pcc": _summ([r["test_pcc"] - r["rr_test_pcc"] for r in rs]),
            "d_rmse": _summ([r["test_rmse"] - r["rr_test_rmse"] for r in rs]),
            "d_mae": _summ([r["test_mae"] - r["rr_test_mae"] for r in rs]),
            "n_pcc_better": int(sum(r["test_pcc"] > r["rr_test_pcc"]
                                    for r in rs)),
            "n_rmse_lower": int(sum(r["test_rmse"] < r["rr_test_rmse"]
                                    for r in rs)),
            "n_mae_lower": int(sum(r["test_mae"] < r["rr_test_mae"]
                                   for r in rs)),
        }
    vs_rr["macro"] = {
        m: _macro(vs_rr, envs, m) for m in ("d_pcc", "d_rmse", "d_mae")
    }
    vs_rr["macro"]["n_pcc_better"] = sum(
        vs_rr[e]["n_pcc_better"] for e in envs)
    vs_rr["macro"]["n_rmse_lower"] = sum(
        vs_rr[e]["n_rmse_lower"] for e in envs)
    vs_rr["macro"]["n_mae_lower"] = sum(
        vs_rr[e]["n_mae_lower"] for e in envs)

    summary = {
        "provenance": provenance,
        "rr_reference_check": rr_check,
        "selected": selected_env,
        "selection": selection,
        "vs_rrblup_selected": vs_rr,
        "candidate_diagnostics": candidates,
        "n_trainings": sum(1 for r in rows if not r["failed"]),
        "n_failures": len(failures),
        "limits": [
            "10 folds share materials -> not independent; descriptive only, "
            "no p-values/CIs",
            "single seed (42); seed-42 split has participated in earlier model "
            "development, so this is not an independent confirmation",
            "validation is used twice: RR alpha and branch/epoch selection; "
            "val-side selection optimism is not corrected",
            "RR training predictions are in-sample, so train residual SD is "
            "smaller than validation/test residual SD; correction magnitude "
            "diagnostics are reported for this reason",
            "test labels are never used to choose the branch; candidate test "
            "metrics for the unselected branch are diagnostic only",
            "marker-only; 'A backbone' is the FT-Transformer architecture "
            "label, not the pedigree matrix wheat.A",
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"[residual] done. trainings={len(rows)} "
          f"selected_residual={selection['macro']['residual']}/"
          f"{selection['macro']['n']} failures={len(failures)} -> {out}/",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
