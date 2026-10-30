#!/usr/bin/env python
"""FT-T candidate with original affine tokenizer, one full marker interaction
block, and a position-dependent factorized linear readout.

First round is the complete model; no component ablation, no grid search, no
test-based selection.  "Position" in the readout means the marker's own
identity/index in the 1,279-column input; no genomic coordinate is used.
Marker-only; "A" is the previous FT-Transformer architecture label, not the
pedigree wheat.A.
"""

from __future__ import annotations

import argparse
import csv
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
from wheat599.direct_ftt import build_direct_model  # noqa: E402

FIELDS = [
    "environment", "fold", "pcc", "rmse", "mae", "pred_sd", "true_sd",
    "sd_ratio", "best_epoch", "best_val_pcc", "n_train", "n_val", "n_test",
    "seconds", "n_params", "tokenizer_params", "attention_scores",
    "pred_sd_initial", "token_norm_initial", "token_norm_final",
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


def _macro(per_env: dict, envs: list[str], metric: str) -> dict:
    vals = [per_env[e][metric]["mean"] for e in envs
            if np.isfinite(per_env[e][metric]["mean"])]
    return _summ(vals)


def _load_reference(path: Path, rule: str | None = None) -> dict:
    if not path.exists():
        return {}
    out = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            if rule is not None and row.get("rule") != rule:
                continue
            if str(row.get("failed", "")).lower() in ("true", "1"):
                continue
            out[(str(row["environment"]), int(row["fold"]))] = {
                "pcc": float(row["test_pcc"]),
                "rmse": float(row["test_rmse"]),
                "mae": float(row["test_mae"]),
            }
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path,
                    default=PROJ / "configs" / "bimsgp_direct.yaml")
    ap.add_argument("--out", type=Path,
                    default=PROJ / "results" / "bimsgp_direct_20261030")
    ap.add_argument("--folds", nargs="*", type=int, default=None)
    ap.add_argument("--a-reference", type=Path,
                    default=PROJ / "results" / "bimsgp_ckpt_rules_20261030"
                    / "per_fold.csv")
    ap.add_argument("--rr-reference", type=Path,
                    default=PROJ / "results" / "bimsgp_diag_fixed_20261029"
                    / "per_fold_rrblup.csv")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text())
    proto, mcfg, tcfg = cfg["protocol"], cfg["model"], cfg["training"]
    if mcfg.get("model_type") != "direct_ftt":
        raise ValueError(f"unsupported model_type: {mcfg.get('model_type')}")
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
    a_ref = _load_reference(args.a_reference, rule="val_pcc")
    rr_ref = _load_reference(args.rr_reference)
    model_meta = {
        "architecture": "original_ft_tokenizer + one_full_marker_attention"
                        " + factorized_linear_readout",
        "d_block": int(mcfg["d_block"]),
        "attention_n_heads": int(mcfg["attention_n_heads"]),
        "ffn_d_hidden": int(mcfg["ffn_d_hidden"]),
        "readout_rank": int(mcfg["readout_rank"]),
        "readout": "y = bias + <H, G^T T1>; V=GH",
        "position_semantics": "marker identity/index; no genomic coordinate"
                              " or positional encoding",
        "tokenizer_initialization": "b_j,w_j ~ Uniform(-1/sqrt(d),1/sqrt(d))",
        "attention_implementation": "PyTorch scaled_dot_product_attention",
        "tokenizer_weight_decay": False,
        "readout_weight_decay": True,
    }
    provenance = {
        "run": out.name,
        "config": str(args.config),
        "data": str(proto["data"]),
        "data_sha256": rb.sha256_of(Path(proto["data"])),
        "folds": [f["fold"] for f in folds],
        "seed": int(tcfg["seed"]),
        "model": model_meta,
        "a_reference": str(args.a_reference),
        "rr_reference": str(args.rr_reference),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=1))

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.45)
    print(f"[direct] device={device} torch={torch.__version__} "
          f"folds={[f['fold'] for f in folds]}", flush=True)

    rows: list[dict] = []
    oof_rows: list[dict] = []
    history_rows: list[dict] = []
    failures: list[dict] = []
    model_info = None

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
                ytr_s = sy.transform(y[tr])
                yva_s = sy.transform(y[va])

                model = build_direct_model(X.shape[1], mcfg,
                                           seed=int(tcfg["seed"]))
                model = model.to(device)
                token_norm_initial = model.tokenizer.token_norm_mean()
                initial_pred = rb.predict(model, Xtr_s, device)
                pred_sd_initial = float(np.std(initial_pred, ddof=1))
                if not 0.005 <= pred_sd_initial <= 0.2:
                    raise RuntimeError(
                        f"initial prediction SD out of range: {pred_sd_initial}")
                if model_info is None:
                    n_params = model.parameter_count()
                    model_info = {
                        "n_params": n_params,
                        "tokenizer_params": model.tokenizer_parameter_count(),
                        "attention_scores_per_sample": model.attention_score_count(),
                        "parameter_breakdown": model.parameter_breakdown(),
                        "n_params_A": 247_425,
                        "n_params_ratio_vs_A": n_params / 247_425,
                    }
                model, hist, fit = rb.train_one_fold(
                    model, Xtr_s, ytr_s, Xva_s, yva_s, tcfg, device,
                    seed=int(tcfg["seed"]))
                token_norm_final = model.tokenizer.token_norm_mean()
                pred = sy.inverse_transform(rb.predict(model, Xte_s, device))
                row = rb.evaluate_fold(env, k, y[te], pred)
                ps = float(np.std(pred, ddof=1))
                ts = float(np.std(y[te], ddof=1))
                row.update({
                    "pred_sd": ps, "true_sd": ts,
                    "sd_ratio": ps / ts if ts > 0 else float("nan"),
                    "best_epoch": fit["best_pcc_epoch"],
                    "best_val_pcc": fit["best_pcc"],
                    "n_train": int(tr.size), "n_val": int(va.size),
                    "n_test": int(te.size),
                    "seconds": round(time.time() - t0, 1),
                    "n_params": model.parameter_count(),
                    "tokenizer_params": model.tokenizer_parameter_count(),
                    "attention_scores": model.attention_score_count(),
                    "pred_sd_initial": pred_sd_initial,
                    "token_norm_initial": token_norm_initial,
                    "token_norm_final": token_norm_final,
                })
                rows.append(row)
                for h in hist:
                    history_rows.append({"environment": env, "fold": k, **h})
                for i, idx in enumerate(te):
                    oof_rows.append({
                        "material_index": int(idx), "environment": env,
                        "fold": k, "y_true": float(y[idx]),
                        "y_pred": float(pred[i]),
                    })
                print(f"  [env {env} fold {k}] pcc={row['pcc']:.4f} "
                      f"rmse={row['rmse']:.4f} pred_sd={ps:.4f} "
                      f"best_epoch={fit['best_pcc_epoch']} "
                      f"{row['seconds']}s"
                      f"{' FAILED:'+row['reason'] if row['failed'] else ''}",
                      flush=True)
            except Exception as exc:  # noqa: BLE001 - per-run isolation
                reason = f"{type(exc).__name__}: {exc}"
                row = {name: float("nan") for name in FIELDS}
                row.update({
                    "environment": env, "fold": k, "failed": True,
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

    with (out / "per_fold.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    with (out / "oof.csv").open("w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["material_index", "environment", "fold",
                           "y_true", "y_pred"], lineterminator="\n")
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

    per_env = {}
    for env in envs:
        rs = [r for r in rows if r["environment"] == env and not r["failed"]]
        per_env[env] = {
            m: _summ([r[m] for r in rs]) for m in
            ("pcc", "rmse", "mae", "pred_sd", "sd_ratio",
             "best_epoch", "best_val_pcc")
        }
    per_env["macro"] = {
        m: _macro(per_env, envs, m) for m in
        ("pcc", "rmse", "mae", "pred_sd", "sd_ratio")
    }

    def vs_ref(ref: dict) -> dict:
        out_ref = {}
        for env in envs:
            diffs = {"pcc": [], "rmse": [], "mae": []}
            for r in rows:
                if r["environment"] != env or r["failed"]:
                    continue
                key = (env, r["fold"])
                if key not in ref:
                    continue
                diffs["pcc"].append(r["pcc"] - ref[key]["pcc"])
                diffs["rmse"].append(r["rmse"] - ref[key]["rmse"])
                diffs["mae"].append(r["mae"] - ref[key]["mae"])
            out_ref[env] = {
                "d_pcc": _summ(diffs["pcc"]),
                "d_rmse": _summ(diffs["rmse"]),
                "d_mae": _summ(diffs["mae"]),
                "n_pcc_better": int(sum(v > 0 for v in diffs["pcc"])),
                "n_rmse_lower": int(sum(v < 0 for v in diffs["rmse"])),
                "n_mae_lower": int(sum(v < 0 for v in diffs["mae"])),
            }
        out_ref["macro"] = {
            m: _macro(out_ref, envs, m) for m in ("d_pcc", "d_rmse", "d_mae")
        }
        out_ref["macro"]["n_pcc_better"] = sum(
            out_ref[e]["n_pcc_better"] for e in envs)
        out_ref["macro"]["n_rmse_lower"] = sum(
            out_ref[e]["n_rmse_lower"] for e in envs)
        out_ref["macro"]["n_mae_lower"] = sum(
            out_ref[e]["n_mae_lower"] for e in envs)
        out_ref["n_reference"] = len(ref)
        return out_ref

    vs_a = vs_ref(a_ref)
    vs_rr = vs_ref(rr_ref)
    go_cfg = cfg.get("decision", {}).get("go", {})
    d_a_pcc = vs_a["macro"]["d_pcc"]["mean"]
    d_a_rmse = vs_a["macro"]["d_rmse"]["mean"]
    d_rr_pcc = vs_rr["macro"]["d_pcc"]["mean"]
    env_pos = sum(vs_a[e]["d_pcc"]["mean"] > 0 for e in envs)
    g1 = bool(np.isfinite(d_a_pcc)
              and d_a_pcc >= float(go_cfg["vs_A_macro_dpcc_min"])
              and env_pos >= int(go_cfg["vs_A_env_dpcc_positive_min"]))
    g2 = bool(np.isfinite(d_a_rmse)
              and d_a_rmse <= float(go_cfg["vs_A_macro_drmse_max"]))
    g3 = bool(np.isfinite(d_rr_pcc)
              and d_rr_pcc >= float(go_cfg["vs_RR_macro_dpcc_min"]))
    go = {
        "criteria": go_cfg,
        "values": {
            "vs_A_macro_dpcc": d_a_pcc,
            "vs_A_env_dpcc_positive": int(env_pos),
            "vs_A_macro_drmse": d_a_rmse,
            "vs_RR_macro_dpcc": d_rr_pcc,
        },
        "G1": g1, "G2": g2, "G3": g3,
        "GO": bool(g1 and g2 and g3),
    }

    summary = {
        "provenance": provenance,
        "model": model_info,
        "summary": per_env,
        "vs_A": vs_a,
        "vs_RRBLUP": vs_rr,
        "go": go,
        "n_runs": len(rows),
        "n_failures": len(failures),
        "limits": [
            "10 folds share materials -> not independent; descriptive only, "
            "no p-values/CIs",
            "single seed (42); seed-42 split has participated in earlier model "
            "development, so this is not an independent confirmation",
            "test is evaluated once per fold after training/checkpoint and is "
            "not used for architecture or hyperparameter selection",
            "first round is the complete model; no component ablation and no "
            "architecture grid",
            "readout 'position' means marker identity/index; no genomic "
            "coordinate or biological position is used",
            "marker-only; 'A' is the previous FT-Transformer architecture, "
            "not the pedigree matrix wheat.A",
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"[direct] GO={go['GO']} G1={g1} G2={g2} G3={g3}; "
          f"runs={len(rows)} failures={len(failures)} -> {out}/",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
