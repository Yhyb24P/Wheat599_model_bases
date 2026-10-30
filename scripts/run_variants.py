#!/usr/bin/env python
"""Val-only variant selection (A/B/C/D), then test only the selected variant."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_bimsgp as rb  # noqa: E402
from wheat599.data import load  # noqa: E402
from wheat599.metrics import pcc, rmse  # noqa: E402

VARIANTS = {
    "A_baseline": {},
    "B_d32": {"d_block": 32},
    "C_1block": {"n_blocks": 1},
    "D_ffn_dropout0.2": {"ffn_dropout": 0.2},
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path("configs/bimsgp.yaml"))
    ap.add_argument("--out", type=Path, default=Path("results/bimsgp_variants_20261030"))
    ap.add_argument("--skip-test", action="store_true")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text())
    proto, mcfg, tcfg = cfg["protocol"], cfg["model"], cfg["training"]
    folds = rb.make_folds(n=599, seed=int(proto["fold_seed"]),
                          test_sizes=proto["test_sizes"],
                          val_size=int(proto["val_size"]))
    data = load(Path(proto["data"]), proto.get("data_sha256"))
    Y = rb.load_y(Path(proto["y_path"]), Path(proto["data"]), proto.get("data_sha256"))
    envs = [str(e) for e in proto["environments"]]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []

    for vname, override in VARIANTS.items():
        vmcfg = copy.deepcopy(mcfg)
        vmcfg.update(override)
        for fold in folds:
            k = fold["fold"]
            tr, va = fold["train"], fold["val"]
            for j, env in enumerate(envs):
                y = Y[:, j]
                t0 = time.time()
                sx = rb.Scaler().fit(data.X[tr])
                sy = rb.Scaler().fit(y[tr])
                model = rb.build_model(data.X.shape[1], vmcfg,
                                       seed=int(cfg["training"]["seed"]))
                model, hist, fit = rb.train_one_fold(
                    model, sx.transform(data.X[tr]), sy.transform(y[tr]),
                    sx.transform(data.X[va]), sy.transform(y[va]),
                    tcfg, device, seed=int(tcfg["seed"]))
                pv = rb.predict(model, sx.transform(data.X[va]), device)
                yv = sy.transform(y[va])
                rows.append({
                    "variant": vname, "environment": env, "fold": k,
                    "val_pcc": float(pcc(yv, pv)),
                    "val_rmse": float(rmse(yv, pv)),
                    "best_epoch": fit["best_pcc_epoch"],
                    "best_val_pcc": fit["best_pcc"],
                    "seconds": round(time.time() - t0, 1),
                    "n_params": int(sum(p.numel() for p in model.parameters())),
                })
                print(f"  [{vname} env {env} fold {k}] val_pcc={rows[-1]['val_pcc']:.4f} "
                      f"val_rmse={rows[-1]['val_rmse']:.4f} ep={fit['best_pcc_epoch']} "
                      f"{rows[-1]['seconds']}s", flush=True)
                del model, hist
                if device == "cuda":
                    torch.cuda.empty_cache()

    with (args.out / "variants_val.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {}
    for vname in VARIANTS:
        vr = [r for r in rows if r["variant"] == vname]
        per_env = {}
        for env in envs:
            er = [r for r in vr if r["environment"] == env]
            per_env[env] = {"val_pcc": float(np.mean([r["val_pcc"] for r in er])),
                            "val_rmse": float(np.mean([r["val_rmse"] for r in er]))}
        summary[vname] = {
            "val_pcc_macro": float(np.mean([per_env[e]["val_pcc"] for e in envs])),
            "val_rmse_macro": float(np.mean([per_env[e]["val_rmse"] for e in envs])),
            "per_env": per_env,
            "n_params": int(vr[0]["n_params"]),
        }
    selected = max(summary, key=lambda v: (summary[v]["val_pcc_macro"],
                                           -summary[v]["val_rmse_macro"]))
    (args.out / "variants_summary.json").write_text(json.dumps(
        {"selected": selected, "summary": summary}, indent=1))
    print(f"[variants] selected={selected} "
          f"val_pcc={summary[selected]['val_pcc_macro']:.4f}", flush=True)

    if args.skip_test:
        return 0
    sel_cfg = copy.deepcopy(cfg)
    sel_cfg["model"].update(VARIANTS[selected])
    sel_path = Path("configs/bimsgp_selected_20261030.yaml")
    sel_path.write_text(yaml.safe_dump(sel_cfg, sort_keys=False))
    sel_out = Path("results/bimsgp_selected_test_20261030")
    print(f"[variants] running test once for {selected} -> {sel_out}", flush=True)
    return rb.main(["--config", str(sel_path), "--out", str(sel_out)])


if __name__ == "__main__":
    raise SystemExit(main())
