#!/usr/bin/env python
"""Per-epoch train/val RMSE trajectory on selected folds; no test evaluation."""

from __future__ import annotations

import argparse
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path("configs/bimsgp.yaml"))
    ap.add_argument("--out", type=Path, default=Path("results/bimsgp_trajectory_20261030"))
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 4])
    ap.add_argument("--envs", nargs="+", default=None)
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text())
    proto, mcfg, tcfg = cfg["protocol"], cfg["model"], cfg["training"]
    folds = rb.make_folds(n=599, seed=int(proto["fold_seed"]),
                          test_sizes=proto["test_sizes"],
                          val_size=int(proto["val_size"]))
    folds = [f for f in folds if f["fold"] in set(args.folds)]
    data = load(Path(proto["data"]), proto.get("data_sha256"))
    Y = rb.load_y(Path(proto["y_path"]), Path(proto["data"]), proto.get("data_sha256"))
    envs = [str(e) for e in proto["environments"]]
    if args.envs:
        envs = [e for e in envs if e in set(args.envs)]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for fold in folds:
        k = fold["fold"]
        tr, va = fold["train"], fold["val"]
        for j, env in enumerate(envs):
            y = Y[:, j]
            t0 = time.time()
            sx = rb.Scaler().fit(data.X[tr])
            sy = rb.Scaler().fit(y[tr])
            model = rb.build_model(data.X.shape[1], mcfg)
            model, hist, best_epoch, best_pcc = rb.train_one_fold(
                model, sx.transform(data.X[tr]), sy.transform(y[tr]),
                sx.transform(data.X[va]), sy.transform(y[va]),
                tcfg, device, seed=int(tcfg["seed"]))
            for h in hist:
                rows.append({"environment": env, "fold": k, **h})
            print(f"  [env {env} fold {k}] best_epoch={best_epoch} "
                  f"best_val_pcc={best_pcc:.4f} {time.time()-t0:.1f}s", flush=True)
            del model, hist
            if device == "cuda":
                torch.cuda.empty_cache()

    with (args.out / "trajectory.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "environment", "fold", "epoch", "train_rmse", "val_rmse",
            "val_pcc", "lr"])
        writer.writeheader()
        writer.writerows(rows)
    (args.out / "summary.json").write_text(json.dumps({
        "config": str(args.config), "folds": [f["fold"] for f in folds],
        "environments": envs, "rows": len(rows)}, indent=1))
    print(f"done -> {args.out}/trajectory.csv", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
