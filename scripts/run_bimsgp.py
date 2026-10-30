#!/usr/bin/env python
"""BiMSGP-protocol FT-Transformer evaluation on Wheat599 — independent runner.

Per the user's frozen 2026-10-29 spec (configs/bimsgp.yaml):
* Standard 10-fold CV built from BGLR wheat.X/Y row order (seed 42):
  9 folds of 60 + 1 of 59 test materials; every material is tested exactly
  once; val = 54 drawn from the remaining 539/540; the rest is train.
* One FT-Transformer per (environment, fold): 1279 numeric tokens,
  d_block=64, 2 blocks, 4 heads, FFN hidden 128 ReGLU, dropout .1/.1/0,
  AdamW lr 5e-4 wd 0.01, batch 8, 100 epochs (no early stopping),
  ReduceLROnPlateau on val RMSE (standardized) patience 10 factor 0.5,
  checkpoint at best val PCC (ties -> earlier epoch), BF16 forward /
  FP32 loss, seed 42.
* X and y scalers are fit on the TRAIN subset of each fold only; val/test
  are transform-only; predictions are inverse-transformed before metrics
  (PCC primary; RMSE/MAE on the same BGLR scale).
* build_model() seeds random/numpy/torch before construction so the initial
  state is reproducible. train_one_fold() also retains the best-val-RMSE state
  for the pre-specified comparison in scripts/run_ckpt_rules.py; this runner
  still reports the best-val-PCC checkpoint.
* Test is evaluated once, after training and checkpoint selection.
* 10 folds share materials (not independent): per-fold + mean/SD(ddof=1),
  descriptive only; failed folds are recorded, never filtered or nanmean'ed.

Stage-1 frozen code (src/wheat599/models/ft_transformer.py, contracts.py,
gate.py, splits.py) is intentionally NOT imported or modified — this runner
is standalone except for read-only data loading and metric functions.
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

import numpy as np
import yaml

from wheat599.data import load, sha256_of
from wheat599.metrics import mae, pcc, rmse


# ---------------------------------------------------------------- folds ----

def make_folds(n: int = 599, seed: int = 42, test_sizes=None, val_size: int = 54):
    """Standard 10-fold CV from a fixed material order.

    Returns a list of dicts {fold, train, val, test} with sorted index arrays.
    Deterministic given (n, seed, test_sizes, val_size).
    """
    test_sizes = list(test_sizes or ([60] * 9 + [n - 60 * 9]))
    if sum(test_sizes) != n:
        raise ValueError(f"test sizes sum to {sum(test_sizes)}, expected {n}")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    folds = []
    pos = 0
    for k, size in enumerate(test_sizes):
        test = np.sort(perm[pos:pos + size])
        pos += size
        rest = np.setdiff1d(np.arange(n), test)
        rng_k = np.random.default_rng(seed + 1000 + k)
        val = np.sort(rng_k.choice(rest, size=val_size, replace=False))
        train = np.sort(np.setdiff1d(rest, val))
        folds.append({"fold": k, "train": train, "val": val, "test": test})
    return folds


def fold_table(folds) -> list[dict]:
    return [
        {"fold": f["fold"], "n_train": int(f["train"].size),
         "n_val": int(f["val"].size), "n_test": int(f["test"].size),
         "train_idx": f["train"].tolist(), "val_idx": f["val"].tolist(),
         "test_idx": f["test"].tolist()}
        for f in folds
    ]


# --------------------------------------------------------------- scalers ----

class Scaler:
    """Minimal standardizer (mean/std from the fit data only)."""

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, a: np.ndarray) -> "Scaler":
        self.mean = a.mean(axis=0)
        sd = a.std(axis=0)
        self.std = np.where(sd > 0, sd, 1.0)
        return self

    def transform(self, a: np.ndarray) -> np.ndarray:
        return (a - self.mean) / self.std

    def inverse_transform(self, a: np.ndarray) -> np.ndarray:
        return a * self.std + self.mean


# ---------------------------------------------------------------- model ----

RR_ALPHAS = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)


def fit_rrblup(X_tr, y_tr, X_val, y_val):
    """The current RR-BLUP baseline: train-only centering, alpha on val RMSE."""
    from sklearn.linear_model import Ridge

    X_tr = np.asarray(X_tr, dtype=np.float64)
    y_tr = np.asarray(y_tr, dtype=np.float64)
    X_val = np.asarray(X_val, dtype=np.float64)
    y_val = np.asarray(y_val, dtype=np.float64)
    mu_x, mu_y = X_tr.mean(axis=0), float(y_tr.mean())
    Xtr_c, Xva_c = X_tr - mu_x, X_val - mu_x
    best_alpha, best_rmse = None, np.inf
    for alpha in RR_ALPHAS:
        m = Ridge(alpha=float(alpha), fit_intercept=True).fit(Xtr_c, y_tr - mu_y)
        score = rmse(y_val, m.predict(Xva_c) + mu_y)
        if np.isfinite(score) and score < best_rmse:
            best_rmse, best_alpha = float(score), float(alpha)
    if best_alpha is None:
        raise RuntimeError("RR-BLUP: no finite validation RMSE")
    model = Ridge(alpha=best_alpha, fit_intercept=True).fit(Xtr_c, y_tr - mu_y)
    return {"model": model, "mu_x": mu_x, "mu_y": mu_y,
            "alpha": best_alpha, "val_rmse": best_rmse}


def predict_rrblup(fit, X):
    X = np.asarray(X, dtype=np.float64)
    return fit["model"].predict(X - fit["mu_x"]) + fit["mu_y"]


def zero_init_output(model):
    """Zero only the final regression linear layer; backbone stays initialized."""
    import torch

    output = model.backbone.output
    last = output[-1] if isinstance(output, torch.nn.Sequential) else output
    if not isinstance(last, torch.nn.Linear):
        raise TypeError(f"unexpected output layer: {type(last).__name__}")
    with torch.no_grad():
        last.weight.zero_()
        if last.bias is not None:
            last.bias.zero_()
    return last


def residual_prediction(rr_pred, correction, y_scaler):
    """RR prediction + s_T * g, with g in standardized phenotype units."""
    rr_pred = np.asarray(rr_pred, dtype=np.float64)
    correction = np.asarray(correction, dtype=np.float64)
    return rr_pred + float(y_scaler.std) * correction


def build_model(p: int, cfg_model: dict, seed: int):
    import random
    import torch
    from rtdl_revisiting_models import FTTransformer

    # Seed before construction: the initial state must not depend on whatever
    # RNG state the caller happened to leave behind.
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    return FTTransformer(
        n_cont_features=p,
        cat_cardinalities=[],
        d_out=1,
        n_blocks=int(cfg_model["n_blocks"]),
        d_block=int(cfg_model["d_block"]),
        attention_n_heads=int(cfg_model["attention_n_heads"]),
        attention_dropout=float(cfg_model["attention_dropout"]),
        ffn_d_hidden=int(cfg_model["ffn_d_hidden"]),
        ffn_d_hidden_multiplier=cfg_model.get("ffn_d_hidden_multiplier"),
        ffn_dropout=float(cfg_model["ffn_dropout"]),
        residual_dropout=float(cfg_model["residual_dropout"]),
    )


def train_one_fold(
    model, X_tr, y_tr, X_val, y_val, cfg_training, device, seed=42,
    base_tr=None, base_val=None,
):
    """100-epoch training loop; returns (model, history, fit_info).

    y_tr/y_val are already standardized (train-fold scaler). LR scheduler and
    both checkpoint rules use val signals only; the test fold never enters this
    function. When base_tr/base_val are given, the model output is a correction
    and the metrics are computed on base + output (the combined prediction).
    The returned model is loaded with the best-val-PCC state; the best-val-RMSE
    state is returned separately for a pre-specified comparison.
    """
    import torch

    torch.manual_seed(seed)
    model = model.to(device)
    groups = model.make_parameter_groups()
    for g in groups:
        g["lr"] = float(cfg_training["lr"])
    if len(groups) > 1:
        groups[1]["weight_decay"] = 0.0
    opt = torch.optim.AdamW(groups, weight_decay=float(cfg_training["weight_decay"]))
    sched_cfg = cfg_training["lr_scheduler"]
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode=sched_cfg["mode"], factor=float(sched_cfg["factor"]),
        patience=int(sched_cfg["patience"]),
    )
    loss_fn = torch.nn.MSELoss()

    Xt = torch.as_tensor(X_tr, dtype=torch.float32, device=device)
    yt = torch.as_tensor(y_tr, dtype=torch.float32, device=device).view(-1)
    Xv = torch.as_tensor(X_val, dtype=torch.float32, device=device)
    yv = torch.as_tensor(y_val, dtype=torch.float32, device=device).view(-1)
    n = Xt.shape[0]
    bs = int(cfg_training["batch_size"])
    epochs = int(cfg_training["epochs"])
    amp = device == "cuda"

    gen = torch.Generator()
    gen.manual_seed(seed)
    best_pcc, best_pcc_epoch, best_pcc_state = -np.inf, -1, None
    best_rmse, best_rmse_epoch, best_rmse_state = np.inf, -1, None
    history = []
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=gen).to(device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
                pred = model(x_cont=Xt[idx], x_cat=None).squeeze(-1)
            loss = loss_fn(pred.float(), yt[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
                pv = model(x_cont=Xv, x_cat=None).squeeze(-1)
            pv = pv.float().cpu().numpy()
        if base_val is None:
            val_truth, val_pred = y_val, pv
        else:
            val_truth = y_val + base_val
            val_pred = pv + base_val
        val_rmse = rmse(val_truth, val_pred)
        val_pcc = pcc(val_truth, val_pred)
        train_rmse = None
        if epoch % 5 == 0 or epoch == epochs - 1:
            ptr = predict(model, X_tr, device)
            if base_tr is None:
                train_rmse = float(rmse(y_tr, ptr))
            else:
                train_rmse = float(rmse(y_tr + base_tr, ptr + base_tr))
        history.append({"epoch": epoch, "val_rmse": float(val_rmse),
                        "val_pcc": float(val_pcc),
                        "train_rmse": train_rmse,
                        "lr": opt.param_groups[0]["lr"]})
        sched.step(val_rmse)
        if np.isfinite(val_rmse) and val_rmse < best_rmse:
            best_rmse, best_rmse_epoch = float(val_rmse), epoch
            best_rmse_state = {k: v.detach().clone()
                               for k, v in model.state_dict().items()}
        if np.isfinite(val_pcc) and val_pcc > best_pcc:
            best_pcc, best_pcc_epoch = float(val_pcc), epoch
            best_pcc_state = {k: v.detach().clone()
                              for k, v in model.state_dict().items()}
    if best_pcc_state is not None:
        model.load_state_dict(best_pcc_state)
    model.eval()
    return model, history, {
        "best_pcc_epoch": int(best_pcc_epoch),
        "best_pcc": float(best_pcc),
        "best_rmse_epoch": int(best_rmse_epoch),
        "best_rmse": float(best_rmse),
        "best_rmse_state": best_rmse_state,
    }


def predict(model, X, device, batch_size: int = 64):
    """Chunked inference (identical math, bounded memory).

    A full-batch forward on the ~485-row train slice materializes an
    attention map of ~12.7 GB (485 x 4 x 1280 x 1280 fp32); chunking is
    required when predicting train/val for diagnostics.
    """
    import torch

    model.eval()
    chunks = []
    with torch.no_grad():
        for i in range(0, X.shape[0], batch_size):
            Xt = torch.as_tensor(X[i:i + batch_size], dtype=torch.float32,
                                 device=device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=(device == "cuda")):
                pred = model(x_cont=Xt, x_cat=None).squeeze(-1)
            chunks.append(pred.float().cpu().numpy())
            del Xt, pred
    return np.concatenate(chunks)


# ------------------------------------------------------------ evaluation ----

def evaluate_fold(env, fold, y_true, y_pred) -> dict:
    """Metric row; failures (non-finite / zero-variance) are explicit."""
    row = {"environment": env, "fold": int(fold),
           "pcc": float("nan"), "rmse": float("nan"), "mae": float("nan"),
           "failed": False, "reason": ""}
    if not np.isfinite(y_pred).all() or not np.isfinite(y_true).all():
        row.update(failed=True, reason="non_finite_prediction")
        return row
    if np.ptp(y_true) == 0 or np.ptp(y_pred) == 0:
        # exact check: np.std of identical floats can be ~1e-17, not 0
        row.update(failed=True,
                   reason="zero_variance_y_true" if np.ptp(y_true) == 0
                   else "constant_prediction")
        return row
    row["pcc"] = float(pcc(y_true, y_pred))
    row["rmse"] = float(rmse(y_true, y_pred))
    row["mae"] = float(mae(y_true, y_pred))
    if not np.isfinite(row["pcc"]):
        row.update(failed=True, reason="non_finite_pcc")
    return row


def summarize(rows: list[dict]) -> dict:
    """mean/SD over the 10 folds; failures excluded from the mean but counted."""
    out = {}
    for env in sorted({r["environment"] for r in rows}):
        env_rows = [r for r in rows if r["environment"] == env]
        valid = [r for r in env_rows if not r["failed"]]
        entry = {"n_folds": len(env_rows), "n_valid": len(valid),
                 "n_failed": len(env_rows) - len(valid)}
        for metric in ("pcc", "rmse", "mae"):
            vals = np.asarray([r[metric] for r in valid], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                entry[metric] = {"mean": float(vals.mean()),
                                 "sd_ddof1": float(vals.std(ddof=1)) if vals.size > 1 else float("nan"),
                                 "n": int(vals.size)}
            else:
                entry[metric] = {"mean": float("nan"), "sd_ddof1": float("nan"), "n": 0}
        out[env] = entry
    return out


# ------------------------------------------------------------------ main ----

def load_y(y_path: Path, data_path: Path, expected_sha: str | None):
    """Y in original units: .npy/.csv override, else the BGLR load()."""
    y_path = Path(y_path)
    if y_path.suffix in (".npy",):
        Y = np.load(y_path)
    elif y_path.suffix in (".csv",):
        Y = np.loadtxt(y_path, delimiter=",", skiprows=1)
    else:
        if y_path != Path(data_path):
            raise ValueError(
                f"y_path {y_path} must be .npy/.csv or equal to data path")
        Y = load(Path(data_path), expected_sha).Y
    Y = np.asarray(Y, dtype=np.float64)
    if Y.shape != (599, 4):
        raise ValueError(f"Y shape {Y.shape}, expected (599, 4)")
    return Y


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=PROJ / "configs" / "bimsgp.yaml")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--folds", nargs="*", type=int, default=None,
                    help="subset of folds (pilot); default all 10")
    ap.add_argument("--manifest-only", action="store_true",
                    help="write the fold manifest and exit (no torch, no training)")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text())
    proto, mcfg, tcfg = cfg["protocol"], cfg["model"], cfg["training"]
    envs = proto["environments"]
    n = 599

    run_name = args.out.name if args.out else time.strftime("bimsgp_%Y%m%d_%H%M%S")
    out = args.out or (PROJ / "results" / run_name)
    out.mkdir(parents=True, exist_ok=True)

    folds = make_folds(n=n, seed=int(proto["fold_seed"]),
                       test_sizes=proto["test_sizes"],
                       val_size=int(proto["val_size"]))
    if args.folds:
        folds = [f for f in folds if f["fold"] in set(args.folds)]
    (out / "splits_manifest.json").write_text(json.dumps({
        "seed": proto["fold_seed"], "test_sizes": proto["test_sizes"],
        "val_size": proto["val_size"], "folds": fold_table(folds)}, indent=1))
    sizes = [(f["fold"], f["train"].size, f["val"].size, f["test"].size) for f in folds]
    print(f"[bimsgp] folds (fold,train,val,test): {sizes}")
    if args.manifest_only:
        print(f"[bimsgp] manifest written: {out / 'splits_manifest.json'}")
        return 0

    data = load(Path(proto["data"]), proto.get("data_sha256"))
    Y = load_y(Path(proto["y_path"]), Path(proto["data"]), proto.get("data_sha256"))
    X = data.X
    provenance = {
        "run": run_name, "config": str(args.config),
        "data": str(proto["data"]), "y_path": str(proto["y_path"]),
        "data_sha256": sha256_of(Path(proto["data"])),
        "folds": [f["fold"] for f in folds],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=1))

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[bimsgp] device={device} torch={torch.__version__}")

    rows, oof_rows, history_rows = [], [], []
    for fold in folds:
        k = fold["fold"]
        tr, va, te = fold["train"], fold["val"], fold["test"]
        for j, env in enumerate(envs):
            y = Y[:, j]
            t0 = time.time()
            sx = Scaler().fit(X[tr])
            sy = Scaler().fit(y[tr])
            Xtr_s = sx.transform(X[tr])
            ytr_s = sy.transform(y[tr])
            Xva_s = sx.transform(X[va])
            yva_s = sy.transform(y[va])
            Xte_s = sx.transform(X[te])
            model = build_model(X.shape[1], mcfg, seed=int(tcfg["seed"]))
            model, hist, fit = train_one_fold(
                model, Xtr_s, ytr_s, Xva_s, yva_s, tcfg, device,
                seed=int(tcfg["seed"]))
            best_epoch = fit["best_pcc_epoch"]
            best_pcc = fit["best_pcc"]
            y_pred_s = predict(model, Xte_s, device)
            y_pred = sy.inverse_transform(y_pred_s)  # back to BGLR units
            row = evaluate_fold(env, k, y[te], y_pred)
            row.update({"best_epoch": int(best_epoch), "best_val_pcc": float(best_pcc),
                        "n_train": int(tr.size), "n_val": int(va.size),
                        "n_test": int(te.size), "seconds": round(time.time() - t0, 1),
                        "n_params": int(sum(p.numel() for p in model.parameters()))})
            rows.append(row)
            history_rows.append({"environment": env, "fold": k,
                                 "n_epochs": len(hist)})
            for i, idx in enumerate(te):
                oof_rows.append({"material_index": int(idx), "environment": env,
                                 "fold": k, "y_true": float(y[idx]),
                                 "y_pred": float(y_pred[i])})
            print(f"  [env {env} fold {k}] pcc={row['pcc']:.4f} "
                  f"rmse={row['rmse']:.4f} best_epoch={best_epoch} "
                  f"{row['seconds']}s{'' if not row['failed'] else ' FAILED:'+row['reason']}")

    with (out / "per_fold.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (out / "oof.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(oof_rows[0].keys()))
        writer.writeheader()
        writer.writerows(oof_rows)
    failures = [r for r in rows if r["failed"]]
    with (out / "failures.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["environment", "fold", "reason"])
        writer.writeheader()
        for r in failures:
            writer.writerow({"environment": r["environment"], "fold": r["fold"],
                             "reason": r["reason"]})
    summary = summarize(rows)
    (out / "summary.json").write_text(json.dumps({
        "provenance": provenance,
        "summary": summary,
        "n_runs": len(rows),
        "n_failures": len(failures),
        "limits": cfg["reporting"]["limits"],
    }, indent=1))
    print(f"[bimsgp] done. runs={len(rows)} failures={len(failures)} -> {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
