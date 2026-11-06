#!/usr/bin/env python
"""MoEGP under the fair nested protocol — two-phase, selection-only artifacts.

Phase "dev" (default): for each (outer fold, environment), inner
KFold(n_splits=3, shuffle=True, random_state=42+outer_fold) over the 90% dev
set evaluates the six pre-registered candidates by mean inner-val PCC and
records ONLY the selection (winner candidate, refit epochs, per-candidate
scores). No refit, no weights saved, and the outer test indices are never
touched in this phase.

Phase "test" (run only after the rules are frozen and released): re-fits the
selected candidate on the full dev set for the recorded number of epochs
(best_epoch is 0-based; recorded refit epochs already include the +1
conversion), then predicts each outer test fold ONCE. No test label is used
for any decision.

Official components preserved: utils.moe.Gating with independent experts and
the official gating loss (mse + aux with the fixed loss_coef). Only the six
pre-registered candidates (all values from the official run.py search space)
are searched.

Failure policy: every fit failure is recorded per fit; a candidate with any
failed inner fit is excluded (NaN score); if all candidates fail the unit is
recorded as failed; the test phase records its own failures. When any failure
exists the summary reports valid k/n and an INCOMPLETE flag and does NOT
report a complete macro mean.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from wheat599.data import load, sha256_of  # noqa: E402
from wheat599.metrics import mae, pcc, rmse  # noqa: E402


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_model(p, cand, mcfg, creat_mlp_layer):
    from utils.moe import Gating

    def factory():
        return creat_mlp_layer(p, 1, list(cand["hidden"]),
                               dropout=float(cand["dropout"]), batch_norm=True)

    model = Gating(p, int(cand["experts"]), factory(),
                   top_k=int(mcfg["top_k"]), loss_coef=float(mcfg["loss_coef"]))
    if mcfg.get("independent_experts", True):
        model.experts = torch.nn.ModuleList(
            [factory() for _ in range(int(cand["experts"]))])
    return model


def fit_moegp(X, y, train_idx, val_idx, cand, mcfg, tcfg, epochs, device,
              seed, creat_mlp_layer, lr_schedule=None):
    """One MoEGP fit; checkpoint by inner-val PCC when val_idx is given.

    Y tensors are built ONLY for the train/val subsets handed in — the outer
    test phenotypes are never materialized or processed. When ``lr_schedule``
    is given (refit phase), the per-epoch learning rate is replayed from the
    median inner-fold trajectory instead of running the scheduler.
    """
    seed_all(seed)
    mu, sd = float(y[train_idx].mean()), float(y[train_idx].std())
    if sd == 0:
        raise ValueError("constant training target")
    Ytr = torch.as_tensor((y[train_idx] - mu) / sd, dtype=torch.float32,
                          device=device).view(-1, 1)
    Yval = (torch.as_tensor((y[val_idx] - mu) / sd, dtype=torch.float32,
                            device=device).view(-1, 1)
            if val_idx is not None else None)
    model = build_model(X.shape[1], cand, mcfg, creat_mlp_layer).to(device)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X[train_idx], Ytr),
        batch_size=int(tcfg["batch_size"]), shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=float(cand["lr"]),
                           weight_decay=float(mcfg["weight_decay"]))
    sched = (torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min")
             if val_idx is not None else None)
    best_pcc, best_epoch, best_state = -math.inf, None, None
    epoch_lrs = []
    for epoch in range(int(epochs)):
        if lr_schedule is not None:
            opt.param_groups[0]["lr"] = float(
                lr_schedule[min(epoch, len(lr_schedule) - 1)])
        epoch_lrs.append(float(opt.param_groups[0]["lr"]))
        model.train()
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            pred, aux = model(xb)
            loss = torch.nn.functional.mse_loss(pred, yb) + aux
            if not torch.isfinite(loss):
                raise ValueError("non-finite training loss")
            loss.backward()
            opt.step()
        if val_idx is not None:
            model.eval()
            with torch.no_grad():
                pv, aux = model(X[val_idx])
                val_mse = torch.nn.functional.mse_loss(pv, Yval).item()
            if not np.isfinite(val_mse):
                raise ValueError("non-finite validation loss")
            sched.step(val_mse + float(aux))
            val_pcc = float(pcc(y[val_idx],
                                pv.float().cpu().numpy().ravel() * sd + mu))
            if np.isfinite(val_pcc) and val_pcc > best_pcc:
                best_pcc, best_epoch = val_pcc, epoch
                best_state = copy.deepcopy(model.state_dict())
    if val_idx is not None:
        if best_state is None:
            raise ValueError("no finite validation PCC during training")
        model.load_state_dict(best_state)
    model.eval()
    return {"model": model, "mu": mu, "sd": sd, "best_epoch": best_epoch,
            "best_pcc": best_pcc, "lrs": epoch_lrs,
            "n_params": int(sum(p.numel() for p in model.parameters()))}


def inner_splits(dev, k, outer_fold, base_seed=42):
    """KFold(shuffle=True, random_state=base_seed + outer_fold) over dev."""
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=k, shuffle=True, random_state=base_seed + outer_fold)
    return [(np.sort(dev[tr]), np.sort(dev[va])) for tr, va in kf.split(dev)]


def write_csv(out, name, fields, items):
    with (out / name).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(items)


def summarize(rows, failures, envs, out):
    ok = [r for r in rows if not r.get("failed")]
    summary = {"n_units": len(rows), "n_failures": len(failures),
               "n_valid": len(ok), "complete": False,
               "per_environment": {}, "selection_counts": {}}
    for env in envs:
        rs = [r for r in ok if r["environment"] == env]
        n_total = sum(1 for r in rows if r["environment"] == env)
        entry = {"n_valid": len(rs), "n_total": n_total}
        if rs:
            entry.update({m: {"mean": float(np.mean([float(r[m]) for r in rs])),
                              "sd_ddof1": (float(np.std([float(r[m]) for r in rs], ddof=1))
                                           if len(rs) > 1 else float("nan"))}
                          for m in ("pcc", "rmse", "mae")})
        if len(rs) < n_total:
            entry["incomplete"] = True
        summary["per_environment"][env] = entry
    summary["complete"] = bool(len(failures) == 0 and len(rows) == len(envs) * 10)
    if summary["complete"]:
        summary["macro"] = {m: float(np.mean(
            [summary["per_environment"][e][m]["mean"] for e in envs]))
            for m in ("pcc", "rmse", "mae")}
    else:
        summary["macro"] = None
        summary["note"] = ("INCOMPLETE: macro omitted; per-environment means are "
                           "valid-fold-only with n reported; failures never filtered")
    for r in ok:
        summary["selection_counts"][r.get("selected", "")] = \
            summary["selection_counts"].get(r.get("selected", ""), 0) + 1
    summary["limits"] = [
        "10 outer folds share materials; descriptive only (no p-values/CIs)",
        "single seed (42); no initialization variance",
        "six pre-registered candidates; no grid expansion",
        "fold0/env1 test was observed by earlier smokes (RR/MoEGP) and the "
        "same outer folds were used across rounds; results are not an "
        "independent confirmation",
    ]
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path,
                    default=ROOT / "configs" / "moegp_nested.yaml")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "results" / "moegp_nested_20261102")
    ap.add_argument("--folds", nargs="*", type=int, default=None)
    ap.add_argument("--environments", nargs="*", default=None)
    ap.add_argument("--phase", choices=["dev", "test"], default="dev")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text())
    source = Path(cfg["source"])
    commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain"], text=True).strip()
    if commit != cfg["source_commit"] or dirty:
        raise RuntimeError("Official MoEGP source must match the configured clean commit")
    sys.path.insert(0, str(source))
    sys.dont_write_bytecode = True
    from utils.moe import creat_mlp_layer

    mcfg, tcfg, candidates = cfg["model"], cfg["training"], cfg["candidates"]
    data = load(Path(cfg["data"]), cfg["data_sha256"])
    folds = json.loads(Path(cfg["splits"]).read_text())["folds"]
    envs = [str(e) for e in (args.environments or data.environments)]
    if len(folds) != 10:
        raise ValueError("expected ten fixed folds")

    scoped = args.out if args.phase == "dev" else args.out / "test"
    scoped.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(
            float(tcfg.get("gpu_memory_fraction", 0.45)))
    seed = int(cfg["seed"])
    X = torch.as_tensor(data.X, dtype=torch.float32, device=device)

    provenance = {
        "config": cfg, "source_commit": commit,
        "splits_sha256": sha256_of(Path(cfg["splits"])),
        "data_sha256": sha256_of(Path(cfg["data"])),
        "device": str(device), "seed": seed, "phase": args.phase,
        "inner_cv": "KFold(shuffle=True, random_state=42+outer_fold)",
        "selection_metric": "inner_val_pcc; tie (<=1e-12) -> earlier candidate "
                            "in the pre-registered config order",
        "refit": "test phase refits on the full dev; best_epoch is 0-based, "
                 "recorded refit epochs = ceil(median(best_epoch + 1)), capped",
        "weights_saved": False,
        "independent_experts": mcfg.get("independent_experts", True),
        "official_loss": "mse + aux (loss_coef fixed)",
        "x_scaling": tcfg["x_scaling"], "y_scaling": tcfg["y_scaling"],
        "checkpoint": tcfg["checkpoint"],
        "test_contact": ("none in this phase" if args.phase == "dev"
                         else "one-shot evaluation after refit; no test-based decisions"),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (scoped / "provenance.json").write_text(json.dumps(provenance, indent=1))

    fold_list = [f for f in folds
                 if not args.folds or f["fold"] in set(args.folds)]

    if args.phase == "dev":
        rows, failures = [], []
        for fold in fold_list:
            k = fold["fold"]
            dev = np.sort(np.concatenate([np.asarray(fold["train_idx"], dtype=int),
                                          np.asarray(fold["val_idx"], dtype=int)]))
            for env in envs:
                j = data.environments.index(env)
                y = data.Y[:, j]
                t0 = time.time()
                try:
                    inner = inner_splits(dev, int(tcfg["inner_folds"]), k)
                    stats = []
                    for ci, cand in enumerate(candidates):
                        fold_pccs, epochs_seen, lrs_seen = [], [], []
                        failed = False
                        for fi, (itr, iva) in enumerate(inner):
                            try:
                                res = fit_moegp(
                                    X, y, itr, iva, cand, mcfg, tcfg,
                                    int(tcfg["epochs"]), device, seed,
                                    creat_mlp_layer)
                                fold_pccs.append(res["best_pcc"])
                                epochs_seen.append(res["best_epoch"])
                                lrs_seen.append(res["lrs"])
                                del res
                            except Exception as exc:  # noqa: BLE001
                                failed = True
                                failures.append({
                                    "environment": env, "fold": k,
                                    "stage": f"inner:{cand['name']}:f{fi}",
                                    "reason": f"{type(exc).__name__}: {exc}"})
                                break
                        score = (float(np.mean(fold_pccs))
                                 if (not failed and len(fold_pccs) == len(inner)
                                     and np.all(np.isfinite(fold_pccs))) else float("nan"))
                        stats.append({"name": cand["name"], "index": ci,
                                      "inner_pcc": score,
                                      "best_epochs": epochs_seen,
                                      "lrs": lrs_seen})
                    finite = [c for c in stats if np.isfinite(c["inner_pcc"])]
                    if not finite:
                        raise RuntimeError("all candidates failed internally")
                    best_score = max(c["inner_pcc"] for c in finite)
                    tied = [c for c in finite
                            if best_score - c["inner_pcc"] <= 1e-12]
                    win = min(tied, key=lambda c: c["index"])
                    one_based = [e + 1 for e in win["best_epochs"]]
                    refit_epochs = int(math.ceil(float(np.median(one_based))))
                    refit_epochs = max(1, min(refit_epochs, int(tcfg["epochs"])))
                    n_ep = int(tcfg["epochs"])
                    lr_mat = np.full((len(win["lrs"]), n_ep), np.nan)
                    for i, seq in enumerate(win["lrs"]):
                        lr_mat[i, :len(seq)] = seq[:n_ep]
                    lr_schedule = np.nanmedian(lr_mat, axis=0)[:refit_epochs].tolist()
                    rows.append({"environment": env, "fold": k,
                                 "selected": win["name"],
                                 "inner_pcc": win["inner_pcc"],
                                 "refit_epochs": refit_epochs,
                                 "lr_schedule": json.dumps(lr_schedule),
                                 "candidate_detail": json.dumps(stats),
                                 "seconds": round(time.time() - t0, 1),
                                 "failed": False, "reason": ""})
                    print(f"[dev f{k} e{env}] sel={win['name']} "
                          f"inner_pcc={win['inner_pcc']:.4f} "
                          f"refit_epochs={refit_epochs} "
                          f"({time.time()-t0:.0f}s)", flush=True)
                    write_csv(scoped, "dev_selection.csv",
                              ["environment", "fold", "selected", "inner_pcc",
                               "refit_epochs", "lr_schedule", "candidate_detail",
                               "seconds", "failed", "reason"], rows)
                    write_csv(scoped, "failures.csv",
                              ["environment", "fold", "stage", "reason"], failures)
                except Exception as exc:  # noqa: BLE001
                    failures.append({"environment": env, "fold": k,
                                     "stage": "dev_unit",
                                     "reason": f"{type(exc).__name__}: {exc}"})
                    rows.append({"environment": env, "fold": k, "selected": "",
                                 "inner_pcc": float("nan"), "refit_epochs": "",
                                 "lr_schedule": "", "candidate_detail": "",
                                 "seconds": round(time.time() - t0, 1),
                                 "failed": True,
                                 "reason": f"{type(exc).__name__}: {exc}"})
                    print(f"[dev f{k} e{env}] FAILED: {exc}", flush=True)
                    write_csv(scoped, "dev_selection.csv",
                              ["environment", "fold", "selected", "inner_pcc",
                               "refit_epochs", "lr_schedule", "candidate_detail",
                               "seconds", "failed", "reason"], rows)
                    write_csv(scoped, "failures.csv",
                              ["environment", "fold", "stage", "reason"], failures)
                    if isinstance(exc, torch.cuda.OutOfMemoryError):
                        raise
        pass  # final files were persisted incrementally after each unit
        ok = [r for r in rows if not r["failed"]]
        inner = [r["inner_pcc"] for r in ok]
        ep = [int(r["refit_epochs"]) for r in ok]
        summary = {
            "phase": "dev", "n_units": len(rows), "n_valid": len(ok),
            "n_failures": len(failures),
            "complete": bool(len(failures) == 0 and len(rows) == len(envs) * 10),
            "selection_counts": {},
            "inner_pcc": ({"mean": float(np.mean(inner)),
                           "sd_ddof1": (float(np.std(inner, ddof=1))
                                        if len(inner) > 1 else float("nan"))}
                          if inner else None),
            "refit_epochs": ({"median": float(np.median(ep)),
                              "min": int(np.min(ep)), "max": int(np.max(ep))}
                             if ep else None),
            "weights_saved": False,
            "test_contact": "none",
        }
        for r in ok:
            summary["selection_counts"][r["selected"]] =                 summary["selection_counts"].get(r["selected"], 0) + 1
        if not summary["complete"]:
            summary["incomplete"] = True
        (scoped / "summary.json").write_text(json.dumps(summary, indent=1))
        print(json.dumps(summary), flush=True)
        return 0

    # ---- phase test: refit from saved selection, predict test once ----
    dev_prov_path = args.out / "provenance.json"
    if not dev_prov_path.exists():
        raise RuntimeError("test phase requires the dev phase provenance.json")
    dev_prov = json.loads(dev_prov_path.read_text())
    if (dev_prov.get("splits_sha256") != sha256_of(Path(cfg["splits"]))
            or dev_prov.get("data_sha256") != sha256_of(Path(cfg["data"]))
            or dev_prov.get("config") != cfg):
        raise RuntimeError("dev provenance does not match the current config/splits/data")
    sel_rows = list(csv.DictReader(open(args.out / "dev_selection.csv", newline="")))
    expected = {(str(f["fold"]), e) for f in fold_list for e in envs}
    got = {(r["fold"], r["environment"]) for r in sel_rows}
    n_failed = sum(1 for r in sel_rows
                   if r["failed"].lower() in ("true", "1"))
    if len(sel_rows) != len(expected) or got != expected or n_failed:
        raise RuntimeError(
            f"test phase refused: {len(sel_rows)} rows, expected {len(expected)}; "
            f"missing={sorted(expected - got)}; failed_rows={n_failed}")
    rows, oof_rows, failures = [], [], []
    for fold in fold_list:
        k = fold["fold"]
        test = np.sort(np.asarray(fold["test_idx"], dtype=int))
        dev = np.sort(np.concatenate([np.asarray(fold["train_idx"], dtype=int),
                                      np.asarray(fold["val_idx"], dtype=int)]))
        for env in envs:
            j = data.environments.index(env)
            y = data.Y[:, j]
            t0 = time.time()
            try:
                sel = next(r for r in sel_rows
                           if r["environment"] == env and int(r["fold"]) == k)
                if sel["failed"].lower() in ("true", "1"):
                    raise RuntimeError(f"dev selection failed for this unit: {sel['reason']}")
                winner = next(c for c in candidates if c["name"] == sel["selected"])
                refit_epochs = int(sel["refit_epochs"])
                lr_schedule = json.loads(sel["lr_schedule"])
                final = fit_moegp(X, y, dev, None, winner, mcfg, tcfg,
                                  refit_epochs, device, seed, creat_mlp_layer,
                                  lr_schedule=lr_schedule)
                with torch.no_grad():
                    pred = final["model"](X[test])[0].detach().float().cpu().numpy().ravel()
                pred = pred * final["sd"] + final["mu"]
                vals = {"pcc": float(pcc(y[test], pred)),
                        "rmse": float(rmse(y[test], pred)),
                        "mae": float(mae(y[test], pred))}
                if not all(np.isfinite(v) for v in vals.values()):
                    raise ValueError("non-finite test metric")
                rows.append({"environment": env, "fold": k, **vals,
                             "selected": winner["name"],
                             "refit_epochs": refit_epochs,
                             "n_params": final["n_params"],
                             "seconds": round(time.time() - t0, 1),
                             "failed": False, "reason": ""})
                oof_rows.extend({"environment": env, "fold": k,
                                 "material_index": int(idx), "y_true": float(y[idx]),
                                 "y_pred": float(p)} for idx, p in zip(test, pred))
                print(f"[test f{k} e{env}] PCC={vals['pcc']:.4f} "
                      f"sel={winner['name']} {time.time()-t0:.0f}s", flush=True)
                del final
            except Exception as exc:  # noqa: BLE001
                failures.append({"environment": env, "fold": k, "stage": "test",
                                 "reason": f"{type(exc).__name__}: {exc}"})
                rows.append({"environment": env, "fold": k, "pcc": float("nan"),
                             "rmse": float("nan"), "mae": float("nan"),
                             "selected": "", "refit_epochs": "", "n_params": "",
                             "seconds": round(time.time() - t0, 1),
                             "failed": True,
                             "reason": f"{type(exc).__name__}: {exc}"})
                print(f"[test f{k} e{env}] FAILED: {exc}", flush=True)
    write_csv(scoped, "per_fold.csv",
              ["environment", "fold", "pcc", "rmse", "mae", "selected",
               "refit_epochs", "n_params", "seconds", "failed", "reason"], rows)
    write_csv(scoped, "oof.csv",
              ["environment", "fold", "material_index", "y_true", "y_pred"],
              oof_rows)
    write_csv(scoped, "failures.csv",
              ["environment", "fold", "stage", "reason"], failures)
    summarize(rows, failures, envs, scoped)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
