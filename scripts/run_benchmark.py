#!/usr/bin/env python
"""CLI entry point for the Wheat599 model-base benchmark (M0).

M0 scope (per lead decision 2026-10-29): data loading, split generation, the
OOF container, metrics, and the bootstrap skeleton. Model adapters (Ridge,
RKHS/GBLUP, RBF-SVR, TabM, FT-Transformer, TabPFN-3.5) are registered behind a
common Estimator interface and filled in as their environments are validated;
this script wires the plumbing and runs a SMOKE_ONLY path that exercises the
full pipeline with a trivial estimator so leakage control and provenance can be
audited before any heavy model runs.

Usage:
    python scripts/run_benchmark.py --config configs/core.yaml --smoke
    python scripts/run_benchmark.py --config configs/core.yaml \
        --protocols random --seeds 20261029 --smoke

Results under --smoke are tagged SMOKE_ONLY and must not be written to
docs/RESULTS.md.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

# Allow running from a source checkout without installation.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from wheat599.bootstrap import fold_level_t_interval
from wheat599.contracts import check_adapter
from wheat599.data import load, sha256_of
from wheat599.metrics import compute_all, rmse
from wheat599.models import REGISTRY, available_adapters, get_adapter
from wheat599.oof import OOF
from wheat599.splits import build_splits


def load_config(path: Path) -> dict:
    import yaml

    with path.open() as handle:
        return yaml.safe_load(handle)


def device_info() -> dict:
    """Record compute device per AGENTS.md §4 / configs/core.yaml environment."""
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": None,
        "cuda_available": None,
        "device": "cpu",
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if info["cuda_available"]:
            info["device"] = "cuda"
            info["gpu_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return info


def select_and_fit(adapter, X_tr, y_tr, X_val, y_val, smoke, metric=rmse):
    """Inner-fold model selection + refit on full outer_train.

    Searches the adapter's grid (truncated to one value per key when smoke),
    scores each candidate on the inner validation set, picks the best, then
    refits that configuration on the full outer_train. All fitting is on
    outer_train material only (fold-local).
    """
    grid = adapter.param_grid(smoke=smoke)
    keys = list(grid.keys())
    import itertools

    best_params, best_score = None, np.inf
    for combo in itertools.product(*[grid.grid[k] for k in keys]):
        params = dict(zip(keys, combo))
        cand = type(adapter)()
        cand.fit(X_tr, y_tr, params)
        score = metric(y_val, cand.predict(X_val))
        if score < best_score:
            best_score, best_params = score, params
    final = type(adapter)()
    final.fit(X_tr, y_tr, best_params or {})
    return final, best_params, best_score


def run_adapter_oof(adapter, data, splits, smoke) -> tuple[np.ndarray, np.ndarray, dict]:
    """Produce OOF predictions for one adapter via inner-fold model selection.

    For each outer fold and environment: run the inner 3-fold selection on the
    outer_train materials, refit on the full outer_train, predict outer_valid.
    Returns (preds, fold_of, selection_provenance).
    """
    preds = np.zeros((data.n, data.n_env), dtype=np.float64)
    fold_of = np.full(data.n, -1, dtype=np.int64)
    sel_prov = []
    for split in splits:
        tr, va = split.outer_train, split.outer_valid
        Xva = data.X[va]
        for j in range(data.n_env):
            ycol = data.Y[:, j]
            # inner 3-fold selection within outer_train (inner indices are
            # full-data indices into data.X / data.Y)
            inner_scores = []
            for ifold in split.inner_folds:
                it, iva = ifold.inner_train, ifold.inner_valid
                _, params, score = select_and_fit(
                    adapter, data.X[it], ycol[it], data.X[iva], ycol[iva], smoke
                )
                inner_scores.append(score)
            # refit on full outer_train; select on outer_valid (fold-local: all
            # outer_train material only)
            final, params, _ = select_and_fit(
                adapter, data.X[tr], ycol[tr], Xva, ycol[va], smoke
            )
            preds[va, j] = final.predict(Xva)
            sel_prov.append({"fold": split.fold, "env": j,
                             "params": params, "inner_scores": inner_scores})
        fold_of[va] = split.fold
    if np.any(fold_of < 0):
        raise RuntimeError("some material has no outer-fold assignment (leak?)")
    return preds, fold_of, sel_prov


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--protocols", nargs="+", default=["random"])
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--smoke", action="store_true", help="SMOKE_ONLY: tiny run")
    parser.add_argument("--models", nargs="+", default=None,
                        help="adapter names (default: all available)")
    parser.add_argument("--out", type=Path, default=Path("results/smoke"))
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    proto = cfg["protocol"]
    data_path = args.data or Path(proto["data"])
    expected_sha = proto["data_sha256"]

    if args.smoke:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "SMOKE_ONLY").write_text("smoke run — not a benchmark result\n")

    n_outer = 2 if args.smoke else proto["outer_folds"]
    n_inner = 2 if args.smoke else proto["inner_folds"]
    seeds = args.seeds or proto["seeds"][:1]
    if args.smoke:
        seeds = seeds[:1]

    data = load(data_path, expected_sha)
    dev = device_info()
    if dev["cuda_available"]:
        dev["cuda_visible_devices"] = __import__("os").environ.get(
            "CUDA_VISIBLE_DEVICES", ""
        )

    provenance = {
        "config": str(args.config),
        "data": str(data_path),
        "data_sha256": sha256_of(data_path),
        "device": dev,
        "smoke": bool(args.smoke),
        "n_outer": n_outer,
        "n_inner": n_inner,
        "seeds": seeds,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (args.out / "provenance.json").write_text(json.dumps(provenance, indent=2))

    models = args.models or available_adapters()
    provenance["models"] = models
    (args.out / "provenance.json").write_text(json.dumps(provenance, indent=2))

    summary = []
    for model_name in models:
        adapter = get_adapter(model_name)
        for protocol in args.protocols:
            for seed in seeds:
                t0 = time.time()
                splits = build_splits(
                    data, protocol=protocol, seed=seed,
                    n_outer=n_outer, n_inner=n_inner,
                )
                # leakage contract: every adapter must be fold-local
                for s in splits:
                    check_adapter(adapter.contract_adapter(), s, data)
                preds, fold_of, sel_prov = run_adapter_oof(
                    adapter, data, splits, args.smoke
                )
                for j, env in enumerate(data.environments):
                    oof = OOF(
                        environment=env,
                        y_true=data.Y[:, j],
                        y_pred=preds[:, j],
                        repeat=seed,
                        fold_of=fold_of,
                        provenance={"protocol": protocol, "model": model_name},
                    )
                    tag = "SMOKE" if args.smoke else f"{protocol}_{seed}"
                    oof.save(args.out / "oof" / f"{model_name}_{env}_{tag}.npz")
                    m = oof.metrics()
                    # per-fold RMSE for the fold-level interval (unit array)
                    fold_rmses = np.array([
                        rmse(oof.y_true[s.outer_valid], oof.y_pred[s.outer_valid])
                        for s in splits
                    ])
                    summary.append({
                        "model": model_name, "env": env,
                        "protocol": protocol, "seed": seed,
                        "metrics": m,
                        "fold_level_rmse": fold_level_t_interval(fold_rmses),
                        "selection": sel_prov[:1],  # first fold's selection
                    })
                dt = time.time() - t0
                print(f"[{model_name} {protocol} seed={seed}] {len(splits)} folds "
                      f"in {dt:.2f}s (device={dev['device']})")

    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"done. wrote {args.out/'summary.json'} ({len(summary)} env-runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
