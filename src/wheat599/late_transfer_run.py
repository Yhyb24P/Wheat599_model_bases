"""Wheat599 development-only evaluation of decoupled TabM transfer."""
import json
import multiprocessing
import platform
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
import yaml

from wheat599.data import load, sha256_of
from wheat599.late_transfer import fit_late_transfer, load_predictor, save_predictor
from wheat599.retrieval_run import check_split, environment_metrics, macro_metrics, verify_versions


def unit_prediction(job):
    """The worker receives only fit phenotypes. Query phenotypes stay in the scorer."""
    fold, inner, x, y, query, recipe, options, out, verify_reload = job
    started = time.perf_counter()
    torch.set_num_threads(options["threads"])
    device = torch.device(options["device"])
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(options["gpu_fraction"], device)
    x, y, query = [torch.as_tensor(v, device=device, dtype=torch.float32) for v in (x, y, query)]
    model = fit_late_transfer(x, y, recipe)
    with torch.inference_mode():
        predictions = {k: v.cpu().numpy() for k, v in model.systems(query).items()}
    if not all(np.isfinite(v).all() for v in predictions.values()):
        raise FloatingPointError("Nonfinite query predictions")
    path = out / "checkpoints" / f"fold_{fold}_inner_{inner}.pt"
    temp = path.with_suffix(".tmp")
    save_predictor(temp, model, recipe, x.shape[1])
    temp.replace(path)
    reload_error = None
    if verify_reload:
        restored = load_predictor(path, device)
        with torch.inference_mode():
            value = restored(query).cpu().numpy()
        np.testing.assert_allclose(value, predictions["hidden_transfer"], atol=1e-6, rtol=1e-6)
        reload_error = float(np.max(np.abs(value - predictions["hidden_transfer"])))
    return fold, inner, predictions, {
        "checkpoint": str(path.relative_to(out)), "checkpoint_sha256": sha256_of(path),
        "reload_max_abs_difference": reload_error, "seconds": time.perf_counter() - started}


def write_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def historical_scores(config):
    result = {}
    rows = json.loads(Path(config["historical_scores"]).read_text())
    for train in ["independent", "shared"]:
        selected = [r["ensemble"] for r in rows if r["training"] == train and
                    r["selection"] == "per_environment" and r["fold"] in config["outer_folds"]]
        if len(selected) != len(config["outer_folds"]):
            raise ValueError("Incomplete historical TabM comparisons")
        result[train + "_per_environment"] = {
            key: float(np.mean([r[key] for r in selected])) for key in ("pcc", "rmse", "mae")}
    fusion = json.loads(Path(config["fusion_scores"]).read_text())["fixed_fusion"]["per_fold"]
    selected = [r for r in fusion if r["fold"] in config["outer_folds"]]
    if len(selected) != len(config["outer_folds"]):
        raise ValueError("Incomplete historical fusion scores")
    result["fixed_fusion"] = {key: float(np.mean([r[key] for r in selected]))
                              for key in ("pcc", "rmse", "mae")}
    return result


def summarize(units, config):
    systems = {}
    for mode in units[0]["metrics"]:
        by_env = units[0]["environments"][mode]
        systems[mode] = {
            "macro": {k: float(np.mean([u["metrics"][mode][k] for u in units]))
                      for k in ("pcc", "rmse", "mae")},
            "per_environment": {
                e: {k: float(np.mean([u["environments"][mode][e][k] for u in units]))
                    for k in ("pcc", "rmse", "mae")} for e in by_env},
            "fold_pcc": {str(f): float(np.mean([u["metrics"][mode]["pcc"] for u in units
                                                if u["fold"] == f])) for f in config["outer_folds"]}}
    return {"units": len(units), "systems": systems, "historical_same_folds": historical_scores(config),
            "score_scope": "Frozen-recipe development scores, mean of inner/environment metrics."}


def run(config, args, runner):
    if args.selection is not None:
        raise ValueError("Late transfer operates on development materials only")
    versions = verify_versions(config)
    data = load(Path(config["data"]), config["data_sha256"])
    split_path = Path(config["splits"])
    manifest = json.loads(split_path.read_text())
    profiles_path = Path(config["profile_source"])
    profiles = yaml.safe_load(profiles_path.read_text())["configurations"]
    recipe = {k: config[k] for k in ("profile_indices", "epochs", "seeds", "batch_size",
              "gradient_clip", "crossfit_folds", "crossfit_seed", "transfer_ridge", "stack_ridge")}
    if len(recipe["profile_indices"]) != data.n_env or len(set(recipe["seeds"])) != len(recipe["seeds"]):
        raise ValueError("Check specialist profiles and seeds")
    recipe["profiles"] = [profiles[i] for i in recipe["profile_indices"]]
    folds = [f for f in manifest["folds"] if f["fold"] in config["outer_folds"]]
    if sorted(f["fold"] for f in folds) != sorted(config["outer_folds"]) or not folds:
        raise ValueError("Invalid development fold list")
    device = config["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    workers = config["cpu_workers"] if device == "cpu" else 1
    options = {"threads": config["threads_per_worker"], "device": device,
               "gpu_fraction": config["gpu_memory_fraction"]}
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    for name in ("checkpoints", "predictions"):
        (out / name).mkdir()
    files = [Path(__file__), Path(__file__).with_name("late_transfer.py"),
             Path(__file__).with_name("genomic_tabm.py"), profiles_path, split_path, runner,
             args.config, Path(config["historical_scores"]), Path(config["fusion_scores"])]
    provenance = {"git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip(),
                  "files_sha256": {str(p): sha256_of(p) for p in files},
                  "versions": versions, "torch": str(torch.__version__), "python": platform.python_version(),
                  "device": device, "threads_per_worker": options["threads"], "workers": workers,
                  "config": config, "recipe": recipe, "data_sha256": config["data_sha256"]}
    write_json(out / "provenance.json", provenance)
    jobs, targets = [], {}
    for fold in folds:
        if len(fold["inner"]) != 3:
            raise ValueError("Expected three inner splits")
        for j in range(3):
            train, val = check_split(fold, j)
            targets[(fold["fold"], j)] = (val, data.Y[val])
            jobs.append((fold["fold"], j, data.X[train].astype("float32"), data.Y[train].astype("float32"),
                         data.X[val].astype("float32"), recipe, options, out, len(jobs) == 0))
    units, failures = [], []
    started = time.perf_counter()
    per_unit_fits = (recipe["crossfit_folds"] + 1) * len(recipe["seeds"]) * data.n_env

    def status(state):
        write_json(out / "execution_status.json", {
            "status": state, "expected_units": len(jobs), "completed_units": len(units),
            "specialist_fits_in_completed_units": per_unit_fits * len(units),
            "elapsed_seconds": time.perf_counter() - started})
        write_json(out / "failures.json", failures)

    def record(result):
        fold, j, predictions, details = result
        indices, truth = targets[(fold, j)]
        path = out / "predictions" / f"fold_{fold}_inner_{j}.npz"
        np.savez_compressed(path, material_ids=indices, **predictions)
        units.append({"fold": fold, "inner": j, **details, "predictions_sha256": sha256_of(path),
                      "metrics": {k: macro_metrics(v, truth) for k, v in predictions.items()},
                      "environments": {k: environment_metrics(v, truth, data.environments)
                                       for k, v in predictions.items()}})
        units.sort(key=lambda u: (u["fold"], u["inner"]))
        write_json(out / "units.json", units)
        status("running")
        print(f"Completed {len(units)}/{len(jobs)}: fold {fold}, inner {j}", flush=True)

    status("running")
    try:
        if workers == 1:
            for job in jobs:
                record(unit_prediction(job))
        else:
            with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool:
                pending = [pool.submit(unit_prediction, job) for job in jobs]
                for future in as_completed(pending):
                    record(future.result())
        summary = summarize(units, config)
        write_json(out / "summary.json", summary)
        status("complete")
        return summary
    except BaseException as exc:
        failures.append({"error": repr(exc), "completed_units": len(units)})
        status("failed")
        raise
