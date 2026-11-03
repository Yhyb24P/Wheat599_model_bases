"""Development selection and frozen CV1 evaluation for locus-context neural features."""
import copy
import itertools
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from wheat599.data import load, sha256_of
from wheat599.get_icm import write
from wheat599.metrics import mae, pcc, rmse

METRICS = ("pcc", "rmse", "mae")
FIELDS = ["fold", "inner_fold", "model", "environment", "gamma", "weight_decay", "steps", *METRICS]
MODELS = ("full", "full_background", "linear", "global", "frozen_full")


def metrics(truth, prediction):
    if prediction.shape != truth.shape or not np.isfinite(prediction).all():
        raise ValueError("Invalid predictions")
    values = [{m: float(f(truth[:, e], prediction[:, e]))
               for m, f in zip(METRICS, (pcc, rmse, mae))}
              for e in range(truth.shape[1])]
    if not all(np.isfinite(list(v.values())).all() for v in values):
        raise ValueError("Nonfinite metric; failed environments cannot be omitted")
    return values


def base_value(source, eta):
    penalties = source["ridge"]
    if len(set(penalties)) != 1:
        raise ValueError("The analytic head requires a shared ridge penalty")
    return {"bandwidth": source["bandwidth"], "fraction": source["fraction"],
            "ridge": penalties[0], "eta": eta}


def candidates(cfg, model):
    steps = [0] if model == "frozen_full" else cfg["checkpoints"]
    return list(itertools.product(cfg["gammas"], cfg["weight_decays"], steps))


def choose(rows, cfg, model):
    scored = []
    for gamma, decay, step in candidates(cfg, model):
        selected = [r for r in rows if r["model"] == model and r["gamma"] == gamma
                    and r["weight_decay"] == decay and r["steps"] == step]
        if len(selected) != 12:
            raise ValueError(f"Incomplete three-fold/four-environment candidate: {model}")
        scored.append((float(np.mean([r["pcc"] for r in selected])), gamma, decay, step))
    ceiling = max(v[0] for v in scored)
    score, gamma, decay, step = next(v for v in scored if ceiling-v[0] <= cfg["tie_tolerance"])
    return {"inner_pcc": score, "gamma": gamma, "weight_decay": decay, "steps": step}


def development_gate(selected, cfg):
    differences = {}
    for control in ("baseline", "frozen_full", "full_background", "linear", "global"):
        differences[control] = [
            item["chosen"]["full"]["inner_pcc"] -
            (item["baseline_pcc"] if control == "baseline" else item["chosen"][control]["inner_pcc"])
            for item in selected]
    details = {}
    for name, delta in differences.items():
        minimum = cfg["minimum_development_gain"] if name == "baseline" else cfg["minimum_mechanism_gain"]
        details[name] = {"mean_delta": float(np.mean(delta)), "per_fold_delta": delta,
                         "improved_folds": int(sum(v > cfg["tie_tolerance"] for v in delta)),
                         "minimum_gain": minimum}
    passed = len(selected) == 10 and all(
        v["mean_delta"] >= v["minimum_gain"] and
        v["improved_folds"] >= cfg["minimum_improved_folds"] for v in details.values())
    return {"pass": bool(passed), "comparisons": details,
            "meaning": "Conditional development selection statistics, not independent generalization"}


def run(cfg, args, runner):
    from wheat599.locus_context import train_context, predict_context

    cfg = copy.deepcopy(cfg)
    torch.set_num_threads(int(cfg["cpu_threads"]))
    data = load(Path(cfg["data"]), cfg["data_sha256"])
    folds = json.loads(Path(cfg["splits"]).read_text())["folds"]
    if len(folds) != 10 or len(data.environments) != 4:
        raise ValueError("This frozen experiment requires ten material folds and four environments")
    source_hashes = {key: sha256_of(Path(cfg[key])) for key in
                     ("splits", "base_choices", "refit_base_choices")}
    base_records = json.loads(Path(cfg["base_choices"]).read_text())
    refit_source = json.loads(Path(cfg["refit_base_choices"]).read_text())
    if (refit_source["splits_sha256"] != source_hashes["splits"]
            or refit_source["base_source_sha256"] != source_hashes["base_choices"]):
        raise ValueError("Base choices do not belong to this split/source")
    if any(item["squared"]["eta"] != cfg["eta"] for item in refit_source["selected"]):
        raise ValueError("Fixed eta differs from the recorded development choice")
    frozen = json.loads(args.selection.read_text()) if args.selection else None
    if frozen and (frozen["config"] != cfg or frozen["source_sha256"] != source_hashes
                   or not frozen["development_gate"]["pass"]
                   or len(frozen["selected"]) != 10
                   or {i["fold"] for i in frozen["selected"]} != {f["fold"] for f in folds}):
        raise ValueError("Frozen selection mismatch or failed development condition")
    if cfg["device"] == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Configured CUDA unavailable; no silent device/precision change")
        torch.cuda.set_per_process_memory_fraction(cfg["gpu_memory_fraction"])
    args.out.mkdir(parents=True, exist_ok=False)
    provenance = {"config": cfg, "source_sha256": source_hashes,
                  "data_sha256": cfg["data_sha256"],
                  "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  "code_sha256": {p.name: sha256_of(p) for p in
                                 (Path(__file__), Path(__file__).with_name("locus_context.py"), Path(runner))},
                  "selection_sha256": sha256_of(args.selection) if args.selection else None,
                  "protocol": "CV1; fit-only neural episodes; inner macro PCC selection; frozen refit",
                  "runtime": {"torch": torch.__version__, "numpy": np.__version__,
                              "python": __import__("sys").version}}
    (args.out / "provenance.json").write_text(json.dumps(provenance, indent=2))
    rows, selected, records, failures, oof = [], [], [], [], []
    start = time.perf_counter()

    def append_metrics(prediction, truth, fold, inner, model, choice):
        for env, value in zip(data.environments, metrics(truth, prediction)):
            rows.append({"fold": fold, "inner_fold": inner, "model": model, "environment": env,
                         **choice, **value})

    for fold in folds:
        k = fold["fold"]
        print(f"fold {k}", flush=True)
        try:
            dev, test = np.array(fold["dev_idx"]), np.array(fold["test_idx"])
            if set(dev) & set(test) or set(dev) | set(test) != set(range(data.n)):
                raise ValueError("Invalid outer split")
            if frozen is None:
                fold_start = len(rows)
                for j, inner in enumerate(fold["inner"]):
                    fit, valid = np.array(inner["train_idx"]), np.array(inner["val_idx"])
                    if set(fit) & set(valid) or set(fit) | set(valid) != set(dev):
                        raise ValueError("Invalid inner split")
                    base = base_value(next(v["homogeneous"] for v in base_records
                                           if v["fold"] == k and v["inner_fold"] == j), cfg["eta"])
                    seed = cfg["seed"] + 100 * k + j
                    X, Y, target = data.X[fit], data.Y[fit], data.X[valid]
                    baseline = predict_context(X, Y, target, base, "full", 0., None, seed, cfg)
                    append_metrics(baseline, data.Y[valid], k, j, "baseline",
                                   {"gamma": 0., "weight_decay": 0., "steps": 0})
                    for variant, gamma, decay in itertools.product(
                            cfg["variants"], cfg["gammas"], cfg["weight_decays"]):
                        states, record = train_context(X, Y, base, variant, gamma, decay, seed, cfg)
                        records.append({"fold": k, "inner_fold": j, "variant": variant,
                                        "gamma": gamma, "weight_decay": decay, **record})
                        for step in cfg["checkpoints"]:
                            prediction = predict_context(X, Y, target, base, variant, gamma,
                                                         states[step], seed, cfg)
                            choice = {"gamma": gamma, "weight_decay": decay, "steps": step}
                            append_metrics(prediction, data.Y[valid], k, j, variant, choice)
                            if variant == "full" and step == 0:
                                append_metrics(prediction, data.Y[valid], k, j, "frozen_full", choice)
                        del states
                    print(f"  inner {j} complete", flush=True)
                unit = rows[fold_start:]
                selected.append({"fold": k, "chosen": {m: choose(unit, cfg, m) for m in MODELS},
                                 "base": base_value(next(v["base"] for v in refit_source["selected"]
                                                         if v["fold"] == k), cfg["eta"]),
                                 "baseline_pcc": float(np.mean([r["pcc"] for r in unit
                                                               if r["model"] == "baseline"]))})
            else:
                item = next(v for v in frozen["selected"] if v["fold"] == k)
                X, Y, target, base = data.X[dev], data.Y[dev], data.X[test], item["base"]
                seed = cfg["seed"] + 100 * k + 99
                for model in ("baseline", *MODELS):
                    choice = ({"gamma": 0., "weight_decay": 0., "steps": 0} if model == "baseline"
                              else {v: item["chosen"][model][v] for v in ("gamma", "weight_decay", "steps")})
                    if model != "baseline" and tuple(choice[v] for v in ("gamma", "weight_decay", "steps")) not in candidates(cfg, model):
                        raise ValueError("Frozen choice outside the fixed candidate set")
                    variant = "full" if model in ("baseline", "frozen_full") else model
                    state = None
                    if model != "baseline":
                        local_cfg = {**cfg, "training_steps": choice["steps"], "checkpoints": [choice["steps"]]}
                        states, record = train_context(X, Y, base, variant, choice["gamma"],
                                                       choice["weight_decay"], seed, local_cfg)
                        state = states[choice["steps"]]
                        records.append({"fold": k, "model": model, **record})
                    prediction = predict_context(X, Y, target, base, variant, choice["gamma"], state, seed, cfg)
                    append_metrics(prediction, data.Y[test], k, "", model, choice)
                    for e, env in enumerate(data.environments):
                        oof.extend({"fold": k, "model": model, "environment": env,
                                    "material_index": int(idx), "y_true": float(t), "y_pred": float(p)}
                                   for idx, t, p in zip(test, data.Y[test, e], prediction[:, e]))
        except Exception as exc:
            failures.append({"fold": k, "reason": repr(exc)})
            raise
        finally:
            write(args.out / "failures.csv", ["fold", "reason"], failures)
            write(args.out / ("validation.csv" if frozen is None else "per_fold.csv"), FIELDS, rows)
            (args.out / "training.json").write_text(json.dumps(records, indent=2))
            if frozen is None:
                (args.out / "selection.json").write_text(json.dumps(
                    {"config": cfg, "source_sha256": source_hashes, "selected": selected,
                     "development_gate": development_gate(selected, cfg) if selected else {"pass": False}},
                    indent=2))
            else:
                write(args.out / "oof.csv", ["fold", "model", "environment", "material_index",
                                            "y_true", "y_pred"], oof)
    if frozen is None:
        print(json.dumps(development_gate(selected, cfg)), flush=True)
        print("Development complete; no outer test predictions.", flush=True)
    else:
        summary = {"n_failures": len(failures), "elapsed_seconds": time.perf_counter()-start, "models": {}}
        for model in ("baseline", *MODELS):
            model_rows = [r for r in rows if r["model"] == model]
            if len(model_rows) != 40:
                raise ValueError("Incomplete model evaluation")
            per = {env: {metric: float(np.mean([r[metric] for r in model_rows if r["environment"] == env]))
                         for metric in METRICS} for env in data.environments}
            pooled = {}
            for env in data.environments:
                values = [r for r in oof if r["model"] == model and r["environment"] == env]
                if len(values) != data.n or len({r["material_index"] for r in values}) != data.n:
                    raise ValueError("OOF materials duplicated or missing")
                pooled[env] = float(pcc(np.array([r["y_true"] for r in values]),
                                        np.array([r["y_pred"] for r in values])))
            summary["models"][model] = {"per_environment": per,
                                       "macro": {m: float(np.mean([v[m] for v in per.values()])) for m in METRICS},
                                       "pooled_oof_pcc": pooled}
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary), flush=True)
