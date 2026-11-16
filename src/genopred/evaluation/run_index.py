"""Machine-readable run index derived read-only from existing run artifacts.

Scans results/**/ for provenance.json / summary.json / execution_status.json and
derives one record per run: dataset, method, phase, status, role, source_commit,
protocol, result_path. Writes only results/run_index.json; never modifies frozen
run directories, never re-hashes, and is not consumed by build_report (site
determinism unchanged). Phase/role come from a small static rule table over the
run directory name and provenance fields, not from physical reclassification.
"""
import json
import subprocess
from pathlib import Path

SCHEMA = "genopred-run-index/1"

_PHASE_SUFFIX = (
    ("_development_", "development"), ("_dev_", "development"),
    ("_selection_", "selection"), ("_manifest_", "selection"),
    ("_outer_", "outer_eval"), ("_final_", "refit"),
    ("_reproduction", "refit"), ("_refit_", "refit"),
    ("_diagnostic_", "diagnostic"), ("_diag_", "diagnostic"),
)
_PHASE_NORMALIZE = {"dev": "development", "development": "development",
                    "selection": "selection", "final": "refit", "refit": "refit",
                    "outer": "outer_eval", "outer_eval": "outer_eval",
                    "diagnostic": "diagnostic", "test": "outer_eval"}


def _load(path):
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _method(prov, name):
    cfg = prov.get("config")
    model = cfg.get("model") if isinstance(cfg, dict) else None
    if isinstance(model, str) and model:
        return model
    if model is None and ("background" in name or name.startswith("a_rr")):
        return "background_ridge"
    for suffix, _ in _PHASE_SUFFIX:
        if suffix in name:
            return name.split(suffix)[0]
    return name.split("_2026")[0].split("_2025")[0]


def _phase(prov, status, name):
    raw = prov.get("phase") or status.get("phase")
    if isinstance(raw, str) and raw in _PHASE_NORMALIZE:
        return _PHASE_NORMALIZE[raw]
    for suffix, phase in _PHASE_SUFFIX:
        if suffix in name:
            return phase
    return "unknown"


def _role(dataset, family, name):
    if family == "baselines":
        return "baseline"
    if family == "diagnostics" or dataset == "diagnostics" or "diagnostic" in name:
        return "diagnostic"
    if dataset == "wheat2000":
        return "cross_dataset"
    return "candidate"


def _status(run_dir, status):
    raw = status.get("status")
    if isinstance(raw, str) and raw:
        return raw
    failures = run_dir / "failures.json"
    if failures.exists():
        data = _load(failures)
        rows = data.get("failures", data) if isinstance(data, dict) else data
        if rows:
            return "failed"
    if (run_dir / "interruption.json").exists():
        return "interrupted"
    if (run_dir / "summary.json").exists():
        return "complete"
    return "unknown"


def _run_dirs(root):
    prov_dirs = {p.parent for p in root.rglob("provenance.json")}
    runs = set(prov_dirs)
    for marker in ("summary.json", "execution_status.json"):
        for p in root.rglob(marker):
            d = p.parent
            if not any(parent in prov_dirs for parent in d.parents):
                runs.add(d)
    return sorted(runs)


def build_index(root):
    root = Path(root)
    results = root / "results"
    try:
        commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        commit = None
    runs = []
    for run_dir in _run_dirs(results):
        prov = _load(run_dir / "provenance.json")
        status = _load(run_dir / "execution_status.json")
        rel = run_dir.relative_to(results)
        parts = rel.parts
        dataset = parts[0]
        family = parts[1] if len(parts) > 2 else ""
        name = run_dir.name
        cfg = prov.get("config") if isinstance(prov.get("config"), dict) else {}
        runs.append({
            "run_id": str(rel),
            "dataset": dataset,
            "family": family,
            "method": _method(prov, name),
            "phase": _phase(prov, status, name),
            "status": _status(run_dir, status),
            "role": _role(dataset, family, name),
            "source_commit": prov.get("git_commit") or prov.get("source_commit"),
            "protocol": prov.get("protocol") or cfg.get("splits"),
            "result_path": str(rel),
        })
    runs.sort(key=lambda r: r["run_id"])
    return {"schema": SCHEMA, "generated_from_commit": commit, "n_runs": len(runs), "runs": runs}


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    index = build_index(root)
    out = Path(args.out) if args.out else root / "results" / "run_index.json"
    out.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n")
    print(f"Wrote {out} ({index['n_runs']} runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())