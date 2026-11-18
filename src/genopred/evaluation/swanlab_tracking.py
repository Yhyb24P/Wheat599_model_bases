"""Opt-in SwanLab experiment tracking for genopred runs (cloud swanlab.cn).

Enabled only when SWANLAB_ENABLE is truthy. Logs run config and final summary
metrics; SwanLab auto-records system hardware (GPU/CPU/memory) and host info,
so one project dashboard distinguishes runs across omen/tim/nhy. No-op when
disabled or swanlab is absent, leaving existing runs unaffected.

Host setup: pip install swanlab; swanlab login -k <API key from swanlab.cn/settings>.
"""
import json
import os
from pathlib import Path


def _dataset(cfg):
    blob = json.dumps(cfg, default=str)
    if 'wheat2000' in blob:
        return 'wheat2000'
    if 'soybase' in blob:
        return 'soybase'
    return 'wheat599'


def init_run(cfg, out):
    if os.environ.get('SWANLAB_ENABLE', '').lower() not in ('1', 'true', 'yes'):
        return None
    try:
        import swanlab
    except ImportError:
        return None
    try:
        return swanlab.init(
            project=os.environ.get('SWANLAB_PROJECT', 'wheat599'),
            experiment_name=Path(out).name,
            description=f"{cfg.get('model', 'unknown')} on {_dataset(cfg)}",
            config={k: cfg[k] for k in ('model', 'seeds', 'traits', 'device', 'trait')
                    if k in cfg},
        )
    except Exception:
        return None


def _flatten(obj, prefix=''):
    flat = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            flat.update(_flatten(v, f'{prefix}{k}/'))
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        import math
        if math.isfinite(obj):
            flat[prefix.rstrip('/')] = obj
    return flat


def finish_run(run, out):
    if run is None:
        return
    try:
        import swanlab
        summary = Path(out) / 'summary.json'
        if summary.exists():
            metrics = _flatten(json.loads(summary.read_text()))
            if metrics:
                swanlab.log(metrics)
        swanlab.finish()
    except Exception:
        try:
            import swanlab
            swanlab.finish()
        except Exception:
            pass