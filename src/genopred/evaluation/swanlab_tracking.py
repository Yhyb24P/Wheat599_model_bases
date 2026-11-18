"""Opt-in SwanLab experiment tracking for genopred runs (cloud swanlab.cn).

Enabled only when SWANLAB_ENABLE is truthy. Organizes runs by dataset project
(wheat599/wheat2000/soybase) with host/model/dataset/phase tags so the dashboard
groups many machines/models/tasks. Runners call log() for live per-fold/epoch
curves; the cli.py hook inits the run and finish_run() logs the final summary.
SwanLab auto-records GPU/CPU/memory while a run is active. No-op when disabled
or swanlab absent, leaving existing runs unaffected.

Host setup: pip install swanlab; swanlab login -k <key from swanlab.cn/settings>.
"""
import json
import os
import socket
from pathlib import Path


def _enabled():
    return os.environ.get('SWANLAB_ENABLE', '').lower() in ('1', 'true', 'yes')


def _dataset(cfg):
    blob = json.dumps(cfg, default=str)
    if 'wheat2000' in blob:
        return 'wheat2000'
    if 'soybase' in blob:
        return 'soybase'
    return 'wheat599'


def init_run(cfg, out, phase=None):
    if not _enabled():
        return None
    try:
        import swanlab
    except ImportError:
        return None
    try:
        dataset = _dataset(cfg)
        model = str(cfg.get('model', 'unknown'))
        tags = [socket.gethostname(), model, dataset]
        if phase:
            tags.append(str(phase))
        if cfg.get('trait'):
            tags.append(str(cfg['trait']))
        return swanlab.init(
            project=os.environ.get('SWANLAB_PROJECT', dataset),
            experiment_name=f'{model}-{Path(out).name}',
            tags=tags,
            config={k: cfg[k] for k in ('model', 'seeds', 'traits', 'trait', 'device',
                                        'min_call_rate', 'min_maf', 'n_quantiles') if k in cfg},
        )
    except Exception:
        return None


def log(metrics, step=None):
    """Live metric logging for runners (per-fold/epoch). No-op if no active run."""
    if not _enabled():
        return
    try:
        import swanlab
        swanlab.log(metrics, step=step)
    except Exception:
        pass


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