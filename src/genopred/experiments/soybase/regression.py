"""SoyBase seven-trait regression baselines on the shared nested protocol.

Reuses the wheat2000 classical/kernel loop with two SoyBase adaptations:
strict fit-only SNP preprocessing per fit set (call-rate/MAF filter, mean
imputation, standardization recomputed on the current fit rows, protocol §15)
and per-trait NaN label masks on the material-level joint holdout splits.
GBLUP is a standalone row (gblup: true): fixed fraction 1.0, ridge inner-selected.
"""
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

import genopred
from genopred.data import soybase
from genopred.data.wheat599 import sha256_of
from genopred.experiments.wheat2000.classical import nested_classical


def fit_only_preprocess(X, fit, min_call_rate, min_maf):
    """SNP filter + impute + standardize using ONLY the fit rows (§15)."""
    sub = X[fit]
    observed = ~np.isnan(sub)
    call_rate = observed.mean(axis=0)
    filled = np.where(observed, sub, 0.0)
    alt_freq = filled.sum(axis=0) / (2.0 * np.maximum(observed.sum(axis=0), 1))
    maf = np.minimum(alt_freq, 1.0 - alt_freq)
    keep = (call_rate >= min_call_rate) & (maf >= min_maf)
    if not keep.any():
        raise ValueError('fit-only SNP filter removed every marker')
    means = (filled.sum(axis=0)[keep] / observed.sum(axis=0)[keep]).astype(X.dtype)
    out = X[:, keep]
    np.copyto(out, means, where=np.isnan(out))
    std = out[fit].std(axis=0)
    out -= means
    out /= np.where(std == 0, 1.0, std)
    return out


def run(cfg, args, runner_path):
    data = soybase.load_soybase_regression(cfg['cache'], cfg['cache_sha256'])
    split_path = Path(cfg['splits'])
    if sha256_of(split_path) != cfg['splits_sha256']:
        raise ValueError('Frozen SoyBase regression splits SHA differs')
    splits = json.loads(split_path.read_text())
    traits = list(data.trait_names)
    if splits['cache_sha256'] != cfg['cache_sha256']:
        raise ValueError('Frozen SoyBase regression splits refer to a different cache')
    if traits != cfg['traits'] or splits['trait_names'] != traits:
        raise ValueError('All seven traits and their order must match the frozen protocol')
    config_sha = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    _pkg = Path(genopred.__file__).parent
    sources = {str(p): sha256_of(Path(p)) for p in
               (Path(__file__), Path(runner_path), Path(args.config),
                _pkg/'experiments'/'wheat2000'/'classical.py',
                _pkg/'models'/'classical'/'classical_baselines.py',
                _pkg/'models'/'kernels'/'kernel_baselines.py',
                _pkg/'data'/'soybase.py', _pkg/'evaluation'/'metrics.py')}
    nested_classical(
        cfg, args, data, splits['folds'], traits,
        sources=sources, config_sha=config_sha,
        prepare=lambda X, fit: fit_only_preprocess(X, fit, cfg['min_call_rate'], cfg['min_maf']),
        labeled=np.isfinite(data.Y),
        provenance_extra={
            'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            'task': 'soybase_regression',
            'preprocessing': 'strict fit-only per fit set: call-rate/MAF filter, mean impute, standardize (protocol §15)',
            'splits_rule': 'material-level joint KFold holdout shared by all traits; per-trait NaN labels masked at fit/score',
            'gblup': 'centered linear kernel with inner-selected ridge' if cfg.get('gblup') else None,
        },
        oof_ids=lambda d, idx: [{'sample_id': str(d.sample_ids[i])} for i in idx],
        oof_id_fields=('sample_id',))
