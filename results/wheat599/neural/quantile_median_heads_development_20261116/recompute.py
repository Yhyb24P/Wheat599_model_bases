"""One-shot independent recomputation of the H2 selection macro from saved predictions."""
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from wheat599.data import load
from wheat599.metrics import pcc

OUT = Path(__file__).resolve().parent
SEEDS = (42, 2026)

data = load(ROOT / 'data/raw/wheat.RData',
            '8710523389007dd8b7c88a5deb74ee16aea0bae158496bc11f0626fb43b5ea21')
folds = json.loads((ROOT / 'results/get_icm_protocol_20261103/splits_manifest.json').read_text())['folds']
summary = json.loads((OUT / 'summary.json').read_text())
n_configs = len(yaml.safe_load((ROOT / 'configs/environment_tabm.yaml').read_text())['configurations'])

scores = []
for fold in folds:
    f = fold['fold']
    arrays = [np.load(OUT / f'predictions_{f}_{j}.npz') for j in range(3)]
    for e in range(4):
        def pred(j, c):
            return np.mean([arrays[j][f'e{e}_c{c}_s{s}'] for s in SEEDS], axis=0)
        cfg_scores = [np.mean([pcc(data.Y[split['val_idx'], e], pred(j, c))
                               for j, split in enumerate(fold['inner'])]) for c in range(n_configs)]
        chosen = int(np.argmax(cfg_scores))
        assert chosen == summary['folds'][f]['selected'][e], (f, e, chosen)
        scores.append(cfg_scores[chosen])
macro = float(np.mean(scores))
ref = summary['macro']['quantile']['pcc']
assert abs(macro - ref) < 1e-6, (macro, ref)
print(f'recomputed macro PCC {macro:.9f} == summary {ref:.9f} (40 env-folds, selection matched)')
