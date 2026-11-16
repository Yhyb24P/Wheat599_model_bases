"""WheatGP unified wiring: split prep, vendor forward and refit API (CPU)."""
import importlib.util
from pathlib import Path

import numpy as np
import torch

from genopred.models.neural.wheatgp_unified import fit, prepare, segments

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('wheatgp_vendor', ROOT/'third_party/WheatGP/model/WheatGP_base.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

CFG = {'pca_components': 10, 'weight_decay': 0.0, 'batch_size': 8,
       'max_epochs': 2, 'patience': 50}


def test_segments_partition():
    x = torch.zeros(3, 45)
    pieces = segments(x)
    assert len(pieces) == 5
    assert torch.equal(torch.cat(pieces, dim=1), x)


def test_fit_select_and_refit_paths():
    rng = np.random.default_rng(0)
    x = rng.integers(0, 2, size=(24, 40)).astype(float)
    y = x[:, :5].mean(1) + rng.normal(0, 0.05, size=24)
    xf, xq = prepare(x, np.arange(16), np.arange(16, 24), 'raw', 42, CFG)
    pred, best_epoch, epochs_run = fit(module.wheatGP_base, xf, y[:16], xq, y[16:],
                                       1e-3, 2, CFG, torch.device('cpu'), 42)
    assert pred.shape == (8,) and np.all(np.isfinite(pred))
    assert epochs_run >= 1 and best_epoch >= 1
    refit, _, _ = fit(module.wheatGP_base, xf, y[:16], xq, None,
                      1e-3, 2, CFG, torch.device('cpu'), 42, select=False)
    assert refit.shape == (8,) and np.all(np.isfinite(refit))
