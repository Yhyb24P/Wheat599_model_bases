"""GSformer wiring against the pinned upstream checkout when it is present."""
from pathlib import Path

import numpy as np
import pytest
import torch

from genopred.models.neural.gsformer_unified import build, fit, load_classes

SOURCE = Path('/tmp/gsformer-official')
pytestmark = pytest.mark.skipif(not SOURCE.exists(), reason='pinned GSformer checkout not present')

CFG = {
    'device': 'cpu', 'source_dir': str(SOURCE),
    'source_commit': '214c3831044ebcfce30db8e6825460bf8831bf1f',
    'variant': 'csa', 'pad_to': 40,
    'architecture': {'seq_num': 8, 'seq_length': 5, 'd_model': 8, 'd_ff': 16,
                     'heads': 2, 'dropout': 0.0, 'dim': 8, 'dim_head': 4},
    'training': {'learning_rate': 1e-3, 'weight_decay': 0.0, 'epochs': 2,
                 'batch_size': 8, 'patience': 5, 'min_loss_reduction': 1e-9},
}


def _classes():
    return load_classes(CFG['source_dir'], CFG['source_commit'])


def test_csa_build_and_fit():
    classes = _classes()
    x = np.random.default_rng(0).integers(0, 2, size=(20, 40)).astype(np.float32)
    y = x[:, :3].mean(1).astype(np.float32)
    model = build('csa', 40, CFG, classes)
    assert model(torch.as_tensor(x[:4])).shape == (4, 1)
    pred, best_epoch, epochs_run, n_params = fit('csa', classes, x, y, np.arange(14), np.arange(14, 20),
                                                 CFG, 42, 2, True)
    assert pred.shape == (6,) and np.all(np.isfinite(pred))
    assert epochs_run >= 1 and best_epoch >= 1 and n_params > 0


def test_nsa_build_and_fit():
    classes = _classes()
    nsa_cfg = dict(CFG, variant='nsa', pad_to=1440,
                   architecture={'dim': 72, 'dim_head': 18, 'heads': 4, 'dropout': 0.0,
                                 'compress_block_size': 4, 'selection_block_size': 4,
                                 'num_selected_blocks': 4})
    x = np.random.default_rng(1).integers(0, 2, size=(20, 1440)).astype(np.float32)
    y = x[:, :3].mean(1).astype(np.float32)
    model = build('nsa', 1440, nsa_cfg, classes)
    assert model(torch.as_tensor(x[:4])).shape == (4, 1)
    pred, best_epoch, epochs_run, n_params = fit('nsa', classes, x, y, np.arange(14), np.arange(14, 20),
                                                 nsa_cfg, 42, 2, True)
    assert pred.shape == (6,) and np.all(np.isfinite(pred))
    assert epochs_run >= 1 and best_epoch >= 1 and n_params > 0
