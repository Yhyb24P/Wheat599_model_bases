import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from wheat599.wheatgp_nested import fit, prepare, segments


def test_train_only_pca_and_official_shapes():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(25, 45))
    cfg = {'pca_components': 10, 'seed': 555}
    tr, va = np.arange(20), np.arange(20, 25)
    first, _ = prepare(x, tr, va, 'pca', cfg)
    x[va] += 1000
    second, _ = prepare(x, tr, va, 'pca', cfg)
    np.testing.assert_allclose(first, second)
    with pytest.raises(ValueError):
        prepare(x, tr, tr, 'raw', cfg)
    spec = importlib.util.spec_from_file_location('wgp', Path('vendor/WheatGP/model/WheatGP_base.py'))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    for width in (250, 1279):
        model = module.wheatGP_base(8 * (width - 20))
        assert model(*segments(torch.randn(2, width))).shape == (2, 1)


def test_training_produces_finite_prediction():
    spec = importlib.util.spec_from_file_location('wgp', Path('vendor/WheatGP/model/WheatGP_base.py'))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    rng = np.random.default_rng(2); x = rng.normal(size=(15, 30)).astype('float32'); y = rng.normal(size=15)
    cfg = {'seed': 555, 'weight_decay': .0001, 'batch_size': 4, 'patience': 2}
    pred, best, _ = fit(module.wheatGP_base, x[:10], y[:10], x[10:], y[10:], .001, 2, cfg, torch.device('cpu'))
    assert pred.shape == (5,) and np.isfinite(pred).all() and 1 <= best <= 2
