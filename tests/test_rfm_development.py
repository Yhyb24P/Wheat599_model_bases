import numpy as np
import pytest
import torch

pytest.importorskip('rfm')
from wheat599.rfm_development import train


def test_official_full_rfm_preserves_fit_when_query_changes():
    rng = np.random.default_rng(17)
    x, y = rng.normal(size=(24, 6)), rng.normal(size=24)
    first, bandwidth = train(x[:18], y[:18], x[18:], 1., .1, 2, 8, torch.device('cpu'))
    changed = x[18:].copy()
    changed[1:] += 5
    second, _ = train(x[:18], y[:18], changed, 1., .1, 2, 8, torch.device('cpu'))
    assert first.shape == (3, 6) and np.isfinite(first).all() and bandwidth > 0
    np.testing.assert_allclose(first[:, 0], second[:, 0], atol=1e-5)
    assert not np.allclose(first[0], first[-1])
