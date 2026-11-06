import numpy as np
import pytest
import torch

pytest.importorskip('tabm')
from wheat599.quantile_tabm import quantile_loss, train


def test_quantile_loss_sums_quantiles_and_trains_members_separately():
    out = torch.tensor([[[0., 1.], [1., 3.]]], requires_grad=True)
    loss = quantile_loss(out, torch.tensor([[2.]]), torch.tensor([.25, .75]))
    assert loss.item() == pytest.approx((.5 + .75 + .25 + .25) / 2)
    loss.backward()
    assert torch.count_nonzero(out.grad) == 4


def test_real_tabm_outputs_finite_points_from_all_quantiles():
    rng = np.random.default_rng(31)
    x, y = rng.normal(size=(32, 8)), rng.normal(size=32)
    config = dict(n_blocks=2, d_block=16, dropout=.1, k=2,
                  learning_rate=.001, weight_decay=.001)
    settings = dict(n_quantiles=5, quantile_min=.001, quantile_max=.999,
        training=dict(max_epochs=2, patience=2, batch_size=8,
                      gradient_clip=5., tie_tolerance=1e-12))
    pred, info = train(x[:24], y[:24], x[24:], y[24:], config, 42, 0, settings, 'cpu')
    assert pred.shape == (8,) and np.isfinite(pred).all()
    assert 0 <= info['best_epoch'] <= 2
