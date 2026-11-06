import numpy as np
import pytest
import torch

pytest.importorskip('tabm')
from wheat599.multihead_mse import multihead_mse, train
from wheat599.quantile_tabm import train as quantile_train


def test_head_sum_keeps_member_losses_independent():
    output = torch.tensor([[[0., 1.], [2., 3.]]], requires_grad=True)
    loss = multihead_mse(output, torch.tensor([[1.]]))
    assert loss.item() == pytest.approx(3.)
    loss.backward()
    np.testing.assert_allclose(output.grad.numpy(), [[[-1., 0.], [1., 2.]]])


def test_quantile_telemetry_replays_original_and_mse_is_finite():
    rng = np.random.default_rng(17)
    x,y = rng.normal(size=(32,8)),rng.normal(size=32)
    c = dict(n_blocks=2,d_block=16,dropout=.1,k=2,learning_rate=.001,weight_decay=.001)
    s = dict(n_quantiles=5,quantile_min=.001,quantile_max=.999,
             training=dict(max_epochs=3,patience=4,batch_size=8,gradient_clip=5.,tie_tolerance=1e-12))
    args = (x[:24],y[:24],x[24:],y[24:],c,42,0,s,'cpu')
    expected,old = quantile_train(*args)
    replay = train(*args,'quantile')
    np.testing.assert_array_equal(expected,replay['prediction'])
    assert replay['best_epoch']==old['best_epoch']
    new = train(*args,'mse')
    assert np.isfinite(new['prediction']).all()
    assert 0 <= new['best_clipped'] <= new['best_steps']
