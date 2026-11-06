import numpy as np
import pytest
import torch
pytest.importorskip('tabm')
from wheat599.midpoint_quantile import levels,train


def test_midpoints_and_training():
    q=levels(99,'cpu').numpy()
    np.testing.assert_allclose(q,(np.arange(99)+.5)/99,atol=1e-7)
    rng=np.random.default_rng(9);x=rng.normal(size=(24,6));y=rng.normal(size=24)
    c=dict(n_blocks=2,d_block=16,dropout=.1,k=2,learning_rate=.001,weight_decay=.001)
    s=dict(n_quantiles=99,training=dict(max_epochs=1,patience=2,batch_size=8,gradient_clip=5.,tie_tolerance=1e-12))
    pred,info=train(x[:18],y[:18],x[18:],y[18:],c,42,0,s,'cpu')
    assert pred.shape==(6,) and np.isfinite(pred).all() and info['best_epoch'] in (0,1)
