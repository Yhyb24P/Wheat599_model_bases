import numpy as np
import pytest
pytest.importorskip('tabm')
from wheat599.fusion_refit import fit


@pytest.mark.parametrize('arm',['original','mean'])
def test_frozen_refit_outputs_and_weights(tmp_path,arm):
    rng=np.random.default_rng(9);x=rng.normal(size=(24,6));y=rng.normal(size=24)
    row=dict(arm=arm,fold=0,environment_index=0,seed=42,config_index=0,epochs=2)
    c=dict(n_blocks=2,d_block=16,dropout=.1,k=2,learning_rate=.001,weight_decay=.001)
    s=dict(n_quantiles=5,quantile_min=.001,quantile_max=.999,eta=.1,
           training=dict(batch_size=8,gradient_clip=5.))
    pred,info=fit(x[:18],y[:18],x[18:],row,c,s,'cpu',tmp_path)
    assert pred.shape==(6,) and np.isfinite(pred).all() and len(info['weight_sha256'])==64
