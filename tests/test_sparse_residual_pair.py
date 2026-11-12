"""Check that both seeds use the same validation-selected training epoch."""
import sys
import types
import numpy as np
import torch
from wheat599.sparse_residual_run import train_pair


def test_pair_selects_one_epoch(monkeypatch):
    class Fake(torch.nn.Module):
        def __init__(self, x, y, **kw):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.1))
        def standardize_y(self, y): return y
        def inverse_y(self, y): return y
        def forward(self, x): return (x[:, :1] * self.weight)[:, :, None].expand(-1, 2, 99)
    monkeypatch.setitem(sys.modules, 'wheat599.sparse_residual',
        types.SimpleNamespace(SparseResidualQuantile=Fake, proximal_step=lambda *a: None,
            make_optimizer=lambda model, lr, wd: torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=wd)))
    x=np.linspace(0,1,12,dtype=np.float32)[:,None]; y=x[:,0]
    cfg=dict(d_block=2,k=2,learning_rate=.001,weight_decay=0.,prox_lambda=0.)
    settings=dict(n_quantiles=99,quantile_min=.001,quantile_max=.999,
                  training=dict(max_epochs=3,patience=2,batch_size=6,gradient_clip=5.,tie_tolerance=1e-12))
    result=train_pair(x,y,x,y,cfg,[42,2026],0,settings,torch.device('cpu'))
    assert len(result)==2 and result[0][1]['best_epoch']==result[1][1]['best_epoch']
    assert result[0][1]['best_epoch']>=1
