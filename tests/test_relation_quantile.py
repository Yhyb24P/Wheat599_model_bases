import numpy as np
import torch

from wheat599.quantile_tabm import train
from wheat599.relation_quantile import train_joint


def test_zero_penalty_preserves_independent_training():
    torch.set_num_threads(1)
    rng=np.random.default_rng(7)
    x=rng.normal(size=(32,12)).astype('float32');y=rng.normal(size=(32,4)).astype('float32')
    c=dict(n_blocks=2,d_block=16,dropout=.1,k=2,learning_rate=.001,weight_decay=.001)
    spec=dict(n_quantiles=5,quantile_min=.001,quantile_max=.999,training=dict(max_epochs=3,patience=2,batch_size=8,gradient_clip=5.,tie_tolerance=1e-12))
    device=torch.device('cpu');indices=[0,1,2,3]
    actual,details=train_joint(x[:24],y[:24],x[24:],y[24:],[c]*4,42,indices,0.0,spec,
                               lambda m:((m-m.mean(0))**2).mean(),device)
    for e in range(4):
        expected,entry=train(x[:24],y[:24,e],x[24:],y[24:,e],c,42,indices[e],spec,device)
        np.testing.assert_allclose(actual[:,e],expected,atol=1e-6,rtol=1e-6)
        assert details['best_epoch'][e]==entry['best_epoch']


def test_positive_penalty_keeps_gradients_and_finite_predictions():
    torch.set_num_threads(1);rng=np.random.default_rng(8)
    x=rng.normal(size=(32,12)).astype('float32');y=rng.normal(size=(32,4)).astype('float32')
    c=dict(n_blocks=2,d_block=16,dropout=.1,k=2,learning_rate=.001,weight_decay=.001)
    spec=dict(n_quantiles=5,quantile_min=.001,quantile_max=.999,training=dict(max_epochs=2,patience=2,batch_size=8,gradient_clip=5.,tie_tolerance=1e-12))
    observed=[]
    def relation(m):
        observed.append((m.requires_grad,m.shape));return (m-m.mean(0)).square().mean()
    pred,details=train_joint(x[:24],y[:24],x[24:],y[24:],[c]*4,42,[0]*4,.1,spec,relation,torch.device('cpu'))
    assert observed and all(x==(True,torch.Size([24,4])) for x in observed)
    assert pred.shape==(8,4) and np.isfinite(pred).all() and len(details['best_epoch'])==4


def test_relation_forward_uses_eval_mode_with_gradient(monkeypatch):
    import wheat599.relation_quantile as module
    original=module.build_model; events=[]
    def wrapped(*args,**kwargs):
        model=original(*args,**kwargs)
        model.register_forward_pre_hook(lambda m,inputs:events.append((len(inputs[0]),m.training,torch.is_grad_enabled())))
        return model
    monkeypatch.setattr(module,'build_model',wrapped)
    torch.set_num_threads(1);rng=np.random.default_rng(9)
    x=rng.normal(size=(32,12)).astype('float32');y=rng.normal(size=(32,4)).astype('float32')
    c=dict(n_blocks=2,d_block=16,dropout=.1,k=2,learning_rate=.001,weight_decay=.001)
    spec=dict(n_quantiles=5,quantile_min=.001,quantile_max=.999,training=dict(max_epochs=1,patience=2,batch_size=8,gradient_clip=5.,tie_tolerance=1e-12))
    train_joint(x[:24],y[:24],x[24:],y[24:],[c]*4,42,[0]*4,.1,spec,
                lambda m:(m-m.mean(0)).square().mean(),torch.device('cpu'))
    full=[v for v in events if v[0]==24]
    assert len(full)==12 and all(training is False and grad is True for _,training,grad in full)
