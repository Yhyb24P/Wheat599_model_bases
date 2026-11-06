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


def test_relation_uses_final_prediction_scale(monkeypatch):
    import wheat599.relation_quantile as module
    original=module.build_model; full_outputs=[];checked=[]
    def wrapped(*args,**kwargs):
        model=original(*args,**kwargs)
        def capture(m,inputs,output):
            if len(inputs[0])==24:
                full_outputs.append(m.inverse_y(output.mean(dim=(1,2))[:,None])[:,0])
        model.register_forward_hook(capture)
        return model
    monkeypatch.setattr(module,'build_model',wrapped)
    torch.set_num_threads(1);rng=np.random.default_rng(13)
    x=rng.normal(size=(32,12)).astype('float32');y=(rng.normal(size=(32,4))*[2,3,4,5]+[10,20,30,40]).astype('float32')
    c=dict(n_blocks=2,d_block=16,dropout=.1,k=2,learning_rate=.001,weight_decay=.001)
    spec=dict(n_quantiles=5,quantile_min=.001,quantile_max=.999,training=dict(max_epochs=1,patience=2,batch_size=8,gradient_clip=5.,tie_tolerance=1e-12))
    def relation(m):
        torch.testing.assert_close(m,torch.stack(full_outputs[-4:],dim=1))
        checked.append(True);return (m-m.mean(0)).square().mean()
    train_joint(x[:24],y[:24],x[24:],y[24:],[c]*4,42,[0]*4,.1,spec,relation,torch.device('cpu'))
    assert len(checked)==3


def test_curve_control_receives_raw_scale_quantiles(monkeypatch):
    import wheat599.relation_quantile as module
    original=module.build_model; full_outputs=[];checked=[]
    def wrapped(*args,**kwargs):
        model=original(*args,**kwargs)
        def capture(m,inputs,output):
            if len(inputs[0])==24:
                full_outputs.append(m.inverse_y(output.mean(dim=1)))
        model.register_forward_hook(capture)
        return model
    monkeypatch.setattr(module,'build_model',wrapped)
    torch.set_num_threads(1);rng=np.random.default_rng(14)
    x=rng.normal(size=(32,12)).astype('float32');y=(rng.normal(size=(32,4))*[2,3,4,5]+[10,20,30,40]).astype('float32')
    c=dict(n_blocks=2,d_block=16,dropout=.1,k=2,learning_rate=.001,weight_decay=.001)
    spec=dict(n_quantiles=5,quantile_min=.001,quantile_max=.999,training=dict(max_epochs=1,patience=2,batch_size=8,gradient_clip=5.,tie_tolerance=1e-12))
    def relation(q):
        assert q.shape==(24,4,5)
        torch.testing.assert_close(q,torch.stack(full_outputs[-4:],dim=1))
        checked.append(True);return (q-q.mean(0)).square().mean()
    train_joint(x[:24],y[:24],x[24:],y[24:],[c]*4,42,[0]*4,10.,spec,relation,torch.device('cpu'),curve=True)
    assert len(checked)==3


def test_permutation_retains_spectrum_and_is_reproducible():
    from wheat599.relation_quantile_controls import permute_kernel
    generator=torch.Generator().manual_seed(11)
    x=torch.randn(12,5,generator=generator,dtype=torch.float64);gram=x@x.T
    first,p=permute_kernel(gram,99);second,p2=permute_kernel(gram,99)
    assert torch.equal(p,p2) and torch.equal(first,second)
    torch.testing.assert_close(torch.linalg.eigvalsh(gram),torch.linalg.eigvalsh(first),atol=1e-12,rtol=1e-12)
    assert not torch.equal(first,gram)


def test_curve_penalty_decomposes_into_center_and_shape():
    from wheat599.relation_regularizer import make_relation, make_relation_curve
    generator=torch.Generator().manual_seed(97)
    x=torch.randn(14,7,generator=generator,dtype=torch.float64)
    y=torch.randn(14,4,generator=generator,dtype=torch.float64)
    curves=torch.randn(14,4,9,generator=generator,dtype=torch.float64)
    gram=x@x.T;center=curves.mean(dim=2);shape=curves-center[:,:,None]
    relation=make_relation(gram,y);whole=make_relation_curve(gram,y)(curves)
    expected=relation(center)+torch.stack([relation(shape[:,:,q]) for q in range(9)]).mean()
    torch.testing.assert_close(whole,expected,atol=1e-12,rtol=1e-12)
