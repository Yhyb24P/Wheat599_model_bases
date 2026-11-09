import numpy as np
import torch

from wheat599.interaction_tabm import InteractionBackbone, make_optimizers
from wheat599.quantile_tabm import build_model, train


def test_bi_interaction_equals_explicit_pairs():
    torch.manual_seed(7);z=torch.randn(4,6);v=torch.randn(6,3)
    fast=.5*((z@v).square()-z.square()@v.square())
    direct=sum(z[:,i,None]*z[:,j,None]*v[i]*v[j] for i in range(6) for j in range(i+1,6))
    torch.testing.assert_close(fast,direct,atol=1e-6,rtol=1e-6)


def test_gated_backbone_members_and_optimizer_partition():
    torch.manual_seed(42);model=InteractionBackbone(12,width=16,members=3,dropout=.1,quantiles=5,rank=4)
    z=torch.randn(8,12);out=model(z)
    assert out.shape==(8,3,5) and not torch.allclose(out[:,0],out[:,1])
    out.square().mean().backward()
    assert model.factors.grad is not None and model.factors.grad.abs().sum()>0
    assert all(p.grad is not None for p in model.parameters())


def test_quantile_training_with_both_muon_variants():
    torch.set_num_threads(1);rng=np.random.default_rng(3)
    x=rng.normal(size=(30,12)).astype('float32');y=rng.normal(size=30).astype('float32')
    c=dict(n_blocks=2,d_block=16,dropout=.1,k=3,learning_rate=.001,weight_decay=.001,muon_lr=.002)
    spec=dict(n_quantiles=5,quantile_min=.001,quantile_max=.999,training=dict(max_epochs=1,patience=2,batch_size=8,gradient_clip=5.,tie_tolerance=1e-12))
    for variant in ('muon','gated'):
        pred,detail=train(x[:20],y[:20],x[20:],y[20:],c,42,0,dict(spec,optimizer_variant=variant),torch.device('cpu'))
        assert pred.shape==(10,) and np.isfinite(pred).all() and detail['best_epoch']>=0


def test_muon_parameter_groups_are_disjoint_and_complete():
    rng=np.random.default_rng(4);x=rng.normal(size=(20,12));y=rng.normal(size=20)
    c=dict(n_blocks=2,d_block=16,dropout=.1,k=3,learning_rate=.001,weight_decay=.001,muon_lr=.002)
    spec=dict(n_quantiles=5)
    for variant in ('muon','gated'):
        model=build_model(x,y,c,42,spec,torch.device('cpu'))
        if variant=='gated':model.backbone=InteractionBackbone(12,width=16,members=3,dropout=.1,quantiles=5,rank=4)
        muon,adam=make_optimizers(model,c,variant)
        m={id(p) for group in muon.param_groups for p in group['params']};a={id(p) for group in adam.param_groups for p in group['params']}
        assert not m&a and m|a=={id(p) for p in model.parameters()}
        assert all(p.ndim==2 for group in muon.param_groups for p in group['params'])
        if variant=='gated':
            assert id(model.backbone.factors) in a
            assert all(id(p) in a for gate in model.backbone.read for p in gate.parameters())
