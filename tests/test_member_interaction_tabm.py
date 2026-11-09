import copy

import numpy as np
import pytest
import torch

from wheat599.member_interaction_tabm import MemberInteractionBackbone
from wheat599.quantile_tabm import build_model, quantile_loss, train, update_epoch


def model_inputs():
    rng=np.random.default_rng(83)
    x=rng.normal(size=(30,12)).astype('float32');y=rng.normal(size=30).astype('float32')
    config=dict(n_blocks=3,d_block=16,dropout=.2,k=3,learning_rate=.001,weight_decay=.001)
    spec=dict(n_quantiles=5,quantile_min=.001,quantile_max=.999,training=dict(max_epochs=2,patience=3,batch_size=8,gradient_clip=5.,tie_tolerance=1e-12))
    return x,y,config,spec


@pytest.mark.parametrize('n_blocks',[2,3,4])
def test_zero_alpha_preserves_official_member_outputs_and_dropout(n_blocks):
    x,y,c,s=model_inputs();c['n_blocks']=n_blocks;base=build_model(x[:20],y[:20],c,42,s,torch.device('cpu'))
    reference=copy.deepcopy(base);base.backbone=MemberInteractionBackbone(base.backbone,12,rank=4)
    for training in (False,True):
        base.train(training);reference.train(training);torch.manual_seed(91);a=base(torch.from_numpy(x[20:]));torch.manual_seed(91);b=reference(torch.from_numpy(x[20:]));torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_member_interactions_match_explicit_pairs_and_alpha_gradients():
    x,y,c,s=model_inputs();base=build_model(x[:20],y[:20],c,42,s,torch.device('cpu'))
    net=MemberInteractionBackbone(base.backbone,12,rank=4);z=base.standardize_x(torch.from_numpy(x[20:]));r=net.base.backbone.blocks[0][0].r;scaled=z[:,None,:]*r
    observed=[];net.keys[0].register_forward_pre_hook(lambda _m,args:observed.append(args[0].detach()))
    out=net(z);v=net.factors
    expected=sum(scaled[:,:,i,None]*scaled[:,:,j,None]*v[i]*v[j] for i in range(12) for j in range(i+1,12))
    torch.testing.assert_close(observed[0],expected,rtol=1e-5,atol=1e-6)
    assert not torch.allclose(observed[0][:,0],observed[0][:,1])
    out.square().mean().backward();assert net.alpha.grad is not None and net.alpha.grad.abs().sum()>0
    net.zero_grad(set_to_none=True);net.alpha.data.fill_(.1);net(z).square().mean().backward()
    assert net.factors.grad is not None and net.factors.grad.abs().sum()>0
    assert all(layer.weight.grad is not None and layer.weight.grad.abs().sum()>0 for layer in net.keys)
    assert all(layer.weight.grad is not None and layer.weight.grad.abs().sum()>0 for layer in net.values)


def test_member_candidate_trains_with_adamw():
    torch.set_num_threads(1);x,y,c,s=model_inputs()
    pred,record=train(x[:20],y[:20],x[20:],y[20:],c,42,0,dict(s,optimizer_variant='member_conditional'),torch.device('cpu'))
    assert pred.shape==(10,) and np.isfinite(pred).all() and record['best_epoch']>=0


def test_zero_alpha_preserves_original_gradients_and_delays_interaction_gradients():
    x,y,c,s=model_inputs();base=build_model(x[:20],y[:20],c,42,s,torch.device('cpu'))
    reference=copy.deepcopy(base)
    rng_before=torch.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]):base.backbone=MemberInteractionBackbone(base.backbone,12,rank=4)
    assert torch.equal(rng_before,torch.get_rng_state())
    base.eval();reference.eval();inputs=torch.from_numpy(x[20:])
    base(inputs).square().mean().backward();reference(inputs).square().mean().backward()
    original=dict(reference.backbone.named_parameters());actual=dict(base.backbone.base.named_parameters())
    assert original.keys()==actual.keys()
    for name in original:torch.testing.assert_close(original[name].grad,actual[name].grad,rtol=0,atol=0)
    assert base.backbone.alpha.grad.abs().sum()>0
    assert base.backbone.factors.grad.abs().sum()==0
    assert all(layer.weight.grad.abs().sum()==0 for layer in base.backbone.keys)
    assert all(layer.weight.grad.abs().sum()==0 for layer in base.backbone.values)


def test_frozen_zero_alpha_matches_one_original_training_epoch():
    torch.set_num_threads(1);x,y,c,s=model_inputs()
    reference=build_model(x[:20],y[:20],c,42,s,torch.device('cpu'));base=copy.deepcopy(reference)
    with torch.random.fork_rng(devices=[]):base.backbone=MemberInteractionBackbone(base.backbone,12,rank=4)
    base.backbone.alpha.requires_grad_(False)
    inputs=torch.from_numpy(x[:20]);target=reference.standardize_y(torch.from_numpy(y[:20,None]));levels=torch.linspace(.001,.999,5)
    reference.train();base.train();torch.manual_seed(17);loss=quantile_loss(reference(inputs),target,levels)
    torch.manual_seed(17);other_loss=quantile_loss(base(inputs),target,levels)
    torch.testing.assert_close(loss,other_loss,rtol=0,atol=0)
    optimizers=[torch.optim.AdamW(m.parameters(),lr=c['learning_rate'],weight_decay=c['weight_decay']) for m in (reference,base)]
    for model,optimizer in zip((reference,base),optimizers):
        torch.manual_seed(27);update_epoch(model,inputs,target,levels,optimizer,torch.Generator().manual_seed(100),s['training'],torch.device('cpu'))
    reference.eval();base.eval();torch.testing.assert_close(reference(inputs),base(inputs),rtol=0,atol=0)
