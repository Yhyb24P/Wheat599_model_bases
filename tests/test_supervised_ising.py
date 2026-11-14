import inspect
import numpy as np
import pytest
import torch

from wheat599.supervised_ising import SupervisedIsing, AdditiveMLP
from wheat599.quantile_tabm import train


def data(p=9):
    rng=np.random.default_rng(5)
    x=rng.integers(0,2,(24,p)).astype('float32');x[:,0]=0
    return x,(x[:,1]-x[:,2]+rng.normal(size=len(x))).astype('float32')


def test_exact_zero_coupling_and_covariance_gradient():
    x,y=data();m=SupervisedIsing(x,y,spins=4).double()
    a=torch.tensor([[.2,-.3,.8,.1]],dtype=torch.float64,requires_grad=True)
    torch.testing.assert_close(m.hidden_mean(a),a.tanh(),atol=1e-14,rtol=1e-14)
    with torch.no_grad():m.coupling.copy_(torch.linspace(-.2,.3,6))
    probs=(a@m.states.T+m.pair_states@m.coupling).softmax(-1)
    mean=m.hidden_mean(a)[0]
    cov=(m.states.T*probs[0])@m.states-mean[:,None]*mean[None,:]
    jac=torch.autograd.functional.jacobian(lambda v:m.hidden_mean(v[None])[0],a[0])
    torch.testing.assert_close(jac,cov,atol=1e-13,rtol=1e-13)
    assert not torch.allclose(mean,a[0].tanh())


def test_ising_pair_sign_no_double_counting_and_stable_large_fields():
    x,y=data();m=SupervisedIsing(x,y,spins=2).double()
    m.coupling.data.fill_(.7)
    torch.testing.assert_close(m.pair_states@m.coupling,torch.tensor([.7,-.7,-.7,.7],dtype=torch.float64))
    p=(m.pair_states@m.coupling).softmax(0)
    assert (p*m.pair_states[:,0]).sum()>0
    a=torch.tensor([[1000.,-1000.]],dtype=torch.float64,requires_grad=True)
    out=m.hidden_mean(a);out.sum().backward()
    assert torch.isfinite(out).all() and torch.isfinite(a.grad).all()
    torch.testing.assert_close(out,torch.tensor([[1.,-1.]],dtype=torch.float64))


def test_matched_initialization_parameter_count_and_gradient():
    x,y=data(1279)
    torch.manual_seed(42);m=SupervisedIsing(x,y,spins=12)
    torch.manual_seed(42);control=SupervisedIsing(x,y,spins=12,coupled=False)
    assert sum(p.numel() for p in m.parameters())==16718
    assert m.states.shape==(4096,12) and m.coupling.numel()==66
    for name in ['beta','intercept','readout']:
        torch.testing.assert_close(getattr(m,name),getattr(control,name),atol=0,rtol=0)
    torch.testing.assert_close(m.field.weight,control.field.weight,atol=0,rtol=0)
    sample=torch.tensor(x[:3]);prediction=m(sample)
    torch.testing.assert_close(prediction,control(sample),atol=1e-6,rtol=1e-5)
    ((prediction[:,0,:]-m.standardize_y(torch.tensor(y[:3,None])))**2).mean().backward()
    for name,p in m.named_parameters():assert p.grad is not None and torch.isfinite(p.grad).all(),name
    assert m.coupling.grad.norm()>0 and m.field.weight.grad.norm()>0
    assert set(inspect.signature(m.forward).parameters)=={'x'}


@pytest.mark.parametrize('kind',['ising','ising_independent','ising_mlp'])
def test_fit_boundaries_and_training_loop(kind):
    x,y=data()
    cls=AdditiveMLP if kind=='ising_mlp' else SupervisedIsing
    options={'width':8} if kind=='ising_mlp' else {'spins':4,'coupled':kind=='ising'}
    m=cls(x,y,**options);m.eval();a=torch.tensor(x[:3]);b=a.clone();b[:,0]=100
    buffers={n:t.clone() for n,t in m.named_buffers()}
    torch.testing.assert_close(m(a),m(b))
    torch.testing.assert_close(m(a[:1]),m(a)[:1],atol=1e-6,rtol=1e-5)
    for n,t in m.named_buffers():torch.testing.assert_close(t,buffers[n])
    cfg=dict(spins=4,width=8,learning_rate=.001,weight_decay=.01)
    settings=dict(optimizer_variant=kind,n_quantiles=1,quantile_min=.001,quantile_max=.999,
                  training=dict(objective='mse',max_epochs=2,patience=1,batch_size=8,gradient_clip=5,tie_tolerance=1e-12))
    first,info=train(x[:16],y[:16],x[16:],y[16:],cfg,42,0,settings,'cpu')
    torch.randn(20)
    second,_=train(x[:16],y[:16],x[16:],y[16:],cfg,42,0,settings,'cpu')
    np.testing.assert_array_equal(first,second)
    assert first.shape==(8,) and np.isfinite(first).all() and 0<=info['best_epoch']<=2
