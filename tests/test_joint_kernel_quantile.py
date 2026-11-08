import numpy as np
import pytest
from scipy.optimize import Bounds, LinearConstraint, minimize
import torch

from wheat599.joint_kernel_quantile import kernel_product, point_coefficients, predict, project_box_zero_sum, solve


def test_projection_enforces_box_and_per_output_sum():
    torch.manual_seed(5)
    x=torch.randn(7,2,3,dtype=torch.float64);lo=torch.tensor([-.9,-.5,-.1]);hi=lo+1
    z=project_box_zero_sum(x,lo,hi)
    assert float(z.sum(0).abs().max())<1e-12
    assert torch.all(z>=lo) and torch.all(z<=hi)


def test_separable_product_matches_explicit_kronecker():
    rng=np.random.default_rng(1);n,e,q=4,2,3
    factors=[rng.normal(size=(d,d)) for d in (n,e,q)]
    k,b,c=[x@x.T for x in factors];beta=rng.normal(size=(n,e,q))
    actual=kernel_product(*[torch.tensor(x) for x in (beta,k,b,c)]).numpy()
    np.testing.assert_allclose(actual.ravel(),np.kron(np.kron(k,b),c)@beta.ravel(),atol=1e-12)


@pytest.mark.parametrize('state',['independent','smooth','shared'])
def test_dual_matches_independent_primal_qp(state):
    rng=np.random.default_rng(91);n,e,q=5,2,3;ridge=.08
    def factor(d):
        a=rng.normal(size=(d,d));return a@a.T+np.eye(d)*.2
    k,b=factor(n),factor(e);c={'shared':np.ones((q,q)),'smooth':factor(q),'independent':np.eye(q)}[state]
    y=rng.normal(size=(n,e));tau=np.array([.23,.51,.77]);size=n*e*q;outputs=e*q
    fit=solve(k,b,c,y,tau,ridge,tolerance=2e-6,max_iterations=30000,check_every=100)
    def root(matrix):
        v,u=np.linalg.eigh(matrix);return (u*np.sqrt(np.maximum(v,0)))@u.T
    design=np.kron(np.kron(root(k),root(b)),root(c))
    intercept=np.tile(np.eye(outputs),(n,1));identity=np.eye(size)
    constraint=np.hstack([design,intercept,identity,-identity]);target=np.repeat(y[:,:,None],q,axis=2).ravel()
    weights=np.tile(tau,n*e);normalization=1/size
    def objective(v):return ridge/2*np.dot(v[:size],v[:size])+normalization*(weights@v[size+outputs:size+outputs+size]+(1-weights)@v[-size:])
    def gradient(v):return np.r_[ridge*v[:size],np.zeros(outputs),normalization*weights,normalization*(1-weights)]
    lower=np.r_[np.full(size+outputs,-np.inf),np.zeros(2*size)]
    start=np.r_[np.zeros(size+outputs),np.maximum(target,0),np.maximum(-target,0)]
    ref=minimize(objective,start,jac=gradient,method='SLSQP',bounds=Bounds(lower,np.inf),constraints=LinearConstraint(constraint,target,target),options={'ftol':1e-11,'maxiter':2000})
    assert ref.success,ref.message
    assert abs(ref.fun-fit['diagnostics']['primal'])<3e-6
    np.testing.assert_allclose(constraint@ref.x,target,atol=1e-8)
    pred=predict(k,b,c,fit)
    assert pred.shape==(n,e,q) and np.isfinite(pred).all()
    coef,bias=point_coefficients(b,c,fit)
    np.testing.assert_allclose(k@coef+bias,pred.mean(axis=2),atol=1e-12)
    reference_prediction=(design@ref.x[:size]+intercept@ref.x[size:size+outputs]).reshape(n,e,q)
    np.testing.assert_allclose(pred,reference_prediction,atol=5e-3,rtol=0)
    if state=='shared':np.testing.assert_allclose(pred[:,:,0]-pred[:,:,1],np.broadcast_to(pred[0,:,0]-pred[0,:,1],(n,e)),atol=1e-10)


def test_solver_rejects_unconverged():
    with pytest.raises(RuntimeError,match='did not converge'):
        solve(np.diag([1.,2.,3.]),np.eye(1),np.eye(2),np.array([[1.],[2.],[4.]]),np.array([.25,.75]),.1,tolerance=1e-15,max_iterations=1)


def test_candidate_grid_preserves_common_eta_and_endpoint_kernels():
    from wheat599.joint_kernel_quantile_run import candidates
    levels=np.linspace(.001,.999,5);cfg={'gamma':16.,'etas':[0.,.3,.6,.8],'ridge_shared':[1e-5,1e-4,1e-3]}
    choices=candidates(cfg,levels)
    assert len(choices)==36
    for state in ('shared','smooth','independent'):
        rows=[v for v in choices if v[0]==state]
        assert len(rows)==12 and {r[1] for r in rows}==set(cfg['etas'])
    np.testing.assert_array_equal(choices[0][3],np.ones((5,5)))
    np.testing.assert_array_equal(choices[-1][3],np.eye(5))
