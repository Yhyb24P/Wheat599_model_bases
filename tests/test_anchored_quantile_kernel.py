import numpy as np
import pytest
import torch
from scipy.optimize import minimize,LinearConstraint,Bounds
from wheat599.anchored_quantile_kernel import solve


def reference(residuals,taus,k,b,lam):
    n=len(k);tasks=len(b);q=len(taus);gram=np.kron(k,b);inverse=np.linalg.inv(gram)
    thresholds=np.concatenate([r.ravel() for r in residuals]);indices=np.concatenate([np.repeat(np.arange(n)*tasks+e,r.shape[1]) for e,r in enumerate(residuals)])
    counts=len(thresholds);coeff=np.concatenate([np.full(r.size,q/(n*tasks*r.shape[1])) for r in residuals]);t=np.concatenate([np.tile(taus,r.size//q) for r in residuals])
    def fun(z):
        g=z[:n*tasks];err=thresholds-g[indices]
        value=(coeff*(z[n*tasks:]-(1-t)*err)).sum()+lam*.5*g@inverse@g
        grad=np.r_[np.bincount(indices,weights=coeff*(1-t),minlength=n*tasks)+lam*inverse@g,coeff]
        return value,grad
    a=np.zeros((counts,n*tasks+counts));a[np.arange(counts),indices]=1;a[:,n*tasks:]=np.eye(counts)
    initial=np.r_[np.zeros(n*tasks),np.maximum(thresholds,0)]
    result=minimize(fun,initial,jac=True,method='SLSQP',constraints=[LinearConstraint(a,thresholds,np.inf)],bounds=Bounds(np.r_[np.full(n*tasks,-np.inf),np.zeros(counts)],np.inf),options={'ftol':1e-11,'maxiter':2000})
    assert result.success,result.message
    return result.fun,result.x[:n*tasks].reshape(n,tasks)


@pytest.mark.parametrize('lam',[.1,1.])
def test_original_pinball_matches_generic_convex_optimizer(lam):
    torch.set_num_threads(1)
    rng=np.random.default_rng(7);a=rng.normal(size=(4,4));k=a@a.T/4+.1*np.eye(4);b=np.array([[1.,.3],[.3,1.]])
    residuals=[rng.normal(size=(4,6)),rng.normal(size=(4,3))];taus=np.array([.1,.5,.9])
    truth,g=reference(residuals,taus,k,b,lam)
    pos,coef,info=solve([torch.tensor(r) for r in residuals],torch.tensor(taus),torch.tensor(k),torch.tensor(b),lam,tolerance=1e-8)
    assert abs(info['primal']-truth)<2e-7
    assert info['relative_gap']<=1e-8
    np.testing.assert_allclose(pos.numpy(),g,atol=2e-5)
    torch.testing.assert_close(pos,torch.tensor(k)@coef@torch.tensor(b))


def test_singular_kernel_and_reject_uncertified_solution():
    residuals=[torch.tensor([[1.,2.,3.],[-1.,0.,1.]],dtype=torch.float64)]
    k=torch.ones((2,2),dtype=torch.float64);b=torch.ones((1,1),dtype=torch.float64);q=torch.tensor([.1,.5,.9],dtype=torch.float64)
    pos,coef,info=solve(residuals,q,k,b,.5,tolerance=1e-7)
    assert info['relative_gap']<=1e-7
    torch.testing.assert_close(pos,k@coef@b)
    with pytest.raises(RuntimeError,match='did not converge'):
        solve(residuals,q,k,b,.5,max_iterations=1,tolerance=1e-14)


def test_joint_initialization_sync_and_reload():
    from pathlib import Path
    import yaml
    from wheat599 import quantile_tabm as qt
    from wheat599.anchored_quantile import train
    from wheat599.robust_icm import features,fit_predict
    rng=np.random.default_rng(71);x=rng.normal(size=(14,8));y=rng.normal(size=(14,4))
    config=dict(n_blocks=1,d_block=8,dropout=.3,k=2,learning_rate=.001,weight_decay=.01)
    settings=dict(n_quantiles=99,quantile_min=.001,quantile_max=.999,network_configs=[config]*4,updates=1,
                  gradient_clip=5.,learning_rate_multiplier=.1,solver=dict(tolerance=1e-6,max_iterations=10000,rho=.1))
    models=[qt.build_model(x[:10],y[:10,e],config,42,settings,'cpu') for e in range(4)]
    for m in models:m.eval()
    with torch.no_grad():original=np.column_stack([m.inverse_y(m(torch.tensor(x[10:],dtype=torch.float32)).mean((1,2))[:,None])[:,0].numpy() for m in models])
    base=dict(bandwidth=1.,fraction=.5,ridge=[1.]*4)
    prepared=features(x,y,np.arange(10),np.arange(10,14),base)
    cfg=yaml.safe_load(Path('configs/robust_icm.yaml').read_text())
    hp,_,state=fit_predict(prepared,1.,.3,1.345,cfg,return_state=True)
    default,_=fit_predict(prepared,1.,.3,1.345,cfg)
    np.testing.assert_array_equal(hp,default)
    saved=[]
    result=train(models,x[:10],y[:10],x[10:],prepared,state,.3,dict(lambda_kernel=1.,lambda_neural=.01),settings,lambda *v:saved.append(v))
    np.testing.assert_allclose(saved[0][1],.5*(original+hp),atol=3e-7,rtol=0)
    assert saved[0][3] is None and saved[1][3]['relative_gap']<=1e-6
    for m in result:assert not m.training
    final=saved[1][2];coef=final['kernel_delta_coefficients'];env=final['kernel_environment']
    cross=prepared['target_features'];design=prepared['train_features']
    scales=np.array([float(m.y_std) for m in result]);means=np.array([float(m.y_mean) for m in result])
    anchor=.5*(hp-means)/scales
    correction=torch.tensor(cross@design.T)@coef@env
    with torch.no_grad():
        replay=[]
        for e,m in enumerate(result):
            m.load_state_dict(final['neural'][e]);a=m(torch.tensor(x[10:],dtype=torch.float32)).mean((1,2)).double()
            replay.append(m.inverse_y((a+torch.tensor(anchor[:,e])+correction[:,e])[:,None])[:,0].numpy())
    np.testing.assert_allclose(np.column_stack(replay),saved[1][1],atol=1e-12,rtol=0)


def test_selection_uses_one_configuration_and_step_across_all_inner_units(tmp_path):
    from types import SimpleNamespace
    from wheat599.anchored_quantile_run import summarize
    y=np.arange(24).reshape(6,4).astype(float)
    folds=[dict(fold=0,inner=[dict(val_idx=list(range(6))) for _ in range(3)])]
    for j in range(3):
        unit=tmp_path/f'unit_0_{j}';unit.mkdir();arrays={}
        for seed in (42,2026):
            arrays[f'original_s{seed}']=-y
            for c in range(3):
                for t in range(3):arrays[f'c{c}_s{seed}_t{t}']=y if (c,t)==(1,1) or ((c,t)==(0,2) and j==0) else -y
        np.savez(unit/'predictions.npz',**arrays)
    result=summarize(SimpleNamespace(Y=y),folds,dict(updates=2,seeds=[42,2026],training=dict(tie_tolerance=1e-12)),tmp_path)
    chosen=result['folds'][0]['selected'];assert chosen['config_index']==1 and chosen['step']==1
    assert result['macro']['joint']['pcc']==pytest.approx(1.)
