from pathlib import Path
import numpy as np
import pytest
from scipy.optimize import minimize
from wheat599.structured_coefficient import expansion,prepared_features,unweighted_posterior,huber_certificate
from wheat599.robust_icm import fit_predict


def test_white_features_equal_structured_posterior_and_effective_kernel():
    rng=np.random.default_rng(4);x=rng.integers(0,2,(13,9));query=rng.integers(0,2,(5,9));y=rng.normal(size=(13,4));ridge=.7
    d,dc,meta=expansion(x,query,1.,.01);p=meta['precision'];r=meta['rbf'];rc=meta['query_rbf']
    np.testing.assert_allclose(d@d.T,r@np.linalg.solve(p,r.T),atol=1e-10)
    coefficient=np.linalg.solve(r.T@r+ridge*p,r.T@y)
    np.testing.assert_allclose(unweighted_posterior(d,dc,y,ridge),rc@coefficient,atol=1e-10)
    theta=rng.normal(size=(13,4));a=np.linalg.solve(meta['lower'].T,theta)
    np.testing.assert_allclose(np.trace(a.T@p@a),np.sum(theta**2),atol=1e-10)


def test_structured_multienvironment_huber_matches_direct_primal():
    import yaml
    rng=np.random.default_rng(19);x=rng.integers(0,2,(10,8));query=rng.integers(0,2,(3,8));y=rng.normal(size=(10,4));d,dc,meta=expansion(x,query,.3,.1)
    prepared=prepared_features(d,dc,y);cfg=yaml.safe_load(Path('configs/robust_icm.yaml').read_text())
    ridge=2.;eta=.6;delta=.8
    pred,diag,state=fit_predict(prepared,ridge,eta,delta,cfg,fixed_thresholds=np.full(4,delta),return_state=True)
    b=(1-eta)*np.eye(4)+eta*prepared['C_Y'];invb=np.linalg.inv(b);p=meta['precision'];r=meta['rbf'];target=prepared['Y'].numpy()
    def objective(v):
        a=v[:-4].reshape(10,4);bias=v[-4:];res=r@a+bias-target;score=np.clip(res,-delta,delta)
        loss=np.where(abs(res)<=delta,.5*res**2,delta*(abs(res)-.5*delta)).sum()+.5*ridge*np.trace(invb@a.T@p@a)
        grad=np.r_[(r.T@score+ridge*p@a@invb).ravel(),score.sum(0)]
        return loss/target.size,grad/target.size
    result=minimize(objective,np.zeros(44),jac=True,method='BFGS',options={'gtol':1e-9,'maxiter':2000})
    assert np.linalg.norm(result.jac,np.inf)<1e-7
    direct=(meta['query_rbf']@result.x[:-4].reshape(10,4)+result.x[-4:])*prepared['scale']+prepared['mean']
    np.testing.assert_allclose(pred,direct,atol=3e-5,rtol=0)
    assert diag['gradient_inf']<=cfg['accepted_gradient']
    cert=huber_certificate(prepared,state,ridge,np.full(4,delta))
    assert -1e-10<=cert['relative_gap']<1e-6
    assert cert['dual_zero_sum_residual']<1e-12 and cert['dual_box_violation']==0.


def test_query_basis_uses_fit_only_and_singular_relationship_is_regularized():
    x=np.array([[0,1],[1,0],[0,1],[1,0]],dtype=float);q=np.array([[1,1],[0,0]],dtype=float)
    d,dc,meta=expansion(x,q,1.,.001)
    assert meta['min_precision_eigenvalue']>=.001-1e-12
    _,sub,_=expansion(x,q[:1],1.,.001)
    np.testing.assert_allclose(dc[:1],sub,atol=1e-12)
    with pytest.raises(ValueError):expansion(x,q,1.,0.)
    with pytest.raises(ValueError):expansion(x+2,q,1.,.1)


def test_development_unit_saves_finite_predictions_and_certificates(tmp_path):
    from wheat599.structured_coefficient_run import run_unit
    from wheat599.distill_quantile import read
    import yaml
    cfg=yaml.safe_load(Path('configs/structured_coefficient.yaml').read_text())
    cfg.update(bandwidths=[.25],epsilons=[.01],lambdas=[.001,.01],etas=[0.,.6],deltas=[1.345],reference_ridges=[.1,1.])
    rng=np.random.default_rng(52);x=rng.integers(0,2,(15,12));q=rng.integers(0,2,(5,12));y=rng.normal(size=(15,4))
    result=run_unit((x,y,q,0,0,cfg,str(tmp_path)))
    assert result['candidate_fits']==4 and result['reference_fits']==2
    root=tmp_path/'unit_0_0';assert read(root/'execution_status.json')['status']=='complete'
    with np.load(root/'predictions.npz') as predictions:
        assert len(predictions.files)==6
        assert all(predictions[k].shape==(5,4) and np.isfinite(predictions[k]).all() for k in predictions.files)
    for record in read(root/'optimization.json'):
        assert record['relative_gap']<=cfg['relative_gap_tolerance']
        assert record['dual_zero_sum_residual']<=1e-12 and record['dual_box_violation']<=1e-12
