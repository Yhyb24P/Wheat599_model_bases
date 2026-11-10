"""RBF expansion with a binary-marker structured coefficient precision."""
import numpy as np
import torch
from scipy.linalg import solve_triangular
from wheat599.kernel_baselines import kernels


def expansion(x_fit, query, bandwidth, epsilon):
    if epsilon<=0 or not np.isfinite(epsilon):raise ValueError('Positive finite precision offset required')
    if not np.isin(x_fit,[0,1]).all() or not np.isin(query,[0,1]).all():raise ValueError('Expected binary DArT markers')
    n=len(x_fit);joined=np.concatenate((x_fit,query))
    g,_,r,rc=kernels(joined,np.arange(n),np.arange(n,len(joined)),bandwidth,'cpu')
    g,r,rc=g.numpy(),r.numpy(),rc.numpy()
    precision=g+epsilon*np.eye(n)
    lower=np.linalg.cholesky(precision)
    # P=L L^T, so D=R L^{-T} and A=L^{-T} theta.
    design=solve_triangular(lower,r.T,lower=True).T
    cross=solve_triangular(lower,rc.T,lower=True).T
    return design,cross,dict(precision=precision,rbf=r,query_rbf=rc,lower=lower,
        min_precision_eigenvalue=float(np.linalg.eigvalsh(precision)[0]),effective_mean_diagonal=float(np.mean(np.sum(design**2,axis=1))))


def spectrum(design,cross, *, center):
    mean=design.mean(0) if center else np.zeros(design.shape[1])
    d=design-mean;dc=cross-mean
    u,root,vt=np.linalg.svd(d,full_matrices=False)
    keep=root>root[0]*1e-12
    values=np.where(keep,root**2,0.)
    return dict(eigenvalues=torch.tensor(values,dtype=torch.float64),vectors=torch.tensor(u,dtype=torch.float64),
                keep=torch.tensor(keep),root=root[keep],train_features=u[:,keep]*root[keep],
                target_features=dc@vt[keep].T,rotation=vt[keep].T,feature_mean=mean)


def prepared_features(design,cross,y):
    prepared=spectrum(design,cross,center=True)
    mean=y.mean(0);scale=y.std(0)
    if np.any(scale<=0) or not np.isfinite(y).all():raise ValueError('Invalid fit phenotypes')
    z=(y-mean)/scale;norm=z/np.linalg.norm(z,axis=0)
    prepared.update(Y=torch.tensor(z,dtype=torch.float64),mean=mean,scale=scale,C_Y=norm.T@norm)
    return prepared


def unweighted_posterior(design,cross,y,ridge):
    """Author K_Model posterior with supplied P, sigma_e2/sigma_a2 absorbed in ridge.

    Fit-label centering/scaling is supplied by caller; no learned intercept here.
    """
    if ridge<=0:raise ValueError('Positive effective ridge required')
    spec=spectrum(design,cross,center=False)
    return posterior_from_spectrum(spec,y,ridge)[0]


def posterior_from_spectrum(spec,y,ridge):
    u=spec['vectors'].numpy()[:,spec['keep'].numpy()];s=spec['root']
    w=(s/(s*s+ridge))[:,None]*(u.T@y)
    return spec['target_features']@w,spec['rotation']@w


def huber_certificate(prepared,state,ridge,thresholds):
    """Fenchel lower bound with a box/zero-sum-feasible Huber dual score."""
    y=prepared['Y'].numpy();d=prepared['train_features'];w=state['weights'];factor=state['factor'];thresholds=np.asarray(thresholds)
    residual=d@w@factor.T+state['intercept']-y
    raw=np.clip(residual,-thresholds,thresholds)
    low=(raw-thresholds).min(0);high=(raw+thresholds).max(0)
    for _ in range(60):
        mid=(low+high)/2;score=np.clip(raw-mid,-thresholds,thresholds)
        positive=score.sum(0)>0;low=np.where(positive,mid,low);high=np.where(positive,high,mid)
    score=np.clip(raw-(low+high)/2,-thresholds,thresholds)
    primal=(np.where(abs(residual)<=thresholds,.5*residual**2,thresholds*(abs(residual)-.5*thresholds)).sum()+.5*ridge*np.sum(w*w))/y.size
    moment=d.T@score@factor
    dual=(-(score*y).sum()-.5*np.sum(score**2)-.5*np.sum(moment**2)/ridge)/y.size
    return dict(primal=float(primal),dual=float(dual),relative_gap=float((primal-dual)/max(1e-12,abs(primal),abs(dual))),
                dual_zero_sum_residual=float(abs(score.sum(0)).max()/y.size),
                dual_box_violation=float(np.maximum(abs(score)-thresholds,0).max()/y.size))


def huber_hessian_product(values,direction,design,factor,targets,ridge,thresholds,precondition,intercept_scale):
    tasks=targets.shape[1];shape=(design.shape[1],tasks)
    weights=values[:-tasks].reshape(shape)/precondition;bias=values[-tasks:]/intercept_scale
    dw=direction[:-tasks].reshape(shape)/precondition;db=direction[-tasks:]/intercept_scale
    residual=design@weights@factor.T+bias-targets
    tangent=(abs(residual)<=thresholds)*(design@dw@factor.T+db)
    return np.r_[((design.T@tangent@factor+ridge*dw)/precondition).ravel(),tangent.sum(0)/intercept_scale]/targets.size


def fit_certified(prepared,ridge,eta,delta,cfg,gap_tolerance):
    from scipy.optimize import minimize
    from wheat599.robust_icm import fit_predict,huber_objective
    # The first solve supplies an initial point; only the final joint certificate accepts a fit.
    pred,diag,state=fit_predict(prepared,ridge,eta,delta,dict(cfg,accepted_gradient=np.inf),return_state=True)
    initial_gradient=diag['gradient_inf']
    certificate=huber_certificate(prepared,state,ridge,diag['thresholds'])
    if certificate['relative_gap']>gap_tolerance or initial_gradient>cfg['accepted_gradient']:
        d=prepared['train_features'];y=prepared['Y'].numpy();factor=state['factor'];tasks=y.shape[1]
        precondition=np.sqrt(prepared['root'][:,None]**2*np.sum(factor**2,axis=0)[None,:]+ridge);intercept_scale=np.sqrt(len(y))
        values=np.r_[(state['weights']*precondition).ravel(),state['intercept']*intercept_scale]
        args=(d,factor,y,ridge,np.asarray(diag['thresholds']),precondition,intercept_scale)
        result=minimize(huber_objective,values,args=args,jac=True,hessp=huber_hessian_product,method='trust-ncg',options=dict(gtol=1e-14,maxiter=cfg['max_iterations']))
        state=dict(weights=result.x[:-tasks].reshape(state['weights'].shape)/precondition,intercept=result.x[-tasks:]/intercept_scale,factor=factor)
        pred=(prepared['target_features']@state['weights']@factor.T+state['intercept'])*prepared['scale']+prepared['mean']
        raw=np.r_[result.jac[:-tasks]*precondition.ravel(),result.jac[-tasks:]*intercept_scale]
        diag.update(gradient_inf=float(abs(raw).max()),iterations=diag['iterations']+int(result.nit),refinement_solver='trust-ncg',refinement_success=bool(result.success),refinement_message=str(result.message),initial_gradient_inf=initial_gradient)
        certificate=huber_certificate(prepared,state,ridge,diag['thresholds'])
    if not np.isfinite(pred).all() or not(-1e-10<=certificate['relative_gap']<=gap_tolerance) or diag['gradient_inf']>cfg['accepted_gradient'] or certificate['dual_zero_sum_residual']>1e-12 or certificate['dual_box_violation']>1e-12:
        raise ValueError(f'Uncertified structured Huber optimization: {certificate}; gradient={diag["gradient_inf"]}')
    diag.update(certificate,number_observations=prepared['Y'].numel(),ridge=ridge,normalized_lambda=ridge/prepared['Y'].numel())
    return pred,diag,state
