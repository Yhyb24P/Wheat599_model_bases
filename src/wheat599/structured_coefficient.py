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
