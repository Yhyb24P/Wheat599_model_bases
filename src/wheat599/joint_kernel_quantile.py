"""Separable multi-environment kernel quantile regression, exact pinball dual."""
import numpy as np
import torch


def project_box_zero_sum(values, lower, upper, iterations=48):
    """Euclidean projection onto per-output box constraints and zero sums."""
    low=(values-upper).amin(dim=0,keepdim=True)
    high=(values-lower).amax(dim=0,keepdim=True)
    for _ in range(iterations):
        mid=(low+high)/2
        projected=torch.maximum(torch.minimum(values-mid,upper),lower)
        positive=projected.sum(dim=0,keepdim=True)>0
        low=torch.where(positive,mid,low)
        high=torch.where(positive,high,mid)
    return torch.maximum(torch.minimum(values-(low+high)/2,upper),lower)


def kernel_product(coefficients, material, environment, quantile):
    return torch.einsum('ab,ibq,qj->iaj',environment,
                        torch.einsum('ij,jeq->ieq',material,coefficients),quantile)


def solve(material, environment, quantile, y, levels, ridge, *, tolerance=1e-6,
          max_iterations=10000, check_every=50, device='cpu'):
    """Solve original nonsmooth objective; return only on certified dual gap."""
    device=torch.device(device)
    tensors=[torch.as_tensor(x,dtype=torch.float64,device=device) for x in (material,environment,quantile,y,levels)]
    k,b,c,y,tau=tensors;n,e=y.shape;q=tau.numel()
    if k.shape!=(n,n) or b.shape!=(e,e) or c.shape!=(q,q) or tau.ndim!=1:
        raise ValueError('Incompatible shapes')
    if any(not torch.isfinite(t).all() for t in tensors) or not 0<ridge or not 0<tolerance or max_iterations<1:
        raise ValueError('Invalid numerical inputs')
    if torch.any(tau<=0) or torch.any(tau>=1):raise ValueError('Quantile levels must be in (0,1)')
    eigenvalues=[]
    for matrix in (k,b,c):
        if not torch.allclose(matrix,matrix.T,rtol=1e-10,atol=1e-10):raise ValueError('Kernel must be symmetric')
        eigen=torch.linalg.eigvalsh(matrix)
        if eigen[0]<-1e-9*max(1.,float(eigen[-1])):raise ValueError('Kernel must be PSD')
        eigenvalues.append(float(eigen[-1]))
    norm=1.0/(n*e*q);scale=norm/ridge
    lipschitz=scale*np.prod(eigenvalues)
    if not np.isfinite(lipschitz) or lipschitz<=0:raise ValueError('Degenerate kernel')
    lo=(tau-1)[None,None,:];hi=tau[None,None,:]
    beta=torch.zeros((n,e,q),dtype=torch.float64,device=device);momentum=beta.clone();acceleration=1.
    diagnostics=None
    for iteration in range(1,max_iterations+1):
        gradient=y[:,:,None]-scale*kernel_product(momentum,k,b,c)
        new=project_box_zero_sum(momentum+gradient/lipschitz,lo,hi)
        next_acceleration=(1+np.sqrt(1+4*acceleration**2))/2
        momentum=new+((acceleration-1)/next_acceleration)*(new-beta)
        beta=new;acceleration=next_acceleration
        if iteration%check_every and iteration!=max_iterations:continue
        fit=scale*kernel_product(beta,k,b,c)
        residual=y[:,:,None]-fit
        # Any empirical tau-quantile minimizes the unpenalized intercept loss.
        sorted_residual=torch.sort(residual,dim=0).values
        positions=tau*n
        indices=torch.ceil(positions).long().clamp(1,n)-1
        intercept=sorted_residual.gather(0,indices[None,None,:].expand(1,e,q))[0]
        tied=(positions-positions.round()).abs()<1e-12
        upper_indices=(indices+1).clamp_max(n-1)
        upper=sorted_residual.gather(0,upper_indices[None,None,:].expand(1,e,q))[0]
        intercept=torch.where(tied[None,:],(intercept+upper)/2,intercept)
        errors=residual-intercept[None,:,:]
        loss=torch.maximum(tau*errors,(tau-1)*errors).mean()
        energy=norm*torch.sum(beta*fit)
        primal=loss+energy/2
        dual=norm*torch.sum(beta*y[:,:,None])-energy/2
        zero=float(torch.max(torch.abs(beta.sum(dim=0))))
        box=float(torch.maximum((lo-beta).clamp_min(0).max(),(beta-hi).clamp_min(0).max()))
        gap=float((primal-dual)/torch.maximum(torch.ones((),device=device,dtype=torch.float64),torch.maximum(primal.abs(),dual.abs())))
        diagnostics={'iteration':iteration,'primal':float(primal),'dual':float(dual),
                     'relative_gap':gap,'zero_sum_residual':zero*norm,'box_violation':box*norm}
        if zero*norm<=1e-10 and box*norm<=1e-10 and -1e-10<=gap<=tolerance:
            return {'beta':beta.cpu().numpy(),'intercept':intercept.cpu().numpy(),
                    'normalization':norm,'ridge':ridge,'diagnostics':diagnostics}
    raise RuntimeError(f'Joint quantile solve did not converge: {diagnostics}')


def predict(cross, environment, quantile, solution):
    beta=solution['beta'];cross=np.asarray(cross);environment=np.asarray(environment);quantile=np.asarray(quantile)
    return (solution['normalization']/solution['ridge'])*np.einsum('ti,ef,jq,ifq->tej',cross,environment,quantile,beta,optimize=True)+solution['intercept'][None,:,:]


def point_coefficients(environment, quantile, solution):
    """Precombine quantile coefficients for a single material-kernel readout."""
    beta=solution['beta'];q=beta.shape[2]
    weights=np.einsum('ef,q,ifq->ie',environment,np.asarray(quantile).sum(axis=0),beta,optimize=True)
    return (solution['normalization']/solution['ridge'])*weights/q,solution['intercept'].mean(axis=1)
