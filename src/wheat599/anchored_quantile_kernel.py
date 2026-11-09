"""Certified convex position updates for anchored neural quantile curves."""
import torch


def solve(residuals, taus, material, environment, regularization, *, tolerance=1e-6, max_iterations=10000, rho=.1):
    """Minimize averaged per-member pinball plus lambda/2 times RKHS norm.

    residuals[e] has flattened member/quantile columns; tau repeats per member.
    Intercept is fixed at its initial value. Only the RKHS correction is learned.
    """
    device=material.device; dtype=torch.float64
    material=material.to(dtype);environment=environment.to(dtype)
    residuals=[r.detach().to(device=device,dtype=dtype) for r in residuals]
    taus=taus.to(device=device,dtype=dtype)
    if regularization<=0 or rho<=0:raise ValueError('Positive regularization and ADMM penalty required')
    n=material.shape[0];tasks=environment.shape[0];q=len(taus)
    if len(residuals)!=tasks or any(r.shape[0]!=n or r.shape[1]%q for r in residuals):raise ValueError('Residual dimensions do not match kernels')
    for k in (material,environment):
        if not torch.isfinite(k).all() or not torch.allclose(k,k.T,atol=1e-10,rtol=1e-10):raise ValueError('Invalid symmetric kernel')
    ek,uk=torch.linalg.eigh(material);eb,ub=torch.linalg.eigh(environment)
    if min(float(ek.min()),float(eb.min())) < -1e-9:raise ValueError('Kernel is not PSD')
    eig=ek.clamp_min(0)[:,None]*eb.clamp_min(0)[None,:]
    positive=eig>1e-12; eig=torch.where(positive,eig,torch.zeros_like(eig)); shrink=rho*eig/(regularization+rho*eig)
    def operator(v):return uk@((uk.T@v@ub)*eig)@ub.T
    def quadratic(v):
        rotated=uk.T@v@ub
        return ((rotated[positive]**2)/eig[positive]).sum()
    sorted_r=[r.sort(dim=1).values.contiguous() for r in residuals]
    coefficients=[q/(n*tasks*r.shape[1]) for r in residuals]
    tau_sum=[taus.sum()*(r.shape[1]//q) for r in residuals]
    def value(v):
        total=torch.zeros((),device=device,dtype=dtype)
        for e,r in enumerate(residuals):
            error=(r-v[:,e,None]).reshape(n,-1,q)
            total+=torch.maximum(taus*error,(taus-1)*error).sum()*coefficients[e]
        return total
    def conjugate(s):
        total=torch.zeros((),device=device,dtype=dtype)
        for e,r in enumerate(residuals):
            rank=torch.ceil(s[:,e]/coefficients[e]+tau_sum[e]).long().clamp(1,r.shape[1])-1
            v=sorted_r[e].gather(1,rank[:,None])[:,0]
            error=(r-v[:,None]).reshape(n,-1,q)
            f=torch.maximum(taus*error,(taus-1)*error).sum((1,2))*coefficients[e]
            total+=(s[:,e]*v-f).sum()
        return total
    z=torch.zeros((n,tasks),device=device,dtype=dtype);u=torch.zeros_like(z)
    for iteration in range(1,max_iterations+1):
        v=torch.empty_like(z)
        for e,r in enumerate(sorted_r):
            count=r.shape[1];a=coefficients[e]/rho;center=z[:,e]-u[:,e]
            low=torch.zeros(n,device=device,dtype=torch.long);high=torch.full_like(low,count)
            for _ in range(count.bit_length()+1):
                mid=(low+high)//2
                threshold=r.gather(1,mid.clamp_max(count-1)[:,None])[:,0]
                before=(low<high)&(mid<count)&(threshold<center+a*(tau_sum[e]-mid))
                low=torch.where(before,mid+1,low);high=torch.where(before,high,mid)
            k=low;free=center+a*(tau_sum[e]-k)
            below=r.gather(1,(k-1).clamp_min(0)[:,None])[:,0]
            above=r.gather(1,k.clamp_max(count-1)[:,None])[:,0]
            v[:,e]=torch.where(k>0,torch.maximum(free,below),free)
            v[:,e]=torch.where(k<count,torch.minimum(v[:,e],above),v[:,e])
        old=z
        z=uk@((uk.T@(v+u)@ub)*shrink)@ub.T
        u=u+v-z
        if iteration%10 and iteration!=max_iterations:continue
        s=-rho*u
        for e,r in enumerate(residuals):
            s[:,e]=s[:,e].clamp(float(-coefficients[e]*tau_sum[e]),float(coefficients[e]*(r.shape[1]-tau_sum[e])))
        primal=value(z)+regularization*.5*quadratic(z)
        dual=-conjugate(s)-(s*operator(s)).sum()/(2*regularization)
        gap=float((primal-dual)/max(1.,abs(float(primal)),abs(float(dual))))
        diagnostics=dict(iterations=iteration,primal=float(primal),dual=float(dual),relative_gap=gap,
                         primal_residual=float((v-z).abs().max()),dual_residual=float((rho*(z-old)).abs().max()))
        if not torch.isfinite(primal+dual):raise FloatingPointError('Nonfinite kernel certificate')
        if -1e-10<=gap<=tolerance:
            # Return whitened coefficients: prediction uses k(query,fit) times G pseudoinverse z.
            rotated=uk.T@z@ub
            coefficients_matrix=uk@torch.where(positive,rotated/eig.clamp_min(1e-12),torch.zeros_like(rotated))@ub.T
            return z,coefficients_matrix,diagnostics
    raise RuntimeError(f'Kernel optimization did not converge: {diagnostics}')
