"""Synthetic in-context regression tasks from fit-only marker rows."""
import numpy as np


def task(x_fit, rng, *, context=128, query=32):
    x=np.asarray(x_fit,dtype=np.float32)
    if x.ndim!=2 or context<2 or query<1 or len(x)<context+query:
        raise ValueError('Not enough fit-only materials for a task')
    ids=rng.choice(len(x),size=context+query,replace=False)
    markers=x[ids];mu=markers[:context].mean(0);sd=markers[:context].std(0)
    z=(markers-mu)/np.where(sd>0,sd,1.)
    p=z.shape[1];kind=int(rng.integers(3));beta=rng.normal(size=p)
    if kind==1:
        mask=np.zeros(p);mask[rng.choice(p,size=min(p,int(rng.integers(4,33))),replace=False)]=1
        beta*=mask
    signal=z@beta/np.sqrt(max(np.count_nonzero(beta),1))
    if kind==2:
        a,b=rng.choice(p,size=2,replace=False)
        signal=signal+rng.uniform(0.,1.)*z[:,a]*z[:,b]
    signal_std=np.std(signal[:context])
    if signal_std<=1e-8:raise ValueError('Constant synthetic signal')
    noise=rng.uniform(.2,1.2)*signal_std
    labels=signal+rng.normal(scale=noise,size=len(ids))
    labels=(labels-labels[:context].mean())/max(labels[:context].std(),1e-8)
    return markers.astype(np.float32),labels.astype(np.float32),ids
