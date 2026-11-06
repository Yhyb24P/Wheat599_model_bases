import numpy as np
from wheat599.genetic_prior import task


def test_task_uses_only_supplied_rows_and_disjoint_context_query():
    rng=np.random.default_rng(1);x=rng.integers(0,2,size=(200,50)).astype(np.float32)
    a,y,ids=task(x,np.random.default_rng(3),context=100,query=30)
    assert a.shape==(130,50) and y.shape==(130,) and len(set(ids))==130
    np.testing.assert_array_equal(a,x[ids]);assert np.isfinite(y).all()
    assert abs(y[:100].mean())<1e-6 and abs(y[:100].std()-1)<1e-6
    unused=np.setdiff1d(np.arange(len(x)),ids)
    altered=x.copy();altered[unused]=1-altered[unused]
    b,z,ids2=task(altered,np.random.default_rng(3),context=100,query=30)
    np.testing.assert_array_equal(ids,ids2)
    np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(y,z)
