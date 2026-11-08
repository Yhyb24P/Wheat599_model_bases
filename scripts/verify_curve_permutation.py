"""Recompute the paired curve-control results from saved validation outputs."""
import json
import sys
from pathlib import Path

import numpy as np
import yaml

from wheat599.data import load


def check(root, run):
    cfg=yaml.safe_load((root/'configs/relation_quantile_curve_permutation.yaml').read_text())
    base=yaml.safe_load((root/cfg['original_config']).read_text())
    settings=yaml.safe_load((root/base['quantile_config']).read_text())
    data=load(root/settings['data'],settings['data_sha256'])
    folds=json.loads((root/settings['splits']).read_text())['folds']
    status=json.loads((run/'status.json').read_text());failures=json.loads((run/'failures.json').read_text())
    records=json.loads((run/'trajectories.json').read_text())
    previous=root/cfg['previous_controls']
    permutations=json.loads((run/'permutations.json').read_text())
    assert status=={'status':'complete','completed':120,'expected':120} and not failures
    assert permutations==json.loads((previous/'permutations.json').read_text())
    keys={(r['fold'],r['inner'],r['mode'],r['seed']) for r in records}
    modes=('curve','curve_shuffled');seeds=settings['seeds']
    assert len(records)==len(keys)==120
    assert keys=={(f['fold'],j,m,s) for f in folds for j in range(3) for m in modes for s in seeds}
    levels=np.linspace(settings['quantile_min'],settings['quantile_max'],settings['n_quantiles'])
    lo=int(np.argmin(abs(levels-.05)));hi=int(np.argmin(abs(levels-.95)))
    summary=json.loads((run/'summary.json').read_text());max_diff=0.;replay_diff=0.
    for fold in folds:
        f=fold['fold'];expected=next(x for x in summary['folds'] if x['fold']==f)
        metrics={m:[] for m in modes};diagnostics={m:[] for m in modes}
        for j,inner in enumerate(fold['inner']):
            truth=data.Y[inner['val_idx']]
            with np.load(run/f'predictions_{f}_{j}.npz') as pred, np.load(run/f'quantiles_{f}_{j}.npz') as quant, np.load(previous/f'predictions_{f}_{j}.npz') as old:
                for mode in modes:
                    ps=[];qs=[]
                    for seed in seeds:
                        p=pred[f'{mode}_s{seed}'];q=quant[f'{mode}_s{seed}']
                        assert p.shape==truth.shape and q.shape==(*truth.shape,len(levels))
                        assert np.isfinite(p).all() and np.isfinite(q).all()
                        np.testing.assert_allclose(q.mean(2),p,atol=1e-6,rtol=1e-6)
                        if mode=='curve':
                            replay_diff=max(replay_diff,float(np.max(np.abs(p-old[f'curve_s{seed}']))))
                            np.testing.assert_allclose(p,old[f'curve_s{seed}'],atol=1e-6,rtol=1e-6)
                        ps.append(p);qs.append(q)
                    p=np.mean(ps,axis=0);q=np.mean(qs,axis=0)
                    t64=truth.astype(np.float64);p64=p.astype(np.float64)
                    metrics[mode].append([[np.corrcoef(t64[:,e],p64[:,e])[0,1],np.sqrt(np.mean((t64[:,e]-p64[:,e])**2)),np.mean(np.abs(t64[:,e]-p64[:,e]))] for e in range(4)])
                    error=truth[:,:,None]-q;shape=q-q.mean(2,keepdims=True);shape-=shape.mean(0,keepdims=True)
                    diagnostics[mode].append(np.stack([np.maximum(levels*error,(levels-1)*error).mean((0,2)),np.sqrt(np.mean(shape**2,axis=(0,2))),np.mean(np.diff(q,axis=2)<0,axis=(0,2)),np.mean((truth>=q[:,:,lo])&(truth<=q[:,:,hi]),axis=0)],axis=1))
        for mode in modes:
            for actual,reference in ((np.mean(metrics[mode],axis=0),expected['metrics'][mode]),(np.mean(diagnostics[mode],axis=0),expected['quantile_diagnostics'][mode])):
                max_diff=max(max_diff,float(np.max(np.abs(actual-reference))))
                np.testing.assert_allclose(actual,reference,atol=1e-12,rtol=1e-12)
    return {'complete':True,'units':30,'trajectories':len(records),'permutations_match':True,'max_replay_difference':replay_diff,'max_summary_difference':max_diff}


if __name__=='__main__':
    print(json.dumps(check(Path(sys.argv[1]),Path(sys.argv[2])),indent=2))
