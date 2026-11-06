"""Compare saved TabICL predictions with the frozen quantile–Huber reference."""
import json
from pathlib import Path
import numpy as np
from wheat599.data import load,sha256_of
from wheat599.metrics import pcc,rmse,mae


def run(cfg,args,runner):
    if args.selection is not None:raise ValueError('Re-summary only')
    data=load(Path(cfg['data']),cfg['data_sha256']);folds=json.loads(Path(cfg['splits']).read_text())['folds']
    source=Path(cfg['tabicl_source']);q=Path(cfg['quantile_source']);h=Path(cfg['reference_source'])
    status=json.loads((source/'execution_status.json').read_text())
    if status['status']!='complete' or status['completed_units']!=30 or status['completed_adaptations']!=60 or json.loads((source/'failures.json').read_text()):
        raise ValueError('Incomplete TabICL run')
    for path in (source,q,h):
        if json.loads((path/'provenance.json').read_text())['sha256'][cfg['splits']]!=sha256_of(Path(cfg['splits'])):
            raise ValueError('Source split mismatch')
    chosen={v['fold']:v['selected'] for v in json.loads((q/'summary.json').read_text())['folds']}
    names=('official','generic','genetic','quantile','huber','quantile_huber');results=[]
    paths=[Path(cfg['splits']),Path(cfg['data']),Path(args.config),Path(__file__),Path(runner),q/'summary.json']
    for fold in folds:
        f=fold['fold'];scores={name:[] for name in names}
        for j,sp in enumerate(fold['inner']):
            ids=sp['val_idx'];fit=set(sp['train_idx']);val=set(ids)
            if fit&val or fit|val!=set(fold['dev_idx']) or (fit|val)&set(fold['test_idx']):raise ValueError('Invalid split')
            pa=source/f'predictions_{f}_{j}.npz';pb=q/f'predictions_{f}_{j}.npz';ph=h/f'references_{f}_{j}.npz';paths += [pa,pb,ph]
            a,b,ref=np.load(pa),np.load(pb),np.load(ph);truth=data.Y[ids]
            quantile=np.column_stack([np.mean([b[f'e{e}_c{c}_s{seed}'] for seed in (42,2026)],axis=0) for e,c in enumerate(chosen[f])]);hub=ref['huber']
            preds={'official':a['official'],'generic':a['generic'],'genetic':a['genetic'],
                   'quantile':quantile,'huber':hub,'quantile_huber':.5*(quantile+hub)}
            for name,pred in preds.items():
                if pred.shape!=truth.shape or not np.isfinite(pred).all():raise ValueError('Invalid predictions')
                scores[name].append([[fn(truth[:,e],pred[:,e]) for fn in (pcc,rmse,mae)] for e in range(4)])
        results.append({'fold':f,'metrics':{name:np.mean(v,axis=0).tolist() for name,v in scores.items()}})
    if len(results)!=10:raise ValueError('Incomplete folds')
    macro={name:dict(zip(('pcc','rmse','mae'),np.mean([r['metrics'][name] for r in results],axis=(0,1)).tolist())) for name in names}
    wins={name:sum(int(np.mean(np.array(r['metrics']['genetic'])[:,0])>np.mean(np.array(r['metrics'][name])[:,0])) for r in results) for name in ('official','generic','quantile_huber')}
    args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'summary.json').write_text(json.dumps({'macro':macro,'genetic_fold_wins':wins,'folds':results,
        'sha256':{str(p):sha256_of(p) for p in paths}},indent=2,allow_nan=False)+'\n')
