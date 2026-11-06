"""Reselect saved quantile configurations for fixed Huber fusion, without fitting."""
import json
from pathlib import Path
import numpy as np
from wheat599.data import load, sha256_of
from wheat599.metrics import pcc, rmse, mae


def read(path):
    return json.loads(Path(path).read_text())


def run(cfg, args, runner):
    if args.selection is not None:
        raise ValueError('Development re-summary only')
    data = load(Path(cfg['data']), cfg['data_sha256'])
    folds = read(cfg['splits'])['folds']
    src, hub = Path(cfg['quantile_source']), Path(cfg['huber_source'])
    status = read(src/'execution_status.json')
    if status != {'status':'complete','completed':1920,'expected':1920} or read(src/'failures.json'):
        raise ValueError('Incomplete source')
    if read(src/'provenance.json')['sha256'][cfg['splits']] != sha256_of(Path(cfg['splits'])):
        raise ValueError('Split mismatch')
    previous = {r['fold']:r['selected'] for r in read(src/'summary.json')['folds']}
    results=[]
    for fold in folds:
        f=fold['fold']; arrays=[np.load(src/f'predictions_{f}_{j}.npz') for j in range(3)]
        refs=[np.load(hub/f'references_{f}_{j}.npz') for j in range(3)]
        chosen=[]
        for e in range(4):
            scores=[]
            for c in range(8):
                vals=[]
                for j,split in enumerate(fold['inner']):
                    tr,val=set(split['train_idx']),set(split['val_idx'])
                    if tr & val or tr | val != set(fold['dev_idx']) or (tr | val) & set(fold['test_idx']):
                        raise ValueError('Invalid development boundary')
                    q=np.mean([arrays[j][f'e{e}_c{c}_s{seed}'] for seed in cfg['seeds']],axis=0)
                    vals.append(pcc(data.Y[split['val_idx'],e],.5*q+.5*refs[j]['huber'][:,e]))
                scores.append(float(np.mean(vals)))
            if not np.isfinite(scores).all():raise ValueError('Nonfinite score')
            chosen.append(int(np.argmax(scores)))
        metrics={k:[] for k in ('original','fusion_selected')}
        for j,split in enumerate(fold['inner']):
            truth=data.Y[split['val_idx']]
            for name,indices in [('original',previous[f]),('fusion_selected',chosen)]:
                q=np.column_stack([np.mean([arrays[j][f'e{e}_c{c}_s{seed}'] for seed in cfg['seeds']],axis=0)
                                   for e,c in enumerate(indices)])
                pred=.5*q+.5*refs[j]['huber']
                metrics[name].append([[fn(truth[:,e],pred[:,e]) for fn in (pcc,rmse,mae)] for e in range(4)])
        results.append({'fold':f,'original':previous[f],'selected':chosen,
                        'metrics':{k:np.mean(v,axis=0).tolist() for k,v in metrics.items()}})
    if len(results)!=10:raise ValueError('Expected ten folds')
    macro={k:dict(zip(('pcc','rmse','mae'),np.mean([r['metrics'][k] for r in results],axis=(0,1)).tolist()))
           for k in ('original','fusion_selected')}
    args.out.mkdir(parents=True,exist_ok=False)
    sources=[Path(cfg['splits']),src/'summary.json',src/'trajectories.json',Path(__file__),Path(runner),Path(args.config)]
    sources += [src/f"predictions_{f['fold']}_{j}.npz" for f in folds for j in range(3)]
    sources += [hub/f"references_{f['fold']}_{j}.npz" for f in folds for j in range(3)]
    (args.out/'summary.json').write_text(json.dumps({'macro':macro,'folds':results,
        'sha256':{str(p):sha256_of(p) for p in sources}},indent=2,allow_nan=False)+'\n')
