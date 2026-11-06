"""Original TabM training with evenly spaced midpoint quantile levels."""
import json
from pathlib import Path
import numpy as np
import torch
from wheat599 import quantile_tabm
from wheat599.data import load
from wheat599.metrics import pcc,rmse,mae


def levels(n,device):
    return (torch.arange(n,device=device,dtype=torch.float32)+.5)/n


def train(x_fit,y_fit,x_val,y_val,config,seed,index,settings,device):
    model=quantile_tabm.build_model(x_fit,y_fit,config,seed,settings,device)
    x=torch.as_tensor(x_fit,dtype=torch.float32,device=device)
    xv=torch.as_tensor(x_val,dtype=torch.float32,device=device)
    y=model.standardize_y(torch.as_tensor(y_fit[:,None],dtype=torch.float32,device=device))
    q=levels(settings['n_quantiles'],device)
    opt=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    generator=torch.Generator().manual_seed(int(seed)*1000+index);spec=settings['training']
    best={'best_pcc':-np.inf,'best_epoch':-1}
    for epoch in range(spec['max_epochs']+1):
        if epoch:quantile_tabm.update_epoch(model,x,y,q,opt,generator,spec,device)
        model.eval()
        with torch.no_grad():pred=model.inverse_y(model(xv).mean(dim=(1,2))[:,None])[:,0].cpu().numpy()
        score=pcc(y_val,pred)
        if not np.isfinite(score):raise ValueError('Nonfinite validation PCC')
        if score>best['best_pcc']+spec['tie_tolerance']:
            best=dict(best_pcc=score,best_epoch=epoch,prediction=pred.copy())
        elif epoch-best['best_epoch']>=spec['patience']:break
    pred=best.pop('prediction');best['epochs_run']=epoch
    return pred,best


def run(cfg,args,runner):
    if cfg['n_quantiles']!=99:raise ValueError('Expected 99 midpoint levels')
    original=quantile_tabm.train
    try:
        quantile_tabm.train=train
        quantile_tabm.run(cfg,args,runner)
    finally:
        quantile_tabm.train=original
    data=load(Path(cfg['data']),cfg['data_sha256'])
    folds=json.loads(Path(cfg['splits']).read_text())['folds']
    summary=json.loads((args.out/'summary.json').read_text())
    reference=Path(cfg['reference_source']);out=[]
    for fold,row in zip(folds,summary['folds']):
        if fold['fold']!=row['fold']:raise ValueError('Fold mismatch')
        values=[]
        for j,split in enumerate(fold['inner']):
            arr=np.load(args.out/f"predictions_{fold['fold']}_{j}.npz")
            hub=np.load(reference/f"references_{fold['fold']}_{j}.npz")['huber']
            pred=np.column_stack([np.mean([arr[f'e{e}_c{c}_s{seed}'] for seed in cfg['seeds']],axis=0)
                                  for e,c in enumerate(row['selected'])])
            truth=data.Y[split['val_idx']];pred=.5*pred+.5*hub
            values.append([[fn(truth[:,e],pred[:,e]) for fn in (pcc,rmse,mae)] for e in range(4)])
        out.append({'fold':fold['fold'],'metrics':np.mean(values,axis=0).tolist()})
    macro=dict(zip(('pcc','rmse','mae'),np.mean([r['metrics'] for r in out],axis=(0,1)).tolist()))
    (args.out/'fusion_summary.json').write_text(json.dumps({'macro':macro,'folds':out},indent=2,allow_nan=False)+'\n')
