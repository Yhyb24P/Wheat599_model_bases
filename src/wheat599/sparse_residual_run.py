"""Development comparison with a shared checkpoint for both seed predictions."""
import time
from pathlib import Path
import numpy as np
import torch
from wheat599.metrics import pcc
from wheat599.quantile_tabm import quantile_loss, run as quantile_run


def train_pair(x_fit, y_fit, x_val, y_val, config, seeds, index, settings, device):
    from wheat599.sparse_residual import SparseResidualQuantile, proximal_step, make_optimizer
    start=time.perf_counter()
    if len(seeds)!=2: raise ValueError('Exactly two independent seeds are required')
    x=torch.as_tensor(x_fit,dtype=torch.float32,device=device)
    xv=torch.as_tensor(x_val,dtype=torch.float32,device=device)
    q=torch.linspace(settings['quantile_min'],settings['quantile_max'],settings['n_quantiles'],device=device)
    models=[]; optimizers=[]; generators=[]
    for seed in seeds:
        torch.manual_seed(int(seed))
        model=SparseResidualQuantile(x_fit,y_fit,width=config['d_block'],members=config['k'],quantiles=settings['n_quantiles']).to(device)
        models.append(model)
        optimizers.append(make_optimizer(model,config['learning_rate'],config['weight_decay']))
        generators.append(torch.Generator().manual_seed(int(seed)*1000+index))
    y=models[0].standardize_y(torch.as_tensor(y_fit[:,None],dtype=torch.float32,device=device))
    best_score=-np.inf;best_epoch=-1;best_predictions=None;spec=settings['training']
    for epoch in range(1,spec['max_epochs']+1):
        for model,opt,gen in zip(models,optimizers,generators):
            model.train()
            for batch in torch.randperm(len(x),generator=gen).to(device).split(spec['batch_size']):
                opt.zero_grad(set_to_none=True);loss=quantile_loss(model(x[batch]),y[batch],q)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite quantile loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),spec['gradient_clip'])
                opt.step();proximal_step(model,opt,config['prox_lambda'])
        preds=[]
        for model in models:
            model.eval()
            with torch.no_grad():preds.append(model.inverse_y(model(xv).mean(dim=(1,2))[:,None])[:,0].cpu().numpy())
        score=pcc(y_val,np.mean(preds,axis=0))
        if not np.isfinite(score):raise ValueError('Nonfinite joint validation PCC')
        if score>best_score+spec['tie_tolerance']:
            best_score,best_epoch,best_predictions=score,epoch,[v.copy() for v in preds]
        elif epoch-best_epoch>=spec['patience']:break
    return [(pred,dict(best_pcc=best_score,best_epoch=best_epoch,epochs_run=epoch,
                       parameters=sum(p.numel() for p in model.parameters()),seconds=time.perf_counter()-start))
            for model,pred in zip(models,best_predictions)]


def run(cfg,args,runner):
    return quantile_run(cfg,args,runner,paired_trainer=train_pair,
        extra_sources=[Path(__file__),Path(__file__).with_name('sparse_residual.py')])
