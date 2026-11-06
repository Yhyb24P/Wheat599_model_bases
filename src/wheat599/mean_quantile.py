"""Quantile regression with per-member mean supervision."""
import numpy as np
import torch
from wheat599 import quantile_tabm
from wheat599.metrics import pcc


def mean_loss(output, target, eta):
    return eta * output.shape[-1] * (output.mean(-1)-target).square().mean()


def train(x_fit, y_fit, x_val, y_val, config, seed, index, settings, device):
    model=quantile_tabm.build_model(x_fit,y_fit,config,seed,settings,device)
    x=torch.as_tensor(x_fit,dtype=torch.float32,device=device)
    xv=torch.as_tensor(x_val,dtype=torch.float32,device=device)
    y=model.standardize_y(torch.as_tensor(y_fit[:,None],dtype=torch.float32,device=device))
    q=torch.linspace(settings['quantile_min'],settings['quantile_max'],settings['n_quantiles'],device=device)
    opt=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    gen=torch.Generator().manual_seed(int(seed)*1000+index);spec=settings['training']
    totals=torch.zeros(4,device=device);steps=0;best={'best_pcc':-np.inf,'best_epoch':-1}
    for epoch in range(spec['max_epochs']+1):
        if epoch:
            model.train();order=torch.randperm(len(x),generator=gen).to(device)
            for batch in order.split(spec['batch_size']):
                opt.zero_grad(set_to_none=True);output=model(x[batch])
                pin=quantile_tabm.quantile_loss(output,y[batch],q)
                mean=mean_loss(output,y[batch],settings['eta']);loss=pin+mean
                if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),spec['gradient_clip'])
                if not torch.isfinite(norm):raise ValueError('Nonfinite gradient')
                coefficient=torch.clamp(spec['gradient_clip']/(norm+1e-6),max=1.)
                totals+=torch.stack([pin.detach(),mean.detach(),norm.detach(),coefficient.detach()]);steps+=1
                opt.step()
        model.eval()
        with torch.no_grad():pred=model.inverse_y(model(xv).mean(dim=(1,2))[:,None])[:,0].cpu().numpy()
        score=pcc(y_val,pred)
        if not np.isfinite(score):raise ValueError('Nonfinite validation PCC')
        if score>best['best_pcc']+spec['tie_tolerance']:
            best=dict(best_pcc=score,best_epoch=epoch,prediction=pred.copy(),steps=steps,
                      totals=totals.cpu().tolist())
        elif epoch-best['best_epoch']>=spec['patience']:break
    prediction=best.pop('prediction');best['epochs_run']=epoch
    return prediction,best


def run(cfg,args,runner):
    if cfg['eta'] != .1:raise ValueError('Frozen eta must be 0.1')
    original=quantile_tabm.train
    try:
        quantile_tabm.train=train
        quantile_tabm.run(cfg,args,runner)
    finally:
        quantile_tabm.train=original
