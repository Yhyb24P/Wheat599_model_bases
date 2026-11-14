"""Development comparison of learned ANOVA-kernel Gaussian processes."""
import time
from pathlib import Path
import numpy as np
import torch
from wheat599.metrics import pcc
from wheat599.quantile_tabm import run as run_quantile


def train(x_fit,y_fit,x_val,y_val,config,seed,index,settings,device):
    from wheat599.adaptive_anova import AdaptiveAnovaGP
    start=time.perf_counter()
    torch.manual_seed(int(seed))
    model=AdaptiveAnovaGP(x_fit,y_fit,rho=config['rho'],rank=8,seed=seed).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=config['learning_rate'])
    q=np.asarray(x_val,dtype=np.float64)
    best=-np.inf;best_epoch=-1;best_pred=None
    for epoch in range(settings['training']['max_epochs']+1):
        if epoch:
            model.train();opt.zero_grad(set_to_none=True)
            loss=model.loss(config['delta_penalty'])
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite marginal likelihood')
            loss.backward();opt.step()
        model.eval()
        with torch.no_grad(): pred=model(q).detach().cpu().numpy().reshape(-1)
        score=pcc(y_val,pred)
        if not np.isfinite(score):raise ValueError('Nonfinite validation PCC')
        if score>best+settings['training']['tie_tolerance']:
            best,best_epoch,best_pred=score,epoch,pred.copy()
        elif epoch-best_epoch>=settings['training']['patience']:break
    return best_pred,dict(best_pcc=float(best),best_epoch=best_epoch,epochs_run=epoch,
                          parameters=sum(p.numel() for p in model.parameters()),seconds=time.perf_counter()-start)


def run(cfg,args,runner):
    return run_quantile(cfg,args,runner,trainer=train,extra_sources=[Path(__file__),Path(__file__).with_name('adaptive_anova.py')])
