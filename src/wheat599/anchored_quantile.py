"""Strong-function-initialized neural/kernel quantile block-coordinate training."""
import copy
from pathlib import Path
import numpy as np
import torch
from wheat599 import quantile_tabm as qt
from wheat599.anchored_quantile_kernel import solve


def transform_head(model):
    head=model.backbone.output
    if head.weight.ndim!=3 or head.weight.shape[-1]!=99:raise ValueError('Expected official member-specific 99-output head')
    with torch.no_grad():
        head.weight.sub_(.5*head.weight.mean(-1,keepdim=True))
        if head.bias is not None:head.bias.sub_(.5*head.bias.mean(-1,keepdim=True))


def train(models, x_fit, y_fit, x_query, prepared, huber_state, eta, parameters, settings, save):
    """No query phenotypes; save is called for every synchronized joint state."""
    device=next(models[0].parameters()).device
    models=copy.deepcopy(models)
    x=torch.as_tensor(x_fit,dtype=torch.float32,device=device);query=torch.as_tensor(x_query,dtype=torch.float32,device=device)
    for model in models:model.eval();transform_head(model)
    centers=[{name:p.detach().clone() for name,p in model.named_parameters()} for model in models]
    scales=np.array([float(model.y_std) for model in models]);means=np.array([float(model.y_mean) for model in models])
    ratio=prepared['scale']/scales
    signal=(1-eta)*np.eye(4)+eta*prepared['C_Y']
    environment=torch.as_tensor(signal*ratio[:,None]*ratio[None,:],dtype=torch.float64,device=device)
    design=prepared['train_features'];cross=prepared['target_features']
    material=torch.as_tensor(design@design.T,dtype=torch.float64,device=device)
    cross_kernel=torch.as_tensor(cross@design.T,dtype=torch.float64,device=device)
    def anchor(features):
        physical=(features@huber_state['weights']@huber_state['factor'].T+huber_state['intercept'])*prepared['scale']+prepared['mean']
        return torch.as_tensor(.5*(physical-means)/scales,dtype=torch.float64,device=device)
    anchor_fit,anchor_query=anchor(design),anchor(cross)
    target=torch.stack([m.standardize_y(torch.as_tensor(y_fit[:,e,None],dtype=torch.float32,device=device))[:,0] for e,m in enumerate(models)],dim=1)
    taus=torch.linspace(settings['quantile_min'],settings['quantile_max'],99,device=device)
    optimizers=[torch.optim.AdamW(m.parameters(),lr=c['learning_rate']*settings['learning_rate_multiplier'],weight_decay=0.) for m,c in zip(models,settings['network_configs'])]
    def outputs(values):return [m(values) for m in models]
    def match(neural):
        residuals=[(target[:,e,None,None].double()-a.detach().double()-anchor_fit[:,e,None,None]).flatten(1) for e,a in enumerate(neural)]
        return solve(residuals,taus,material,environment,parameters['lambda_kernel'],**settings['solver'])
    zero=torch.zeros_like(anchor_fit);coef=zero.clone();position=zero;certificate=None
    for step in range(settings['updates']+1):
        with torch.no_grad():
            neural=outputs(x);predicted=outputs(query)
            location=anchor_query+cross_kernel@coef@environment
            point=np.column_stack([m.inverse_y((a.mean((1,2)).double()+location[:,e])[:,None])[:,0].cpu().numpy() for e,(m,a) in enumerate(zip(models,predicted))])
        state=dict(neural=[{k:v.detach().cpu().clone() for k,v in m.state_dict().items()} for m in models],
                   kernel_delta_coefficients=coef.cpu(),kernel_initial_state=huber_state,
                   kernel_mean=prepared['mean'],kernel_scale=prepared['scale'],kernel_environment=environment.cpu(),
                   kernel_train_features=design,kernel_intercept_fixed=True)
        save(step,point,state,certificate)
        if step==settings['updates']:break
        if step==0:position,coef,certificate=match(neural)
        for opt in optimizers:opt.zero_grad(set_to_none=True)
        # A valid subgradient of the neural block with kernel held fixed, not an envelope gradient.
        neural=outputs(x)
        pin=sum(qt.quantile_loss(a.double()+(anchor_fit[:,e]+position[:,e])[:,None,None],target[:,e,None].double(),taus.double()) for e,a in enumerate(neural))/4
        sp=sum((p-centers[e][name]).square().sum() for e,m in enumerate(models) for name,p in m.named_parameters())
        objective=pin+.5*parameters['lambda_neural']*sp
        if not torch.isfinite(objective):raise FloatingPointError('Nonfinite joint objective')
        objective.backward()
        for model,opt in zip(models,optimizers):
            torch.nn.utils.clip_grad_norm_(model.parameters(),settings['gradient_clip']);opt.step()
        with torch.no_grad():position,coef,certificate=match(outputs(x))
    return models
