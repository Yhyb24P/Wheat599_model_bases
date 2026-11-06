"""Paired fixed-residual and profiled Huber ICM training for independent TabM."""
import numpy as np
import torch
from torch.nn import functional as F

from wheat599.genomic_tabm import GenomicTabM
from wheat599.metrics import pcc
from wheat599.profiled_huber import solve, profiled_neural_loss
from wheat599.reml_icm import profile


def train_unit(x_fit, y_fit, x_val, y_val, prepared, ridge, eta, thresholds,
               configurations, seed, *, joint, epochs, patience, device='cpu'):
    """One four-environment trajectory; all kernel inputs were frozen on fit rows."""
    if len(configurations) != 4 or y_fit.shape[1] != 4:
        raise ValueError('Exactly four configurations and environments are required')
    x = torch.as_tensor(x_fit, dtype=torch.float32, device=device)
    xv = torch.as_tensor(x_val, dtype=torch.float32, device=device)
    target = torch.as_tensor(prepared['Y'].numpy(), dtype=torch.float32, device=device)
    thresholds_t = torch.as_tensor(thresholds, dtype=torch.float32, device=device)
    counts = [int(config['k']) for config in configurations]
    models, optimizers = [], []
    for environment, config in enumerate(configurations):
        torch.manual_seed(int(seed) + environment)
        model = GenomicTabM(x_fit, y_fit[:, environment:environment+1], variant='raw',
                            n_blocks=config['n_blocks'], d_block=config['d_block'],
                            dropout=config['dropout'], k=config['k']).to(device)
        with torch.no_grad():
            model.backbone.output.weight.zero_()
            model.backbone.output.bias.zero_()
        models.append(model)
        optimizers.append(torch.optim.AdamW(model.parameters(), lr=config['learning_rate'],
                                             weight_decay=config['weight_decay']))
    signal = (1-eta)*np.eye(4) + eta*prepared['C_Y']
    eigenvalues, vectors = np.linalg.eigh(signal)
    factor = vectors * np.sqrt(np.maximum(eigenvalues, 0))
    design, cross = prepared['train_features'], prepared['target_features']
    fit_y = prepared['Y'].numpy()
    def outputs(values):
        return torch.stack([F.pad(model(values)[..., 0], (0, max(counts)-count))
                            for model, count in zip(models, counts)], dim=-1)
    eig, basis, y = prepared['eigenvalues'], prepared['vectors'], prepared['Y']
    _, alpha, intercept = profile(eig, basis.T @ y, basis.sum(0),
                                  torch.as_tensor(signal), ridge * torch.eye(4, dtype=y.dtype))
    previous = (prepared['root'][:, None] * alpha.numpy()[prepared['keep']] @ factor,
                intercept.numpy())
    def kernel(neural):
        nonlocal previous
        w, c, fit, diag = solve(design, factor, fit_y, neural.detach().cpu().numpy(),
                                ridge, thresholds, member_counts=counts, initial=previous)
        previous = (w, c)
        return fit, cross @ w @ factor.T + c, diag
    for model in models:
        model.eval()
    with torch.no_grad():
        zero = outputs(x)
    fixed_fit, fixed_val, _ = kernel(zero)
    best = {'pcc': -np.inf, 'epoch': -1, 'prediction': None}
    history = []
    for epoch in range(int(epochs) + 1):
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        for model in models:
            model.train()
        neural = outputs(x)
        fit_branch, val_branch, diag = kernel(neural) if joint else (fixed_fit, fixed_val, None)
        for model in models:
            model.eval()
        with torch.no_grad():
            vals = [model(xv)[..., 0].mean(1) for model in models]
            pred = np.stack([v.cpu().numpy() for v in vals], axis=1)
            pred = (pred + val_branch) * prepared['scale'] + prepared['mean']
            score = float(np.mean([pcc(y_val[:, e], pred[:, e]) for e in range(4)]))
        if not np.isfinite(score):
            raise ValueError('Nonfinite validation PCC')
        history.append({'epoch': epoch, 'validation_pcc': score, 'kernel': diag})
        if score > best['pcc']:
            best = {'pcc': score, 'epoch': epoch, 'prediction': pred.copy()}
        elif epoch - best['epoch'] >= patience:
            break
        if epoch == epochs:
            break
        loss = profiled_neural_loss(neural, torch.as_tensor(fit_branch, dtype=x.dtype, device=device),
                                    target, thresholds_t, counts)
        loss.backward()
        for model, optimizer in zip(models, optimizers):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
    return best, history
