"""Background-conditioned additive ridge: math core (conditional analytic LOO gates).

Experiment orchestration (nested PCC selection / full-dev refit) lives in
genopred.experiments.wheat599.background_ridge.
"""
import torch

from genopred.models.classical.kernel_ridge import gates, solve


def kernel(g, h, gram, global_penalty, background_penalty):
    return gram * (1.0 / global_penalty + (g @ h.T) / background_penalty)


def fit_candidate(x, gram, y, gl, bl, cfg):
    w = torch.nn.Parameter(torch.randn(x.shape[1], cfg['gates'], device=x.device, dtype=x.dtype) * cfg['gate_init_std'])
    opt = torch.optim.Adam([w], lr=cfg['gate_lr'])
    best_loss, best_w = float('inf'), None
    for step in range(cfg['gate_steps'] + 1):
        opt.zero_grad(set_to_none=True)
        q = gates(x, w)
        alpha, _, diag = solve(kernel(q, q, gram, gl, bl), y)
        loss = torch.mean((alpha / diag)**2)
        if not torch.isfinite(loss):
            raise ValueError('Non-finite conditional LOO loss')
        if loss.item() < best_loss:
            best_loss, best_w = loss.item(), w.detach().clone()
        if step < cfg['gate_steps']:
            loss.backward()
            opt.step()
    with torch.no_grad():
        q = gates(x, best_w)
        alpha, intercept, _ = solve(kernel(q, q, gram, gl, bl), y)
    return best_w, alpha, intercept, best_loss