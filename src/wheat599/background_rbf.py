"""Background-conditioned and Gaussian genomic kernels for ridge prediction."""

import numpy as np
import torch

from wheat599.kernel_ridge import gates, solve


def inputs(X, train, target, features, bandwidth, redundancy_shrinkage=None):
    """Construct kernels with distance and scale determined only by training rows."""
    x, xt, gram, cross = features
    # For binary markers, squared Euclidean distance equals Hamming distance.
    ztr = torch.as_tensor(X[train], dtype=x.dtype, device=x.device)
    zt = torch.as_tensor(X[target], dtype=x.dtype, device=x.device)
    if redundancy_shrinkage is None:
        marker_weights = torch.ones(ztr.shape[1], dtype=x.dtype, device=x.device)
    else:
        centered = ztr - ztr.mean(0)
        norms = torch.linalg.vector_norm(centered, dim=0)
        standardized = centered / torch.where(norms > 0, norms, torch.ones_like(norms))
        corr = standardized.T @ standardized
        corr.fill_diagonal_(0)
        raw = torch.rsqrt(1 + corr.square().sum(1))
        marker_weights = (1 - redundancy_shrinkage) + redundancy_shrinkage * raw / raw.mean()
    train_norm = (ztr.square() * marker_weights).sum(1)
    dtrain = (train_norm[:, None] + train_norm[None, :] - 2 * (ztr * marker_weights) @ ztr.T).clamp_min(0)
    dcross = ((zt.square() * marker_weights).sum(1)[:, None] + train_norm[None, :] - 2 * (zt * marker_weights) @ ztr.T).clamp_min(0)
    distances = dtrain[torch.triu_indices(len(train), len(train), offset=1, device=x.device).unbind()]
    d0 = torch.quantile(distances, 0.5)
    if d0 <= 0:
        raise ValueError('Training Hamming median must be positive')
    scale = torch.diagonal(gram).mean()
    if scale <= 0:
        raise ValueError('Additive kernel has zero scale')
    return x, xt, gram / scale, cross / scale, torch.exp(-bandwidth * dtrain / d0), torch.exp(-bandwidth * dcross / d0)


def fit_predict(prepared, ytrain, cfg, seed):
    """Joint projected Adam minimization of conditional analytic LOO MSE."""
    x, xt, gram, cross, rbf, rbf_cross = prepared
    ym, ys = float(np.mean(ytrain)), float(np.std(ytrain))
    if ys == 0:
        raise ValueError('Constant target')
    y = torch.as_tensor(((ytrain - ym) / ys)[:, None], dtype=x.dtype, device=x.device)
    torch.manual_seed(seed)
    w = torch.nn.Parameter(torch.randn(x.shape[1], cfg['gates'], dtype=x.dtype, device=x.device) * cfg['gate_init_std'])
    weights = torch.nn.Parameter(torch.tensor(cfg['initial_weights'], dtype=x.dtype, device=x.device))
    opt = torch.optim.Adam([w, weights], lr=cfg['learning_rate'])

    def components(w_now):
        q = gates(x, w_now) - 1.0 / cfg['gates']
        bg = gram * (q @ q.T)
        scale = torch.diagonal(bg).mean().clamp_min(cfg['kernel_scale_floor'])
        return q, bg / scale, scale

    best_loss, best_w, best_weights = float('inf'), None, None
    for step in range(cfg['steps'] + 1):
        opt.zero_grad(set_to_none=True)
        _, bg, _ = components(w)
        alpha, _, diagonal = solve(weights[0] * gram + weights[1] * bg + weights[2] * rbf, y)
        loss = ((alpha / diagonal) ** 2).mean()
        if not torch.isfinite(loss):
            raise ValueError('Non-finite conditional LOO loss')
        if loss.item() < best_loss:
            best_loss, best_w, best_weights = loss.item(), w.detach().clone(), weights.detach().clone()
        if step < cfg['steps']:
            loss.backward()
            opt.step()
            with torch.no_grad():
                weights.clamp_(min=0)
    with torch.no_grad():
        q, bg, scale = components(best_w)
        alpha, intercept, _ = solve(best_weights[0] * gram + best_weights[1] * bg + best_weights[2] * rbf, y)
        qt = gates(xt, best_w) - 1.0 / cfg['gates']
        kcross = best_weights[0] * cross + best_weights[1] * cross * (qt @ q.T) / scale + best_weights[2] * rbf_cross
        pred = (intercept + kcross @ alpha).cpu().numpy().ravel() * ys + ym
    return pred, best_weights.cpu().tolist(), best_loss
