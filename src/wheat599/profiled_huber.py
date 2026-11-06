"""Profile a shared ICM Huber branch against all neural ensemble members."""
import numpy as np
from scipy.optimize import minimize


def solve(design, factor, targets, neural, ridge, thresholds, *, maxiter=300,
          ftol=1e-11, gtol=1e-7, accepted_gradient=1e-5, member_counts=None, initial=None):
    design = np.asarray(design, dtype=float)
    factor = np.asarray(factor, dtype=float)
    targets = np.asarray(targets, dtype=float)
    neural = np.asarray(neural, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)
    n, tasks = targets.shape
    if neural.ndim != 3 or neural.shape[0] != n or neural.shape[2] != tasks:
        raise ValueError('neural must be material x member x environment')
    if design.shape[0] != n or factor.shape != (tasks, tasks) or thresholds.shape != (tasks,):
        raise ValueError('Incompatible kernel or threshold shapes')
    if not all(np.isfinite(x).all() for x in (design, factor, targets, neural, thresholds)) or ridge <= 0 or np.any(thresholds <= 0):
        raise ValueError('Nonfinite input or nonpositive regularization/threshold')
    rank, members = design.shape[1], neural.shape[1]
    counts = np.full(tasks, members) if member_counts is None else np.asarray(member_counts, dtype=int)
    if counts.shape != (tasks,) or np.any(counts < 1) or np.any(counts > members):
        raise ValueError('Invalid member counts')
    member_weight = (np.arange(members)[:, None] < counts[None, :]) / counts[None, :]
    precondition = np.sqrt(np.sum(design**2, axis=0)[:, None] *
                           np.sum(factor**2, axis=0)[None, :] + ridge)
    intercept_scale = np.sqrt(n)
    def objective(vector):
        weights = vector[:-tasks].reshape(rank, tasks) / precondition
        intercept = vector[-tasks:] / intercept_scale
        prediction = design @ weights @ factor.T + intercept
        residual = prediction[:, None, :] + neural - targets[:, None, :]
        score = np.clip(residual, -thresholds, thresholds)
        abs_residual = np.abs(residual)
        loss = (np.where(abs_residual <= thresholds, .5 * residual**2,
                         thresholds * (abs_residual - .5 * thresholds)) * member_weight).sum()
        loss += .5 * ridge * np.sum(weights**2)
        aggregate = (score * member_weight).sum(axis=1)
        gradient = np.r_[((design.T @ aggregate @ factor + ridge * weights) / precondition).ravel(),
                         aggregate.sum(axis=0) / intercept_scale]
        return loss / targets.size, gradient / targets.size
    start = np.zeros(rank * tasks + tasks) if initial is None else np.r_[
        (initial[0] * precondition).ravel(), initial[1] * intercept_scale]
    result = minimize(objective, start, jac=True, method='L-BFGS-B',
                      options=dict(maxiter=maxiter, ftol=ftol, gtol=gtol, maxls=40))
    raw_gradient = np.r_[result.jac[:-tasks] * precondition.ravel(), result.jac[-tasks:] * intercept_scale]
    gradient_inf = float(np.max(np.abs(raw_gradient)))
    if not np.isfinite(result.fun) or not np.isfinite(result.x).all() or gradient_inf > accepted_gradient:
        raise ValueError(f'Profiled Huber solve failed: {result.message}; gradient={gradient_inf}')
    weights = result.x[:-tasks].reshape(rank, tasks) / precondition
    intercept = result.x[-tasks:] / intercept_scale
    prediction = design @ weights @ factor.T + intercept
    return weights, intercept, prediction, dict(objective=float(result.fun),
        gradient_inf=gradient_inf, iterations=int(result.nit))


def profiled_neural_loss(neural, kernel_prediction, targets, thresholds, member_counts=None):
    """Envelope gradient: treat the converged kernel solution as constant."""
    import torch
    residual = neural + kernel_prediction[:, None, :] - targets[:, None, :]
    absolute = residual.abs()
    losses = torch.where(absolute <= thresholds, .5 * residual.square(),
                         thresholds * (absolute - .5 * thresholds))
    if member_counts is None:
        return losses.mean()
    counts = torch.as_tensor(member_counts, device=neural.device)
    weights = (torch.arange(neural.shape[1], device=neural.device)[:, None] < counts) / counts
    return (losses * weights).sum() / (neural.shape[0] * neural.shape[2])
