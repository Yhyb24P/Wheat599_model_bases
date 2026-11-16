"""Shared ridge solve and background gate primitives."""
import torch


def solve(k, y):
    """Generalized ridge solve with an unpenalized intercept."""
    n = y.shape[0]
    chol = torch.linalg.cholesky(k + torch.eye(n, device=y.device, dtype=y.dtype))
    inv = torch.cholesky_inverse(chol)
    ones = torch.ones_like(y)
    u = inv @ ones
    denom = ones.T @ u
    proj = inv - (u @ u.T) / denom
    return proj @ y, (u.T @ y) / denom, torch.diagonal(proj).reshape(-1, 1)


def gates(x, w):
    return torch.softmax(x @ w, dim=1)
