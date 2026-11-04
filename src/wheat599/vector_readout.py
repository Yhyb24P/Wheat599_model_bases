"""Vector-valued readout over a fixed candidate set, plus score-geometry shapes.

The scalar readout solves ``C + D H C = D S``.  With a shared effect correlation
``Omega`` and per-candidate scale ``d_l`` (prior ``c_l ~ N(0, d_l Omega)``) the same
profile system becomes

    C + D H C Omega = D S Omega,

which is solved exactly by eigendecomposing ``Omega = U diag(lambda) U^T`` and solving q
independent k-by-k systems in the rotated frame,

    (I + d_l lambda_j H) c~_{:,j} = d_l lambda_j s~_{:,j},   S~ = S U,  C = C~ U^T.

``lambda_j = 0`` and ``d_l = 0`` are exact zero columns, so no pseudo-inverse is used
and singular ``Omega`` is safe.  ``Omega = I`` reproduces the scalar readout, which is
checked by a test.

``residual_shapes`` returns three PSD candidates built from the observable response
geometry ``R = (M Y)^T (M Y) / n`` (trace-normalised, zero energy gives the identity).
That matrix still contains the additive signal, so it is a response geometry, not a
noise or effect covariance estimate.

``covariance_shapes`` returns four PSD candidates built only from the score geometry
``G = S^T S / k`` (no centring, one candidate keeps its direction): the identity, the
diagonal, the half-diagonal/half-full mixture and the full matrix, all normalised to
trace ``q``.  These are score-geometry shapes, not denoised effect covariances: no
noise is estimated and no true Omega is used.
"""

import torch


def _validate(evidence, indices, d, omega):
    indices = torch.as_tensor(indices, dtype=torch.long, device=evidence.device).reshape(-1)
    d = torch.as_tensor(d, dtype=evidence.dtype, device=evidence.device)
    if d.shape != (indices.numel(),):
        raise ValueError('d must be one non-negative value per candidate')
    if bool((d < 0).any()):
        raise ValueError('d must be non-negative')
    omega = torch.as_tensor(omega, dtype=evidence.dtype, device=evidence.device)
    environments = evidence.environments
    if omega.shape != (environments, environments):
        raise ValueError('omega must be environments by environments')
    symmetric = (omega + omega.T) / 2
    values = torch.linalg.eigvalsh(symmetric)
    scale = max(1.0, float(values.abs().max()))
    if float(values.min()) < -1e-10 * scale:
        raise ValueError('omega must be positive semidefinite')
    return indices, d, symmetric


def readout(evidence, indices, d, omega):
    """Exact ``(B, C)`` for the shared-correlation prior ``c_l ~ N(0, d_l Omega)``."""
    indices, d, omega = _validate(evidence, indices, d, omega)
    if indices.numel() == 0:
        return evidence._ridge_b.clone(), evidence.y.new_zeros((0, evidence.environments))
    values, vectors = torch.linalg.eigh(omega)
    values = values.clamp_min(0.0)
    statistics = evidence.statistics(indices)
    columns = evidence.phi(indices)
    hessian = statistics['H']
    score = statistics['S'] @ vectors
    identity = torch.eye(indices.numel(), dtype=evidence.dtype, device=evidence.device)
    rotated = torch.empty_like(score)
    for direction in range(evidence.environments):
        scaled = d * values[direction]
        system = identity + scaled.unsqueeze(1) * hessian
        rotated[:, direction] = torch.linalg.solve(system, scaled * score[:, direction])
    coefficients = rotated @ vectors.T
    additive = evidence.a.T @ torch.cholesky_solve(evidence.y - columns @ coefficients,
                                                   evidence._factor)
    return additive, coefficients


def residual_shapes(evidence):
    """Ordered candidates ``(diag, half, full)`` from the response geometry ``M Y``.

    ``R = (M Y)^T (M Y) / n`` includes the additive signal; it is a response geometry,
    not a noise or effect covariance estimate.
    """
    environments = evidence.environments
    filtered = evidence.m @ evidence.y
    response = filtered.T @ filtered / evidence.materials
    response = (response + response.T) / 2
    identity = torch.eye(environments, dtype=evidence.dtype, device=evidence.device)
    scale = response.diagonal().sum() / environments
    if float(scale.abs()) <= 1e-30:
        return identity, identity.clone(), identity.clone()
    response = response / scale
    response = (response + response.T) / 2
    diagonal = torch.diag(response.diagonal())
    half = 0.5 * diagonal + 0.5 * response
    half = (half + half.T) / 2
    return diagonal, half, response


def covariance_shapes(evidence, indices):
    """Ordered candidates ``(I, diag, half, full)`` built from score geometry only."""
    indices = torch.as_tensor(indices, dtype=torch.long, device=evidence.device).reshape(-1)
    if indices.numel() == 0:
        raise ValueError('covariance shapes need at least one candidate')
    environments = evidence.environments
    score = evidence.restricted_score(indices)
    gram = score.T @ score / score.shape[0]
    gram = (gram + gram.T) / 2
    scale = gram.diagonal().sum() / environments
    identity = torch.eye(environments, dtype=evidence.dtype, device=evidence.device)
    if float(scale.abs()) <= 1e-30:
        return identity, identity.clone(), identity.clone(), identity.clone()
    gram = gram / scale
    gram = (gram + gram.T) / 2
    diagonal = torch.diag(gram.diagonal())
    half = 0.5 * diagonal + 0.5 * gram
    half = (half + half.T) / 2
    return identity, diagonal, half, gram
