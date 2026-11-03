"""Locus-preserving context features with a differentiable analytic ICM head.

Feature map, for centred markers ``z = X - mean(X_support)`` (raw scale, never the
kernel-normalised Z) and a learned projection ``V`` of rank ``r``:

    c(x)      = z V / sqrt(p)
    c_-j(x)   = c(x) - z_j V_j / sqrt(p)
    phi_j(x)  = z_j (h(c_-j(x)) - h(0)),        Phi(x) = vec phi(x)  in R^(p r)

``c_-j`` is computed exactly as a subtraction of the focus row's own term, so the
modulating input has zero partial derivative with respect to the focus marker; the
claim stops there.  Normalisation of ``Phi`` uses support statistics that keep their
gradient during training, so the normalised feature is not claimed to be free of the
focus marker.

The material kernel is the weighted average of the retained base kernel and the new
feature kernel,

    K = (K_base + gamma K_phi) / (1 + gamma),     K_phi = Phi~ Phi~^T,

which keeps the average diagonal at one on both branches, so the retained ridge
scale carries over unchanged.  The prediction is the retained multi-environment ICM
on that kernel with B = (1 - eta) I + eta Corr(Y_support): B is eigendecomposed, the
joint system decouples into one n-by-n Cholesky solve per environment, each with an
unpenalised intercept, and the task rotation is undone afterwards.

Variants: ``full`` (self-excluding context), ``full_background`` (context without the
self-exclusion), ``linear`` (bias-free linear h), and ``global`` (one learned weight
per marker times z, i.e. a marker-weighted additive kernel with no background).
"""

import hashlib
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from wheat599.kernel_baselines import kernels as base_kernels
from wheat599.multienv_kernel import task_correlation

VARIANTS = ('full', 'full_background', 'linear', 'global')
TASK_TOLERANCE = 1e-10


class LocusContextFeatures(torch.nn.Module):
    """Per-marker features; only V and h (or the global weights) are learned."""

    def __init__(self, markers, rank=8, variant='full'):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f'Unknown variant {variant!r}')
        if markers < 1 or rank < 1:
            raise ValueError('Marker count and rank must be positive')
        self.markers, self.rank, self.variant = int(markers), int(rank), variant
        self.scale = 1.0 / math.sqrt(self.markers)
        if variant == 'global':
            self.weight = torch.nn.Parameter(torch.ones(self.markers))
            return
        self.projection = torch.nn.Parameter(torch.empty(self.markers, self.rank))
        torch.nn.init.normal_(self.projection, std=1.0)
        if variant == 'linear':
            self.linear = torch.nn.Linear(self.rank, self.rank, bias=False)
            torch.nn.init.normal_(self.linear.weight, std=1.0 / math.sqrt(self.rank))
            return
        self.layer1 = torch.nn.Linear(self.rank, 2 * self.rank)
        self.layer2 = torch.nn.Linear(2 * self.rank, 2 * self.rank)
        self.layer3 = torch.nn.Linear(2 * self.rank, self.rank, bias=False)

    def response(self, context):
        if self.variant == 'linear':
            return self.linear(context)
        return self.layer3(F.silu(self.layer2(F.silu(self.layer1(context)))))

    def contexts(self, z):
        """Per-marker background inputs, shape materials by markers by rank."""
        aggregate = (z @ self.projection) * self.scale
        if self.variant == 'full_background':
            return aggregate.unsqueeze(1)
        own = z.unsqueeze(-1) * self.projection.unsqueeze(0) * self.scale
        return aggregate.unsqueeze(1) - own

    def forward(self, z):
        if self.variant == 'global':
            return z * self.weight
        context = self.contexts(z)
        modulation = self.response(context) - self.response(
            torch.zeros(1, self.rank, dtype=z.dtype, device=z.device))
        return (z.unsqueeze(-1) * modulation).reshape(z.shape[0], -1)


def feature_statistics(features):
    """Column centring and mean-squared-norm scaling from the support rows.

    Statistics intentionally keep their graph: the training objective is defined on
    the normalised features, so gradients must flow through the support mean and
    scale as well.
    """
    center = features.mean(dim=0, keepdim=True)
    centered = features - center
    scale = centered.square().sum(dim=1).mean().clamp_min(1e-12).sqrt()
    return center, scale


def task_matrix(standardised_support, eta):
    correlation = torch.as_tensor(
        task_correlation(standardised_support.detach().cpu().numpy().astype(np.float64)),
        dtype=torch.float64)
    environments = correlation.shape[0]
    return (1 - eta) * torch.eye(environments, dtype=torch.float64) + eta * correlation


def icm_prediction(kernel, kernel_target, labels, task, ridge):
    """Differentiable ICM prediction with one solve per eigen-rotated environment.

    ``labels`` are the support rows in the standardised space; the return value is on
    the same scale.  Gradients flow through ``kernel`` and ``kernel_target`` only.
    """
    if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
        raise ValueError('Kernel must be square')
    if kernel_target.shape[1] != kernel.shape[0] or labels.shape[0] != kernel.shape[0]:
        raise ValueError('Kernel, cross kernel and labels disagree on the support rows')
    if not torch.isfinite(kernel).all() or not torch.isfinite(kernel_target).all():
        raise ValueError('Non-finite kernel input')
    values, vectors = torch.linalg.eigh(task)
    scale = max(1.0, float(values.abs().max()))
    if float(values.min()) < -TASK_TOLERANCE * scale:
        raise ValueError('Task covariance has a negative eigenvalue beyond rounding')
    if float(values.min()) <= 0:
        raise ValueError('Task covariance must be strictly positive definite')
    rotated = labels @ vectors
    environments = rotated.shape[1]
    identity = torch.eye(kernel.shape[0], dtype=torch.float64, device=kernel.device)
    systems = values[:, None, None] * kernel.unsqueeze(0) + ridge * identity.unsqueeze(0)
    right = torch.cat((rotated.T.unsqueeze(-1),
                       torch.ones((environments, kernel.shape[0], 1), dtype=torch.float64,
                                  device=kernel.device)), dim=-1)
    solved = torch.cholesky_solve(right, torch.linalg.cholesky(systems))
    ones = torch.ones(kernel.shape[0], 1, dtype=torch.float64, device=kernel.device)
    intercept = (solved[:, :, 0] @ ones).squeeze(-1) / (solved[:, :, 1] @ ones).squeeze(-1)
    coefficients = solved[:, :, 0] - intercept[:, None] * solved[:, :, 1]
    rotated_prediction = intercept[:, None] + values[:, None] * (kernel_target @ coefficients.T).T
    return rotated_prediction.T @ vectors.T


def _base_tensors(markers, support, query, base):
    """Retained additive and RBF kernels; all statistics from the support rows."""
    additive, additive_cross, rbf, rbf_cross = base_kernels(
        markers, np.asarray(support), np.asarray(query), base['bandwidth'], torch.device('cpu'))
    fraction = float(base['fraction'])
    kernel = fraction * additive + (1 - fraction) * rbf
    cross = fraction * additive_cross + (1 - fraction) * rbf_cross
    return kernel.double(), cross.double()


def _combine(kernel, cross, features, features_target, gamma):
    if gamma == 0:
        return kernel, cross
    kernel = kernel + gamma * (features @ features.T)
    cross = cross + gamma * (features_target @ features.T)
    return kernel / (1 + gamma), cross / (1 + gamma)


def _episode(length, fraction, generator):
    order = torch.randperm(length, generator=generator)
    size = min(length - 1, max(2, int(round(fraction * length))))
    return order[:size].numpy(), order[size:].numpy()


def _prepare(markers, phenotypes, support, query, base):
    support, query = np.asarray(support), np.asarray(query)
    labels = torch.as_tensor(phenotypes, dtype=torch.float64)
    mean = labels[support].mean(0)
    deviation = labels[support].std(0, unbiased=False)
    if torch.any(deviation <= 0):
        raise ValueError('Constant environment in the support rows')
    standardised = (labels - mean) / deviation
    kernel, cross = _base_tensors(markers, support, query, base)
    return standardised, kernel, cross


def _model(variant, markers, rank, seed):
    torch.manual_seed(int(seed))
    return LocusContextFeatures(markers, rank=rank, variant=variant)


def train_context(X_fit, Y_fit, base, variant, gamma, weight_decay, seed, cfg):
    """Train the feature map on episodes inside the fitting subset only.

    Each step draws a support/query split of the passed rows, standardises phenotypes
    and the task matrix on the support rows, adds the feature kernel with the fixed
    gamma, and takes one Adam step on the four-environment query MSE.  Checkpoints are
    detached CPU copies keyed by training step, so ``states[0]`` is the initial model.
    """
    markers = np.asarray(X_fit)
    phenotypes = np.asarray(Y_fit, dtype=np.float64)
    if markers.ndim != 2 or phenotypes.shape[0] != markers.shape[0]:
        raise ValueError('Marker and phenotype rows disagree')
    if variant not in VARIANTS:
        raise ValueError(f'Unknown variant {variant!r}')
    if not gamma >= 0 or not weight_decay >= 0:
        raise ValueError('Gamma and weight decay must be non-negative')
    rank = int(cfg['rank'])
    steps = int(cfg.get('training_steps', cfg.get('steps', 0)))
    checkpoints = [int(s) for s in cfg['checkpoints']]
    if steps < 0 or any(s < 0 or s > steps for s in checkpoints):
        raise ValueError('Checkpoints must lie inside the training budget')
    device = torch.device(cfg.get('device', 'cpu'))
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    if device.type == 'cuda' and device.index is None:
        device = torch.device('cuda:0')
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(float(cfg.get('gpu_memory_fraction', 0.2)),
                                                   device=device.index)
        torch.cuda.reset_peak_memory_stats(device)
    model = _model(variant, markers.shape[1], rank, seed).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg['learning_rate']),
                                 weight_decay=float(weight_decay))
    stream = torch.Generator().manual_seed(int(seed))
    support_fraction = float(cfg['support_fraction'])
    states, curve = {}, []
    digest = hashlib.sha256()
    start = time.perf_counter()
    loss = None
    for step in range(steps + 1):
        if step and gamma > 0:
            support, query = _episode(markers.shape[0], support_fraction, stream)
            digest.update(np.sort(support).astype('<i8').tobytes())
            standardised, kernel, cross = _prepare(markers, phenotypes, support, query, base)
            support_mean = torch.as_tensor(markers[support].mean(0), dtype=torch.float32,
                                           device=device)
            z_support = torch.as_tensor(markers[support], dtype=torch.float32,
                                        device=device) - support_mean
            z_query = torch.as_tensor(markers[query], dtype=torch.float32,
                                      device=device) - support_mean
            kernel, cross = kernel.to(device), cross.to(device)
            standardised = standardised.to(device)
            phi_support = model(z_support)
            phi_query = model(z_query)
            center, scale = feature_statistics(phi_support)
            feature_kernel, feature_cross = _combine(
                kernel, cross, (phi_support - center) / scale,
                (phi_query - center) / scale, gamma)
            support_index = torch.as_tensor(support)
            query_index = torch.as_tensor(query)
            task = task_matrix(standardised[support_index].cpu(), base['eta']).to(device)
            prediction = icm_prediction(feature_kernel, feature_cross,
                                        standardised[support_index], task,
                                        float(base['ridge']))
            target = standardised[query_index]
            loss = (prediction - target).square().mean()
            if not torch.isfinite(loss):
                raise ValueError(f'Non-finite training loss at step {step}')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        if step in checkpoints:
            states[step] = {name: value.detach().cpu().clone() for name, value in
                            model.state_dict().items()}
            curve.append({'step': step, 'loss': None if loss is None else float(loss.detach())})
    cost = {'seconds': round(time.perf_counter() - start, 3), 'steps': steps,
            'ms_per_step': round((time.perf_counter() - start) / max(1, steps) * 1000, 2),
            'support_fraction': support_fraction,
            'feature_parameters': int(sum(p.numel() for p in model.parameters())),
            'peak_megabytes': (round(float(torch.cuda.max_memory_allocated(device) / 2 ** 20), 1)
                               if device.type == 'cuda' else 0.0),
            'device': str(device)}
    record = {'variant': variant, 'gamma': float(gamma), 'weight_decay': float(weight_decay),
              'seed': int(seed), 'checkpoints': checkpoints, 'curve': curve, 'cost': cost,
              'episode_digest': digest.hexdigest(), 'features_trained': bool(gamma > 0)}
    return states, record


def predict_context(X_fit, Y_fit, X_target, base, variant, gamma, state, seed, cfg):
    """Predict target rows from fitting rows only; returns the original phenotype scale.

    ``gamma == 0`` is the retained base model and accepts ``state=None``; otherwise the
    supplied checkpoint is required.  Statistics (phenotype moments, task correlation,
    feature centring and scale) come from the fitting rows alone.
    """
    markers = np.asarray(X_fit)
    target_markers = np.asarray(X_target)
    phenotypes = np.asarray(Y_fit, dtype=np.float64)
    if target_markers.shape[1] != markers.shape[1]:
        raise ValueError('Target markers disagree with the fitting markers')
    support = np.arange(markers.shape[0])
    query = np.arange(markers.shape[0], markers.shape[0] + target_markers.shape[0])
    all_markers = np.concatenate([markers, target_markers])
    standardised, kernel, cross = _prepare(all_markers, phenotypes, support, query, base)
    if gamma == 0:
        if state is not None:
            raise ValueError('gamma = 0 does not accept a trained checkpoint')
        task = task_matrix(standardised, base['eta'])
        prediction = icm_prediction(kernel, cross, standardised, task, float(base['ridge']))
    else:
        if state is None:
            raise ValueError('A trained checkpoint is required when gamma > 0')
        if variant not in VARIANTS:
            raise ValueError(f'Unknown variant {variant!r}')
        device = torch.device(cfg.get('device', 'cpu'))
        model = _model(variant, markers.shape[1], int(cfg['rank']), seed)
        model.load_state_dict({name: value.clone() for name, value in state.items()})
        model.to(device=device, dtype=torch.float32).eval()
        support_mean = torch.as_tensor(markers.mean(0), dtype=torch.float32, device=device)
        z_support = torch.as_tensor(markers, dtype=torch.float32, device=device) - support_mean
        z_target = torch.as_tensor(target_markers, dtype=torch.float32,
                                   device=device) - support_mean
        kernel, cross = kernel.to(device), cross.to(device)
        standardised = standardised.to(device)
        with torch.no_grad():
            phi_support = model(z_support)
            phi_target = model(z_target)
            center, scale = feature_statistics(phi_support)
            kernel, cross = _combine(kernel, cross, (phi_support - center) / scale,
                                     (phi_target - center) / scale, gamma)
        task = task_matrix(standardised.cpu(), base['eta']).to(device)
        prediction = icm_prediction(kernel, cross, standardised, task,
                                    float(base['ridge'])).detach().cpu()
    labels = torch.as_tensor(phenotypes, dtype=torch.float64)
    mean = labels.mean(0)
    deviation = labels.std(0, unbiased=False)
    return (prediction * deviation + mean).detach().cpu().numpy().astype(np.float64)
