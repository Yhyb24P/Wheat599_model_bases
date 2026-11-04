"""Independent synthetic tasks for material--marker inference.

These are explicit computational priors, not additional measured crop data.
No function reads Wheat599, learns a prior from its phenotypes, or consumes a
query phenotype while preparing the network input.
"""

from dataclasses import dataclass
import math

import torch


FAMILIES = ('dense_additive', 'sparse_additive', 'pairwise', 'hidden_additive')
DESIGNS = ('independent', 'linked', 'admixture')


@dataclass
class ContextData:
    x: torch.Tensor
    y: torch.Tensor
    query: torch.Tensor
    y_mean: torch.Tensor
    y_scale: torch.Tensor

    def restore(self, prediction):
        return prediction * self.y_scale + self.y_mean


@dataclass
class SyntheticTask:
    context_x: torch.Tensor
    context_y: torch.Tensor
    query_x: torch.Tensor
    query_signal: torch.Tensor
    query_y: torch.Tensor
    metadata: dict

    def prepare(self):
        data = prepare_context(self.context_x, self.context_y, self.query_x)
        # Only the supervised loss sees this clean simulated query target.
        target = (self.query_signal - data.y_mean) / data.y_scale
        return data, target


def prepare_context(context_x, context_y, query_x):
    """Context-only centering, global genotype RMS and per-environment Y scale.

    Global genotype scaling preserves marginal-frequency differences and handles
    individual constant markers without dividing by their zero variances.
    """
    if (context_x.ndim != 2 or context_y.ndim != 2 or query_x.ndim != 2
            or len(context_x) < 3 or len(context_x) != len(context_y)
            or context_x.shape[1] != query_x.shape[1]
            or context_x.shape[1] == 0 or context_y.shape[1] == 0):
        raise ValueError('Incompatible context/query dimensions')
    arrays = (context_x, context_y, query_x)
    if any(a.device != context_x.device or a.dtype != context_x.dtype for a in arrays):
        raise ValueError('Context and query must share device and floating dtype')
    if not context_x.is_floating_point() or not all(torch.isfinite(a).all() for a in arrays):
        raise ValueError('Expected finite floating-point inputs')
    mean = context_x.mean(0)
    z = context_x - mean
    genotype_scale = z.square().mean().sqrt()
    ym, ys = context_y.mean(0), context_y.std(0, unbiased=False)
    if genotype_scale <= 0 or torch.any(ys <= 0):
        raise ValueError('Constant genotype matrix or context phenotype')
    return ContextData(z / genotype_scale, (context_y - ym) / ys,
                       (query_x - mean) / genotype_scale, ym, ys)


def _choice(values, rng):
    if not values:
        raise ValueError('Empty prior choice')
    return values[int(torch.randint(len(values), (), generator=rng))]


def _genotypes(rows, markers, design, rng, block_size):
    if design == 'independent':
        frequency = .1 + .8 * torch.rand(markers, generator=rng)
        x = (torch.rand(rows, markers, generator=rng) < frequency).float()
        hidden = None
    elif design == 'linked':
        # No physical-order claim: group membership and final columns are shuffled.
        blocks = math.ceil(markers / block_size)
        frequency = .1 + .8 * torch.rand(blocks, generator=rng)
        hidden = (torch.rand(rows, blocks, generator=rng) < frequency).float()
        groups = torch.arange(markers) // block_size
        flip_rate = .02 + .23 * torch.rand(markers, generator=rng)
        flipped = torch.rand(rows, markers, generator=rng) < flip_rate
        x = torch.where(flipped, 1 - hidden[:, groups], hidden[:, groups])
    elif design == 'admixture':
        # A held-out population-structure prior, not a breeding simulator.
        ancestry = torch.rand(rows, 1, generator=rng)
        frequencies = .05 + .9 * torch.rand(2, markers, generator=rng)
        probability = ancestry * frequencies[0] + (1 - ancestry) * frequencies[1]
        x = (torch.rand(rows, markers, generator=rng) < probability).float()
        hidden = None
    else:
        raise ValueError(f'Unknown genotype design: {design}')
    swap = torch.rand(markers, generator=rng) < .5
    x = torch.where(swap, 1 - x, x)
    return x[:, torch.randperm(markers, generator=rng)], hidden


def _effects(count, environments, rng):
    # Random shared and environment-specific effects, including negative sharing.
    loading = torch.randn(environments, generator=rng)
    loading = loading / loading.square().mean().sqrt().clamp_min(1e-6)
    share = .8 * torch.rand((), generator=rng)
    shared = torch.randn(count, 1, generator=rng) * loading
    private = torch.randn(count, environments, generator=rng)
    scale = (share * loading.square() + 1 - share).sqrt()
    covariance = (share * torch.outer(loading, loading)
                  + (1 - share) * torch.eye(environments)) / torch.outer(scale, scale)
    effects = (share.sqrt() * shared + (1 - share).sqrt() * private)
    return effects / scale / math.sqrt(count), covariance


def sample_task(config, seed, *, family=None, design=None, device='cpu'):
    """Draw one task with its own CPU RNG, independent of model initialization.

    Hidden-additive tasks always use linked markers: latent binary causes are
    omitted from the observed matrix, which contains multiple noisy proxies.
    Signal amplitude uses context feature geometry, not realized query outcomes
    or realized effect norms. Conditional Gaussian effects remain Gaussian. Noise and
    signal are available to evaluation, but only noisy context Y enters a model.
    """
    rng = torch.Generator(device='cpu').manual_seed(seed)
    n = int(_choice(config['context_sizes'], rng))
    p = int(_choice(config['marker_counts'], rng))
    m, q = int(config['query_size']), int(config['environments'])
    family = family or _choice(config.get('families', FAMILIES), rng)
    design = design or _choice(config.get('designs', ('independent', 'linked')), rng)
    if family not in FAMILIES or design not in DESIGNS:
        raise ValueError('Unknown task family or genotype design')
    if n < 3 or p < 2 or m < 1 or q < 1 or config.get('block_size', 8) < 1:
        raise ValueError('Invalid task dimensions')
    if any(int(v) < 1 for key in ('active_counts', 'pair_counts')
           for v in config.get(key, (1, 4, 16))):
        raise ValueError('Effect counts must be positive')
    if family == 'hidden_additive':
        design = 'linked'
    x, hidden = _genotypes(n + m, p, design, rng, config.get('block_size', 8))
    z = x - x[:n].mean(0)
    if family == 'dense_additive':
        count = p
        features = z
    elif family == 'sparse_additive':
        count = min(p, int(_choice(config.get('active_counts', (1, 4, 16)), rng)))
        active = torch.randperm(p, generator=rng)[:count]
        features = z[:, active]
    elif family == 'pairwise':
        count = min(p // 2, int(_choice(config.get('pair_counts', (1, 4, 16)), rng)))
        active = torch.randperm(p, generator=rng)[:2 * count]
        features = z[:, active[:count]] * z[:, active[count:]]
    else:
        count = min(hidden.shape[1], int(_choice(config.get('active_counts', (1, 4, 16)), rng)))
        active = torch.randperm(hidden.shape[1], generator=rng)[:count]
        features = hidden[:, active]
    features = features - features[:n].mean(0)
    feature_scale = features[:n].square().mean().sqrt()
    # Constant context features need not be constant on future queries. Retain
    # their original units instead of amplifying them by an arbitrary tiny floor.
    safe_scale = torch.where(feature_scale > 0, feature_scale, torch.ones_like(feature_scale))
    coefficients, covariance = _effects(count, q, rng)
    signal = (features / safe_scale) @ coefficients
    lower, upper = config.get('heritability_range', (.1, .8))
    if not 0 < lower <= upper < 1:
        raise ValueError('Require heritability range strictly inside (0,1)')
    h2 = lower + (upper - lower) * torch.rand(q, generator=rng)
    signal = signal * h2.sqrt()
    y = signal + torch.randn(n + m, q, generator=rng) * (1 - h2).sqrt()
    metadata = {'seed': seed, 'family': family, 'design': design, 'context_size': n,
                'query_size': m, 'markers': p, 'environments': q, 'effects': count,
                'nominal_h2': h2.tolist(),
                'effect_correlation': covariance.tolist(),
                'constant_context_signal_design': bool(feature_scale <= 0)}
    return SyntheticTask(x[:n].to(device), y[:n].to(device), x[n:].to(device),
                         signal[n:].to(device), y[n:].to(device), metadata)
