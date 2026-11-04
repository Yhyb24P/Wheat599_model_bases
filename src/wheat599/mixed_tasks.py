"""Mixed additive-and-interaction synthetic tasks for the joint round-two protocol.

``signal = additive_scale * F_A + gamma * F_I`` where each block is context-centered and
divided by its own context root-mean-square (a block that is constant on the context keeps
its units instead of being amplified), and the noise scale is drawn per environment from
``noise_scales``. Nothing is calibrated to a realized heritability: moving ``gamma`` changes
the realized interaction share, which is the point of the knob.

Draw order is the pairing guarantee. Every configuration choice (design, background, gamma,
additive scale, noise scales, pair count, active count) is drawn from the task's own
generator *before* any data randomness, and the interaction block is drawn even when
``gamma = 0``. Overriding a knob afterwards therefore leaves genotypes, additive direction,
interaction direction and noise draws bit-identical at a fixed seed.

No query row enters any centering or scale, no true support is placed in metadata, and
nothing here reads measured data.
"""
import torch

from wheat599.matrix_tasks import SyntheticTask, _choice, _effects, _genotypes

BACKGROUNDS = ('dense', 'sparse', 'hidden')
DESIGNS = ('independent', 'linked', 'admixture')


def pair_space(markers):
    """Number of unordered distinct-marker pairs; self-pairs are excluded by construction."""
    return markers * (markers - 1) // 2


def decode_pairs(index, markers):
    """Bijective row-major decode of ``0 <= t < pair_space(markers)`` into pairs ``j < k``."""
    t = torch.as_tensor(index, dtype=torch.int64)
    row = 2 * markers - 1
    left = torch.floor((row - torch.sqrt((row * row - 8 * t).to(torch.float64))) / 2).to(torch.int64)
    return left, left + 1 + (t - left * (row - left) // 2)


def normalize_features(features, context_rows):
    """Context-only centering plus one global RMS per block; returns (normalized, scale)."""
    centered = features - features[:context_rows].mean(dim=0, keepdim=True)
    scale = centered[:context_rows].square().mean().sqrt()
    safe = torch.where(scale > 0, scale, torch.ones_like(scale))
    return centered / safe, float(scale)


def draw_pairs(count, markers, rng):
    """Distinct unordered pairs from the full pair space; endpoints may repeat across pairs."""
    chosen = torch.randperm(pair_space(markers), generator=rng)[:count]
    return decode_pairs(chosen, markers)


def sample_mixed_task(config, seed, *, design=None, background=None, gamma=None,
                      return_support=False):
    """Draw one mixed task; the returned object is the existing SyntheticTask structure.

    ``return_support`` is diagnostic-only, like the frozen generator's pair-support option:
    it returns the drawn additive columns and interaction pairs separately, never inside
    metadata, and does not change the random draw sequence.
    """
    rng = torch.Generator(device='cpu').manual_seed(seed)
    n = int(_choice(config['context_sizes'], rng))
    p = int(_choice(config['marker_counts'], rng))
    m, q = int(config['query_size']), int(config['environments'])
    drawn_design = _choice(config.get('designs', ('independent', 'linked')), rng)
    drawn_background = _choice(config.get('backgrounds', BACKGROUNDS), rng)
    drawn_gamma = float(_choice(config.get('gammas', (0., .15, .5, 1.)), rng))
    additive_scale = float(_choice(config.get('additive_scales', (.5, 1., 2.)), rng))
    noise_scale = torch.tensor([float(_choice(config.get('noise_scales', (.5, 1.)), rng))
                                for _ in range(q)])
    pair_count = int(_choice(config.get('pair_counts', (1, 4, 8)), rng))
    active_count = int(_choice(config.get('active_counts', (4, 16)), rng))
    design = drawn_design if design is None else str(design)
    background = drawn_background if background is None else str(background)
    gamma = drawn_gamma if gamma is None else float(gamma)
    if background not in BACKGROUNDS:
        raise ValueError(f'Unknown additive background: {background}')
    if design not in DESIGNS:
        raise ValueError(f'Unknown genotype design: {design}')
    if not 0 <= gamma:
        raise ValueError('Interaction intensity must be non-negative')
    if n < 3 or p < 2 or m < 1 or q < 1 or config.get('block_size', 8) < 1:
        raise ValueError('Invalid task dimensions')
    if pair_space(p) < 1:
        raise ValueError('At least two markers are required to form a pair')
    # A hidden additive background is only defined through the linked construction, which is
    # the single structure choice this generator makes on its own; it never sets gamma.
    auto_linked = background == 'hidden' and design != 'linked'
    if background == 'hidden':
        design = 'linked'

    x, hidden = _genotypes(n + m, p, design, rng, config.get('block_size', 8))
    z = x - x[:n].mean(dim=0, keepdim=True)
    if background == 'dense':
        additive, additive_support = z, torch.arange(p)
    elif background == 'sparse':
        additive_support = torch.randperm(p, generator=rng)[:min(active_count, p)]
        additive = z[:, additive_support]
    else:
        additive_support = torch.randperm(hidden.shape[1], generator=rng)[:min(active_count, hidden.shape[1])]
        additive = hidden[:, additive_support]
    pair_j, pair_k = draw_pairs(min(pair_count, pair_space(p)), p, rng)
    interaction = z[:, pair_j] * z[:, pair_k]

    additive, additive_rms = normalize_features(additive, n)
    interaction, interaction_rms = normalize_features(interaction, n)
    additive_effects, additive_correlation = _effects(additive.shape[1], q, rng)
    interaction_effects, interaction_correlation = _effects(interaction.shape[1], q, rng)
    signal = additive_scale * (additive @ additive_effects) \
        + gamma * (interaction @ interaction_effects)
    observations = signal + torch.randn(n + m, q, generator=rng) * noise_scale

    metadata = {'seed': seed, 'background': background, 'design': design, 'gamma': gamma,
                'additive_scale': additive_scale, 'noise_scale': noise_scale.tolist(),
                'context_size': n, 'query_size': m, 'markers': p, 'environments': q,
                'effects': int(additive.shape[1]), 'interaction_count': int(interaction.shape[1]),
                'auto_linked': bool(auto_linked),
                'context_rms': [float(additive_rms), float(interaction_rms)],
                # Diagnostic only, never a model input: the realized per-environment variance of
                # the clean signal over context rows. The RMS geometry above does not determine it.
                'context_signal_variance': signal[:n].var(dim=0, unbiased=False).tolist()}
    task = SyntheticTask(x[:n].clone(), observations[:n].clone(), x[n:].clone(),
                         signal[n:].clone(), observations[n:].clone(), metadata)
    if return_support:
        return task, {'additive_support': additive_support.numpy(),
                      'pairs': torch.stack([pair_j, pair_k], dim=1).numpy()}
    return task
