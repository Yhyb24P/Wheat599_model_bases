"""Synthetic task stream and shared training loop for the posterior corrector.

Tasks are drawn from the current fitting subset only; query rows enter through
the operator, never through preprocessing or phenotype labels.

Intercept convention. The retained pipeline always fits an unpenalized
per-environment intercept, so tasks live under the corresponding
translation-invariant model with a flat intercept prior. The denoised query target
L y + R W from the generating operator is the teacher of *that* model: it is not
the conditional expectation of a distribution with a known zero intercept, and the
shared-target comparison between the corrector and the direct model is stated with
that scope. The simulator draws no intercept; the profile operator absorbs the
context sample mean.

Scale convention. The generating model has per-environment marginal variance
1 + ridge on average across materials (the additive kernel is normalized to mean
diagonal one, so individual materials differ), while the retained pipeline fits
on phenotypes standardized to unit variance. Each task therefore divides
phenotypes and effects by the fixed c = sqrt(1 + ridge) and uses the
correspondingly rescaled generating covariance, so the denoised query target is
the flat-intercept conditional expectation expressed on the scale the pipeline
uses. The inference operator keeps the grid ridge and the covariance estimated
from those context phenotypes, which is the retained pipeline's own assumption;
the difference between the two operators is recorded, not claimed away.
"""

import math
import time

import numpy as np
import torch

from wheat599.multienv_kernel import task_correlation
from wheat599.posterior_objective import correction_loss, direct_loss, query_targets
from wheat599.posterior_operators import ConditionalLikelihood
from wheat599.posterior_synthetic import prepare_genotypes, sample_task


def covariance_pool(specifications, environments, dtype, device):
    """Fixed generating task covariances covering weak, strong and negative correlation."""
    matrices = []
    for values in specifications:
        matrix = torch.tensor(values, dtype=dtype, device=device)
        if matrix.shape != (environments, environments):
            raise ValueError('Task covariance pool entry has the wrong shape')
        symmetric = (matrix + matrix.T) / 2
        if torch.linalg.eigvalsh(symmetric).min() <= 0:
            raise ValueError('Task covariance pool entry is not positive definite')
        scale = torch.sqrt(torch.diag(symmetric))
        matrices.append(symmetric / torch.outer(scale, scale))
    return matrices


def draw_configuration(cfg, generator):
    """Sample one kernel and regularization configuration together with a context share."""
    def choice(values):
        return values[int(torch.randint(len(values), (1,), generator=generator).item())]
    span = cfg['context_fraction_max'] - cfg['context_fraction_min']
    return {'bandwidth': choice(cfg['bandwidths']), 'fraction': choice(cfg['additive_fractions']),
            'ridge': choice(cfg['ridge']), 'eta': choice(cfg['etas']),
            'covariance_index': int(torch.randint(len(cfg['covariance_pool']), (1,),
                                                  generator=generator).item()),
            'shared_scales': cfg['shared_scales'],
            'context_fraction': cfg['context_fraction_min']
            + span * float(torch.rand((), generator=generator, dtype=torch.float64).item())}


def split_rows(count, configuration, generator):
    """Split the fitting rows into context and query; held-out rows are never touched."""
    size = max(2, int(round(configuration['context_fraction'] * count)))
    order = torch.randperm(count, generator=generator)
    return order[:size].numpy(), order[size:].numpy()


def build_task(markers, configuration, pool, environments, seed, device):
    """Assemble one task: both operators, network inputs and the denoised query target.

    A fixed seed reproduces the task on any device: the row split uses a CPU stream
    and the phenotype draw a device stream, both from that seed.
    """
    rows = torch.as_tensor(markers, dtype=torch.float64, device=device)
    count = rows.shape[0]
    context, query = split_rows(count, configuration, torch.Generator().manual_seed(seed))
    task_generator = torch.Generator(device=device)
    task_generator.manual_seed(seed)
    z, rbf, z_query, rbf_cross = prepare_genotypes(rows[context], configuration['bandwidth'],
                                                  target=rows[query])
    task_true = pool[configuration['covariance_index']]
    if not configuration['shared_scales'] and not torch.equal(task_true, torch.diag(task_true.diagonal())):
        raise ValueError('Independent scales require a diagonal task covariance')
    simulated = sample_task(z, rbf, task_true, configuration['fraction'], configuration['ridge'],
                            task_generator, shared_scales=configuration['shared_scales'])
    scale = math.sqrt(1 + configuration['ridge'])
    response = simulated['y'] / scale
    effects = simulated['effects'] / scale
    generating = ConditionalLikelihood(z, rbf, task_true / scale ** 2, configuration['fraction'],
                                       configuration['ridge'] / scale ** 2)
    identity = torch.eye(environments, dtype=torch.float64, device=device)
    eta = configuration['eta']
    estimated = (1 - eta) * identity + eta * torch.tensor(
        task_correlation(response.detach().cpu().numpy().astype(np.float64)),
        dtype=torch.float64, device=device)
    inference = ConditionalLikelihood(z, rbf, estimated, configuration['fraction'],
                                      configuration['ridge'])
    mean, baseline, target, delta_target = query_targets(inference, generating, response, effects,
                                                         z_query, rbf_cross)
    return {'configuration': configuration, 'context': context, 'query': query,
            'context_rows': rows[context], 'query_rows': rows[query], 'response': response,
            'effects': effects, 'likelihood': inference, 'generating': generating,
            'estimated_covariance': estimated, 'generating_covariance': task_true / scale ** 2,
            'score': inference.sufficient_score(response),
            'diagonal': inference.hessian_diagonal_blocks(batch_size=64),
            'gaussian_mean': mean, 'baseline': baseline, 'target': target,
            'delta_target': delta_target, 'z_query': z_query, 'rbf_cross': rbf_cross,
            'rho': float(simulated['rho']), 'inclusion': float(simulated['inclusion']),
            'covariance_gap': float((estimated - task_true / scale ** 2).norm())}


def _step(model, task, mode):
    """Loss plus the mapped prediction-space output, comparable with its target."""
    if mode == 'correction':
        correction = model(task['likelihood'], task['score'], task['diagonal'], task['gaussian_mean'])
        output = task['likelihood'].prediction_correction(correction, task['z_query'], task['rbf_cross'])
        return correction_loss(task['likelihood'], correction, task['z_query'], task['rbf_cross'],
                               task['delta_target']), output
    prediction = model(task['context_rows'], task['response'], task['query_rows'],
                       task['estimated_covariance'], task['configuration']['fraction'],
                       task['configuration']['ridge'], task['configuration']['bandwidth'])
    return direct_loss(prediction, task['target']), prediction


def fixed_validation_sequence(cfg, count, seed):
    """Pre-draw validation configurations and seeds, disjoint from the training stream."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return [(draw_configuration(cfg, generator),
             int(torch.randint(2 ** 31, (1,), generator=generator).item())) for _ in range(count)]


def score_tasks(model, markers, validation, pool, environments, mode, device):
    """Score the same fixed validation sequence for every checkpoint and both models.

    The correction model is scored against the ideal prediction increment and the
    direct model against the absolute denoised query prediction; the ICM baseline
    power is the same increment norm in both cases.
    """
    errors, powers, gaps = [], [], []
    with torch.no_grad():
        for configuration, seed in validation:
            task = build_task(markers, configuration, pool, environments, seed, device)
            _, output = _step(model, task, mode)
            reference = task['delta_target'] if mode == 'correction' else task['target']
            errors.append(float((output - reference).square().mean()))
            powers.append(float(task['delta_target'].square().mean()))
            gaps.append(task['covariance_gap'])
            del task
    return {'error': float(np.mean(errors)), 'baseline': float(np.mean(powers)),
            'covariance_gap': float(np.mean(gaps))}


def train(model, markers, cfg, device, mode='correction', log=None):
    """Train one model on synthetic tasks from one fitting subset and record cost.

    The checkpoint rule is fixed before any real phenotype is read: among the
    recorded evaluations on a pre-drawn validation task sequence, the lowest
    synthetic error wins; ties use the earliest checkpoint.
    """
    environments = cfg['environments']
    model = model.to(device=device, dtype=torch.float64)
    pool = covariance_pool(cfg['covariance_pool'], environments, torch.float64, device)
    generator = torch.Generator()
    generator.manual_seed(cfg['seed'])
    validation = fixed_validation_sequence(cfg, cfg['evaluation_tasks'], cfg['validation_seed'])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])
    history = []
    best = {'error': math.inf, 'step': 0, 'state': None}
    sync = device.type == 'cuda'
    start = time.perf_counter()
    if sync:
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    prepared = time.perf_counter() - start
    train_seconds = evaluation_seconds = 0.0
    for step in range(cfg['training_steps']):
        block = time.perf_counter()
        configuration = draw_configuration(cfg, generator)
        seed = int(torch.randint(2 ** 31, (1,), generator=generator).item())
        task = build_task(markers, configuration, pool, environments, seed, device)
        loss, _ = _step(model, task, mode)
        train_loss = float(loss.detach())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'])
        optimizer.step()
        del task, loss
        if sync:
            torch.cuda.synchronize(device)
        train_seconds += time.perf_counter() - block
        if (step + 1) % cfg['evaluation_every'] == 0 or step + 1 == cfg['training_steps']:
            block = time.perf_counter()
            record = {'step': step + 1, 'train_loss': train_loss,
                      **score_tasks(model, markers, validation, pool, environments, mode, device)}
            if sync:
                torch.cuda.synchronize(device)
            evaluation_seconds += time.perf_counter() - block
            history.append(record)
            if record['error'] < best['error']:
                best = {'error': record['error'], 'step': step + 1,
                        'state': {key: value.detach().clone()
                                  for key, value in model.state_dict().items()}}
            if log:
                log(record)
    cost = {'training_tasks': cfg['training_steps'], 'evaluation_tasks': cfg['evaluation_tasks'],
            'evaluation_checkpoints': len(history), 'prepared_seconds': prepared,
            'training_seconds': train_seconds, 'evaluation_seconds': evaluation_seconds,
            'wall_seconds': prepared + train_seconds + evaluation_seconds,
            'seconds_per_training_task': train_seconds / max(1, cfg['training_steps']),
            'peak_megabytes': float(torch.cuda.max_memory_allocated(device) / 2 ** 20) if sync else 0.0,
            'mode': mode, 'seed': cfg['seed']}
    return {'history': history, 'cost': cost, 'checkpoint': best}
