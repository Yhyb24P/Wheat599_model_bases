"""Stage-2 group weights learned inside the current fit, with a frozen final Huber.

The module exposes one entry point, ``run_unit``, which receives only fit
markers/phenotypes and validation markers. It returns three arms:

* ``identity``: D_g = I, solved by the generic non-separable kernel Huber;
* ``scalar``: one fit-internal ``beta`` shrinks evidence into positive weights;
* ``network``: a tiny shared MLP maps evidence to positive group weights.

Evidence uses only support-material split halves. Surrogate training uses a
support/query split inside the same fit; query phenotypes enter only the ridge
loss. The final D_g is rebuilt from the full fit evidence, and the final solver
is the exact non-separable kernel Huber with the original fixed thresholds.
No validation phenotype is accepted by this module.
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from scipy.optimize import minimize

from wheat599.grouped_kernel import fit_group_assignment

EVIDENCE_SEEDS = (11, 22, 33)
EPISODE_SEEDS = (701, 702)
TRAIN_STEPS = 100
LEARNING_RATE = 0.01
TORCH_SEED = 42
GROUPS = 64
TAU = 0.25
RIDGE_FLOOR = 1e-12


# -- small numeric helpers -------------------------------------------------
def _matrix(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError('Expected a finite matrix')
    return x


def _squared(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    distance = (np.square(left).sum(1)[:, None] + np.square(right).sum(1)[None, :]
                - 2.0 * left @ right.T)
    return np.maximum(distance, 0.0)


def _off_diagonal(distance):
    return distance[np.triu_indices(len(distance), 1)]


def _correlation(left, right):
    left = np.asarray(left, dtype=np.float64).ravel()
    right = np.asarray(right, dtype=np.float64).ravel()
    left = left - left.mean()
    right = right - right.mean()
    denominator = np.sqrt((left @ left) * (right @ right))
    if denominator <= 0 or not np.isfinite(denominator):
        return 0.0
    return float((left @ right) / denominator)


class Context:
    def __init__(self, x, bandwidth):
        x = _matrix(x)
        if len(x) < 4:
            raise ValueError('At least four fit materials are required')
        self.x = x
        self.bandwidth = float(bandwidth)
        if not np.isfinite(self.bandwidth) or self.bandwidth <= 0:
            raise ValueError('bandwidth must be finite and positive')
        grouping = fit_group_assignment(x, groups=GROUPS)
        self.assignment = np.asarray(grouping['assignment'], dtype=np.int64)
        self.anchors = np.asarray(grouping['anchors'], dtype=np.int64)
        self.groups = int(grouping['groups'])
        if self.groups < 1:
            raise ValueError('No nonconstant marker groups')
        self.constant = x.std(0) <= 0
        variance = np.square(x.std(0))
        weights = np.zeros(self.groups, dtype=np.float64)
        for position in range(self.groups):
            members = (self.assignment == position) & ~self.constant
            weights[position] = float(variance[members].sum())
        total = float(weights.sum())
        if total <= 0:
            raise ValueError('All fit marker variance is zero')
        self.weights = weights / total
        self.sizes = np.asarray([((self.assignment == position) & ~self.constant).sum()
                                 for position in range(self.groups)], dtype=np.int64)
        distance = _squared(x, x)
        scale = float(np.median(_off_diagonal(distance)))
        if scale <= 0:
            raise ValueError('Zero global distance median')
        self.d0 = scale
        self.group_d0 = self.weights * self.d0
        self.positive = self.weights > 0
        self.tau = TAU

    def components(self, x_query):
        """Additive, global RBF, and individual group kernels on fit/query."""
        query = _matrix(x_query)
        if query.shape[1] != self.x.shape[1]:
            raise ValueError('Query marker count differs from fit')
        markers = self.x.shape[1]
        mean = self.x.mean(0)
        centered = self.x - mean
        centered_query = query - mean
        additive = centered @ centered.T / markers
        additive_cross = centered_query @ centered.T / markers
        additive_scale = float(np.diagonal(additive).mean())
        if additive_scale <= 0:
            raise ValueError('Zero additive kernel scale')
        additive = additive / additive_scale
        additive_cross = additive_cross / additive_scale
        fit_distance = _squared(self.x, self.x)
        cross_distance = _squared(query, self.x)
        rbf = np.exp(-self.bandwidth * fit_distance / self.d0)
        rbf_cross = np.exp(-self.bandwidth * cross_distance / self.d0)
        group_fit, group_cross = [], []
        for position in range(self.groups):
            if not self.positive[position]:
                group_fit.append(None)
                group_cross.append(None)
                continue
            members = np.flatnonzero((self.assignment == position) & ~self.constant)
            if members.size == 0:
                raise ValueError('Positive group weight without members')
            fit_block = np.exp(-self.bandwidth
                               * _squared(self.x[:, members], self.x[:, members])
                               / self.group_d0[position])
            cross_block = np.exp(-self.bandwidth
                                 * _squared(query[:, members], self.x[:, members])
                                 / self.group_d0[position])
            group_fit.append(fit_block)
            group_cross.append(cross_block)
        return {'additive': additive, 'additive_cross': additive_cross,
                'rbf': rbf, 'rbf_cross': rbf_cross,
                'group_fit': group_fit, 'group_cross': group_cross}

    def evidence(self, y):
        """Support-only split-half correlation difference, averaged over seeds."""
        y = _matrix(y)
        if len(y) != len(self.x) or not np.isfinite(y).all():
            raise ValueError('Phenotypes must align with fit rows')
        tasks = y.shape[1]
        mean, scale = y.mean(0), y.std(0)
        if np.any(scale <= 0):
            raise ValueError('Constant phenotype')
        standardized = (y - mean) / scale
        evidence = np.zeros((self.groups, tasks), dtype=np.float64)
        for seed in EVIDENCE_SEEDS:
            order = np.random.default_rng(int(seed)).permutation(len(self.x))
            first, second = order[:len(order) // 2], order[len(order) // 2:]
            global_cross = np.exp(
                -self.bandwidth * _squared(self.x[second], self.x[first]) / self.d0)
            group_cross = []
            for position in range(self.groups):
                if not self.positive[position]:
                    group_cross.append(None)
                    continue
                members = np.flatnonzero(
                    (self.assignment == position) & ~self.constant)
                group_cross.append(np.exp(
                    -self.bandwidth * _squared(self.x[second][:, members],
                                               self.x[first][:, members])
                    / self.group_d0[position]))
            for task in range(tasks):
                phenotype = -0.5 * (
                    standardized[second, task][:, None]
                    - standardized[first, task][None, :]) ** 2
                global_corr = _correlation(global_cross, phenotype)
                for position in range(self.groups):
                    if not self.positive[position]:
                        continue
                    evidence[position, task] += (
                        _correlation(group_cross[position], phenotype) - global_corr)
        evidence /= len(EVIDENCE_SEEDS)
        if not np.isfinite(evidence).all():
            raise FloatingPointError('Nonfinite evidence')
        z = np.zeros_like(evidence)
        for task in range(tasks):
            values = evidence[self.positive, task]
            deviation = float(values.std())
            if deviation > 0:
                z[self.positive, task] = (values - values.mean()) / deviation
        return evidence, z


# -- weights ---------------------------------------------------------------
def _softmax_weights(logits, positive):
    logits = logits.masked_fill(~positive[:, None], float('-inf'))
    weights = torch.softmax(logits, dim=0)
    return torch.where(positive[:, None], weights, torch.zeros_like(weights))


def _scalar_weights(beta, z, log_omega, positive):
    r = torch.tanh(beta * z)
    return _softmax_weights(log_omega[:, None] + r, positive)


class SharedWeightNet(nn.Module):
    def __init__(self, tasks):
        super().__init__()
        width = 2 + int(tasks)
        self.first = nn.Linear(width, 8, dtype=torch.float64)
        self.second = nn.Linear(8, 1, dtype=torch.float64)
        nn.init.zeros_(self.second.weight)
        nn.init.zeros_(self.second.bias)

    def forward(self, z, log_omega, one_hot, positive):
        groups, tasks = z.shape
        env = one_hot.unsqueeze(0).expand(groups, -1, -1)
        features = torch.cat([
            z.unsqueeze(-1),
            log_omega[:, None, None].expand(groups, tasks, 1),
            env], dim=-1)
        logits = torch.tanh(self.first(features))
        logits = self.second(logits).squeeze(-1)
        return _softmax_weights(log_omega[:, None] + torch.tanh(logits), positive)


def _weights_to_d(weights, omega, positive):
    d = np.zeros_like(weights)
    for position in range(len(omega)):
        if positive[position]:
            d[position] = np.sqrt(weights[position] / omega[position])
    return d


def _json_weights(context, weights, d):
    return {
        'assignment': [int(value) for value in context.assignment],
        'anchors': [int(value) for value in context.anchors],
        'sizes': [int(value) for value in context.sizes],
        'omega': [float(value) for value in context.weights],
        'w': [[float(value) for value in row] for row in weights],
        'D': [[float(value) for value in row] for row in d],
        'd0': float(context.d0),
        'group_d0': [float(value) for value in context.group_d0],
        'constant_markers': np.flatnonzero(context.constant).astype(int).tolist(),
        'tau': float(TAU),
    }


# -- surrogate ridge -------------------------------------------------------
def _episode(x_fit, y_fit, seed, fraction, bandwidth):
    order = np.random.default_rng(int(seed)).permutation(len(x_fit))
    split = max(2, (2 * len(order)) // 3)
    support_ids, query_ids = order[:split], order[split:]
    support = x_fit[support_ids]
    query = x_fit[query_ids]
    support_y = y_fit[support_ids]
    query_y = y_fit[query_ids]
    context = Context(support, bandwidth)
    _, z = context.evidence(support_y)
    components = context.components(query)
    mean, scale = support_y.mean(0), support_y.std(0)
    if np.any(scale <= 0):
        raise ValueError('Constant support phenotype')
    support_y = (support_y - mean) / scale
    query_y = (query_y - mean) / scale
    tasks = y_fit.shape[1]
    one_hot = torch.eye(tasks, dtype=torch.float64)
    support_kernel = (fraction * components['additive']
                      + (1 - fraction) * (1 - TAU) * components['rbf'])
    cross_kernel = (fraction * components['additive_cross']
                    + (1 - fraction) * (1 - TAU) * components['rbf_cross'])
    group_support = [torch.tensor(block, dtype=torch.float64) if block is not None
                     else None for block in components['group_fit']]
    group_cross = [torch.tensor(block, dtype=torch.float64) if block is not None
                   else None for block in components['group_cross']]
    return {
        'context': context,
        'group_coefficient': (1 - fraction) * TAU,
        'z': torch.tensor(z, dtype=torch.float64),
        'log_omega': torch.where(
            torch.tensor(context.positive, dtype=torch.bool),
            torch.log(torch.tensor(context.weights, dtype=torch.float64)),
            torch.zeros(context.groups, dtype=torch.float64)),
        'positive': torch.tensor(context.positive, dtype=torch.bool),
        'one_hot': one_hot,
        'support_kernel': torch.tensor(support_kernel, dtype=torch.float64),
        'cross_kernel': torch.tensor(cross_kernel, dtype=torch.float64),
        'group_support': group_support,
        'group_cross': group_cross,
        'support_y': torch.tensor(support_y, dtype=torch.float64),
        'query_y': torch.tensor(query_y, dtype=torch.float64),
    }


def _episode_loss(episode, weights, ridge):
    tasks = weights.shape[1]
    losses = []
    for task in range(tasks):
        train = episode['support_kernel'].clone()
        cross = episode['cross_kernel'].clone()
        for position, block in enumerate(episode['group_support']):
            if block is None:
                continue
            coefficient = episode['group_coefficient'] * weights[position, task]
            train = train + coefficient * block
            cross = cross + coefficient * episode['group_cross'][position]
        device = train.device
        matrix = torch.zeros((len(train) + 1, len(train) + 1), dtype=torch.float64,
                             device=device)
        matrix[:len(train), :len(train)] = train + ridge * torch.eye(
            len(train), dtype=torch.float64, device=device)
        matrix[:len(train), len(train)] = 1.0
        matrix[len(train), :len(train)] = 1.0
        right = torch.cat([episode['support_y'][:, task],
                           torch.zeros(1, dtype=torch.float64, device=device)])
        solution = torch.linalg.solve(matrix, right)
        prediction = cross @ solution[:len(train)] + solution[len(train)]
        losses.append(F.mse_loss(prediction, episode['query_y'][:, task]))
    return torch.stack(losses).mean()


def _train_scalar(episodes, ridge, device):
    beta = torch.tensor(0.0, dtype=torch.float64, requires_grad=True, device=device)
    optimizer = torch.optim.Adam([beta], lr=LEARNING_RATE)
    losses = []
    for _ in range(TRAIN_STEPS):
        optimizer.zero_grad(set_to_none=True)
        episode_losses = []
        for episode in episodes:
            weights = _scalar_weights(beta, episode['z'], episode['log_omega'],
                                      episode['positive'])
            episode_losses.append(_episode_loss(episode, weights, ridge))
        loss = torch.stack(episode_losses).mean()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            beta.clamp_(0.0, 2.0)
        losses.append(float(loss.detach()))
        if not np.isfinite(losses[-1]):
            raise FloatingPointError('Nonfinite scalar surrogate loss')
    return beta, losses


def _train_network(episodes, ridge, tasks, device):
    torch.manual_seed(TORCH_SEED)
    network = SharedWeightNet(tasks).to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=LEARNING_RATE)
    losses = []
    for _ in range(TRAIN_STEPS):
        optimizer.zero_grad(set_to_none=True)
        episode_losses = []
        for episode in episodes:
            weights = network(episode['z'], episode['log_omega'],
                              episode['one_hot'], episode['positive'])
            episode_losses.append(_episode_loss(episode, weights, ridge))
        loss = torch.stack(episode_losses).mean()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        if not np.isfinite(losses[-1]):
            raise FloatingPointError('Nonfinite network surrogate loss')
    return network, losses


# -- exact non-separable kernel Huber -------------------------------------
def _task_signal(y_fit, eta):
    mean, scale = y_fit.mean(0), y_fit.std(0)
    if np.any(scale <= 0):
        raise ValueError('Constant fit phenotype')
    normalized = (y_fit - mean) / scale
    norm = np.linalg.norm(normalized, axis=0)
    if np.any(norm <= 0):
        raise ValueError('Zero phenotype norm')
    z = normalized / norm
    correlation = z.T @ z
    signal = (1.0 - float(eta)) * np.eye(y_fit.shape[1]) + float(eta) * correlation
    return mean, scale, normalized, signal


def _full_kernels(context, components, weights, signal, fraction):
    markers, tasks = context.x.shape[1], signal.shape[0]
    train = np.empty((len(context.x) * tasks, len(context.x) * tasks), dtype=np.float64)
    cross = np.empty((len(components['rbf_cross']) * tasks,
                      len(context.x) * tasks), dtype=np.float64)
    shared = fraction * components['additive'] \
        + (1 - fraction) * (1 - TAU) * components['rbf']
    shared_cross = fraction * components['additive_cross'] \
        + (1 - fraction) * (1 - TAU) * components['rbf_cross']
    coefficient = (1 - fraction) * TAU
    for first in range(tasks):
        for second in range(tasks):
            block = shared.copy()
            cross_block = shared_cross.copy()
            for position in range(context.groups):
                if not context.positive[position]:
                    continue
                group_scale = np.sqrt(weights[position, first]
                                      * weights[position, second])
                if group_scale == 0:
                    continue
                block += coefficient * group_scale * components['group_fit'][position]
                cross_block += coefficient * group_scale \
                    * components['group_cross'][position]
            train[first * len(context.x):(first + 1) * len(context.x),
                  second * len(context.x):(second + 1) * len(context.x)] = \
                signal[first, second] * block
            cross[first * len(components['rbf_cross']):
                  (first + 1) * len(components['rbf_cross']),
                  second * len(context.x):(second + 1) * len(context.x)] = \
                signal[first, second] * cross_block
    return train, cross


def _generic_huber(kernel, cross, y_fit, x_val, ridge, thresholds, solver):
    mean, scale, normalized, _ = _task_signal(y_fit, 0.0)
    # signal was already folded into kernel/cross by the caller.
    eigenvalues, vectors = np.linalg.eigh(kernel)
    tolerance = 1e-8 * max(1.0, float(np.max(np.abs(eigenvalues))))
    if float(eigenvalues.min()) < -tolerance:
        raise ValueError('Non-PSD full kernel')
    eigenvalues = np.maximum(eigenvalues, 0.0)
    keep = eigenvalues > 1e-12
    root = np.sqrt(eigenvalues[keep])
    design = vectors[:, keep] * root
    target = cross @ vectors[:, keep] / root
    targets = normalized.T.reshape(-1)
    thresholds = np.asarray(thresholds, dtype=np.float64)
    tasks = y_fit.shape[1]
    if thresholds.shape != (tasks,):
        raise ValueError('Thresholds must have one value per task')
    threshold_vector = np.repeat(thresholds, len(y_fit))
    ridge = float(ridge)
    if not np.isfinite(ridge) or ridge <= 0:
        raise ValueError('ridge must be finite and positive')
    precondition = np.sqrt(eigenvalues[keep] + ridge)
    intercept_scale = np.sqrt(len(y_fit))

    def objective(values):
        weights = values[:-tasks] / precondition
        intercept = values[-tasks:] / intercept_scale
        prediction = design @ weights + np.repeat(intercept, len(y_fit))
        residual = prediction - targets
        score = np.clip(residual, -threshold_vector, threshold_vector)
        magnitude = np.abs(residual)
        loss = (np.where(magnitude <= threshold_vector, 0.5 * residual ** 2,
                         threshold_vector * (magnitude - 0.5 * threshold_vector)).sum()
                + 0.5 * ridge * float(weights @ weights)) / targets.size
        weight_gradient = (design.T @ score + ridge * weights) / targets.size
        gradient = np.r_[weight_gradient / precondition,
                         score.reshape(tasks, len(y_fit)).sum(axis=1)
                         / targets.size / intercept_scale]
        return loss, gradient

    # GLS/ridge initialization with an unpenalized environment intercept.
    gram = np.diag(eigenvalues[keep]) + ridge * np.eye(len(precondition))
    sums = np.stack([design[task * len(y_fit):(task + 1) * len(y_fit)].sum(0)
                     for task in range(tasks)], axis=1)
    system = np.zeros((len(precondition) + tasks, len(precondition) + tasks),
                      dtype=np.float64)
    system[:len(precondition), :len(precondition)] = gram
    system[:len(precondition), len(precondition):] = sums
    system[len(precondition):, :len(precondition)] = sums.T
    system[len(precondition):, len(precondition):] = len(y_fit) * np.eye(tasks)
    right = np.r_[design.T @ targets,
                  np.asarray([normalized[:, task].sum() for task in range(tasks)])]
    solution = np.linalg.solve(system, right)
    initial_weights = solution[:len(precondition)]
    initial_intercept = solution[len(precondition):]
    initial_prediction = design @ initial_weights \
        + np.repeat(initial_intercept, len(y_fit))
    initial_residual = initial_prediction - targets
    initial_residual = initial_residual.reshape(tasks, len(y_fit)).T
    mad = np.maximum(1.4826 * np.median(
        np.abs(initial_residual - np.median(initial_residual, axis=0)), axis=0), 1e-6)
    initial = np.r_[initial_weights * precondition,
                    initial_intercept * intercept_scale]
    initial_loss = objective(initial)[0]
    result = minimize(objective, initial, jac=True, method='L-BFGS-B',
                      options={'maxiter': int(solver['max_iterations']), 'maxls': 40,
                               'ftol': float(solver['function_tolerance']),
                               'gtol': float(solver['gradient_tolerance'])})
    raw_gradient = np.r_[result.jac[:-tasks] * precondition,
                         result.jac[-tasks:] * intercept_scale]
    gradient_inf = float(np.max(np.abs(raw_gradient)))
    if not np.isfinite(result.fun) or not np.isfinite(result.x).all() \
            or gradient_inf > float(solver['accepted_gradient']):
        raise ValueError('Full-kernel Huber optimization failed fixed accuracy')
    weights = result.x[:-tasks] / precondition
    intercept = result.x[-tasks:] / intercept_scale
    prediction = target @ weights + np.repeat(intercept, len(x_val))
    prediction = prediction.reshape(tasks, len(x_val)).T * scale + mean
    residual = (design @ weights + np.repeat(intercept, len(y_fit))
                - targets).reshape(tasks, len(y_fit)).T
    diagnostics = {
        'iterations': int(result.nit), 'optimizer_success': bool(result.success),
        'message': str(result.message), 'objective': float(result.fun),
        'initial_objective': float(initial_loss), 'gradient_inf': gradient_inf,
        'residual_mad_scale': mad.tolist(), 'thresholds': thresholds.tolist(),
        'thresholds_fixed': True,
        'clipped_fraction': float(np.mean(np.abs(residual) > thresholds)),
        'ridge': ridge, 'full_kernel_rows': int(kernel.shape[0]),
    }
    if not np.isfinite(prediction).all():
        raise FloatingPointError('Nonfinite full-kernel prediction')
    return prediction, diagnostics


def _solve_arm(context, query, weights, y_fit, ridge, thresholds, eta, fraction, solver):
    components = context.components(query)
    mean, scale, normalized, signal = _task_signal(y_fit, eta)
    kernel, cross = _full_kernels(context, components, weights, signal, fraction)
    prediction, diagnostics = _generic_huber(
        kernel, cross, y_fit, query, ridge, thresholds, solver)
    return prediction, diagnostics


# -- entry point -----------------------------------------------------------
def run_unit(x_fit, y_fit, x_val, base, choice, thresholds, solver, device='cpu'):
    x_fit = _matrix(x_fit)
    y_fit = _matrix(y_fit)
    x_val = _matrix(x_val)
    if len(x_fit) != len(y_fit) or x_fit.shape[1] != x_val.shape[1]:
        raise ValueError('Fit/validation shapes are inconsistent')
    tasks = y_fit.shape[1]
    thresholds = np.asarray(thresholds, dtype=np.float64)
    if thresholds.shape != (tasks,):
        raise ValueError('thresholds must have one value per environment')
    if not np.isfinite(thresholds).all() or np.any(thresholds <= 0):
        raise ValueError('thresholds must be finite and positive')
    ridge = float(base['ridge'][0]) * float(choice['multiplier'])
    fraction = float(base['fraction'])
    context = Context(x_fit, base['bandwidth'])
    if context.groups < 1:
        raise ValueError('No marker groups')
    _, z = context.evidence(y_fit)
    z_tensor = torch.tensor(z, dtype=torch.float64, device=device)
    omega_tensor = torch.tensor(context.weights, dtype=torch.float64, device=device)
    positive = torch.tensor(context.positive, dtype=torch.bool, device=device)
    log_omega = torch.where(positive, torch.log(omega_tensor),
                            torch.zeros_like(omega_tensor))
    episodes = [_episode(x_fit, y_fit, seed, fraction, base['bandwidth'])
                for seed in EPISODE_SEEDS]
    for episode in episodes:
        episode['z'] = episode['z'].to(device)
        episode['log_omega'] = episode['log_omega'].to(device)
        episode['positive'] = episode['positive'].to(device)
        episode['one_hot'] = episode['one_hot'].to(device)
        episode['support_kernel'] = episode['support_kernel'].to(device)
        episode['cross_kernel'] = episode['cross_kernel'].to(device)
        episode['support_y'] = episode['support_y'].to(device)
        episode['query_y'] = episode['query_y'].to(device)
        episode['group_support'] = [None if block is None else block.to(device)
                                    for block in episode['group_support']]
        episode['group_cross'] = [None if block is None else block.to(device)
                                  for block in episode['group_cross']]
    scalar_beta, scalar_losses = _train_scalar(episodes, ridge, device)
    network, network_losses = _train_network(episodes, ridge, tasks, device)
    identity_weights = np.repeat(context.weights[:, None], tasks, axis=1)
    scalar_weights = _scalar_weights(
        scalar_beta.detach().cpu(), z_tensor.cpu(), log_omega.cpu(), positive.cpu()
    ).numpy()
    with torch.no_grad():
        network_weights = network(
            z_tensor, log_omega, torch.eye(tasks, dtype=torch.float64, device=device),
            positive).cpu().numpy()
    for table in (identity_weights, scalar_weights, network_weights):
        if not np.isfinite(table).all():
            raise FloatingPointError('Nonfinite group weights')
    identity_d = _weights_to_d(identity_weights, context.weights, context.positive)
    scalar_d = _weights_to_d(scalar_weights, context.weights, context.positive)
    network_d = _weights_to_d(network_weights, context.weights, context.positive)
    arms = {}
    for name, weights, d, training in (
            ('identity', identity_weights, identity_d, {'steps': 0}),
            ('scalar', scalar_weights, scalar_d,
             {'steps': TRAIN_STEPS, 'beta': float(scalar_beta.detach()),
              'losses': scalar_losses}),
            ('network', network_weights, network_d,
             {'steps': TRAIN_STEPS, 'losses': network_losses,
              'parameters': int(sum(p.numel() for p in network.parameters())),
              'seed': TORCH_SEED})):
        prediction, diagnostics = _solve_arm(
            context, x_val, weights, y_fit, ridge, thresholds,
            float(choice['eta']), fraction, solver)
        arms[name] = {
            'prediction': np.asarray(prediction, dtype=np.float64),
            'diagnostics': diagnostics,
            'weights': _json_weights(context, weights, d),
            'training': training,
        }
    return {
        'arms': arms,
        'metadata': {
            'evidence_seeds': list(EVIDENCE_SEEDS),
            'episode_seeds': list(EPISODE_SEEDS),
            'groups': int(context.groups), 'tau': float(TAU),
            'ridge': ridge, 'fraction': fraction,
            'eta': float(choice['eta']), 'thresholds': thresholds.tolist(),
            'torch_seed': TORCH_SEED,
            'identity_d_in_range': bool(
                np.all((identity_d[context.positive] >= np.exp(-1) - 1e-12)
                       & (identity_d[context.positive] <= np.exp(1) + 1e-12))),
        },
    }
