"""Joint round-two pilot: synthetic training, checkpoint selection, risk gate, confirmation.

Schema is configs/joint_inference.yaml. Boundaries this module enforces.

Task identity. Every task seed is ``role_base + index * RUN_STRIDE + offset`` where the role
base comes from config (training, selection, gate train, gate selection, confirmation) and the
run identity is an enumerated index, never the model seed times a constant, so a model seed can
never walk a training stream into an evaluation panel. ``panel_ids`` and ``training_ids`` are
exposed and asserted pairwise disjoint by tests.

Pairing. One seed serves a whole cell block: the same seed with different ``background`` or
``gamma`` overrides leaves genotypes, additive direction, interaction direction and noise
bit-identical, so level comparisons are paired by construction.

Labels. Models only ever see context genotypes and noisy context phenotypes. Clean query signals
enter the supervised loss and the synthetic evaluation, never a model input. The additive
lambda_A cross-validation scores held-out noisy context phenotypes and re-runs the context-only
preprocessing inside every fold. Gate features are the frozen checkpoint's context-only
statistics and the gate target is a measured synthetic risk difference, never gamma or support.

Order. Training selection, checkpoint hashes and the gate specification are written to disk before
the confirmation tasks are constructed; confirmation is then evaluated once for every method. A
failed exact certificate or dual-gap bound in the interaction reference is recorded and re-raised;
nothing is filtered. This is a limited pilot: no claim of sufficient optimization or convergence.
"""
import hashlib
import json
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch import nn

from wheat599.joint_evidence import Evidence
from wheat599.joint_network import JointInference
from wheat599.matrix_tasks import prepare_context
from wheat599.mixed_tasks import sample_mixed_task
from wheat599.sparse_interaction import SparseInteraction

RUN_STRIDE = 100_000_000
ROLES = ('training', 'selection', 'gate_train', 'gate_selection', 'confirmation')
PANEL_SPEC = {'selection': 'selection', 'gate_selection': 'gate', 'confirmation': 'confirmation'}


# -- task identity ---------------------------------------------------------
def role_base(config, role):
    if role == 'training':
        return int(config['training']['role_base'])
    if role == 'gate_train':
        return int(config['gate']['train_base'])
    if role == 'gate_selection':
        return int(config['gate']['selection_base'])
    if role not in ('selection', 'confirmation'):
        raise ValueError(f'Unknown panel role: {role}')
    return int(config[role]['role_base'])


def task_seed(config, role, index, offset=0):
    return role_base(config, role) + int(index) * RUN_STRIDE + int(offset)


def role_bases(config, role):
    """How many base seeds a panel uses per background; gate_train is a flat task count."""
    if role == 'gate_train':
        return None
    if role == 'gate_selection':
        return int(config['gate']['selection_bases_per_background'])
    return int(config[role]['bases_per_background'])


def panel_ids(config, role):
    base = role_base(config, role)
    count = (int(config['gate']['train_tasks']) if role == 'gate_train'
             else role_bases(config, role) * len(config['tasks']['backgrounds']))
    return {base + index * RUN_STRIDE for index in range(count)}


def training_ids(config):
    # One stream per model seed: every variant at the same seed consumes the same task ids.
    runs = len(config['seeds'])
    steps = int(config['training']['steps'])
    base = role_base(config, 'training')
    return {base + run * RUN_STRIDE + step for run in range(runs) for step in range(1, steps + 1)}


def role_tasks(config, role):
    tasks = dict(config['tasks'])
    if role in PANEL_SPEC:
        spec = config[PANEL_SPEC[role]]
        for key in ('query_size', 'marker_counts', 'context_sizes'):
            if key in spec:
                tasks[key] = spec[key]
    return tasks


def build_panel(config, role):
    """Cell blocks share a seed; gate_train is a flat enumeration with drawn cells."""
    tasks = role_tasks(config, role)
    panel = []
    if role == 'gate_train':
        for index in range(int(config['gate']['train_tasks'])):
            panel.append(sample_mixed_task(tasks, task_seed(config, role, index)))
        return panel
    index = 0
    for background in config['tasks']['backgrounds']:
        for _ in range(role_bases(config, role)):
            seed = task_seed(config, role, index)
            index += 1
            for gamma in config['tasks']['gammas']:
                panel.append(sample_mixed_task(tasks, seed, background=background,
                                               gamma=float(gamma)))
    return panel


# -- evidence and metrics --------------------------------------------------
def context(task, dtype=torch.float64):
    data, target = task.prepare()
    return (replace(data, x=data.x.to(dtype), y=data.y.to(dtype), query=data.query.to(dtype),
                    y_mean=data.y_mean.to(dtype), y_scale=data.y_scale.to(dtype)), target.to(dtype))


def error(prediction, target):
    return (prediction - target).square().mean()


def mse(prediction, target):
    value = float(error(prediction.detach().cpu(), target))
    if value != value:
        raise FloatingPointError('Nonfinite synthetic MSE')
    return value


def folds_of(rows, count):
    return torch.tensor_split(torch.arange(rows), count)


def inner_context(task, hold, width):
    """Context-only preprocessing refit on the fold-training rows, scoring held-out noisy Y."""
    keep = torch.ones(width, dtype=torch.bool)
    keep[hold] = False
    inner = prepare_context(task.context_x[keep], task.context_y[keep], task.context_x[hold])
    truth = ((torch.as_tensor(task.context_y[hold]).double() - inner.y_mean.double())
             / inner.y_scale.double())
    return inner.x.double(), inner.y.double(), inner.query.double(), truth


def additive_cv(task, config):
    values = [float(value) for value in config['lambda_a_grid']]
    rows = len(task.context_x)
    totals = {value: 0.0 for value in values}
    for hold in folds_of(rows, int(config.get('context_cv_folds', 3))):
        x, y, query, truth = inner_context(task, hold, rows)
        for value in values:
            totals[value] += mse(Evidence(x, y, lambda_a=value).ridge(query), truth) * int(len(hold))
    losses = {value: totals[value] / rows for value in values}
    best = min(values, key=lambda value: (round(losses[value], 12), values.index(value)))
    return best, losses


# -- stage one: training ---------------------------------------------------
def build_network(config, variant, seed):
    """Seeded construction with the dtype fixed before any optimizer or checkpoint transfer."""
    torch.manual_seed(seed)
    return JointInference(variant=variant, **config['network']).double()


def state_hash(state):
    digest = hashlib.sha256()
    for key in sorted(state):
        digest.update(key.encode())
        digest.update(state[key].detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def lambda_stream(config, run_index):
    """One independent fixed stream per run index: lambda_A is a known hyper-parameter per step.

    Two variants at the same model seed share the run index, hence the same task ids and the same
    lambda_A sequence; nothing here reads the generating parameters.
    """
    generator = torch.Generator().manual_seed(int(config.get('lambda_draw_seed', 0)) + int(run_index))
    grid = [float(value) for value in config['lambda_a_grid']]
    return grid, [grid[int(torch.randint(len(grid), (), generator=generator))]
                  for _ in range(int(config['training']['steps']))]


def panel_lambdas(config, panel):
    """Evaluation-time lambda_A: the shared context-only cross-validation, cached per panel task."""
    return [additive_cv(task, config)[0] for task in panel]


def selection_score(config, network, panel, lambdas, device):
    network.eval()
    values = []
    with torch.no_grad():
        for task, lambda_a in zip(panel, lambdas):
            data, target = context(task)
            evidence = Evidence(data.x.to(device), data.y.to(device), lambda_a=lambda_a)
            values.append(mse(network(evidence, data.query.to(device))['prediction'], target))
    network.train()
    return sum(values) / len(values)


def train_run(config, variant, run_index, seed, panel, lambdas, device):
    training = config['training']
    checkpoints = sorted(int(value) for value in training['checkpoints'])
    steps = int(training['steps'])
    network = build_network(config, variant, int(seed)).to(device)
    grid, drawn_lambdas = lambda_stream(config, run_index)
    optimizer = torch.optim.AdamW(network.parameters(), lr=float(training['learning_rate']),
                                  weight_decay=float(training['weight_decay']))
    history, states = [], {}
    for step in range(steps + 1):
        drawn, loss = None, None
        if step:
            task = sample_mixed_task(role_tasks(config, 'training'),
                                     task_seed(config, 'training', run_index, step))
            data, target = context(task)
            evidence = Evidence(data.x.to(device), data.y.to(device),
                                lambda_a=drawn_lambdas[step - 1])
            optimizer.zero_grad(set_to_none=True)
            loss = error(network(evidence, data.query.to(device))['prediction'], target.to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite training loss at step {step}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), float(training['gradient_clip']))
            optimizer.step()
            drawn = task.metadata
        if step in checkpoints:
            states[step] = {key: value.detach().cpu().clone()
                            for key, value in network.state_dict().items()}
            history.append({'step': step, 'selection_mse': selection_score(config, network, panel,
                                                                           lambdas, device),
                            'training_loss': float(loss) if step else None,
                            'drawn_background': drawn['background'] if drawn else None,
                            'drawn_gamma': drawn['gamma'] if drawn else None,
                            'state_sha256': state_hash(states[step])})
    chosen = min(history, key=lambda record: (round(record['selection_mse'], 12), record['step']))
    counts = {value: drawn_lambdas.count(value) for value in grid}
    return {'variant': variant, 'run_index': int(run_index), 'model_seed': int(seed),
            'training_lambda_draws': {'grid': grid, 'counts': counts,
                                      'at_checkpoints': {record['step']: drawn_lambdas[record['step'] - 1]
                                                         if record['step'] else None
                                                         for record in history}},
            'parameters': sum(p.numel() for p in network.parameters()),
            'selected_step': int(chosen['step']), 'selected_mse': chosen['selection_mse'],
            'history': history, 'state': states[chosen['step']]}


def network_at(config, variant, state, device):
    network = build_network(config, variant, int(config['seeds'][0])).to(device)
    network.load_state_dict({key: value.to(device) for key, value in state.items()})
    network.eval()
    return network


# -- stage two: context-only risk gate -------------------------------------
class RiskNet(nn.Module):
    def __init__(self, features, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(features, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, values):
        return self.net(values).squeeze(-1)


@torch.no_grad()
def gate_rows(config, runs, panel, lambdas, device):
    rows = []
    for run in runs:
        network = network_at(config, run['variant'], run['state'], device)
        for task, lambda_a in zip(panel, lambdas):
            data, target = context(task)
            evidence = Evidence(data.x.to(device), data.y.to(device), lambda_a=lambda_a)
            output = network(evidence, data.query.to(device))
            rows.append({'run': f"{run['variant']}_{run['model_seed']}",
                         'features': output['gate_features'].detach().cpu().double(),
                         'joint_mse': mse(output['prediction'], target),
                         'additive_mse': mse(output['additive_prediction'], target)})
    return rows


def gate_predictions(gate, epoch, rows):
    net = RiskNet(int(gate['features']), int(gate['hidden'])).double()
    net.load_state_dict(gate['states'][epoch])
    out = []
    for row in rows:
        value = float(net((row['features'].to(torch.float64) - gate['center']) / gate['scale']).detach())
        flag = 1.0 if value < 0 else 0.0
        out.append({'run': row['run'], 'gate': flag,
                    'mse': (1 - flag) * row['additive_mse'] + flag * row['joint_mse']})
    return out


def fit_gate(config, train_rows, selection_rows):
    spec = config['gate']
    features = torch.stack([row['features'] for row in train_rows]).to(torch.float64)
    center, spread = features.mean(0), features.std(0, unbiased=False)
    spread = torch.where(spread > 0, spread, torch.ones_like(spread))
    target = torch.tensor([row['joint_mse'] - row['additive_mse'] for row in train_rows],
                          dtype=torch.float64)
    torch.manual_seed(int(spec.get('seed', 4242)))
    net = RiskNet(features.shape[1], int(spec['width'])).double()
    optimizer = torch.optim.Adam(net.parameters(), lr=float(spec['learning_rate']))
    epochs = sorted(int(value) for value in spec['checkpoints'])
    states = {}
    for epoch in range(int(spec['epochs']) + 1):
        if epoch:
            optimizer.zero_grad(set_to_none=True)
            loss = (net((features - center) / spread) - target).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite gate loss at epoch {epoch}')
            loss.backward()
            optimizer.step()
        if epoch in epochs:
            states[epoch] = {key: value.detach().cpu().clone() for key, value in net.state_dict().items()}
    gate = {'features': int(features.shape[1]), 'hidden': int(spec['width']), 'center': center,
            'scale': spread, 'states': states, 'seed': int(spec.get('seed', 4242))}
    candidates = [('always_off', float(np.mean([row['additive_mse'] for row in selection_rows])), None),
                  ('always_on', float(np.mean([row['joint_mse'] for row in selection_rows])), None)]
    for epoch in epochs:
        scored = gate_predictions(gate, epoch, selection_rows)
        candidates.append((f'learned_{epoch}', float(np.mean([row['mse'] for row in scored])), epoch))
    best = min(candidates, key=lambda item: (round(item[1], 12),
                                            0 if item[0] == 'always_off' else 1,
                                            item[2] if item[2] is not None else -1))
    return gate, {'kind': best[0], 'gated_mse': best[1], 'epoch': best[2],
                  'ties_prefer': 'always_off',
                  'candidates': [{'kind': name, 'gated_mse': value, 'epoch': epoch}
                                 for name, value, epoch in candidates]}


def apply_gate(gate, choice, row):
    if choice['kind'] == 'always_off':
        return row['additive_mse'], 0.0
    if choice['kind'] == 'always_on':
        return row['joint_mse'], 1.0
    scored = gate_predictions(gate, choice['epoch'], [row])[0]
    return scored['mse'], scored['gate']


# -- interaction reference -------------------------------------------------
def make_solver(config, x, y, lambda_a, device):
    options = dict(config['statistical']['solver'])
    return SparseInteraction(x.numpy(), y.numpy(), lambda_a=float(lambda_a),
                             lambda_e=float(config['statistical']['lambda_e']),
                             device=str(device), dtype=torch.float64, **options)


def pair_indices(evidence, active):
    """Lexicographic upper-triangular position of j<k, without materialising a lookup table."""
    markers = int(evidence.markers)
    positions = [int(left) * (2 * markers - int(left) - 1) // 2 + (int(right) - int(left) - 1)
                 for left, right in np.asarray(active)]
    return torch.tensor(positions, dtype=torch.long)


def solver_result(config, evidence, x, y, query, lambda_a, ratio, device):
    """Fit the exact group lasso, then relax its selected support with the configured weak ridge."""
    candidate = make_solver(config, x, y, lambda_a, device).fit([float(ratio)], query=query.numpy())[0]
    if candidate.status != 'optimal':
        raise RuntimeError(f"Interaction reference certificate failed: {candidate.status} "
                           f"(gap {candidate.dual_gap})")
    original = torch.as_tensor(np.asarray(candidate.prediction), dtype=torch.float64)
    indices = pair_indices(evidence, candidate.active) if len(candidate.active) else torch.zeros(0, dtype=torch.long)
    relaxed = original
    if len(indices) and candidate.active.size:
        precision = torch.full((int(indices.numel()),), float(config['statistical']['relaxed_variance']),
                               dtype=torch.float64)
        additive, coefficients = evidence.readout(indices, precision)
        relaxed = evidence.predict(query, indices, additive, coefficients).cpu()
    return {'original': original, 'relaxed': relaxed, 'pairs': int(indices.numel()),
            'indices': indices}


def reference_cv(config, task, lambda_a, device):
    """Fold-CV the ratio and relaxation, then refit on the full context in pipeline units.

    Both the interaction solver and Evidence receive the same context-only preprocessed rows, so
    the original and relaxed predictions share the scale of the additive fit and of the clean
    target; raw marker and phenotype units never enter a score here.
    """
    data, target = context(task)
    x, y, query = data.x, data.y, data.query
    full = Evidence(x, y, lambda_a=float(lambda_a))
    rows = len(x)
    ratios = [float(value) for value in config['statistical']['ratios']]
    scores = []
    for order, ratio in enumerate(ratios):
        totals = {'original': 0.0, 'relaxed': 0.0}
        pairs = []
        for hold in folds_of(rows, int(config.get('context_cv_folds', 3))):
            inner_x, inner_y, inner_query, truth = inner_context(task, hold, rows)
            outcome = solver_result(config, Evidence(inner_x, inner_y, lambda_a=lambda_a),
                                    inner_x, inner_y, inner_query, lambda_a, ratio, device)
            for kind in totals:
                totals[kind] += mse(outcome[kind], truth) * int(len(hold))
            pairs.append(outcome['pairs'])
        for kind in ('original', 'relaxed'):
            scores.append({'ratio': ratio, 'kind': kind, 'fold_mse': totals[kind] / rows,
                           'order': order, 'fold_pairs': pairs})
    best = min(scores, key=lambda item: (round(item['fold_mse'], 12),
                                        0 if item['kind'] == 'original' else 1, item['order']))
    outcome = solver_result(config, full, x, y, query, lambda_a, best['ratio'], device)
    # Only the selected candidate is scored on the full context; the fold grid above is the
    # genuinely computed per-ratio evidence.
    return {'grid': scores, 'chosen': {key: best[key] for key in ('ratio', 'kind', 'fold_mse')},
            'full_context_pairs': outcome['pairs'],
            'mse': mse(outcome[best['kind']], target),
            'kind_mse': {kind: mse(outcome[kind], target)
                         for kind in ('original', 'relaxed') if outcome['pairs'] or kind == 'original'}}


# -- driver ----------------------------------------------------------------
def record_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def describe(values):
    if not values:
        return None
    tensor = torch.tensor(values, dtype=torch.float64)
    return {'count': int(tensor.numel()), 'mean': float(tensor.mean()),
            'median': float(tensor.median()), 'p90': float(torch.quantile(tensor, .9)),
            'worst': float(tensor.max())}


def summarize(records, training_markers):
    buckets = {}
    for record in records:
        markers = int(record['metadata']['markers'])
        layer = 'train_range' if markers in training_markers else 'beyond_train_range'
        key = f"{layer}_{markers}_{record['background']}_{record['gamma']}"
        bucket = buckets.setdefault(key, {'additive': [], 'neural': {}, 'gated': {}, 'delta': {},
                                          'worse': {}, 'fixed_readout': {}, 'reference': []})
        bucket['additive'].append(record['additive_mse'])
        for name, value in record['neural'].items():
            bucket['neural'].setdefault(name, []).append(value)
            bucket['delta'].setdefault(name, []).append(value - record['additive_mse'])
            bucket['worse'].setdefault(name, []).append(float(value > record['additive_mse']))
        for name, value in record['gated'].items():
            bucket['gated'].setdefault(name, []).append(value)
        for name, value in record['fixed_readout'].items():
            bucket['fixed_readout'].setdefault(name, []).append(value)
        bucket['reference'].append(record['reference']['mse'])
    return {key: {'additive': describe(values['additive']),
                  'neural': {name: describe(list_) for name, list_ in values['neural'].items()},
                  'gated': {name: describe(list_) for name, list_ in values['gated'].items()},
                  'fixed_variance_readout': {name: describe(list_)
                                              for name, list_ in values['fixed_readout'].items()},
                  'delta_vs_additive': {name: describe(list_) for name, list_ in values['delta'].items()},
                  'worse_than_additive_fraction': {name: float(np.mean(list_))
                                                   for name, list_ in values['worse'].items()},
                  'reference': describe(values['reference'])}
            for key, values in buckets.items()}


def run(config, args, runner):
    if args.selection is not None:
        raise ValueError('This synthetic pilot has no selection-input path')
    if config['device'] == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Requested CUDA is unavailable; choose CPU explicitly')
    device = torch.device(config['device'])
    torch.set_num_threads(int(config.get('cpu_threads', 2)))
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(float(config['gpu_memory_fraction']))
    args.out.mkdir(parents=True, exist_ok=False)
    here = Path(__file__).parent
    sources = [here / name for name in ('joint_training.py', 'joint_evidence.py', 'joint_network.py',
                                       'mixed_tasks.py', 'matrix_tasks.py', 'sparse_interaction.py')]
    sources += [Path(args.config), Path(runner)]
    start, failures = time.perf_counter(), []
    (args.out / 'provenance.json').write_text(json.dumps({
        'config': config,
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'source_sha256': {str(path): record_hash(path) for path in sources},
        'torch_version': torch.__version__, 'device': str(device),
        'run_stride': RUN_STRIDE, 'role_bases': {role: role_base(config, role) for role in ROLES},
        'scope': 'Synthetic mixed tasks only; no measured genotype or phenotype read',
        'note': 'Limited pilot; no claim of sufficient optimization, convergence or SOTA'}, indent=2))
    try:
        selection = build_panel(config, 'selection')
        selection_lambdas = panel_lambdas(config, selection)
        runs = []
        for variant in config['variants']:
            for seed_index, seed in enumerate(config['seeds']):
                runs.append(train_run(config, variant, seed_index, seed, selection,
                                      selection_lambdas, device))
        for record in runs:
            path = args.out / f"{record['variant']}_{record['model_seed']}.pt"
            torch.save(record['state'], path)
            record['checkpoint_sha256'] = record_hash(path)
        # Weights live only in the .pt files; the in-memory states stay for the later stages.
        (args.out / 'training.json').write_text(json.dumps(
            [{key: value for key, value in record.items() if key != 'state'} for record in runs],
            indent=2))

        gate_train = build_panel(config, 'gate_train')
        gate_selection = build_panel(config, 'gate_selection')
        gate, choice = fit_gate(config,
                                gate_rows(config, runs, gate_train,
                                          panel_lambdas(config, gate_train), device),
                                gate_rows(config, runs, gate_selection,
                                          panel_lambdas(config, gate_selection), device))
        gate_path = args.out / 'gate.pt'
        torch.save({'center': gate['center'], 'scale': gate['scale'], 'states': gate['states'],
                    'selected': choice, 'features': gate['features'], 'hidden': gate['hidden'],
                    'seed': gate['seed']}, gate_path)
        (args.out / 'gate.json').write_text(json.dumps({
            'selected': choice, 'feature_dim': gate['features'], 'hidden': gate['hidden'],
            'seed': gate['seed'], 'gate_sha256': record_hash(gate_path),
            'gate_file': gate_path.name,
            'scaler': 'mean and population std fitted on gate_train rows only',
            'target': 'joint clean-query MSE minus additive clean-query MSE',
            'features': 'context-only gate_features from the frozen checkpoint',
            'policy': 'g=1 when predicted delta < 0; always_off and always_on are candidates; '
                      'ties prefer always_off; the gate never reads the confirmation panel'}, indent=2))

        confirmation = build_panel(config, 'confirmation')
        (args.out / 'confirmation_panel.json').write_text(json.dumps(
            {'note': 'constructed only after every selection above was written to disk',
             'task_metadata': [task.metadata for task in confirmation]}, indent=2))
        records = []
        for task in confirmation:
            lambda_a, additive_losses = additive_cv(task, config)
            data, target = context(task)
            evidence = Evidence(data.x.to(device), data.y.to(device), lambda_a=lambda_a)
            additive_mse = mse(evidence.ridge(data.query.to(device)), target)
            entry = {'metadata': task.metadata, 'background': task.metadata['background'],
                     'gamma': task.metadata['gamma'], 'lambda_a': lambda_a,
                     'additive_cv_losses': additive_losses, 'additive_mse': additive_mse,
                     'neural': {}, 'gated': {}, 'gate_flag': {}, 'fixed_readout': {}}
            for record in runs:
                name = f"{record['variant']}_{record['model_seed']}"
                network = network_at(config, record['variant'], record['state'], device)
                with torch.no_grad():
                    output = network(evidence, data.query.to(device))
                joint = mse(output['prediction'], target)
                entry['neural'][name] = joint
                value, flag = apply_gate(gate, choice,
                                         {'run': name, 'features': output['gate_features'].detach().cpu().double(),
                                          'joint_mse': joint, 'additive_mse': additive_mse})
                entry['gated'][name], entry['gate_flag'][name] = value, flag
                indices = output['indices']
                if int(indices.numel()):
                    precision = torch.full((int(indices.numel()),),
                                           float(config['statistical']['same_candidate_variance']),
                                           dtype=torch.float64, device=device)
                    additive, coefficients = evidence.readout(indices, precision)
                    entry['fixed_readout'][name] = mse(
                        evidence.predict(data.query.to(device), indices, additive, coefficients), target)
            entry['reference'] = reference_cv(config, task, lambda_a, device)
            records.append(entry)
            # Written per task so a late failure cannot discard completed confirmation records.
            (args.out / 'confirmation.json').write_text(json.dumps(records, indent=2, default=str))
        (args.out / 'summary.json').write_text(json.dumps(
            summarize(records, {int(value) for value in config['tasks']['marker_counts']}), indent=2))
    except Exception as exc:
        failures.append({'reason': repr(exc)})
        raise
    finally:
        (args.out / 'timings.json').write_text(json.dumps(
            {'total_seconds': time.perf_counter() - start}, indent=2))
        (args.out / 'failures.json').write_text(json.dumps(failures, indent=2))
