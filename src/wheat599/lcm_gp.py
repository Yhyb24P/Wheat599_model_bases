"""Locus-Complement Modulation GP: additive path plus self-excluded background modulation.

For centered genotypes z (column means taken from the fitting subset only) and
standardized phenotypes y, the prediction for environment e is

    y_e = mu_e + sum_j beta_je z_j
          + (1/sqrt p) sum_j z_j (u_j * t_e)^T ( h(c_{-j}) - h(0) ),
    c_{-j} = (1/sqrt p) sum_{k != j} v_k z_k.

Because c_{-j} is assembled by subtracting the focus marker's own term from the
global aggregate *before* the nonlinear response, the modulating branch is
independent of z_j given the other markers: d c_{-j} / d z_j = 0.

Objective: mean squared error over materials and environments, plus lambda_A on the
additive coefficients, lambda_N on background/network/readout parameters, and
lambda_S times the row norms of U. The group penalty is applied by a proximal step
after each update, so exact zeros are produced rather than implied by descent alone.
Variants realize the attachment's ablations; the additive_only branch is checked
against the closed-form ridge solution under the same penalty.
"""
import csv
import itertools
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from wheat599.data import load, sha256_of
from wheat599.metrics import mae, pcc, rmse

VARIANTS = ('full', 'additive_only', 'linear_background', 'full_background', 'dense_modulation')


class LCMNetwork(torch.nn.Module):
    """Configurable LCM-GP; forward maps centered genotypes to environment predictions."""

    def __init__(self, markers, environments, rank=8, variant='full'):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError('Unknown LCM variant')
        self.markers, self.environments, self.variant, self.rank = markers, environments, variant, rank
        self.scale = 1.0 / math.sqrt(markers)
        self.interact = variant != 'additive_only'
        self.bias = torch.nn.Parameter(torch.zeros(environments))
        self.coefficients = torch.nn.Parameter(torch.zeros(markers, environments))
        if not self.interact:
            return
        self.background = torch.nn.Parameter(torch.zeros(markers, rank))
        self.modulation = torch.nn.Parameter(torch.zeros(markers, rank))
        self.readout = torch.nn.Parameter(torch.zeros(environments, rank))
        self.linear_map = None
        if variant == 'linear_background':
            self.linear_map = torch.nn.Linear(rank, rank, bias=False)
        else:
            hidden = 2 * rank
            self.layer1 = torch.nn.Linear(rank, hidden)
            self.layer2 = torch.nn.Linear(hidden, hidden)
            self.layer3 = torch.nn.Linear(hidden, rank, bias=False)

    def initialize(self):
        self.coefficients.data.normal_(std=0.01)
        if not self.interact:
            return self
        small = 1.0 / math.sqrt(self.markers)
        for tensor in (self.background, self.modulation, self.readout):
            tensor.data.normal_(std=small)
        if self.linear_map is not None:
            self.linear_map.weight.data.normal_(std=0.01)
        else:
            for layer in (self.layer1, self.layer2, self.layer3):
                layer.weight.data.normal_(std=1.0 / math.sqrt(self.rank))
                if layer.bias is not None:
                    layer.bias.data.zero_()
        return self

    def network(self, values):
        if self.linear_map is not None:
            return self.linear_map(values)
        return self.layer3(F.silu(self.layer2(F.silu(self.layer1(values)))))

    def forward(self, z):
        prediction = z @ self.coefficients + self.bias
        if not self.interact:
            return prediction
        aggregate = (z @ self.background) * self.scale
        own = z.unsqueeze(-1) * self.background.unsqueeze(0) * self.scale
        if self.variant == 'full_background':
            context = aggregate.unsqueeze(1)
        else:
            context = aggregate.unsqueeze(1) - own
        zero = torch.zeros(1, self.rank, dtype=z.dtype, device=z.device)
        response = self.network(context) - self.network(zero)
        weights = torch.einsum('ijr,jr,er->ije', response, self.modulation, self.readout)
        return prediction + self.scale * torch.einsum('ij,ije->ie', z, weights)

    def penalized(self):
        """Every parameter counted in ||theta||^2: background, readout and the response network."""
        if self.linear_map is not None:
            return [self.background, self.readout, self.linear_map.weight]
        items = [self.background, self.readout, self.layer1.weight, self.layer2.weight,
                 self.layer3.weight]
        items.extend(layer.bias for layer in (self.layer1, self.layer2) if layer.bias is not None)
        return items

    def parameter_groups(self, lambda_a, lambda_n):
        """Additive and network parameters carry L2 decay; U is optimized and shrunk by the prox."""
        groups = [{'params': [self.coefficients], 'weight_decay': 2 * lambda_a, 'name': 'additive'},
                  {'params': [self.bias], 'weight_decay': 0.0, 'name': 'intercept'}]
        if self.interact:
            groups.append({'params': self.penalized(), 'weight_decay': 2 * lambda_n, 'name': 'network'})
            groups.append({'params': [self.modulation], 'weight_decay': 0.0, 'name': 'modulation'})
        return groups

    def proximal(self, threshold):
        """Group soft-threshold on rows of U; rows below the threshold become exactly zero."""
        if not self.interact or threshold <= 0:
            return
        with torch.no_grad():
            norm = self.modulation.norm(dim=1)
            factor = (1.0 - threshold / norm.clamp_min(1e-15)).clamp_min(0.0)
            self.modulation.mul_(factor[:, None])

    def zero_rows(self):
        return int((self.modulation.abs().sum(dim=1) == 0).sum()) if self.interact else 0

    def count(self):
        return sum(parameter.numel() for parameter in self.parameters())

    def additive_weight(self):
        return self.coefficients

    def set_additive(self, reference):
        """Start the additive path at the exact ridge optimum for its own penalty."""
        self.coefficients.data.copy_(reference.to(self.coefficients.dtype))
        self.bias.data.zero_()


COMPARATORS = {'additive_mlp': 'AdditiveMLP', 'context_modulation': 'ContextModulation'}
TRAINER_METHODS = ('initialize', 'parameter_groups', 'proximal', 'zero_rows', 'count', 'set_additive')


def comparator_class(name):
    """Resolve a comparator builder from lcm_comparators, which owns those modules."""
    import wheat599.lcm_comparators as comparators
    if name not in COMPARATORS or not hasattr(comparators, COMPARATORS[name]):
        raise AttributeError(f'Comparator {name} is not implemented in lcm_comparators.py')
    builder = getattr(comparators, COMPARATORS[name])
    for method in TRAINER_METHODS:
        if not hasattr(builder, method):
            raise AttributeError(f'Comparator {name} lacks the trainer interface method {method}')
    return builder


def prepare(genotypes, phenotypes, fit, target, device, validation=None, dtype=torch.float32):
    """Centering, standardization and every statistic come from the fitting rows only.

    Training runs in float32, measured about 20 times cheaper than float64 at this
    size; analytic references such as ``ridge_reference`` stay float64 and are compared
    after an explicit cast.
    """
    x = torch.as_tensor(genotypes, dtype=dtype, device=device)
    y = torch.as_tensor(phenotypes, dtype=dtype, device=device)
    fit_rows = torch.as_tensor(np.asarray(fit), device=device)
    target_rows = torch.as_tensor(np.asarray(target), device=device)
    center = x[fit_rows].mean(0)
    fit_y = y[fit_rows]
    mean, deviation = fit_y.mean(0), fit_y.std(0, unbiased=False)
    if torch.any(deviation <= 0):
        raise ValueError('Constant phenotype in fitting subset')
    prepared = {'z_fit': x[fit_rows] - center, 'z_target': x[target_rows] - center,
                'y_fit': (fit_y - mean) / deviation, 'mean': mean, 'scale': deviation}
    if validation is not None:
        rows = torch.as_tensor(np.asarray(validation), device=device)
        prepared['z_validation'] = x[rows] - center
        prepared['y_validation'] = (y[rows] - mean) / deviation
    return prepared


def ridge_reference(z, y, lambda_a):
    """Closed-form ridge, unpenalized intercept, for the additive-only objective.

    The mean-normalized objective is stationary where the sum-of-squares form with
    penalty n * q * lambda_A is, so both describe the same optimum. Solved in float64
    regardless of the training dtype.
    """
    z64, y64 = z.double(), y.double()
    penalty = z64.shape[0] * y64.shape[1] * lambda_a
    gram = z64.T @ z64 + penalty * torch.eye(z64.shape[1], dtype=z64.dtype, device=z64.device)
    return torch.linalg.solve(gram, z64.T @ (y64 - y64.mean(0)))


def fit_model(prepared, cfg, variant, parameters, seed, epochs=None, factory=None):
    """Deterministic full-batch Adam with proximal group steps and validation early stopping.

    ``factory`` allows a comparator module to be trained under the same objective,
    schedule and stopping rule; it must provide parameter_groups, proximal, zero_rows
    and count with the same meaning as LCMNetwork.
    """
    device = prepared['z_fit'].device
    torch.manual_seed(seed)
    builder = factory or (comparator_class(variant) if variant in COMPARATORS else LCMNetwork)
    model = builder(prepared['z_fit'].shape[1], cfg['environments'], rank=parameters['rank'],
                    variant=variant).to(device=device, dtype=prepared['z_fit'].dtype).initialize()
    sparse = parameters['lambda_S'] if variant != 'dense_modulation' else 0.0
    optimizer = torch.optim.Adam(model.parameter_groups(parameters['lambda_A'], parameters['lambda_N']),
                                 lr=cfg['learning_rate'])
    limit = cfg['max_epochs'] if epochs is None else epochs
    z, y = prepared['z_fit'], prepared['y_fit']
    key = ('ridge', parameters['lambda_A'])
    if key not in prepared:
        prepared[key] = ridge_reference(z, y, parameters['lambda_A'])
    reference = prepared[key].to(z.dtype)
    model.set_additive(reference)
    holdout, holdout_y = prepared.get('z_validation'), prepared.get('y_validation')
    best, curve = {'loss': math.inf, 'epoch': 0, 'state': None}, []
    if holdout is not None:
        with torch.no_grad():
            value = float((model(holdout) - holdout_y).square().mean())
        best = {'loss': value, 'epoch': 0,
                'state': {key: item.detach().clone() for key, item in model.state_dict().items()}}
        curve.append({'epoch': 0, 'validation_mse': value})
    threshold = cfg['learning_rate'] * sparse
    if variant == 'additive_only':
        # The additive branch is already at the exact optimum of its own objective, so
        # re-training it only reproduces the closed-form ridge solution.
        model.eval()
        with torch.no_grad():
            trained = float(((model(z) - y).square().mean()
                             + parameters['lambda_A'] * model.additive_weight().square().sum()).cpu())
            ridge_objective = float(((z @ reference - y).square().mean()
                                     + parameters['lambda_A'] * reference.square().sum()).cpu())
            value = float((model(holdout) - holdout_y).square().mean()) if holdout is not None else None
        return model, {'epochs': 0, 'last_epoch': 0, 'validation_mse': value,
                       'zero_modulation_rows': 0, 'parameters': model.count(),
                       'objective': trained, 'ridge_objective': ridge_objective,
                       'objective_excess_over_ridge': trained - ridge_objective,
                       'train_mse': float((model(z) - y).square().mean().detach().cpu()),
                       'curve': [{'epoch': 0, 'validation_mse': value}] if value is not None else []}
    stall = 0
    epoch = 0
    for epoch in range(1, limit + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = (model(z) - y).square().mean()
        loss.backward()
        optimizer.step()
        model.proximal(threshold)
        if holdout is not None and epoch % cfg['evaluate_every'] == 0:
            with torch.no_grad():
                value = float((model(holdout) - holdout_y).square().mean())
            curve.append({'epoch': epoch, 'validation_mse': value})
            if value < best['loss'] - cfg['min_improvement']:
                best = {'loss': value, 'epoch': epoch,
                        'state': {key: item.detach().clone() for key, item in model.state_dict().items()}}
                stall = 0
            else:
                stall += 1
            if stall * cfg['evaluate_every'] >= cfg.get('patience', cfg['max_epochs']):
                break
    if holdout is not None and best['state'] is not None:
        model.load_state_dict(best['state'])
    model.eval()
    with torch.no_grad():
        trained = float(((model(z) - y).square().mean()
                         + parameters['lambda_A'] * model.additive_weight().square().sum()).cpu())
        ridge_objective = float(((z @ reference - y).square().mean()
                                 + parameters['lambda_A'] * reference.square().sum()).cpu())
    record = {'last_epoch': epoch,
              'epochs': best['epoch'] if holdout is not None else limit,
              'validation_mse': best['loss'] if holdout is not None else None,
              'zero_modulation_rows': model.zero_rows(), 'parameters': model.count(),
              'objective': trained, 'ridge_objective': ridge_objective,
              'objective_excess_over_ridge': trained - ridge_objective,
              'train_mse': float((model(z) - y).square().mean().detach().cpu()), 'curve': curve}
    return model, record


def predict(model, prepared, which='target'):
    value = model(prepared[f'z_{which}'])
    return (value * prepared['scale'] + prepared['mean']).detach().cpu().numpy()


def effective_signature(item):
    """Identity of the actual training run, so provably identical searches are not repeated.

    dense_modulation is the full model with the group penalty switched off, so its
    lambda_S entry is fixed at zero; additive_only never trains and depends only on
    the additive penalty.
    """
    variant = 'full' if item['variant'] == 'dense_modulation' else item['variant']
    if variant == 'additive_only':
        return (variant, item['rank'], item['lambda_A'])
    return (variant, item['rank'], item['lambda_A'],
            0.0 if item['variant'] == 'dense_modulation' else item['lambda_S'], item['lambda_N'])


def configurations(cfg):
    names = list(cfg['variants']) + list(cfg.get('comparators', []))
    grid = list(itertools.product(names, cfg['ranks'], cfg['lambda_additive'],
                                  cfg['lambda_sparse'], cfg['lambda_network']))
    return [{'variant': v, 'rank': r, 'lambda_A': a, 'lambda_S': s, 'lambda_N': n}
            for v, r, a, s, n in grid
            if (v == 'additive_only' and s == 0.0 and n == 0.0)
            or (v in COMPARATORS and s == 0.0)
            or (v not in ('additive_only',) and v not in COMPARATORS
                and (v != 'dense_modulation' or s == 0.0))]


def write(path, fields, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def used_before(parameters, grid, index):
    """True when an earlier grid entry produced exactly the training reused here."""
    key = effective_signature(parameters)
    return any(effective_signature(item) == key for item in grid[:index])


def selected_records(scores, epochs, cfg, grid):
    """Per variant, the grid entry with the best mean inner PCC; ties take the earliest.

    The refit epoch count is fixed here, as the median of the inner best epochs of the
    chosen entry, so the later development refit cannot consult the test data at all.
    """
    chosen = {}
    for variant in list(cfg['variants']) + list(cfg.get('comparators', [])):
        entries = [index for index, item in enumerate(grid) if item['variant'] == variant]
        if not entries:
            continue
        means = [float(np.mean(scores[index])) for index in entries]
        ceiling = max(means)
        position = next(i for i, value in enumerate(means) if ceiling - value <= cfg['tie_tolerance'])
        index = entries[position]
        median = int(np.median(epochs[index]))
        chosen[variant] = {'configuration': grid[index], 'inner_pcc': ceiling, 'grid_index': index,
                           'epochs': cfg['max_epochs'] if median < 0 else median,
                           'inner_best_epochs': [int(item) for item in epochs[index]]}
    return chosen


def run(cfg, args, runner):
    """CV1 with inner-fold configuration choice, full development refit and one test pass."""
    data = load(Path(cfg['data']), cfg['data_sha256'])
    folds = json.loads(Path(cfg['splits']).read_text())['folds']
    split_hash = sha256_of(Path(cfg['splits']))
    grid = configurations(cfg)
    for name in cfg.get('comparators', []):
        comparator_class(name)
    frozen = json.loads(args.selection.read_text()) if args.selection else None
    if frozen and (frozen['config'] != cfg or frozen['splits_sha256'] != split_hash
                   or len(frozen['selected']) != len(folds)):
        raise ValueError('Frozen selection does not match configuration or splits')
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'provenance.json').write_text(json.dumps({
        'config': cfg, 'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'splits_sha256': split_hash, 'model_sha256': sha256_of(Path(__file__)),
        'runner_sha256': sha256_of(runner), 'grid_size': len(grid),
        'selection_sha256': sha256_of(args.selection) if args.selection else None,
        'rule': 'per variant, mean inner-fold macro PCC across environments selects the grid entry; '
                'earliest entry on ties; refit runs the median of inner best epochs',
        'protocol': 'CV1 material-level; centering, standardization and ridge initialization from the '
                    'fitting rows only; float32 training with float64 analytic references; '
                    'test evaluated once after the frozen selection',
        'objective': 'mean squared error plus lambda_A on additive coefficients, lambda_N on background, '
                     'response network and readout, and proximal group penalty lambda_S on rows of U'},
        indent=2))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(cfg['gpu_memory_fraction'])
    rows, oof, failures, selected, diagnostics = [], [], [], [], []
    start = time.perf_counter()
    for fold in folds:
        k = fold['fold']
        print(f'fold {k}', flush=True)
        try:
            if frozen is None:
                scores = {index: [] for index in range(len(grid))}
                best_epochs = {index: [] for index in range(len(grid))}
                for j, inner in enumerate(fold['inner']):
                    fit, val = np.array(inner['train_idx']), np.array(inner['val_idx'])
                    if set(fit) & set(val) or set(fit) | set(val) != set(fold['dev_idx']):
                        raise ValueError('Invalid development split')
                    prepared = prepare(data.X, data.Y, fit, val, device, validation=val)
                    trained = {}
                    for index, parameters in enumerate(grid):
                        key = effective_signature(parameters)
                        if key in trained:
                            # Provably the same training (same objective, seed and budget);
                            # both grid entries keep their own records without refitting.
                            record, prediction = trained[key]
                        else:
                            model, record = fit_model(prepared, cfg, parameters['variant'], parameters,
                                                      seed=cfg['seed'] + 100 * k + j)
                            prediction = predict(model, prepared)
                            trained[key] = (record, prediction)
                            del model
                        if not np.isfinite(prediction).all():
                            raise ValueError('Non-finite validation prediction')
                        scores[index].append([float(pcc(data.Y[val, e], prediction[:, e]))
                                              for e in range(data.Y.shape[1])])
                        best_epochs[index].append(record['epochs'])
                        diagnostics.append({'fold': k, 'unit': j, 'variant': parameters['variant'],
                                            'rank': parameters['rank'], 'lambda_A': parameters['lambda_A'],
                                            'lambda_S': parameters['lambda_S'],
                                            'lambda_N': parameters['lambda_N'], 'best_epoch': record['epochs'],
                                            'validation_mse': record['validation_mse'],
                                            'zero_modulation_rows': record['zero_modulation_rows'],
                                            'parameters': record['parameters'],
                                            'objective': record['objective'],
                                            'ridge_objective': record['ridge_objective'],
                                            'objective_excess_over_ridge': record['objective_excess_over_ridge'],
                                            'last_epoch': record['last_epoch'],
                                            'shared_training': used_before(parameters, grid, index)})
                        del record
                    del prepared, trained
                mean_scores = {index: np.mean(np.array(scores[index]), axis=0) for index in scores}
                for index, values in mean_scores.items():
                    if not np.isfinite(values).all():
                        raise ValueError('Non-finite inner PCC')
                chosen = selected_records(mean_scores, best_epochs, cfg, grid)
                for variant, item in chosen.items():
                    rows.append({'stage': 'inner', 'fold': k, 'unit': '', 'variant': variant,
                                 'environment': 'macro', 'epochs': item['epochs'],
                                 'pcc': float(item['inner_pcc']), **item['configuration']})
                for index, parameters in enumerate(grid):
                    for position, env in enumerate(data.environments):
                        rows.append({'stage': 'grid', 'fold': k, 'unit': '', 'variant': parameters['variant'],
                                     'environment': env, 'pcc': float(mean_scores[index][position]),
                                     **parameters})
                selected.append({'fold': k, 'chosen': chosen})
                (args.out / 'selection.json').write_text(json.dumps(
                    {'config': cfg, 'splits_sha256': split_hash, 'selected': selected}, indent=2))
                (args.out / 'diagnostics.json').write_text(json.dumps(diagnostics, indent=2))
            else:
                entry = next(item for item in frozen['selected'] if item['fold'] == k)
                dev, test = np.array(fold['dev_idx']), np.array(fold['test_idx'])
                if set(dev) & set(test) or len(set(dev) | set(test)) != data.n:
                    raise ValueError('Invalid outer split')
                prepared = prepare(data.X, data.Y, dev, test, device)
                for variant, item in entry['chosen'].items():
                    parameters = item['configuration']
                    epochs = item['epochs']
                    model, record = fit_model(prepared, cfg, variant, parameters,
                                              seed=cfg['seed'] + 100 * k + 99, epochs=epochs)
                    prediction = predict(model, prepared)
                    for e, env in enumerate(data.environments):
                        metrics = {'pcc': pcc(data.Y[test, e], prediction[:, e]),
                                   'rmse': rmse(data.Y[test, e], prediction[:, e]),
                                   'mae': mae(data.Y[test, e], prediction[:, e])}
                        if not all(np.isfinite(value) for value in metrics.values()):
                            raise ValueError('Non-finite test metric')
                        rows.append({'stage': 'test', 'fold': k, 'unit': 'development', 'variant': variant,
                                     'environment': env, 'epochs': epochs,
                                     'zero_modulation_rows': record['zero_modulation_rows'],
                                     **metrics, **parameters})
                        oof.extend({'variant': variant, 'environment': env, 'fold': k,
                                    'material_index': int(index), 'y_true': float(data.Y[index, e]),
                                    'y_pred': float(value)}
                                   for index, value in zip(test, prediction[:, e]))
                    del model, record, prediction
        except Exception as exc:
            failures.append({'fold': k, 'reason': repr(exc)})
            print(f'FAILED {k}: {exc}', flush=True)
        finally:
            write(args.out / ('validation.csv' if frozen is None else 'per_fold.csv'),
                  ['stage', 'fold', 'unit', 'variant', 'environment', 'rank', 'lambda_A', 'lambda_S',
                   'lambda_N', 'epochs', 'pcc'] if frozen is None else
                  ['stage', 'fold', 'unit', 'variant', 'environment', 'rank', 'lambda_A', 'lambda_S',
                   'lambda_N', 'epochs', 'zero_modulation_rows', 'pcc', 'rmse', 'mae'], rows)
            write(args.out / 'failures.csv', ['fold', 'reason'], failures)
            (args.out / 'diagnostics.json').write_text(json.dumps(diagnostics, indent=2))
            if frozen:
                write(args.out / 'oof.csv',
                      ['variant', 'environment', 'fold', 'material_index', 'y_true', 'y_pred'], oof)
    if failures:
        raise RuntimeError('Incomplete LCM run; inspect failures')
    if frozen is None:
        print('Development selection complete; no test predictions made.', flush=True)
        return
    names = list(cfg['variants']) + list(cfg.get('comparators', []))
    summary = {'n_models': len(names), 'n_failures': 0, 'variants': {}}
    for variant in names:
        entries = [row for row in rows if row['stage'] == 'test' and row['variant'] == variant]
        if not entries:
            continue
        per = {}
        for env in data.environments:
            values = [row for row in entries if row['environment'] == env]
            per[env] = {metric: float(np.mean([row[metric] for row in values]))
                        for metric in ('pcc', 'rmse', 'mae')}
        summary['variants'][variant] = {
            'per_environment': per,
            'macro': {metric: float(np.mean([per[env][metric] for env in data.environments]))
                      for metric in ('pcc', 'rmse', 'mae')},
            'folds': len({row['fold'] for row in entries}),
            'mean_zero_modulation_rows': float(np.mean([row['zero_modulation_rows'] for row in entries])),
            'configurations': sorted({(row['rank'], row['lambda_A'], row['lambda_S'], row['lambda_N'])
                                      for row in entries})}
    summary['elapsed_seconds'] = time.perf_counter() - start
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)
