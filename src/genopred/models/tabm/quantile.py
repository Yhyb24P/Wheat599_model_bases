"""Quantile models, training objectives and fit-level training methods."""
import math
import time

import numpy as np
import torch
from torch import nn

from genopred.evaluation.metrics import pcc
from genopred.models.tabm.backbone import GenomicTabM, PreLNCrossAttention


def quantile_loss(output, target, quantiles):
    """Sum over quantiles; average over materials and independent members."""
    error = target[:, None, :] - output
    return torch.maximum(quantiles * error, (quantiles - 1) * error).sum(-1).mean()


def build_model(x_fit, y_fit, config, seed, settings, device):
    import tabm
    torch.manual_seed(int(seed))
    model = GenomicTabM(x_fit, y_fit[:, None], variant='raw',
        **{k: config[k] for k in ('n_blocks', 'd_block', 'dropout', 'k')})
    # Rebuild only the official backbone so its output has one coordinate per quantile.
    model.backbone = tabm.TabM.make(n_num_features=x_fit.shape[1], d_out=settings['n_quantiles'],
        **{k: config[k] for k in ('n_blocks', 'd_block', 'dropout', 'k')})
    return model.to(device)


def update_epoch(model, x, y, q, opt, generator, spec, device, *, loss_fn=None):
    model.train()
    optimizers=opt if isinstance(opt,list) else [opt]
    order = torch.randperm(len(x), generator=generator).to(device)
    for batch in order.split(spec['batch_size']):
        for optimizer in optimizers:optimizer.zero_grad(set_to_none=True)
        if loss_fn is not None:
            if torch.device(device).type == 'cuda':
                torch.compiler.cudagraph_mark_step_begin()
            loss = loss_fn(model, x[batch], y[batch], q, spec['basis_penalty'])
        else:
            output = model(x[batch])
            if spec.get('objective', 'pinball') == 'mse':
                if output.shape != (len(batch), 1, 1):
                    raise ValueError('Point-MSE model must return [batch,1,1]')
                loss = torch.nn.functional.mse_loss(output[:, 0, :], y[batch])
            elif spec.get('objective') == 'shape':
                output, penalty = model.shape_transform(output)
                loss = (2 / len(q)) * quantile_loss(output, y[batch], q) + spec['shape_penalty'] * penalty
            elif spec.get('objective') == 'midpoint':
                loss = (2 / len(q)) * quantile_loss(output, y[batch], q)
            else:
                loss = quantile_loss(output, y[batch], q)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite quantile loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), spec['gradient_clip'])
        for optimizer in optimizers:optimizer.step()


def train(x_fit, y_fit, x_val, y_val, config, seed, index, settings, device):
    start=time.perf_counter()
    variant=settings.get('optimizer_variant','adam')
    objective = settings['training'].get('objective', 'pinball')
    if objective not in ('pinball', 'mse', 'midpoint', 'shape'):
        raise ValueError('Unknown training objective')
    if variant.startswith('shape_') and objective != ('midpoint' if variant == 'shape_midpoint' else 'shape'):
        raise ValueError('Shape model and loss do not match')
    if variant in ('shared_basis', 'independent_basis') and objective != 'pinball':
        raise ValueError('Connection-basis models require the original pinball objective')
    if variant in ('treewidth_ising', 'treewidth_independent'):
        from genopred.models.structured.treewidth_ising import TreewidthIsing
        if objective != 'mse':
            raise ValueError('Treewidth Ising comparisons require point MSE')
        torch.manual_seed(int(seed))
        model = TreewidthIsing(x_fit, y_fit, spins=config['spins'],
                              coupled=variant == 'treewidth_ising').to(device)
    elif variant in ('ising', 'ising_independent', 'ising_mlp'):
        from genopred.models.structured.supervised_ising import SupervisedIsing, AdditiveMLP
        if settings['training'].get('objective') != 'mse':
            raise ValueError('Supervised Ising comparisons require point MSE')
        torch.manual_seed(int(seed))
        model = (AdditiveMLP(x_fit, y_fit, width=config['width']) if variant == 'ising_mlp' else
                 SupervisedIsing(x_fit, y_fit, spins=config['spins'], coupled=variant == 'ising')).to(device)
    elif variant == 'query':
        torch.manual_seed(int(seed))
        model = QueryQuantile(x_fit, y_fit, width=config['width'], heads=config['heads'],
            dropout=config['dropout'], quantiles=settings['n_quantiles'],
            quantile_min=settings['quantile_min'], quantile_max=settings['quantile_max']).to(device)
    elif variant == 'conditional':
        torch.manual_seed(int(seed))
        model = ConditionalQuantile(x_fit, y_fit, quantiles=settings['n_quantiles']).to(device)
    else:
        outputs = 2 if variant == 'shape_shared' else settings['n_quantiles'] + 1 if variant in ('shape_free', 'shape_shrink') else settings['n_quantiles']
        model = build_model(x_fit, y_fit, config, seed, {**settings, 'n_quantiles': outputs}, device)
        if variant in ('shape_free', 'shape_shared', 'shape_shrink'):
            from genopred.models.tabm.shape import ShapeTransform
            model.shape_transform = ShapeTransform(variant.removeprefix('shape_'), settings['n_quantiles'], dtype=torch.float32).to(device)
    if variant in ('gated','layered'):
        from genopred.models.tabm.interactions import InteractionBackbone, LayerInteractionBackbone
        backbone=InteractionBackbone if variant=='gated' else LayerInteractionBackbone
        model.backbone=backbone(x_fit.shape[1],width=config['d_block'],members=config['k'],
                                           dropout=config['dropout'],quantiles=settings['n_quantiles']).to(device)
    if variant=='member_conditional':
        from genopred.models.tabm.interactions import MemberInteractionBackbone
        # Preserve the original CPU RNG stream after initializing the new modules.
        with torch.random.fork_rng(devices=[]):
            model.backbone=MemberInteractionBackbone(model.backbone,x_fit.shape[1]).to(device)
    if variant in ('shared_basis', 'independent_basis'):
        from genopred.models.tabm.interactions import ConnectionBasis
        with torch.random.fork_rng(devices=[]):
            model.backbone.backbone.blocks[0][0] = ConnectionBasis(
                model.backbone.backbone.blocks[0][0], config['basis_rank'],
                shared=variant == 'shared_basis').to(device)
    x = torch.as_tensor(x_fit, dtype=torch.float32, device=device)
    xv = torch.as_tensor(x_val, dtype=torch.float32, device=device)
    y = model.standardize_y(torch.as_tensor(y_fit[:, None], dtype=torch.float32, device=device))
    q = ((torch.arange(settings['n_quantiles'], device=device) + .5) / settings['n_quantiles']
         if variant in ('shape_midpoint', 'shape_free', 'shape_shared', 'shape_shrink') else
         model.quantiles if variant == 'query' else
         torch.linspace(settings['quantile_min'], settings['quantile_max'],
                        settings['n_quantiles'], device=device))
    if variant in ('adam','member_conditional','conditional','query','ising','ising_independent','ising_mlp',
                   'treewidth_ising','treewidth_independent',
                   'shared_basis','independent_basis',
                   'shape_midpoint','shape_free','shape_shared','shape_shrink'):
        opt = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'],weight_decay=config['weight_decay'])
    else:
        from genopred.models.tabm.interactions import make_optimizers
        opt=make_optimizers(model,config,variant)
    generator = torch.Generator().manual_seed(int(seed) * 1000 + config.get('batch_seed_index', index))
    spec = {**settings['training'], 'shape_penalty': config.get('shape_penalty', 0.)}
    loss_fn = None
    if variant in ('shared_basis', 'independent_basis'):
        from genopred.models.tabm.interactions import basis_loss
        spec['basis_penalty'] = config['basis_penalty']
        loss_fn = (torch.compile(basis_loss, mode='reduce-overhead', dynamic=True)
                   if torch.device(device).type == 'cuda' else basis_loss)
    probabilistic = variant in ('shape_midpoint', 'shape_free', 'shape_shared', 'shape_shrink')
    best_score, best_epoch, best_prediction, best_distribution = -np.inf, -1, None, None
    best_basis = None
    fixed_epochs = settings.get('fixed_epochs')
    if fixed_epochs is not None and (not isinstance(fixed_epochs, int) or fixed_epochs < 0):
        raise ValueError('Invalid fixed epoch count')
    for epoch in range((spec['max_epochs'] if fixed_epochs is None else fixed_epochs) + 1):
        if epoch:
            update_epoch(model, x, y, q, opt, generator, spec, device, loss_fn=loss_fn)
        if fixed_epochs is not None and epoch != fixed_epochs:
            continue
        model.eval()
        with torch.no_grad():
            output = model(xv)
            if variant in ('shape_free', 'shape_shared', 'shape_shrink'):
                output, _ = model.shape_transform(output)
            prediction = model.inverse_y(output.mean(dim=(1, 2))[:, None])[:, 0].cpu().numpy()
        score = pcc(y_val, prediction) if fixed_epochs is None else None
        if fixed_epochs is None and not np.isfinite(score):
            raise ValueError('Nonfinite validation PCC')
        if fixed_epochs is not None or score > best_score + spec['tie_tolerance']:
            best_score, best_epoch, best_prediction = score, epoch, prediction.copy()
            if loss_fn is not None:
                layer = model.backbone.backbone.blocks[0][0]
                with torch.no_grad():
                    delta_rms = layer.penalty().sqrt().item()
                    best_basis = {'delta_rms': delta_rms,
                                  'relative_delta_rms': delta_rms / layer.base.weight.square().mean().sqrt().item()}
                    if layer.shared:
                        c = layer.coefficients - layer.coefficients.mean(0, keepdim=True)
                        best_basis['coefficient_rms'] = c.square().mean().sqrt().item()
            if probabilistic:
                best_distribution = model.inverse_y(output).cpu().numpy()
            if fixed_epochs is not None:
                break
        elif epoch - best_epoch >= spec['patience']:
            break
    details = dict(best_pcc=best_score, best_epoch=best_epoch, epochs_run=epoch,
                   parameters=sum(p.numel() for p in model.parameters()),seconds=time.perf_counter()-start)
    if best_basis is not None:
        details['best_basis'] = best_basis
        details['compiled_training_loss'] = torch.device(device).type == 'cuda'
    if probabilistic:
        details['_distribution'] = best_distribution
    return best_prediction, details


def mixture_scores(distributions, truth):
    """Per-material CRPS and 90% interval for the exact equal-weight atom mixture."""
    atoms = np.concatenate(distributions, axis=1).reshape(len(truth), -1).astype(np.float64)
    atoms.sort(axis=1)
    n = atoms.shape[1]
    crps = np.abs(atoms - np.asarray(truth)[:, None]).mean(axis=1) - (atoms @ (2 * np.arange(1, n + 1) - n - 1)) / n**2
    lo, hi = np.quantile(atoms, [.05, .95], axis=1, method='inverted_cdf')
    return np.column_stack([crps, (truth >= lo) & (truth <= hi), hi - lo])


class QueryQuantile(nn.Module):
    """One environment, one seed; forward returns standardized [batch, 1, queries]."""
    def __init__(self, fit_x, fit_y, *, width=64, heads=4, dropout=0.1,
                 quantiles=99, quantile_min=0.001, quantile_max=0.999):
        super().__init__()
        x = torch.as_tensor(fit_x, dtype=torch.float32).detach().cpu()
        y = torch.as_tensor(fit_y, dtype=torch.float32).detach().cpu().reshape(-1, 1)
        if x.ndim != 2 or x.shape[0] != len(y) or len(y) < 2:
            raise ValueError('Expected matching fit materials and one target environment')
        if not torch.isfinite(x).all() or not torch.isfinite(y).all():
            raise ValueError('Nonfinite fitting data')
        if width < 1 or heads < 1 or width % heads or not 0 <= dropout < 1:
            raise ValueError('Invalid attention dimensions or dropout')
        if quantiles < 2 or not 0 < quantile_min < quantile_max < 1:
            raise ValueError('Invalid quantile grid')
        std = x.std(0, unbiased=False)
        self.register_buffer('x_mean', x.mean(0))
        self.register_buffer('x_constant', std == 0)
        self.register_buffer('x_std', torch.where(std == 0, torch.ones_like(std), std))
        self.register_buffer('y_mean', y.mean(0))
        scale = y.std(0, unbiased=False)
        if torch.any(scale <= 0):
            raise ValueError('Constant fitting target')
        self.register_buffer('y_std', scale)
        self.register_buffer('quantiles', torch.linspace(quantile_min, quantile_max, quantiles))
        self.register_buffer('frequencies', torch.arange(33, dtype=torch.float32))
        self.identity = nn.Parameter(torch.empty(x.shape[1], width))
        self.state = nn.Parameter(torch.empty(x.shape[1], width))
        nn.init.uniform_(self.identity, -width ** -0.5, width ** -0.5)
        nn.init.uniform_(self.state, -width ** -0.5, width ** -0.5)
        self.token_encoder = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(),
                                           nn.Linear(2 * width, width))
        self.query_encoder = nn.Sequential(nn.Linear(33, width), nn.GELU(),
                                           nn.Linear(width, width))
        self.reads = nn.ModuleList([PreLNCrossAttention(width, heads, 2 * width, dropout, dropout, dropout) for _ in range(2)])
        self.readout = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width),
                                     nn.GELU(), nn.Linear(width, 1))

    def standardize_y(self, y):
        return (y - self.y_mean) / self.y_std

    def inverse_y(self, prediction):
        return prediction * self.y_std + self.y_mean

    def forward(self, x, tau=None):
        if x.ndim != 2 or x.shape[1] != self.identity.shape[0] or not torch.isfinite(x).all():
            raise ValueError('Expected finite material-by-marker queries')
        tau = self.quantiles if tau is None else torch.as_tensor(tau, dtype=self.quantiles.dtype, device=self.quantiles.device)
        if tau.ndim != 1 or tau.numel() == 0 or not torch.isfinite(tau).all() or torch.any((tau <= 0) | (tau >= 1)):
            raise ValueError('Quantile queries must be a nonempty vector inside (0,1)')
        z = ((x - self.x_mean) / self.x_std).masked_fill(self.x_constant, 0)
        memory = self.token_encoder(self.identity[None] + z[..., None] * self.state[None])
        basis = torch.cos(math.pi * tau[:, None] * self.frequencies[None])
        query = self.query_encoder(basis)[None].expand(len(x), -1, -1)
        for block in self.reads:
            query = block(query, memory)
        return self.readout(query).squeeze(-1).unsqueeze(1)


def entmax15(x):
    x = x / 2
    ordered = x.sort(dim=-1, descending=True).values
    rho = torch.arange(1, x.shape[-1] + 1, device=x.device, dtype=x.dtype)
    mean = ordered.cumsum(-1) / rho
    variance = ordered.square().cumsum(-1) / rho - mean.square()
    tau = mean - ((1 - rho * variance).clamp_min(0) / rho).sqrt()
    support = (tau <= ordered).sum(-1, keepdim=True).clamp_min(1)
    threshold = tau.gather(-1, support - 1)
    return (x - threshold).clamp_min(0).square()


class ConditionalQuantile(nn.Module):
    def __init__(self, fit_x, fit_y, members=4, width=32, layers=3, quantiles=99):
        super().__init__()
        x = torch.as_tensor(fit_x, dtype=torch.float32)
        y = torch.as_tensor(fit_y, dtype=torch.float32).reshape(-1, 1)
        self.register_buffer('x_mean', x.mean(0))
        std = x.std(0, unbiased=False)
        self.register_buffer('x_constant', std == 0)
        self.register_buffer('x_std', torch.where(std == 0, torch.ones_like(std), std))
        self.register_buffer('y_mean', y.mean(0))
        self.register_buffer('y_std', y.std(0, unbiased=False).clamp_min(1e-12))
        p = x.shape[1]
        self.connections = nn.ParameterList()
        self.tables = nn.ParameterList()
        for layer in range(layers):
            d = p + layer * width
            self.connections.append(nn.Parameter(torch.randn(members, width, 2, d)))
            # Near-copy of first input, with independent perturbations per member.
            self.tables.append(nn.Parameter(torch.tensor([-2., -2., 2., 2.]).expand(members, width, 4).clone() + .05 * torch.randn(members, width, 4)))
        self.linear = nn.Parameter(torch.randn(members, p) * .001)
        self.readout = nn.Parameter(torch.randn(members, layers * width, quantiles) * .01)
        self.bias = nn.Parameter(torch.zeros(members, quantiles))

    def standardize_y(self, y):
        return (y - self.y_mean) / self.y_std

    def inverse_y(self, y):
        return y * self.y_std + self.y_mean

    def forward(self, x):
        if not torch.isfinite(x).all() or torch.any((x < 0) | (x > 1)):
            raise ValueError('Conditional units require finite marker inputs in [0,1]')
        features = x[:, None, :].expand(-1, self.linear.shape[0], -1)
        hidden = []
        for logits, table in zip(self.connections, self.tables):
            choice = entmax15(logits)
            uv = torch.einsum('bmd,musd->bmus', features, choice)
            u, v = uv.unbind(-1)
            c = table.sigmoid()
            h = (c[..., 0]*(1-u)*(1-v) + c[..., 1]*(1-u)*v +
                 c[..., 2]*u*(1-v) + c[..., 3]*u*v)
            hidden.append(h)
            features = torch.cat((features, h), dim=-1)
        z = ((x - self.x_mean) / self.x_std).masked_fill(self.x_constant, 0)
        return torch.einsum('bmp,mpq->bmq', torch.cat(hidden, dim=-1), self.readout) + (z @ self.linear.T)[..., None] + self.bias
