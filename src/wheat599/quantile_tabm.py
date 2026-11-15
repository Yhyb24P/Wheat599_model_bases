"""Independent full-marker TabM with per-member quantile regression."""
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from wheat599.data import load, sha256_of
from wheat599.genomic_tabm import GenomicTabM
from wheat599.metrics import pcc, rmse, mae


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
        from wheat599.treewidth_ising import TreewidthIsing
        if objective != 'mse':
            raise ValueError('Treewidth Ising comparisons require point MSE')
        torch.manual_seed(int(seed))
        model = TreewidthIsing(x_fit, y_fit, spins=config['spins'],
                              coupled=variant == 'treewidth_ising').to(device)
    elif variant in ('ising', 'ising_independent', 'ising_mlp'):
        from wheat599.supervised_ising import SupervisedIsing, AdditiveMLP
        if settings['training'].get('objective') != 'mse':
            raise ValueError('Supervised Ising comparisons require point MSE')
        torch.manual_seed(int(seed))
        model = (AdditiveMLP(x_fit, y_fit, width=config['width']) if variant == 'ising_mlp' else
                 SupervisedIsing(x_fit, y_fit, spins=config['spins'], coupled=variant == 'ising')).to(device)
    elif variant == 'query':
        from wheat599.query_quantile import QueryQuantile
        torch.manual_seed(int(seed))
        model = QueryQuantile(x_fit, y_fit, width=config['width'], heads=config['heads'],
            dropout=config['dropout'], quantiles=settings['n_quantiles'],
            quantile_min=settings['quantile_min'], quantile_max=settings['quantile_max']).to(device)
    elif variant == 'conditional':
        from wheat599.conditional_quantile import ConditionalQuantile
        torch.manual_seed(int(seed))
        model = ConditionalQuantile(x_fit, y_fit, quantiles=settings['n_quantiles']).to(device)
    else:
        outputs = 2 if variant == 'shape_shared' else settings['n_quantiles'] + 1 if variant in ('shape_free', 'shape_shrink') else settings['n_quantiles']
        model = build_model(x_fit, y_fit, config, seed, {**settings, 'n_quantiles': outputs}, device)
        if variant in ('shape_free', 'shape_shared', 'shape_shrink'):
            from wheat599.shape_quantile import ShapeTransform
            model.shape_transform = ShapeTransform(variant.removeprefix('shape_'), settings['n_quantiles'], dtype=torch.float32).to(device)
    if variant in ('gated','layered'):
        from wheat599.interaction_tabm import InteractionBackbone, LayerInteractionBackbone
        backbone=InteractionBackbone if variant=='gated' else LayerInteractionBackbone
        model.backbone=backbone(x_fit.shape[1],width=config['d_block'],members=config['k'],
                                           dropout=config['dropout'],quantiles=settings['n_quantiles']).to(device)
    if variant=='member_conditional':
        from wheat599.member_interaction_tabm import MemberInteractionBackbone
        # Preserve the original CPU RNG stream after initializing the new modules.
        with torch.random.fork_rng(devices=[]):
            model.backbone=MemberInteractionBackbone(model.backbone,x_fit.shape[1]).to(device)
    if variant in ('shared_basis', 'independent_basis'):
        from wheat599.shared_basis_tabm import ConnectionBasis
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
        from wheat599.interaction_tabm import make_optimizers
        opt=make_optimizers(model,config,variant)
    generator = torch.Generator().manual_seed(int(seed) * 1000 + config.get('batch_seed_index', index))
    spec = {**settings['training'], 'shape_penalty': config.get('shape_penalty', 0.)}
    loss_fn = None
    if variant in ('shared_basis', 'independent_basis'):
        from wheat599.shared_basis_tabm import basis_loss
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


_FIT_CONTEXT = None


def _init_fit_worker(x, y, folds, configs, settings, workers):
    global _FIT_CONTEXT
    torch.set_num_threads(settings['cpu_threads'])
    device = torch.device(settings['device'])
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(settings['gpu_memory_fraction'] / workers)
    _FIT_CONTEXT = (x, y, {f['fold']: f for f in folds}, configs, settings, device)


def _fit_worker(task):
    fold, inner, environment, index, seed = task
    x, y, folds, configs, settings, device = _FIT_CONTEXT
    split = folds[fold]['inner'][inner]
    tr, val = np.asarray(split['train_idx']), np.asarray(split['val_idx'])
    try:
        prediction, details = train(x[tr], y[tr, environment], x[val], y[val, environment],
                                    configs[index], seed, index, settings, device)
    except Exception as error:
        raise RuntimeError(f'Fit failed: fold/inner/environment/config/seed={task}') from error
    return environment, index, seed, prediction, details


def read(path):
    return json.loads(Path(path).read_text())


def run(cfg, args, runner, *, trainer=None, paired_trainer=None, extra_sources=()):
    trainer = train if trainer is None else trainer
    workers = cfg.get('workers', 1)
    if not isinstance(workers, int) or workers < 1:
        raise ValueError('workers must be a positive integer')
    if workers > 1 and (trainer is not train or paired_trainer is not None):
        raise ValueError('Custom or paired trainers require workers=1')
    if args.selection is not None:
        raise ValueError('Development only')
    torch.set_num_threads(cfg['cpu_threads'])
    device = torch.device(cfg['device'])
    if device.type == 'cuda' and workers == 1:
        torch.cuda.set_per_process_memory_fraction(cfg['gpu_memory_fraction'])
    data = load(Path(cfg['data']), cfg['data_sha256'])
    folds = read(cfg['splits'])['folds']
    configs = yaml.safe_load(Path(cfg['architecture_config']).read_text())['configurations']
    if len(folds) != 10 or any(len(f['inner']) != 3 for f in folds) or not configs:
        raise ValueError('Expected 10 folds, 3 inner splits and at least one configuration')
    source = Path(cfg['reference_source'])
    info = read(source/'provenance.json')
    if info['sha256'][cfg['splits']] != sha256_of(Path(cfg['splits'])):
        raise ValueError('Reference split mismatch')
    baseline=Path(cfg['baseline_source']) if cfg.get('baseline_source') else None
    if baseline:
        audit=read(cfg['baseline_audit'])['sha256'];baseline_choices=read(baseline/'summary.json')['folds']
        for fold in folds:
            for j in range(3):
                path=baseline/f"predictions_{fold['fold']}_{j}.npz"
                if sha256_of(path)!=audit[str(path)]:raise ValueError('Baseline prediction hash mismatch')
    args.out.mkdir(parents=True, exist_ok=False)
    def write(name, obj):
        (args.out/name).write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')
    sources = [Path(cfg['splits']), Path(cfg['architecture_config']), Path(__file__),
               Path(__file__).with_name('genomic_tabm.py'), Path(runner), Path(args.config),
               source/'provenance.json'] + [source/f"references_{f['fold']}_{j}.npz"
                                           for f in folds for j in range(3)]
    if baseline:
        sources += [Path(__file__).with_name('interaction_tabm.py'),baseline/'summary.json',Path(cfg['baseline_audit'])]
        sources += [baseline/f"predictions_{f['fold']}_{j}.npz" for f in folds for j in range(3)]
    if cfg.get('optimizer_variant')=='member_conditional':sources.append(Path(__file__).with_name('member_interaction_tabm.py'))
    if cfg.get('optimizer_variant')=='conditional':sources.append(Path(__file__).with_name('conditional_quantile.py'))
    if cfg.get('optimizer_variant')=='query':
        sources.extend(Path(__file__).with_name(name) for name in ('query_quantile.py','induced_ftt.py'))
    if cfg.get('optimizer_variant') in ('ising','ising_independent','ising_mlp'):
        sources.append(Path(__file__).with_name('supervised_ising.py'))
    if cfg.get('optimizer_variant') in ('treewidth_ising','treewidth_independent'):
        sources.extend(Path(__file__).with_name(name) for name in ('treewidth_ising.py','supervised_ising.py'))
    if cfg.get('optimizer_variant') in ('shared_basis','independent_basis'):
        sources.append(Path(__file__).with_name('shared_basis_tabm.py'))
    if cfg.get('optimizer_variant') in ('shape_free','shape_shared','shape_shrink'):
        sources.append(Path(__file__).with_name('shape_quantile.py'))
    sources.extend(Path(p) for p in extra_sources)
    comparison={}
    for name,path in cfg.get('comparison_sources',{}).items():
        path=Path(path);p=read(path/'provenance.json')
        if p['sha256'][cfg['splits']]!=sha256_of(Path(cfg['splits'])):raise ValueError('Comparison split mismatch')
        if read(path/'execution_status.json')!={'status':'complete','completed':1920,'expected':1920} or read(path/'failures.json'):raise ValueError('Incomplete comparison')
        comparison[name]=read(path/'summary.json');sources += [path/'summary.json',path/'provenance.json',path/'execution_status.json',path/'failures.json']
    write('provenance.json', {'execution': {'workers': workers,
        'gpu_fraction_per_worker': cfg.get('gpu_memory_fraction', 0) / workers if device.type == 'cuda' else None}, 'config': cfg, 'torch_version':str(torch.__version__), 'cuda_version':torch.version.cuda, 'git_commit': subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'sha256': {str(p): sha256_of(p) for p in sources}})
    expected = len(folds)*3*4*len(configs)*len(cfg['seeds'])
    state = dict(status='running', completed=0, expected=expected)
    records, failures, start = [], [], time.perf_counter()
    pool = None
    try:
        if workers > 1:
            pool = ProcessPoolExecutor(max_workers=workers,
                mp_context=multiprocessing.get_context('spawn'), initializer=_init_fit_worker,
                initargs=(data.X, data.Y, folds, configs, cfg, workers))
        for fold in folds:
            f = fold['fold']
            for inner, split in enumerate(fold['inner']):
                tr, val = np.array(split['train_idx']), np.array(split['val_idx'])
                if (set(tr) & set(val) or set(tr) | set(val) != set(fold['dev_idx'])
                        or (set(tr) | set(val)) & set(fold['test_idx'])):
                    raise ValueError('Invalid split')
                predictions, pending = {}, {}
                def serial_outputs():
                    for environment in range(4):
                        for c, config in enumerate(configs):
                            if paired_trainer is not None:
                                outputs = paired_trainer(data.X[tr], data.Y[tr, environment],
                                    data.X[val], data.Y[val, environment], config, cfg['seeds'], c, cfg, device)
                            else:
                                outputs = [trainer(data.X[tr], data.Y[tr, environment],
                                    data.X[val], data.Y[val, environment], config, seed, c, cfg, device)
                                    for seed in cfg['seeds']]
                            for seed, (pred, details) in zip(cfg['seeds'], outputs):
                                yield environment, c, seed, pred, details
                tasks = [(f, inner, e, c, seed) for e in range(4)
                         for c in range(len(configs)) for seed in cfg['seeds']]
                outputs = pool.map(_fit_worker, tasks, chunksize=1) if pool else serial_outputs()
                try:
                    for environment, c, seed, pred, details in outputs:
                        distribution = details.pop('_distribution', None)
                        if distribution is not None:
                            key = (environment, c)
                            pending.setdefault(key, []).append(distribution)
                            if len(pending[key]) == len(cfg['seeds']):
                                predictions[f'e{environment}_c{c}_prob'] = mixture_scores(pending.pop(key), data.Y[val, environment])
                        predictions[f'e{environment}_c{c}_s{seed}'] = pred
                        records.append(dict(fold=f, inner=inner, environment=environment,
                                            config_index=c, seed=seed, **details))
                        state['completed'] += 1
                        write('execution_status.json', state)
                    if pending:
                        raise ValueError('Incomplete predictive mixtures')
                finally:
                    np.savez_compressed(args.out/f'predictions_{f}_{inner}.npz', **predictions)
                    write('trajectories.json', records)
        if state['completed'] != expected:
            raise ValueError('Incomplete experiment')
        results, probabilistic_results = [], []
        for fold in folds:
            f = fold['fold']
            arrays = [np.load(args.out/f'predictions_{f}_{j}.npz') for j in range(3)]
            refs = [np.load(source/f'references_{f}_{j}.npz') for j in range(3)]
            chosen = []
            for e in range(4):
                scores = []
                for c in range(len(configs)):
                    scores.append(np.mean([pcc(data.Y[split['val_idx'], e],
                        np.mean([arrays[j][f'e{e}_c{c}_s{seed}'] for seed in cfg['seeds']], axis=0))
                        for j, split in enumerate(fold['inner'])]))
                chosen.append(int(np.argmax(scores)))
            methods=('quantile','huber','independent_tabm','fixed_fusion') + (('original_quantile','original_fusion') if baseline else ())
            per_method = {k: [] for k in methods}
            for j, split in enumerate(fold['inner']):
                truth = data.Y[split['val_idx']]
                quantile = np.column_stack([np.mean([arrays[j][f'e{e}_c{chosen[e]}_s{seed}']
                                                    for seed in cfg['seeds']], axis=0) for e in range(4)])
                if baseline:
                    old=np.load(baseline/f'predictions_{f}_{j}.npz')
                    old_choices=next(v['selected'] for v in baseline_choices if v['fold']==f)
                    original=np.column_stack([np.mean([old[f'e{e}_c{old_choices[e]}_s{seed}'] for seed in cfg['seeds']],axis=0) for e in range(4)])
                for method in per_method:
                    pred = quantile if method == 'quantile' else original if method=='original_quantile' else .5*(original+refs[j]['huber']) if method=='original_fusion' else refs[j][method]
                    metrics = [[fun(truth[:, e], pred[:, e]) for fun in (pcc, rmse, mae)] for e in range(4)]
                    if not np.isfinite(metrics).all():
                        raise ValueError('Nonfinite metrics')
                    per_method[method].append(metrics)
            results.append(dict(fold=f, selected=chosen,
                                metrics={k: np.mean(v, axis=0).tolist() for k, v in per_method.items()}))
            if cfg.get('optimizer_variant', '').startswith('shape_'):
                probabilistic_results.append(np.array([[arrays[j][f'e{e}_c{chosen[e]}_prob'].mean(axis=0)
                    for e in range(4)] for j in range(3)]).mean(axis=0))
        summary = {method: dict(zip(('pcc','rmse','mae'),
            np.mean([r['metrics'][method] for r in results], axis=(0, 1)).tolist()))
            for method in methods}
        for name,previous in comparison.items():
            for result in results:
                result['metrics'][name]=next(v['metrics']['quantile'] for v in previous['folds'] if v['fold']==result['fold'])
            summary[name]=dict(zip(('pcc','rmse','mae'),np.mean([r['metrics'][name] for r in results],axis=(0,1)).tolist()))
        result = {'macro': summary, 'folds': results}
        if probabilistic_results:
            values = np.stack(probabilistic_results)
            result['probabilistic'] = {'metrics': ['crps', 'coverage90', 'width90'],
                'macro': values.mean(axis=(0, 1)).tolist(), 'per_environment': values.mean(axis=0).tolist(),
                'folds': values.tolist()}
        write('summary.json', result)
        state['status'] = 'complete'
    except BaseException as exc:
        state['status'] = 'failed'
        failures.append({'completed': state['completed'], 'error': repr(exc)})
        raise
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
        write('execution_status.json', state)
        write('trajectories.json', records)
        write('failures.json', failures)
        write('timing.json', {'seconds': time.perf_counter()-start})
