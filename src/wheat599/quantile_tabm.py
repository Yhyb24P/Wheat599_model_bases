"""Independent full-marker TabM with per-member quantile regression."""
import json
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


def train(x_fit, y_fit, x_val, y_val, config, seed, index, settings, device):
    import tabm
    torch.manual_seed(int(seed))
    model = GenomicTabM(x_fit, y_fit[:, None], variant='raw',
        **{k: config[k] for k in ('n_blocks', 'd_block', 'dropout', 'k')})
    # Rebuild only the official backbone so its output has one coordinate per quantile.
    model.backbone = tabm.TabM.make(n_num_features=x_fit.shape[1], d_out=settings['n_quantiles'],
        **{k: config[k] for k in ('n_blocks', 'd_block', 'dropout', 'k')})
    model = model.to(device)
    x = torch.as_tensor(x_fit, dtype=torch.float32, device=device)
    xv = torch.as_tensor(x_val, dtype=torch.float32, device=device)
    y = model.standardize_y(torch.as_tensor(y_fit[:, None], dtype=torch.float32, device=device))
    q = torch.linspace(settings['quantile_min'], settings['quantile_max'],
                       settings['n_quantiles'], device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'],
                            weight_decay=config['weight_decay'])
    generator = torch.Generator().manual_seed(int(seed) * 1000 + index)
    spec = settings['training']
    best_score, best_epoch, best_prediction = -np.inf, -1, None
    for epoch in range(spec['max_epochs'] + 1):
        if epoch:
            model.train()
            order = torch.randperm(len(x), generator=generator).to(device)
            for batch in order.split(spec['batch_size']):
                opt.zero_grad(set_to_none=True)
                loss = quantile_loss(model(x[batch]), y[batch], q)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite quantile loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), spec['gradient_clip'])
                opt.step()
        model.eval()
        with torch.no_grad():
            prediction = model.inverse_y(model(xv).mean(dim=(1, 2))[:, None])[:, 0].cpu().numpy()
        score = pcc(y_val, prediction)
        if not np.isfinite(score):
            raise ValueError('Nonfinite validation PCC')
        if score > best_score + spec['tie_tolerance']:
            best_score, best_epoch, best_prediction = score, epoch, prediction.copy()
        elif epoch - best_epoch >= spec['patience']:
            break
    return best_prediction, dict(best_pcc=best_score, best_epoch=best_epoch, epochs_run=epoch)


def read(path):
    return json.loads(Path(path).read_text())


def run(cfg, args, runner):
    if args.selection is not None:
        raise ValueError('Development only')
    torch.set_num_threads(cfg['cpu_threads'])
    device = torch.device(cfg['device'])
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(cfg['gpu_memory_fraction'])
    data = load(Path(cfg['data']), cfg['data_sha256'])
    folds = read(cfg['splits'])['folds']
    configs = yaml.safe_load(Path(cfg['architecture_config']).read_text())['configurations']
    if len(folds) != 10 or any(len(f['inner']) != 3 for f in folds) or len(configs) != 8:
        raise ValueError('Expected 10 folds, 3 inner splits and 8 configurations')
    source = Path(cfg['reference_source'])
    info = read(source/'provenance.json')
    if info['sha256'][cfg['splits']] != sha256_of(Path(cfg['splits'])):
        raise ValueError('Reference split mismatch')
    args.out.mkdir(parents=True, exist_ok=False)
    def write(name, obj):
        (args.out/name).write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')
    sources = [Path(cfg['splits']), Path(cfg['architecture_config']), Path(__file__),
               Path(__file__).with_name('genomic_tabm.py'), Path(runner), Path(args.config),
               source/'provenance.json'] + [source/f"references_{f['fold']}_{j}.npz"
                                           for f in folds for j in range(3)]
    write('provenance.json', {'config': cfg, 'git_commit': subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'sha256': {str(p): sha256_of(p) for p in sources}})
    expected = len(folds)*3*4*len(configs)*len(cfg['seeds'])
    state = dict(status='running', completed=0, expected=expected)
    records, failures, start = [], [], time.perf_counter()
    try:
        for fold in folds:
            f = fold['fold']
            for inner, split in enumerate(fold['inner']):
                tr, val = np.array(split['train_idx']), np.array(split['val_idx'])
                if (set(tr) & set(val) or set(tr) | set(val) != set(fold['dev_idx'])
                        or (set(tr) | set(val)) & set(fold['test_idx'])):
                    raise ValueError('Invalid split')
                predictions = {}
                for environment in range(4):
                    for c, config in enumerate(configs):
                        for seed in cfg['seeds']:
                            pred, details = train(data.X[tr], data.Y[tr, environment],
                                data.X[val], data.Y[val, environment], config, seed, c, cfg, device)
                            predictions[f'e{environment}_c{c}_s{seed}'] = pred
                            records.append(dict(fold=f, inner=inner, environment=environment,
                                                config_index=c, seed=seed, **details))
                            state['completed'] += 1
                            write('execution_status.json', state)
                np.savez_compressed(args.out/f'predictions_{f}_{inner}.npz', **predictions)
                write('trajectories.json', records)
        if state['completed'] != expected:
            raise ValueError('Incomplete experiment')
        results = []
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
            per_method = {k: [] for k in ('quantile', 'huber', 'independent_tabm', 'fixed_fusion')}
            for j, split in enumerate(fold['inner']):
                truth = data.Y[split['val_idx']]
                quantile = np.column_stack([np.mean([arrays[j][f'e{e}_c{chosen[e]}_s{seed}']
                                                    for seed in cfg['seeds']], axis=0) for e in range(4)])
                for method in per_method:
                    pred = quantile if method == 'quantile' else refs[j][method]
                    metrics = [[fun(truth[:, e], pred[:, e]) for fun in (pcc, rmse, mae)] for e in range(4)]
                    if not np.isfinite(metrics).all():
                        raise ValueError('Nonfinite metrics')
                    per_method[method].append(metrics)
            results.append(dict(fold=f, selected=chosen,
                                metrics={k: np.mean(v, axis=0).tolist() for k, v in per_method.items()}))
        summary = {method: dict(zip(('pcc','rmse','mae'),
            np.mean([r['metrics'][method] for r in results], axis=(0, 1)).tolist()))
            for method in ('quantile','huber','independent_tabm','fixed_fusion')}
        write('summary.json', {'macro': summary, 'folds': results})
        state['status'] = 'complete'
    except BaseException as exc:
        state['status'] = 'failed'
        failures.append({'completed': state['completed'], 'error': repr(exc)})
        raise
    finally:
        write('execution_status.json', state)
        write('trajectories.json', records)
        write('failures.json', failures)
        write('timing.json', {'seconds': time.perf_counter()-start})
