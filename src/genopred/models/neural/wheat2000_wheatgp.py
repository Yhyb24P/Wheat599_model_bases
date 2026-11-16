"""Wheat2000 SOTA: official WheatGP network, official training recipe.

Model, slicing and optimiser follow the upstream repository at commit
a061b8be (train.py): raw genotype split into five contiguous slices, Adam
lr 0.005 / weight decay 1e-4, batch 64, StepLR(90, 0.1), max 300 epochs,
patience 50. Protocol adaptations only: canonical Wheat2000 folds, per-trait
single output, seeds 42/2026 (the upstream script seeds 555), inner
early-stop per seed, per-seed median best epoch, full-development refit and a
seed-averaged test prediction. No local Wheat599 variant is used.
"""
import importlib.util
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from joblib import Parallel, delayed

from genopred.data.wheat599 import sha256_of
from genopred.evaluation.metrics import pcc, rmse, mae
from genopred.data.wheat2000 import load_wheat2000

STATE = {}


def load_model_class(source, expected_sha):
    if sha256_of(Path(source)) != expected_sha:
        raise ValueError('Official WheatGP model file SHA differs')
    spec = importlib.util.spec_from_file_location('wheatgp_official_model', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def model_lstm_dim(module, n_features):
    width = n_features // 5
    lens = [width] * 4 + [n_features - 4 * width]
    conv = module.ConvPart()
    with torch.no_grad():
        return int(sum(conv(torch.zeros(1, 1, length)).flatten(1).shape[1] for length in lens))


def slices(x):
    width = x.shape[1] // 5
    return torch.split(x, [width] * 4 + [x.shape[1] - 4 * width], dim=1)


def validation_pcc(y, prediction):
    if not np.isfinite(prediction).all():
        raise ValueError('Nonfinite prediction')
    if np.std(y) == 0 or np.std(prediction) == 0:
        return 0.0
    return pcc(y, prediction)


def fit(module, x_fit, y_fit, x_query, y_query, cfg, seed, select, epochs=None):
    torch.manual_seed(seed); np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    device = torch.device(cfg['device'])
    xf = torch.as_tensor(x_fit, dtype=torch.float32, device=device)
    xq = torch.as_tensor(x_query, dtype=torch.float32, device=device)
    parts_x = [part.to(device) for part in slices(xf)]
    parts_q = [part.to(device) for part in slices(xq)]
    lstm_dim = model_lstm_dim(module, x_fit.shape[1])
    model = module.wheatGP_base(lstm_dim).to(device)
    center, scale = float(y_fit.mean()), float(y_fit.std())
    if scale <= 0:
        raise ValueError('Constant training target')
    y = torch.as_tensor((y_fit - center) / scale, dtype=torch.float32, device=device)[:, None]
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg['training']['learning_rate']),
                                 weight_decay=float(cfg['training']['weight_decay']))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=90, gamma=0.1)
    generator = torch.Generator(device='cpu').manual_seed(seed)
    best, best_epoch, best_pred = -np.inf, 0, None
    budget = int(cfg['training']['max_epochs']) if epochs is None else int(epochs)
    for epoch in range(1, budget + 1):
        model.train()
        for idx in torch.randperm(len(y), generator=generator).split(int(cfg['training']['batch_size'])):
            idx = idx.to(device)
            loss = torch.nn.functional.mse_loss(model(*[p[idx] for p in parts_x]), y[idx])
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()
        if select:
            model.eval()
            with torch.no_grad():
                pred = model(*parts_q).flatten().cpu().numpy() * scale + center
            score = validation_pcc(y_query, pred)
            if np.isfinite(score) and score > best:
                best, best_epoch, best_pred = float(score), epoch, pred.copy()
            elif epoch - best_epoch >= int(cfg['training']['patience']):
                break
    if not select:
        model.eval()
        with torch.no_grad():
            best_pred = model(*parts_q).flatten().cpu().numpy() * scale + center
        best_epoch = budget
    if best_pred is None or not np.isfinite(best_pred).all():
        raise ValueError('Invalid prediction')
    return best_pred, best_epoch, epoch


def init_worker(x, y, cfg, module):
    torch.set_num_threads(int(cfg['cpu_threads']))
    device = torch.device(cfg['device'])
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(float(cfg['gpu_memory_fraction']))
    STATE.update(x=x, y=y, cfg=cfg, module=module, device=device)


def dev_job(trait, seed, train_idx, val_idx):
    try:
        cfg = STATE['cfg']
        pred, best, epochs = fit(STATE['module'], STATE['x'][train_idx], STATE['y'][train_idx, trait],
                                 STATE['x'][val_idx], STATE['y'][val_idx, trait], cfg, int(seed), True)
        return {'pred': pred, 'best_epoch': best, 'epochs_run': epochs}
    except Exception as exc:  # noqa: BLE001
        return {'error': f'{type(exc).__name__}: {exc}'}


def final_job(trait, seed, dev_idx, test_idx, epochs):
    try:
        cfg = STATE['cfg']
        pred, _, _ = fit(STATE['module'], STATE['x'][dev_idx], STATE['y'][dev_idx, trait],
                         STATE['x'][test_idx], STATE['y'][test_idx, trait], cfg, int(seed),
                         False, epochs=epochs)
        return {'pred': pred}
    except Exception as exc:  # noqa: BLE001
        return {'error': f'{type(exc).__name__}: {exc}'}


def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=str) + '\n')


def development(cfg, folds, traits, args, module):
    args.out.mkdir(parents=True, exist_ok=True)
    initargs = (np.asarray(cfg['_x']), np.asarray(cfg['_y']), cfg, module)
    init_worker(*initargs)
    for fold in folds:
        k = int(fold['fold'])
        jobs, keys = [], []
        for trait in range(len(traits)):
            for seed in cfg['seeds']:
                for inner in fold['inner']:
                    jobs.append((trait, int(seed), np.asarray(inner['train_idx']),
                                 np.asarray(inner['val_idx'])))
                    keys.append((trait, int(seed)))
        started = time.perf_counter()
        results = Parallel(n_jobs=int(cfg['workers']), initializer=init_worker, initargs=initargs)(
            delayed(dev_job)(*job) for job in jobs)
        preds, records = {}, []
        for key, res in zip(keys, results):
            if 'error' in res:
                raise RuntimeError(f'development fit failed {key}: {res["error"]}')
            preds.setdefault(key, []).append(res['pred'])
            records.append({'trait': traits[key[0]], 'seed': key[1],
                            'best_epoch': res['best_epoch'], 'epochs_run': res['epochs_run']})
        selected = []
        for trait in range(len(traits)):
            scores = []
            for seed in cfg['seeds']:
                fold_scores = []
                for inner_index, inner in enumerate(fold['inner']):
                    val = np.asarray(inner['val_idx'])
                    fold_scores.append(float(pcc(cfg['_y'][val, trait],
                                                 preds[(trait, int(seed))][inner_index])))
                scores.append(float(np.mean(fold_scores)))
            epochs = {str(int(seed)): int(sorted(r['best_epoch'] for r in records
                                                 if r['trait'] == traits[trait]
                                                 and r['seed'] == int(seed))[1])
                      for seed in cfg['seeds']}
            selected.append({'trait': traits[trait], 'epochs': epochs, 'scores': scores})
        write(args.out / f'selection_{k}.json', {'fold': k, 'selected': selected})
        write(args.out / f'trajectories_{k}.json', records)
        print(f'fold {k} selected in {time.perf_counter()-started:.0f}s', flush=True)


def merge_selection(cfg, folds, traits, args):
    document = {'folds': []}
    for fold in folds:
        document['folds'].append(json.loads((args.out / f'selection_{fold["fold"]}.json').read_text()))
    write(args.out / 'selection.json', document)
    print(f'merged {len(document["folds"])} folds', flush=True)


def final(cfg, folds, traits, args, module):
    document = json.loads(Path(args.selection).read_text())
    by_fold = {int(entry['fold']): entry for entry in document['folds']}
    fold = folds[0]
    k = int(fold['fold'])
    selection = by_fold[k]
    epochs = {row['trait']: row['epochs'] for row in selection['selected']}
    initargs = (np.asarray(cfg['_x']), np.asarray(cfg['_y']), cfg, module)
    init_worker(*initargs)
    args.out.mkdir(parents=True, exist_ok=True)
    dev = np.asarray(fold['dev_idx'])
    test = np.asarray(fold['test_idx'])
    jobs, keys = [], []
    for trait in range(len(traits)):
        for seed in cfg['seeds']:
            jobs.append((trait, int(seed), dev, test, int(epochs[traits[trait]][str(int(seed))])))
            keys.append((trait, int(seed)))
    results = Parallel(n_jobs=int(cfg['workers']), initializer=init_worker, initargs=initargs)(
        delayed(final_job)(*job) for job in jobs)
    arrays = {}
    for key, res in zip(keys, results):
        if 'error' in res:
            raise RuntimeError(f'final fit failed {key}: {res["error"]}')
        arrays[f'{traits[key[0]]}_{key[1]}'] = np.asarray(res['pred'], dtype=np.float64)
    arrays['test_ids'] = test
    np.savez_compressed(args.out / f'predictions_{k}.npz', **arrays)
    print(f'fold {k} final complete', flush=True)


def report(cfg, folds, traits, args):
    rows, seedrows = [], []
    y = np.asarray(cfg['_y'])
    for fold in folds:
        k = int(fold['fold'])
        test = np.asarray(fold['test_idx'])
        z = np.load(args.out / f'predictions_{k}.npz')
        if not np.array_equal(np.asarray(z['test_ids']), test):
            raise ValueError('Saved test ids disagree with the manifest')
        for e, trait in enumerate(traits):
            preds = [np.asarray(z[f'{trait}_{seed}']) for seed in cfg['seeds']]
            mean = np.mean(preds, axis=0)
            for seed, pred in zip(cfg['seeds'], preds):
                seedrows.append({'fold': k, 'trait': trait, 'seed': int(seed),
                                 'pcc': float(pcc(y[test, e], pred)),
                                 'rmse': float(rmse(y[test, e], pred)),
                                 'mae': float(mae(y[test, e], pred))})
            rows.append({'fold': k, 'trait': trait,
                         'pcc': float(pcc(y[test, e], mean)),
                         'rmse': float(rmse(y[test, e], mean)),
                         'mae': float(mae(y[test, e], mean))})
    per_trait = {t: {m: float(np.mean([r[m] for r in rows if r['trait'] == t]))
                     for m in ('pcc', 'rmse', 'mae')} for t in traits}
    write(args.out / 'summary.json', {
        'n_trait_folds': len(rows),
        'primary': {m: float(np.mean([r[m] for r in rows])) for m in ('pcc', 'rmse', 'mae')},
        'per_trait': per_trait, 'seeds': list(cfg['seeds'])})
    with (args.out / 'per_fold.csv').open('w') as handle:
        handle.write('fold,trait,pcc,rmse,mae\n')
        for r in rows:
            handle.write(f'{r["fold"]},{r["trait"]},{r["pcc"]:.10g},{r["rmse"]:.10g},{r["mae"]:.10g}\n')
    with (args.out / 'per_seed.csv').open('w') as handle:
        handle.write('fold,trait,seed,pcc,rmse,mae\n')
        for r in seedrows:
            handle.write(f'{r["fold"]},{r["trait"]},{r["seed"]},{r["pcc"]:.10g},{r["rmse"]:.10g},{r["mae"]:.10g}\n')
    print(json.dumps({'primary': {m: float(np.mean([r[m] for r in rows])) for m in ('pcc', 'rmse', 'mae')}}), flush=True)


def run(cfg, args, runner):
    data = load_wheat2000(Path(cfg['data']), cfg['data_sha256'],
                          Path(cfg['cache']), cfg['cache_sha256'])
    split_path = Path(cfg['splits'])
    if sha256_of(split_path) != cfg['splits_sha256']:
        raise ValueError('Frozen Wheat2000 splits SHA differs')
    splits = json.loads(split_path.read_text())
    traits = list(data.trait_names)
    if traits != cfg['traits'] or splits['trait_names'] != traits:
        raise ValueError('Trait names/order differ from the frozen protocol')
    folds = splits['folds']
    if args.folds:
        wanted = set(int(f) for f in args.folds)
        folds = [f for f in folds if int(f['fold']) in wanted]
    if getattr(args, 'traits', None):
        traits = [t for t in traits if t in set(args.traits)]
    module = load_model_class(cfg['source'], cfg['source_sha256'])
    cfg = {**cfg, '_x': np.asarray(data.X), '_y': np.asarray(data.Y)}
    args.out.mkdir(parents=True, exist_ok=True)
    write(args.out / 'provenance.json', {
        'config': {k: v for k, v in cfg.items() if not k.startswith('_')},
        'data_sha256': cfg['data_sha256'], 'cache_sha256': cfg['cache_sha256'],
        'splits_sha256': cfg['splits_sha256'], 'traits': traits, 'phase': args.phase,
        'folds': [int(f['fold']) for f in folds], 'seeds': cfg['seeds'],
        'source': {'url': 'https://github.com/Breed-AI/WheatGP', 'commit': cfg['source_commit'],
                   'file_sha256': sha256_of(Path(cfg['source']))},
        'recipe': 'official train.py: raw five-slice input, Adam lr .005 wd 1e-4, batch 64, '
                  'StepLR(90,.1), 300 epochs, patience 50; seeds 42/2026 replace upstream 555',
    })
    phase = args.phase
    if phase == 'dev':
        development(cfg, folds, traits, args, module)
    elif phase == 'merge':
        merge_selection(cfg, folds, traits, args)
    elif phase == 'final':
        if not args.selection:
            raise ValueError('final requires --selection')
        final(cfg, folds, traits, args, module)
    elif phase == 'report':
        report(cfg, folds, traits, args)
    else:
        raise ValueError('unknown phase')