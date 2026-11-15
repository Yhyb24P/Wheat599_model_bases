"""Wheat2000 SOTA: official MtCro entry (torch_rechub MMOE), minimal adaptation.

The author repository trains multi-gate mixture-of-experts (torch_rechub
MMOE) on PCs with Adam lr 1e-4 / weight decay 1e-5, 100 epochs, batch 640
and early-stopping patience 30. The author's Wheat2000 folder ships only
phenotypes, so this run fits PCA on the current training markers (95%
variance, the same preprocessing used by the author's Wheat599 template) and
predicts all eight traits jointly. Protocol adaptations only: canonical folds,
seeds 42/2026, inner early stop per seed, per-seed median epoch, full-dev
refit and seed-averaged test prediction.
"""
import csv
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import torch
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from torch_rechub.basic.features import DenseFeature
from torch_rechub.models.multi_task import MMOE

from wheat599.data import sha256_of
from wheat599.metrics import pcc, rmse, mae
from wheat599.wheat2000_data import load_wheat2000

STATE = {}


def init_worker(x, y, cfg):
    torch.set_num_threads(int(cfg['cpu_threads']))
    device = torch.device(cfg['device'])
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(float(cfg['gpu_memory_fraction']))
    STATE.update(x=x, y=y, cfg=cfg, device=device)


def build(n_features, cfg, n_tasks):
    names = [str(i) for i in range(n_features)]
    features = [DenseFeature(name) for name in names]
    model = MMOE(features, ['regression'] * n_tasks, int(cfg['architecture']['experts']),
                 {'dims': list(cfg['architecture']['expert_dims']),
                  'dropout': float(cfg['architecture']['dropout'])},
                 [{'dims': list(cfg['architecture']['tower_dims'])}] * n_tasks)
    return model, names


def fit_predict(x, y, train, query, cfg, seed, select, epochs=None):
    torch.manual_seed(seed); np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    device = torch.device(cfg['device'])
    pca = PCA(n_components=float(cfg['architecture']['pca_variance']), svd_solver='full',
              random_state=seed)
    train_x = pca.fit_transform(x[train]).astype(np.float32)
    query_x = pca.transform(x[query]).astype(np.float32)
    model, names = build(train_x.shape[1], cfg, y.shape[1])
    model.to(device)
    xt = torch.as_tensor(train_x, device=device)
    yt = torch.as_tensor(y[train], dtype=torch.float32, device=device)
    qt = torch.as_tensor(query_x, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg['training']['learning_rate']),
                                 weight_decay=float(cfg['training']['weight_decay']))
    generator = torch.Generator().manual_seed(seed)
    best, best_epoch, best_state = -np.inf, 0, None
    budget = int(cfg['training']['epochs']) if epochs is None else int(epochs)
    batch_size = int(cfg['training']['batch_size'])
    for epoch in range(1, budget + 1):
        model.train()
        order = torch.randperm(len(train), generator=generator)
        for start in range(0, len(train), batch_size):
            rows = order[start:start + batch_size].to(device)
            inputs = {name: xt[rows, i].unsqueeze(1) for i, name in enumerate(names)}
            loss = torch.nn.functional.mse_loss(model(inputs), yt[rows])
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward(); optimizer.step()
        if select:
            model.eval()
            with torch.no_grad():
                rows = torch.arange(len(query), device=device)
                pred = model({name: qt[rows, i].unsqueeze(1) for i, name in enumerate(names)})
            scores = [pcc(y[query, e], pred[:, e].cpu().numpy()) for e in range(y.shape[1])]
            score = float(np.mean(scores)) if all(np.isfinite(scores)) else -np.inf
            if score > best:
                best, best_epoch = score, epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            elif epoch - best_epoch >= int(cfg['training']['patience']):
                break
    if select:
        if best_state is None:
            raise ValueError('No finite validation PCC')
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        rows = torch.arange(len(query), device=device)
        pred = model({name: qt[rows, i].unsqueeze(1) for i, name in enumerate(names)})
    return pred.cpu().numpy(), best_epoch, epoch


def dev_job(train_idx, val_idx, seed):
    try:
        cfg = STATE['cfg']
        pred, best, epochs = fit_predict(STATE['x'], STATE['y'], train_idx, val_idx,
                                         cfg, int(seed), True)
        return {'pred': pred, 'best_epoch': best, 'epochs_run': epochs}
    except Exception as exc:  # noqa: BLE001
        return {'error': f'{type(exc).__name__}: {exc}'}


def final_job(dev_idx, test_idx, seed, epochs):
    try:
        cfg = STATE['cfg']
        pred, _, _ = fit_predict(STATE['x'], STATE['y'], dev_idx, test_idx,
                                 cfg, int(seed), False, epochs=epochs)
        return {'pred': pred}
    except Exception as exc:  # noqa: BLE001
        return {'error': f'{type(exc).__name__}: {exc}'}


def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=str) + '\n')


def development(cfg, folds, traits, args):
    args.out.mkdir(parents=True, exist_ok=True)
    initargs = (np.asarray(cfg['_x']), np.asarray(cfg['_y']), cfg)
    init_worker(*initargs)
    for fold in folds:
        k = int(fold['fold'])
        jobs, keys = [], []
        for inner in fold['inner']:
            for seed in cfg['seeds']:
                jobs.append((np.asarray(inner['train_idx']), np.asarray(inner['val_idx']), int(seed)))
                keys.append((int(seed),))
        results = Parallel(n_jobs=int(cfg['workers']), initializer=init_worker, initargs=initargs)(
            delayed(dev_job)(*job) for job in jobs)
        preds, records = {}, []
        for i, (key, res) in enumerate(zip(keys, results)):
            if 'error' in res:
                raise RuntimeError(f'development fit failed {key}: {res["error"]}')
            preds.setdefault(key, []).append(res['pred'])
            records.append({'seed': key[0], 'inner': i // len(cfg['seeds']),
                            'best_epoch': res['best_epoch'], 'epochs_run': res['epochs_run']})
        epochs = {str(int(seed)): int(sorted(r['best_epoch'] for r in records
                                             if r['seed'] == int(seed))[1])
                  for seed in cfg['seeds']}
        write(args.out / f'selection_{k}.json', {'fold': k, 'epochs': epochs})
        write(args.out / f'trajectories_{k}.json', records)
        print(f'fold {k} selected epochs {epochs}', flush=True)


def merge_selection(cfg, folds, traits, args):
    document = {'folds': []}
    for fold in folds:
        document['folds'].append(json.loads((args.out / f'selection_{fold["fold"]}.json').read_text()))
    write(args.out / 'selection.json', document)
    print(f'merged {len(document["folds"])} folds', flush=True)


def final(cfg, folds, traits, args):
    document = json.loads(Path(args.selection).read_text())
    by_fold = {int(entry['fold']): entry for entry in document['folds']}
    fold = folds[0]
    k = int(fold['fold'])
    epochs = by_fold[k]['epochs']
    initargs = (np.asarray(cfg['_x']), np.asarray(cfg['_y']), cfg)
    init_worker(*initargs)
    args.out.mkdir(parents=True, exist_ok=True)
    dev = np.asarray(fold['dev_idx'])
    test = np.asarray(fold['test_idx'])
    jobs = [(dev, test, int(seed), int(epochs[str(int(seed))])) for seed in cfg['seeds']]
    results = Parallel(n_jobs=int(cfg['workers']), initializer=init_worker, initargs=initargs)(
        delayed(final_job)(*job) for job in jobs)
    arrays = {}
    for seed, res in zip(cfg['seeds'], results):
        if 'error' in res:
            raise RuntimeError(f'final fit failed seed {seed}: {res["error"]}')
        arrays[f'seed_{int(seed)}'] = np.asarray(res['pred'], dtype=np.float64)
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
        preds = [np.asarray(z[f'seed_{int(seed)}']) for seed in cfg['seeds']]
        mean = np.mean(preds, axis=0)
        for e, trait in enumerate(traits):
            for seed, pred in zip(cfg['seeds'], preds):
                seedrows.append({'fold': k, 'trait': trait, 'seed': int(seed),
                                 'pcc': float(pcc(y[test, e], pred[:, e])),
                                 'rmse': float(rmse(y[test, e], pred[:, e])),
                                 'mae': float(mae(y[test, e], pred[:, e]))})
            rows.append({'fold': k, 'trait': trait,
                         'pcc': float(pcc(y[test, e], mean[:, e])),
                         'rmse': float(rmse(y[test, e], mean[:, e])),
                         'mae': float(mae(y[test, e], mean[:, e]))})
    write(args.out / 'summary.json', {
        'n_trait_folds': len(rows),
        'primary': {m: float(np.mean([r[m] for r in rows])) for m in ('pcc', 'rmse', 'mae')},
        'per_trait': {t: {m: float(np.mean([r[m] for r in rows if r['trait'] == t]))
                          for m in ('pcc', 'rmse', 'mae')} for t in traits},
        'seeds': list(cfg['seeds'])})
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
    cfg = {**cfg, '_x': np.asarray(data.X), '_y': np.asarray(data.Y)}
    args.out.mkdir(parents=True, exist_ok=True)
    write(args.out / 'provenance.json', {
        'config': {k: v for k, v in cfg.items() if not k.startswith('_')},
        'data_sha256': cfg['data_sha256'], 'cache_sha256': cfg['cache_sha256'],
        'splits_sha256': cfg['splits_sha256'], 'traits': traits, 'phase': args.phase,
        'folds': [int(f['fold']) for f in folds], 'seeds': cfg['seeds'],
        'package': {'torch_rechub': importlib.metadata.version('torch-rechub'),
                    'torch': torch.__version__},
        'source': {'url': 'https://github.com/chaodian12/mtcro',
                   'upstream': 'torch_rechub MMOE (author entry)'},
        'adaptations': ['PCA95 fitted on each training split (author ships no Wheat2000 PCs)',
                        'eight traits as eight regression tasks; eight experts per the author call',
                        'canonical folds and seeds 42/2026 instead of the author 90/10 split'],
    })
    phase = args.phase
    if phase == 'dev':
        development(cfg, folds, traits, args)
    elif phase == 'merge':
        merge_selection(cfg, folds, traits, args)
    elif phase == 'final':
        if not args.selection:
            raise ValueError('final requires --selection')
        final(cfg, folds, traits, args)
    elif phase == 'report':
        report(cfg, folds, traits, args)
    else:
        raise ValueError('unknown phase')