"""Wheat2000 SOTA: official GSformer CSA/NSA architectures and recipe.

Model classes are imported unchanged from the pinned author checkout
(upstream/gsformer/code, commit 214c3831). The author training rules are kept:
Adam lr 5e-5 / weight decay 1e-3, MSE, batch 48, 25 epochs, early stop when
the validation loss improves by less than 1e-6 for five epochs. Adaptations
only: all 33709 markers zero-padded to 33768 = 469 x 72 (CSA: eight chunks of
4221; NSA: 469 tokens, 118 compressed blocks so the default 20 selected blocks
apply unchanged), canonical folds, seeds 42/2026, per-seed median best-PCC
epoch refit and seed-averaged test prediction.
"""
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from joblib import Parallel, delayed

from genopred.data.wheat599 import sha256_of
from genopred.evaluation.metrics import pcc, rmse, mae
from genopred.data.wheat2000 import load_wheat2000

STATE = {}


def load_classes(source_dir, commit):
    source = Path(source_dir)
    if not (source / 'code' / 'CSAformer.py').is_file():
        raise FileNotFoundError('GSformer source checkout missing')
    sys.path.insert(0, str(source / 'code'))
    sys.dont_write_bytecode = True
    try:
        import matplotlib.pyplot  # noqa: F401  (imported but unused by NSAformer)
    except ModuleNotFoundError:
        import types
        matplotlib = types.ModuleType('matplotlib')
        pyplot = types.ModuleType('matplotlib.pyplot')
        matplotlib.pyplot = pyplot
        sys.modules['matplotlib'] = matplotlib
        sys.modules['matplotlib.pyplot'] = pyplot
    from CSAformer import CSAformer
    from NSAformer import Transformer
    return CSAformer, Transformer


def build(variant, n_features, cfg, classes):
    CSAformer, Transformer = classes
    a = cfg['architecture']
    if variant == 'csa':
        return CSAformer(before_length=n_features, seq_length=int(a['seq_length']),
                         seq_num=int(a['seq_num']), head=int(a['heads']),
                         d_model=int(a['d_model']), d_ff=int(a['d_ff']),
                         dropout=float(a['dropout']), N=1)
    if variant == 'nsa':
        return Transformer(num_tokens=n_features, add_num=0, dim=int(a['dim']), depth=1,
                           dim_head=int(a['dim_head']), heads=int(a['heads']),
                           dropout=float(a['dropout']), num_length=n_features,
                           ff_expansion_factor=2)
    raise ValueError('unknown variant')


def fit(variant, classes, x, y, cfg, seed, train, query, select, epochs=None):
    torch.manual_seed(seed); np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    device = torch.device(cfg['device'])
    model = build(variant, x.shape[1], cfg, classes).to(device)
    xt = torch.as_tensor(x, device=device)
    yt = torch.as_tensor(y, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg['training']['learning_rate']),
                                 weight_decay=float(cfg['training']['weight_decay']))
    batch_size = int(cfg['training']['batch_size'])
    best, best_epoch, best_state = -np.inf, 0, None
    previous, stale = np.inf, 0
    budget = int(cfg['training']['epochs']) if epochs is None else int(epochs)
    for epoch in range(1, budget + 1):
        model.train()
        for start in range(0, len(train), batch_size):
            rows = torch.as_tensor(train[start:start + batch_size]).to(device)
            loss = torch.nn.functional.mse_loss(model(xt[rows]).ravel(), yt[rows])
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward(); optimizer.step()
        if not select:
            continue
        model.eval()
        with torch.no_grad():
            rows = torch.as_tensor(query).to(device)
            pred = model(xt[rows]).ravel()
        val_loss = float(torch.nn.functional.mse_loss(pred, yt[rows]))
        score = pcc(y[query], pred.cpu().numpy())
        if np.isfinite(score) and score > best:
            best, best_epoch = float(score), epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if previous - val_loss < float(cfg['training']['min_loss_reduction']):
            stale += 1
        else:
            stale = 0
        previous = val_loss
        if stale >= int(cfg['training']['patience']):
            break
    if select:
        if best_state is None:
            raise ValueError('No finite validation PCC')
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        rows = torch.as_tensor(query).to(device)
        pred = model(xt[rows]).ravel().cpu().numpy()
    return pred, best_epoch, epoch


def init_worker(x, y, cfg, classes):
    torch.set_num_threads(int(cfg['cpu_threads']))
    device = torch.device(cfg['device'])
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(float(cfg['gpu_memory_fraction']))
    STATE.update(x=x, y=y, cfg=cfg, classes=classes, device=device)


def dev_job(trait, seed, train_idx, val_idx):
    try:
        cfg = STATE['cfg']
        pred, best, epochs = fit(cfg['variant'], STATE['classes'], STATE['x'],
                                 STATE['y'][:, trait], cfg, int(seed), train_idx, val_idx, True)
        return {'pred': pred, 'best_epoch': best, 'epochs_run': epochs}
    except Exception as exc:  # noqa: BLE001
        return {'error': f'{type(exc).__name__}: {exc}'}


def final_job(trait, seed, dev_idx, test_idx, epochs):
    try:
        cfg = STATE['cfg']
        pred, _, _ = fit(cfg['variant'], STATE['classes'], STATE['x'], STATE['y'][:, trait],
                         cfg, int(seed), dev_idx, test_idx, False, epochs=epochs)
        return {'pred': pred}
    except Exception as exc:  # noqa: BLE001
        return {'error': f'{type(exc).__name__}: {exc}'}


def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=str) + '\n')


def development(cfg, folds, traits, args, classes):
    args.out.mkdir(parents=True, exist_ok=True)
    initargs = (np.asarray(cfg['_x']), np.asarray(cfg['_y']), cfg, classes)
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
                per_inner = [float(pcc(cfg['_y'][np.asarray(inner['val_idx']), trait],
                                       preds[(trait, int(seed))][i]))
                             for i, inner in enumerate(fold['inner'])]
                scores.append(float(np.mean(per_inner)))
            epochs = {str(int(seed)): int(sorted(r['best_epoch'] for r in records
                                                 if r['trait'] == traits[trait]
                                                 and r['seed'] == int(seed))[1])
                      for seed in cfg['seeds']}
            selected.append({'trait': traits[trait], 'epochs': epochs, 'scores': scores})
        write(args.out / f'selection_{k}.json', {'fold': k, 'selected': selected})
        write(args.out / f'trajectories_{k}.json', records)
        print(f'fold {k} selected', flush=True)


def merge_selection(cfg, folds, traits, args):
    document = {'folds': []}
    for fold in folds:
        document['folds'].append(json.loads((args.out / f'selection_{fold["fold"]}.json').read_text()))
    write(args.out / 'selection.json', document)
    print(f'merged {len(document["folds"])} folds', flush=True)


def final(cfg, folds, traits, args, classes):
    document = json.loads(Path(args.selection).read_text())
    by_fold = {int(entry['fold']): entry for entry in document['folds']}
    fold = folds[0]
    k = int(fold['fold'])
    epochs = {row['trait']: row['epochs'] for row in by_fold[k]['selected']}
    initargs = (np.asarray(cfg['_x']), np.asarray(cfg['_y']), cfg, classes)
    init_worker(*initargs)
    args.out.mkdir(parents=True, exist_ok=True)
    dev = np.asarray(fold['dev_idx']); test = np.asarray(fold['test_idx'])
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
    pad = int(cfg['pad_to'])
    if pad < data.X.shape[1]:
        raise ValueError('pad_to smaller than the marker count')
    x = np.zeros((data.X.shape[0], pad), dtype=np.float32)
    x[:, :data.X.shape[1]] = data.X
    classes = load_classes(cfg['source_dir'], cfg['source_commit'])
    cfg = {**cfg, '_x': x, '_y': np.asarray(data.Y)}
    args.out.mkdir(parents=True, exist_ok=True)
    write(args.out / 'provenance.json', {
        'config': {k: v for k, v in cfg.items() if not k.startswith('_')},
        'variant': cfg['variant'], 'data_sha256': cfg['data_sha256'],
        'cache_sha256': cfg['cache_sha256'], 'splits_sha256': cfg['splits_sha256'],
        'traits': traits, 'phase': args.phase, 'folds': [int(f['fold']) for f in folds],
        'seeds': cfg['seeds'],
        'source': {'url': 'https://github.com/hajudien/GSformer', 'commit': cfg['source_commit'],
                   'files': {name: sha256_of(Path(cfg['source_dir'])/'code'/name)
                             for name in ('CSAformer.py', 'NSAformer.py', 'native_sparse_attention.py')}},
        'adaptations': ['all 33709 markers zero-padded to 33768 = 469 x 72',
                        'CSA eight chunks of 4221; NSA 469 tokens with default 20 selected blocks',
                        'published wheat.Y as target; GY1-GY4 mapping not assumed',
                        'canonical folds, seeds 42/2026, median best-PCC epoch refit',
                        'matplotlib import stubbed when absent (unused by NSAformer)'],
    })
    phase = args.phase
    if phase == 'dev':
        development(cfg, folds, traits, args, classes)
    elif phase == 'merge':
        merge_selection(cfg, folds, traits, args)
    elif phase == 'final':
        if not args.selection:
            raise ValueError('final requires --selection')
        final(cfg, folds, traits, args, classes)
    elif phase == 'report':
        report(cfg, folds, traits, args)
    else:
        raise ValueError('unknown phase')