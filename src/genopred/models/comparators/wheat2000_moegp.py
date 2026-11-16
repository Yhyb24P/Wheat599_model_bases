"""Wheat2000 SOTA: official MoEGP gating network and recipe.

The gating model, auxiliary loss and search defaults come from the author
repository (commit d3a3c44, utils/moe.Gating): lr 1e-4, dropout 0.3,
hidden [512], six experts, top-k 2, loss coefficient 0.1, batch 32,
100 epochs, fit-only phenotype z-scoring. Protocol adaptations only: all
33709 markers as input (the author pipeline likewise trains on raw SNPs),
one model per trait, canonical folds, seeds 42/2026, inner early stop,
per-seed median best epoch refit and seed-averaged test prediction.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from joblib import Parallel, delayed

from genopred.data.wheat599 import sha256_of
from genopred.evaluation.metrics import pcc, rmse, mae
from genopred.models.comparators.moegp_worker import fit_moegp
from genopred.data.wheat2000 import load_wheat2000

STATE = {}


def init_worker(x, y, cfg):
    sys.path.insert(0, cfg['source_dir'])
    sys.dont_write_bytecode = True
    from utils.moe import creat_mlp_layer
    torch.set_num_threads(int(cfg['cpu_threads']))
    device = torch.device(cfg['device'])
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(float(cfg['gpu_memory_fraction']))
    STATE.update(x=torch.as_tensor(x, dtype=torch.float32, device=device),
                 y=y, cfg=cfg, device=device, create=creat_mlp_layer)


def dev_job(trait, seed, train_idx, val_idx):
    try:
        cfg = STATE['cfg']
        res = fit_moegp(STATE['x'], STATE['y'][:, trait], np.asarray(train_idx),
                        np.asarray(val_idx), cfg['candidate'], cfg['model'], cfg['training'],
                        int(cfg['training']['epochs']), STATE['device'], int(seed), STATE['create'])
        return {'pred': res['val_pred'], 'best_epoch': res['best_epoch'],
                'epochs_run': len(res['lrs'])}
    except Exception as exc:  # noqa: BLE001
        return {'error': f'{type(exc).__name__}: {exc}'}


def final_job(trait, seed, dev_idx, test_idx, epochs):
    try:
        cfg = STATE['cfg']
        res = fit_moegp(STATE['x'], STATE['y'][:, trait], np.asarray(dev_idx), None,
                        cfg['candidate'], cfg['model'], cfg['training'], int(epochs),
                        STATE['device'], int(seed), STATE['create'], lr_schedule=None)
        model = res['model']
        rows = torch.as_tensor(np.asarray(test_idx), device=STATE['device'])
        with torch.no_grad():
            pred = model(STATE['x'][rows])[0].detach().float().cpu().numpy().ravel()
        return {'pred': pred * res['sd'] + res['mu']}
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
        for trait in range(len(traits)):
            for seed in cfg['seeds']:
                for inner in fold['inner']:
                    jobs.append((trait, int(seed), inner['train_idx'], inner['val_idx']))
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


def final(cfg, folds, traits, args):
    document = json.loads(Path(args.selection).read_text())
    by_fold = {int(entry['fold']): entry for entry in document['folds']}
    fold = folds[0]
    k = int(fold['fold'])
    epochs = {row['trait']: row['epochs'] for row in by_fold[k]['selected']}
    initargs = (np.asarray(cfg['_x']), np.asarray(cfg['_y']), cfg)
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
    cfg = {**cfg, '_x': np.asarray(data.X), '_y': np.asarray(data.Y)}
    args.out.mkdir(parents=True, exist_ok=True)
    write(args.out / 'provenance.json', {
        'config': {k: v for k, v in cfg.items() if not k.startswith('_')},
        'data_sha256': cfg['data_sha256'], 'cache_sha256': cfg['cache_sha256'],
        'splits_sha256': cfg['splits_sha256'], 'traits': traits, 'phase': args.phase,
        'folds': [int(f['fold']) for f in folds], 'seeds': cfg['seeds'],
        'source': {'url': 'https://github.com/CNRRI-RGRT/MoEGP', 'commit': cfg['source_commit']},
        'adaptations': ['raw 33709 markers as input; official recipe lr1e-4/dropout.3/hidden512/6 experts/topk2/batch32/100ep',
                        'one model per trait; canonical folds; seeds 42/2026; median best epoch refit',
                        'fit-only phenotype z-scoring'],
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