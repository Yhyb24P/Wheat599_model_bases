"""Wheat2000 candidate: per-trait quantile TabM fused 0.5 with multi-trait Huber ICM.

Phases (run on two GPU shards with --folds):
  dev    inner 3-fold selection per (fold, trait): 8 configs x 2 seeds; selection
         scores seed-averaged predictions; per-seed refit epoch is the median of
         its three inner best epochs.
  final  refit every (fold, trait, seed) on the full development set for the
         frozen epoch count; test predictions are saved, not scored.
  report scores after all fits: TabM seed mean, optional 0.5 fusion with the
         Huber ICM predictions, per-trait and pooled metrics.

The frozen protocol supplies the splits; this module only adds the candidate.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
from joblib import Parallel, delayed

from genopred.data.wheat599 import sha256_of
from genopred.evaluation.metrics import pcc, rmse, mae
from genopred.models.tabular.quantile_tabm import train as train_tabm
from genopred.data.wheat2000 import load_wheat2000

STATE = {}


def load_protocol(cfg):
    data = load_wheat2000(Path(cfg['data']), cfg['data_sha256'],
                          Path(cfg['cache']), cfg['cache_sha256'])
    split_path = Path(cfg['splits'])
    if sha256_of(split_path) != cfg['splits_sha256']:
        raise ValueError('Frozen Wheat2000 splits SHA differs')
    splits = json.loads(split_path.read_text())
    if list(data.trait_names) != cfg['traits'] or splits['trait_names'] != cfg['traits']:
        raise ValueError('Trait names/order differ from the frozen protocol')
    return data, splits['folds'], list(data.trait_names)


def settings(cfg, fixed_epochs=None):
    out = {'n_quantiles': int(cfg['n_quantiles']), 'quantile_min': float(cfg['quantile_min']),
           'quantile_max': float(cfg['quantile_max']),
           'training': dict(cfg['training']),
           'device': cfg['device'], 'cpu_threads': int(cfg['cpu_threads'])}
    if fixed_epochs is not None:
        out['fixed_epochs'] = int(fixed_epochs)
    return out


def init_worker(x, y, cfg):
    torch.set_num_threads(int(cfg['cpu_threads']))
    device = torch.device(cfg['device'])
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(float(cfg['gpu_memory_fraction']))
    STATE.update(x=x, y=y, cfg=cfg, device=device)


def dev_job(trait, config_index, seed, train_idx, val_idx):
    try:
        cfg = STATE['cfg']
        x, y = STATE['x'], STATE['y']
        pred, details = train_tabm(x[train_idx], y[train_idx, trait], x[val_idx],
                                   y[val_idx, trait], cfg['configurations'][config_index],
                                   int(seed), int(config_index), settings(cfg), STATE['device'])
        return {'pred': pred, 'best_epoch': details['best_epoch'],
                'epochs_run': details['epochs_run'], 'seconds': details['seconds']}
    except Exception as exc:  # noqa: BLE001
        return {'error': f'{type(exc).__name__}: {exc}'}


def final_job(trait, seed, epoch, dev_ids, test_ids):
    try:
        cfg = STATE['cfg']
        x, y = STATE['x'], STATE['y']
        pred, details = train_tabm(x[dev_ids], y[dev_ids, trait], x[test_ids],
                                   y[test_ids, trait], cfg['configurations'][cfg['_chosen'][trait]],
                                   int(seed), int(cfg['_chosen'][trait]), settings(cfg, epoch),
                                   STATE['device'])
        return {'pred': pred, 'seconds': details['seconds']}
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
            for config_index in range(len(cfg['configurations'])):
                for seed in cfg['seeds']:
                    for inner in fold['inner']:
                        jobs.append((trait, config_index, int(seed),
                                     np.asarray(inner['train_idx']), np.asarray(inner['val_idx'])))
                        keys.append((trait, config_index, int(seed)))
        started = time.perf_counter()
        results = Parallel(n_jobs=int(cfg['workers']), initializer=init_worker, initargs=initargs)(
            delayed(dev_job)(*job) for job in jobs)
        preds, records = {}, []
        for key, res in zip(keys, results):
            if 'error' in res:
                raise RuntimeError(f'development fit failed {key}: {res["error"]}')
            preds.setdefault(key, []).append(res['pred'])
            records.append({'trait': traits[key[0]], 'config_index': key[1], 'seed': key[2],
                            'best_epoch': res['best_epoch'], 'epochs_run': res['epochs_run'],
                            'seconds': res['seconds']})
        selected = []
        for trait in range(len(traits)):
            scores = []
            for config_index in range(len(cfg['configurations'])):
                fold_scores = []
                for inner_index, inner in enumerate(fold['inner']):
                    val = np.asarray(inner['val_idx'])
                    mean_pred = np.mean([preds[(trait, config_index, int(seed))][inner_index]
                                         for seed in cfg['seeds']], axis=0)
                    fold_scores.append(float(pcc(cfg['_y'][val, trait], mean_pred)))
                scores.append(float(np.mean(fold_scores)))
            winner = int(np.argmax(scores))
            epochs = {}
            for seed in cfg['seeds']:
                best = [r['best_epoch'] for r in records if r['trait'] == traits[trait]
                        and r['config_index'] == winner and r['seed'] == int(seed)]
                epochs[str(int(seed))] = int(sorted(best)[1])
            selected.append({'trait': traits[trait], 'config_index': winner,
                             'epochs': epochs, 'scores': scores})
        write(args.out / f'selection_{k}.json',
              {'fold': k, 'selected': selected, 'config_sha256': sha256_of(args.config),
               'splits_sha256': cfg['splits_sha256']})
        write(args.out / f'trajectories_{k}.json', records)
        print(f'fold {k} selected in {time.perf_counter()-started:.0f}s', flush=True)


def merge_selection(cfg, folds, traits, args):
    document = {'folds': [], 'splits_sha256': cfg['splits_sha256'],
                'config_sha256': sha256_of(args.config)}
    for fold in folds:
        k = int(fold['fold'])
        document['folds'].append(json.loads((args.out / f'selection_{k}.json').read_text()))
    write(args.out / 'selection.json', document)
    print(f'merged {len(document["folds"])} folds', flush=True)


def final(cfg, folds, traits, args):
    document = json.loads(Path(args.selection).read_text())
    by_fold = {int(entry['fold']): entry for entry in document['folds']}
    fold = folds[0]
    k = int(fold['fold'])
    selection = by_fold[k]
    chosen = {row['trait']: row for row in selection['selected']}
    cfg = {**cfg, '_chosen': {traits.index(name): row['config_index']
                              for name, row in chosen.items()}}
    epochs = {row['trait']: row['epochs'] for row in selection['selected']}
    initargs = (np.asarray(cfg['_x']), np.asarray(cfg['_y']), cfg)
    init_worker(*initargs)
    args.out.mkdir(parents=True, exist_ok=True)
    for fold in folds:
        if int(fold['fold']) != k:
            raise ValueError('One final invocation handles one fold')
        dev = np.asarray(fold['dev_idx'])
        test = np.asarray(fold['test_idx'])
        jobs, keys = [], []
        for trait in range(len(traits)):
            for seed in cfg['seeds']:
                jobs.append((trait, int(seed), int(epochs[traits[trait]][str(int(seed))]),
                             dev, test))
                keys.append((trait, int(seed)))
        results = Parallel(n_jobs=int(cfg['workers']), initializer=init_worker, initargs=initargs)(
            delayed(final_job)(*job) for job in jobs)
        arrays, seconds = {}, {}
        for key, res in zip(keys, results):
            if 'error' in res:
                raise RuntimeError(f'final fit failed {key}: {res["error"]}')
            arrays[f'{traits[key[0]]}_{key[1]}'] = np.asarray(res['pred'], dtype=np.float64)
            seconds[f'{traits[key[0]]}_{key[1]}'] = res['seconds']
        arrays['test_ids'] = test
        np.savez_compressed(args.out / f'predictions_{k}.npz', **arrays)
        write(args.out / f'timings_{k}.json', seconds)
        print(f'fold {k} final fits complete', flush=True)


def report(cfg, folds, traits, args):
    rows, seed_rows, oof, tabm_rows = [], [], [], []
    data_y = np.asarray(cfg['_y'])
    huber = {}
    if args.huber:
        for path in sorted(Path(args.huber).glob('predictions_*.npz')):
            z = np.load(path)
            fold = int(path.stem.split('_')[1])
            huber[fold] = {name: z[name] for name in z.files if name not in ('test_ids',)}
    for fold in folds:
        k = int(fold['fold'])
        test = np.asarray(fold['test_idx'])
        z = np.load(args.out / f'predictions_{k}.npz')
        if not np.array_equal(np.asarray(z['test_ids']), test):
            raise ValueError('Saved test ids disagree with the manifest')
        for trait in traits:
            e = traits.index(trait)
            preds = [np.asarray(z[f'{trait}_{seed}']) for seed in cfg['seeds']]
            tabm = np.mean(preds, axis=0)
            tabm_value = {'fold': k, 'trait': trait,
                          'pcc': float(pcc(data_y[test, e], tabm)),
                          'rmse': float(rmse(data_y[test, e], tabm)),
                          'mae': float(mae(data_y[test, e], tabm))}
            if all(np.isfinite(v) for v in (tabm_value['pcc'], tabm_value['rmse'], tabm_value['mae'])):
                tabm_rows.append(tabm_value)
            if k in huber and trait in huber[k]:
                fused = 0.5 * (tabm + np.asarray(huber[k][trait]))
                method = 'fusion'
            else:
                fused = tabm
                method = 'tabm'
            value = {'fold': k, 'trait': trait, 'method': method,
                     'pcc': float(pcc(data_y[test, e], fused)),
                     'rmse': float(rmse(data_y[test, e], fused)),
                     'mae': float(mae(data_y[test, e], fused))}
            if not all(np.isfinite(v) for v in (value['pcc'], value['rmse'], value['mae'])):
                raise ValueError('Non-finite or undefined test metric')
            rows.append(value)
            for seed, pred in zip(cfg['seeds'], preds):
                seed_rows.append({'fold': k, 'trait': trait, 'seed': int(seed),
                                  'pcc': float(pcc(data_y[test, e], pred)),
                                  'rmse': float(rmse(data_y[test, e], pred)),
                                  'mae': float(mae(data_y[test, e], pred))})
            oof.extend({'fold': k, 'trait': trait, 'material_index': int(i),
                        'y_true': float(data_y[i, e]), 'y_pred': float(v)}
                       for i, v in zip(test, fused))
    trait_metrics = {trait: {m: float(np.mean([r[m] for r in rows if r['trait'] == trait]))
                             for m in ('pcc', 'rmse', 'mae')} for trait in traits}
    pooled = {}
    for trait in traits:
        truth = np.array([r['y_true'] for r in oof if r['trait'] == trait])
        pred = np.array([r['y_pred'] for r in oof if r['trait'] == trait])
        pooled[trait] = {'pcc': float(pcc(truth, pred)), 'rmse': float(rmse(truth, pred)),
                         'mae': float(mae(truth, pred))}
    tabm_trait = {trait: {m: float(np.mean([r[m] for r in tabm_rows if r['trait'] == trait]))
                          for m in ('pcc', 'rmse', 'mae')} for trait in traits}
    write(args.out / 'summary.json', {
        'n_trait_folds': len(rows), 'method': rows[0]['method'] if rows else None,
        'primary': {m: float(np.mean([r[m] for r in rows])) for m in ('pcc', 'rmse', 'mae')},
        'per_trait': trait_metrics, 'pooled_oof': pooled,
        'tabm_only': {'primary': {m: float(np.mean([r[m] for r in tabm_rows]))
                                  for m in ('pcc', 'rmse', 'mae')},
                      'per_trait': tabm_trait},
        'seeds': list(cfg['seeds'])})
    with (args.out / 'per_fold.csv').open('w') as handle:
        handle.write('fold,trait,pcc,rmse,mae\n')
        for r in rows:
            handle.write(f'{r["fold"]},{r["trait"]},{r["pcc"]:.10g},{r["rmse"]:.10g},{r["mae"]:.10g}\n')
    with (args.out / 'per_fold_tabm.csv').open('w') as handle:
        handle.write('fold,trait,pcc,rmse,mae\n')
        for r in tabm_rows:
            handle.write(f'{r["fold"]},{r["trait"]},{r["pcc"]:.10g},{r["rmse"]:.10g},{r["mae"]:.10g}\n')
    with (args.out / 'per_seed.csv').open('w') as handle:
        handle.write('fold,trait,seed,pcc,rmse,mae\n')
        for r in seed_rows:
            handle.write(f'{r["fold"]},{r["trait"]},{r["seed"]},{r["pcc"]:.10g},{r["rmse"]:.10g},{r["mae"]:.10g}\n')
    with (args.out / 'oof.csv').open('w') as handle:
        handle.write('fold,trait,material_index,y_true,y_pred\n')
        for r in oof:
            handle.write(f'{r["fold"]},{r["trait"]},{r["material_index"]},{r["y_true"]:.10g},{r["y_pred"]:.10g}\n')
    print(json.dumps({'primary': {m: float(np.mean([r[m] for r in rows])) for m in ('pcc', 'rmse', 'mae')}}), flush=True)


def run(cfg, args, runner):
    data, folds, traits = load_protocol(cfg)
    if getattr(args, 'traits', None):
        if not set(args.traits) <= set(traits):
            raise ValueError('Unknown trait requested')
        traits = [t for t in traits if t in set(args.traits)]
    if args.folds:
        wanted = set(int(f) for f in args.folds)
        folds = [f for f in folds if int(f['fold']) in wanted]
        if len(folds) != len(wanted):
            raise ValueError('Requested folds missing from the manifest')

    cfg = {**cfg, '_x': np.asarray(data.X), '_y': np.asarray(data.Y)}
    args.out.mkdir(parents=True, exist_ok=True)
    tag = 'all' if not args.folds else '-'.join(map(str, sorted(int(f) for f in args.folds)))
    write(args.out / f'provenance_{tag}.json', {
        'config': {k: v for k, v in cfg.items() if not k.startswith('_')},
        'data_sha256': cfg['data_sha256'], 'cache_sha256': cfg['cache_sha256'],
        'splits_sha256': cfg['splits_sha256'], 'traits': traits,
        'source_sha256': {str(p): sha256_of(Path(p)) for p in
                          (Path(__file__), Path(args.config), Path(runner))},
        'phase': args.phase, 'folds': [int(f['fold']) for f in folds],
        'seeds': cfg['seeds'], 'workers': cfg['workers'],
        'protocol': 'inner 3-fold selection; per-seed median best epoch; full-dev refit; '
                    '0.5 fixed fusion with Huber ICM when report is given --huber',
    })
    if args.phase == 'dev':
        development(cfg, folds, traits, args)
    elif args.phase == 'final':
        if not args.selection:
            raise ValueError('final requires --selection')
        final(cfg, folds, traits, args)
    elif args.phase == 'merge':
        merge_selection(cfg, folds, traits, args)
    elif args.phase == 'report':
        report(cfg, folds, traits, args)
    else:
        raise ValueError('unknown phase')
