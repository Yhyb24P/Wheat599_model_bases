"""Wheat2000 multi-trait Huber ICM: inner selection, full-dev refit, test once.

The material kernel follows the frozen Wheat599 Huber ICM (additive/RBF mix,
fit-centred markers, selected lambda). The task kernel uses all eight traits
jointly. Selection mirrors the frozen robust_icm runner: per fold, the shared
base kernel/lambda is chosen inside the three inner splits, then eta/delta/
penalty-multiplier are selected by mean inner macro PCC. Final refit predicts
each outer test fold once and saves per-trait arrays.
"""
import itertools
import json
from pathlib import Path

import numpy as np

from wheat599.data import sha256_of
from wheat599.heterogeneous_icm import choose
from wheat599.metrics import pcc, rmse, mae
from wheat599.robust_icm import features, fit_predict
from wheat599.wheat2000_data import load_wheat2000

METRICS = ('pcc', 'rmse', 'mae')


def load(cfg):
    data = load_wheat2000(Path(cfg['data']), cfg['data_sha256'],
                          Path(cfg['cache']), cfg['cache_sha256'])
    split_path = Path(cfg['splits'])
    if sha256_of(split_path) != cfg['splits_sha256']:
        raise ValueError('Frozen Wheat2000 splits SHA differs')
    splits = json.loads(split_path.read_text())
    if list(data.trait_names) != cfg['traits'] or splits['trait_names'] != cfg['traits']:
        raise ValueError('Trait names/order differ from the frozen protocol')
    return data, splits['folds'], list(data.trait_names)


def base_config(cfg):
    return {key: cfg[key] for key in ('bandwidths', 'additive_fractions', 'ridge')}


def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=str) + '\n')


def development(cfg, data, folds, traits, args):
    args.out.mkdir(parents=True, exist_ok=True)
    grid = list(itertools.product(cfg['etas'], cfg['huber_deltas'], cfg['ridge_multipliers']))
    squared = [(eta, None, 1.0) for eta in cfg['etas']]
    base_cfg = base_config(cfg)
    selections = []
    for fold in folds:
        k = int(fold['fold'])
        splits = [(np.asarray(inner['train_idx']), np.asarray(inner['val_idx']))
                  for inner in fold['inner']]
        for j, (train, valid) in enumerate(splits):
            if set(train) & set(valid) or set(train) | set(valid) != set(fold['dev_idx']) \
                    or set(np.r_[train, valid]) & set(fold['test_idx']):
                raise ValueError('Invalid development split')
        _, homo, _ = choose(data.X, data.Y, splits, base_cfg, 'cpu')
        rows = []
        for j, (train, valid) in enumerate(splits):
            prepared = features(data.X, data.Y, train, valid, homo)
            for model, configs in (('squared', squared), ('huber', grid)):
                for eta, delta, multiplier in configs:
                    pred, _ = fit_predict(prepared, homo['ridge'][0] * multiplier, eta, delta, cfg)
                    entry = {'model': model, 'inner': j, 'eta': eta, 'delta': delta,
                             'multiplier': multiplier}
                    for e, trait in enumerate(traits):
                        entry[trait] = {'pcc': float(pcc(data.Y[valid, e], pred[:, e])),
                                        'rmse': float(rmse(data.Y[valid, e], pred[:, e])),
                                        'mae': float(mae(data.Y[valid, e], pred[:, e]))}
                    rows.append(entry)
        chosen = {}
        for model, configs in (('squared', squared), ('huber', grid)):
            means = []
            for eta, delta, multiplier in configs:
                values = [r[trait]['pcc'] for r in rows if r['model'] == model
                          and (r['eta'], r['delta'], r['multiplier']) == (eta, delta, multiplier)]
                means.append(float(np.mean(values)))
            index = next(i for i, score in enumerate(means)
                         if max(means) - score <= cfg['tie_tolerance'])
            eta, delta, multiplier = configs[index]
            chosen[model] = {'eta': eta, 'delta': delta, 'multiplier': multiplier,
                             'inner_pcc': means[index]}
        selections.append({'fold': k, 'base': homo, **chosen})
        write(args.out / f'selection_{k}.json', selections[-1])
        write(args.out / f'validation_{k}.json', rows)
        print(f'fold {k} selected', flush=True)
    deltas = [s['huber']['inner_pcc'] - s['squared']['inner_pcc'] for s in selections]
    gate = {'pcc_gain': float(np.mean(deltas)), 'folds_improved': int(sum(v > 0 for v in deltas)),
            'per_fold_delta': deltas}
    gate['pass'] = (gate['pcc_gain'] >= float(cfg['minimum_development_gain'])
                    and gate['folds_improved'] >= int(cfg['minimum_development_folds']))
    write(args.out / 'development_gate.json', gate)
    write(args.out / 'selection.json', {
        'splits_sha256': cfg['splits_sha256'], 'config_sha256': sha256_of(args.config),
        'development_gate': gate, 'selected': selections})
    print(json.dumps(gate), flush=True)


def final(cfg, data, folds, traits, args):
    selection = json.loads(args.selection.read_text())
    if selection['splits_sha256'] != cfg['splits_sha256'] \
            or selection['config_sha256'] != sha256_of(args.config):
        raise ValueError('Selection/config/splits mismatch')
    by_fold = {s['fold']: s for s in selection['selected']}
    args.out.mkdir(parents=True, exist_ok=True)
    for fold in folds:
        k = int(fold['fold'])
        entry = by_fold[k]
        choice = entry['huber'] if selection['development_gate']['pass'] else entry['squared']
        train, test = np.asarray(fold['dev_idx']), np.asarray(fold['test_idx'])
        if set(train) & set(test) or len(set(train) | set(test)) != data.X.shape[0]:
            raise ValueError('Invalid outer split')
        prepared = features(data.X, data.Y, train, test, entry['base'])
        pred, _ = fit_predict(prepared, entry['base']['ridge'][0] * choice['multiplier'],
                              choice['eta'], choice['delta'], cfg)
        arrays = {trait: np.asarray(pred[:, e], dtype=np.float64)
                  for e, trait in enumerate(traits)}
        arrays['test_ids'] = test
        np.savez_compressed(args.out / f'predictions_{k}.npz', **arrays)
        write(args.out / f'choice_{k}.json', {'fold': k, 'choice': choice,
                                              'base': entry['base']})
        print(f'fold {k} final complete', flush=True)


def report(cfg, data, folds, traits, args):
    rows = []
    for fold in folds:
        k = int(fold['fold'])
        test = np.asarray(fold['test_idx'])
        z = np.load(args.out / f'predictions_{k}.npz')
        if not np.array_equal(np.asarray(z['test_ids']), test):
            raise ValueError('Saved test ids disagree with the manifest')
        for e, trait in enumerate(traits):
            rows.append({'fold': k, 'trait': trait,
                         'pcc': float(pcc(data.Y[test, e], z[trait])),
                         'rmse': float(rmse(data.Y[test, e], z[trait])),
                         'mae': float(mae(data.Y[test, e], z[trait]))})
    write(args.out / 'summary.json', {
        'n_trait_folds': len(rows),
        'primary': {m: float(np.mean([r[m] for r in rows])) for m in METRICS},
        'per_trait': {t: {m: float(np.mean([r[m] for r in rows if r['trait'] == t]))
                          for m in METRICS} for t in traits}})
    with (args.out / 'per_fold.csv').open('w') as handle:
        handle.write('fold,trait,pcc,rmse,mae\n')
        for r in rows:
            handle.write(f'{r["fold"]},{r["trait"]},{r["pcc"]:.10g},{r["rmse"]:.10g},{r["mae"]:.10g}\n')
    print(json.dumps({'primary': {m: float(np.mean([r[m] for r in rows])) for m in METRICS}}), flush=True)


def run(cfg, args, runner):
    data, folds, traits = load(cfg)
    if args.folds:
        wanted = set(int(f) for f in args.folds)
        folds = [f for f in folds if int(f['fold']) in wanted]
        if len(folds) != len(wanted):
            raise ValueError('Requested folds missing from the manifest')
    if getattr(args, 'traits', None):
        traits = [t for t in traits if t in set(args.traits)]
    args.out.mkdir(parents=True, exist_ok=True)
    write(args.out / 'provenance.json', {
        'config': cfg, 'data_sha256': cfg['data_sha256'], 'cache_sha256': cfg['cache_sha256'],
        'splits_sha256': cfg['splits_sha256'], 'traits': traits, 'phase': args.phase,
        'folds': [int(f['fold']) for f in folds],
        'source_sha256': {str(p): sha256_of(Path(p)) for p in
                          (Path(__file__), Path(args.config), Path(runner))},
        'rule': 'inner 3-fold macro PCC selects eta/delta/multiplier and the shared base; '
                'development gate as frozen robust_icm; full-dev refit; test once',
    })
    if args.phase == 'dev':
        development(cfg, data, folds, traits, args)
    elif args.phase == 'final':
        if not args.selection:
            raise ValueError('final requires --selection')
        final(cfg, data, folds, traits, args)
    elif args.phase == 'report':
        report(cfg, data, folds, traits, args)
    else:
        raise ValueError('unknown phase')
