"""Development-only paired Huber ICM and TabM convex prediction fusion.

Recreates predictions using the existing frozen Huber development selections.
Weights are selected on inner validation and scores remain conditional selection
scores; no outer test predictions or labels are read.
"""
import csv
import json
from pathlib import Path

import numpy as np
import torch

from wheat599.data import load, sha256_of
from wheat599.retrieval_run import check_split, macro_pcc, macro_metrics
from wheat599.robust_icm import features, fit_predict


def choose_weight(units, weights):
    scores = []
    for weight in weights:
        score = float(np.mean([macro_pcc(weight * tabm + (1 - weight) * icm, truth)
                               for tabm, icm, truth in units]))
        if not np.isfinite(score):
            raise ValueError('Nonfinite fusion score')
        scores.append(score)
    best = max(range(len(weights)), key=lambda i: (round(scores[i], 12), -i))
    return float(weights[best]), scores


def run(cfg, args, runner):
    if args.selection is not None:
        raise ValueError('Development-only fusion')
    torch.set_num_threads(int(cfg.get('cpu_threads', 2)))
    data = load(Path(cfg['data']), cfg['data_sha256'])
    splits_path = Path(cfg['splits'])
    manifest = json.loads(splits_path.read_text())
    source = Path(cfg['retrieval_source'])
    source_info = json.loads((source / 'provenance.json').read_text())
    if source_info['splits_sha256'] != sha256_of(splits_path):
        raise ValueError('Split mismatch')
    if source_info['config']['data_sha256'] != cfg['data_sha256']:
        raise ValueError('Data mismatch')
    selections = json.loads((source / 'selection.json').read_text())
    huber_path = Path(cfg['huber_selection'])
    huber = json.loads(huber_path.read_text())
    if huber['splits_sha256'] != sha256_of(splits_path):
        raise ValueError('Huber split mismatch')
    bases_path = Path(huber['config']['base_source']) / 'base_choices.json'
    bases = json.loads(bases_path.read_text())
    if huber['base_source_sha256'] != sha256_of(bases_path):
        raise ValueError('Huber base choices mismatch')
    with (source / 'predictions.csv').open() as handle:
        raw_rows = [row for row in csv.DictReader(handle) if row['variant'] == 'raw']
    args.out.mkdir(parents=True, exist_ok=False)
    results, prediction_rows = [], []
    for fold in manifest['folds']:
        k = int(fold['fold'])
        if k not in cfg['outer_folds']:
            continue
        selected = selections[str(k)]['raw']['config_index']
        choice = next(row['huber'] for row in huber['selected'] if row['fold'] == k)
        units = []
        for j in range(3):
            fit, val = check_split(fold, j)
            rows = [row for row in raw_rows if int(row['fold']) == k
                    and int(row['inner']) == j and int(row['config_index']) == selected]
            paired = {}
            for row in rows:
                key = (int(row['seed']), int(row['material_index']))
                if key in paired:
                    raise ValueError('Duplicate prediction')
                paired[key] = [float(row[f'y_pred_{e}']) for e in range(4)]
            seeds = source_info['config']['seeds']
            if set(paired) != {(seed, int(i)) for seed in seeds for i in val}:
                raise ValueError('Prediction IDs do not match validation materials')
            tabm = np.mean([np.array([paired[(seed, int(i))] for i in val])
                            for seed in seeds], axis=0)
            base = next(row['homogeneous'] for row in bases
                        if row['fold'] == k and row['inner_fold'] == j)
            prepared = features(data.X, data.Y, fit, val, base)
            icm, _ = fit_predict(prepared, base['ridge'][0] * choice['multiplier'],
                                 choice['eta'], choice['delta'], huber['config'])
            units.append((tabm, icm, data.Y[val]))
            for material, a, b in zip(val, tabm, icm):
                prediction_rows.append({'fold': k, 'inner': j, 'material_index': int(material),
                                        'tabm': a.tolist(), 'icm': b.tolist()})
        weight, scores = choose_weight(units, cfg['weights'])
        results.append({'fold': k, 'tabm_weight': weight, 'grid_scores': scores,
                        'tabm_pcc': float(np.mean([macro_pcc(a, y) for a, b, y in units])),
                        'icm_pcc': float(np.mean([macro_pcc(b, y) for a, b, y in units])),
                        'fusion_pcc': max(scores),
                        'fusion_rmse': float(np.mean([macro_metrics(weight*a+(1-weight)*b,y)['rmse']
                                                       for a,b,y in units]))})
    (args.out / 'selection.json').write_text(json.dumps(results, indent=2))
    (args.out / 'predictions.json').write_text(json.dumps(prediction_rows))
    (args.out / 'provenance.json').write_text(json.dumps({
        'config': cfg, 'scope': 'conditional inner-validation selection only',
        'source_sha256': sha256_of(Path(__file__)),
        'retrieval_predictions_sha256': sha256_of(source / 'predictions.csv'),
        'retrieval_selection_sha256': sha256_of(source / 'selection.json'),
        'huber_selection_sha256': sha256_of(huber_path),
        'splits_sha256': sha256_of(splits_path)}, indent=2))
    return results
