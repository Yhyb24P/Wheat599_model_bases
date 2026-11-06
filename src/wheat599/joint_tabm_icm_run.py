"""Development-only paired trajectories for profiled and fixed Huber ICM."""
import csv
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from wheat599.data import load, sha256_of
from wheat599.joint_tabm_icm import train_unit
from wheat599.metrics import pcc
from wheat599.robust_icm import features, fit_predict


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
    source = Path(cfg['huber_source'])
    selection, diagnostics = read(source/'selection.json'), read(source/'optimization.json')
    bases = read(cfg['base_choices'])
    environment_selection = read(cfg['environment_selection'])
    environment_config = yaml.safe_load(Path(cfg['environment_config']).read_text())
    references = {}
    with Path(cfg['environment_predictions']).open() as handle:
        for row in csv.DictReader(handle):
            if row['variant'] == 'independent':
                key = (int(row['fold']), int(row['inner']), str(row['environment']),
                       int(row['seed']), int(row['config_index']), int(row['material_index']))
                if key in references:
                    raise ValueError('Duplicate independent TabM prediction')
                references[key] = float(row['y_pred'])
    if len(folds) != 10 or any(len(fold['inner']) != 3 for fold in folds):
        raise ValueError('Expected thirty inner splits')
    if read(source/'provenance.json')['splits_sha256'] != sha256_of(Path(cfg['splits'])):
        raise ValueError('Huber split mismatch')
    args.out.mkdir(parents=True, exist_ok=False)
    def write(name, obj):
        (args.out/name).write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')
    paths = [Path(__file__), Path(__file__).with_name('joint_tabm_icm.py'),
             Path(__file__).with_name('profiled_huber.py'), Path(runner), Path(args.config),
             Path(cfg['splits']), Path(cfg['environment_config']), Path(cfg['environment_selection']),
             Path(cfg['base_choices']), Path(cfg['environment_predictions']), source/'selection.json', source/'optimization.json']
    write('provenance.json', {'config': cfg,
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'sha256': {str(p): sha256_of(p) for p in paths}})
    state = {'status': 'running', 'completed': 0, 'expected': 120}
    records, failures, predictions = [], [], {}
    start = time.perf_counter()
    try:
        for fold in folds:
            f = fold['fold']
            chosen = next(v for v in environment_selection if v['fold'] == f and
                          v['training'] == 'independent' and v['selection'] == 'per_environment')
            configs = [environment_config['configurations'][i] for i in chosen['config_indices']]
            huber = next(v['huber'] for v in selection['selected'] if v['fold'] == f)
            for inner_number, inner in enumerate(fold['inner']):
                train, val = np.array(inner['train_idx']), np.array(inner['val_idx'])
                if (set(train) & set(val) or set(train) | set(val) != set(fold['dev_idx'])
                        or (set(train) | set(val)) & set(fold['test_idx'])):
                    raise ValueError('Invalid development split')
                base = next(v['homogeneous'] for v in bases if v['fold'] == f and v['inner_fold'] == inner_number)
                matches = [v for v in diagnostics if v['fold'] == f and v['inner_fold'] == inner_number and
                           v['model'] == 'huber' and all(v[k] == huber[k] for k in ('eta', 'delta', 'multiplier'))]
                if len(matches) != 1:
                    raise ValueError('Missing frozen thresholds')
                prepared = features(data.X, data.Y, train, val, base)
                huber_prediction, _ = fit_predict(prepared, base['ridge'][0]*huber['multiplier'],
                    huber['eta'], huber['delta'], selection['config'], fixed_thresholds=matches[0]['thresholds'])
                tabm_prediction = np.array([[np.mean([references[(f, inner_number, str(env), seed,
                    chosen['config_indices'][e], int(material))] for seed in cfg['seeds']])
                    for e, env in enumerate(data.environments)] for material in val])
                np.savez_compressed(args.out/f'references_{f}_{inner_number}.npz',
                                    huber=huber_prediction, independent_tabm=tabm_prediction,
                                    fixed_fusion=.5*(huber_prediction+tabm_prediction))
                for seed in cfg['seeds']:
                    for variant in ('fixed', 'joint'):
                        best, curve = train_unit(data.X[train], data.Y[train], data.X[val], data.Y[val],
                            prepared, base['ridge'][0]*huber['multiplier'], huber['eta'], matches[0]['thresholds'],
                            configs, seed, joint=variant == 'joint', epochs=cfg['epochs'],
                            patience=cfg['patience'], device=device)
                        key = f'{f}_{inner_number}_{seed}_{variant}'
                        predictions[key] = best.pop('prediction')
                        records.append(dict(fold=f, inner=inner_number, seed=seed, variant=variant,
                                            config_indices=chosen['config_indices'], **best, history=curve))
                        state['completed'] += 1
                        write('execution_status.json', state)
                np.savez_compressed(args.out/f'predictions_{f}_{inner_number}.npz', **predictions)
                predictions.clear()
                write('trajectories.json', records)
        if state['completed'] != state['expected']:
            raise ValueError('Incomplete experiment')
        per_fold = []
        for fold in folds:
            scores = {'fold': fold['fold']}
            for variant in ('fixed', 'joint', 'huber', 'independent_tabm', 'fixed_fusion'):
                values = []
                for j, inner in enumerate(fold['inner']):
                    if variant in ('fixed', 'joint'):
                        saved = np.load(args.out/f"predictions_{fold['fold']}_{j}.npz")
                        pred = np.mean([saved[f"{fold['fold']}_{j}_{seed}_{variant}"] for seed in cfg['seeds']], axis=0)
                    else:
                        saved = np.load(args.out/f"references_{fold['fold']}_{j}.npz")
                        pred = saved[variant]
                    truth = data.Y[inner['val_idx']]
                    values.append(float(np.mean([pcc(truth[:, e], pred[:, e]) for e in range(4)])))
                scores[variant] = float(np.mean(values))
            scores['delta'] = scores['joint'] - scores['fixed']
            per_fold.append(scores)
        write('summary.json', {'folds': per_fold, 'fixed': float(np.mean([v['fixed'] for v in per_fold])),
                               'joint': float(np.mean([v['joint'] for v in per_fold])),
                               'joint_minus_fixed': float(np.mean([v['delta'] for v in per_fold])),
                               'joint_wins': sum(v['delta'] > 0 for v in per_fold),
                               'references_development': {k: float(np.mean([v[k] for v in per_fold]))
                                                          for k in ('huber','independent_tabm','fixed_fusion')}})
        state['status'] = 'complete'
    except BaseException as error:
        state['status'] = 'failed'
        failures.append({'error': repr(error), 'completed': state['completed']})
        raise
    finally:
        write('trajectories.json', records)
        write('failures.json', failures)
        write('execution_status.json', state)
        write('timing.json', {'seconds': time.perf_counter()-start})
