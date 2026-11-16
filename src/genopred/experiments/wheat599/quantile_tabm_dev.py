"""Wheat599 independent full-marker quantile TabM development orchestration.

Mechanism (train/build_model/update_epoch/quantile_loss/mixture_scores) lives in
genopred.models.tabm.quantile (WP-B); this module owns the dev run/fit-worker/read
extracted from the original quantile_tabm.py at base 93c23fb. Source list anchored
to the tabm package directory.
"""
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import yaml

import genopred.models
import genopred.models.tabm
from genopred.data.wheat599 import load, sha256_of
from genopred.models.tabm.quantile import train, mixture_scores
from genopred.evaluation.metrics import pcc, rmse, mae

_tabm = Path(genopred.models.tabm.__file__).parent
_models = Path(genopred.models.__file__).parent
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
               _tabm/'backbone.py', Path(runner), Path(args.config),
               source/'provenance.json'] + [source/f"references_{f['fold']}_{j}.npz"
                                           for f in folds for j in range(3)]
    if baseline:
        sources += [_tabm/'interactions.py',baseline/'summary.json',Path(cfg['baseline_audit'])]
        sources += [baseline/f"predictions_{f['fold']}_{j}.npz" for f in folds for j in range(3)]
    if cfg.get('optimizer_variant')=='member_conditional':sources.append(_tabm/'interactions.py')
    if cfg.get('optimizer_variant')=='conditional':sources.append(_tabm/'quantile.py')
    if cfg.get('optimizer_variant')=='query':
        sources.extend((_tabm/('quantile.py' if name=='query_quantile.py' else 'backbone.py')) for name in ('query_quantile.py','induced_ftt.py'))
    if cfg.get('optimizer_variant') in ('ising','ising_independent','ising_mlp'):
        sources.append(_models/'structured'/'supervised_ising.py')
    if cfg.get('optimizer_variant') in ('treewidth_ising','treewidth_independent'):
        sources.extend(_models/'structured'/name for name in ('treewidth_ising.py','supervised_ising.py'))
    if cfg.get('optimizer_variant') in ('shared_basis','independent_basis'):
        sources.append(_tabm/'interactions.py')
    if cfg.get('optimizer_variant') in ('shape_free','shape_shared','shape_shrink'):
        sources.append(_tabm/'shape.py')
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
