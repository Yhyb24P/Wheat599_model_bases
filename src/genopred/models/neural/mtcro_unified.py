"""MtCro (torch_rechub MMOE) under the unified protocol, marker-only.

Author recipe kept: multi-gate mixture of experts with 4096-wide expert and
tower layers, Adam lr 1e-4 / weight decay 1e-5, 100 epochs, batch 640,
early-stopping patience 30. The author's maize script passes the task count in
the n_expert position, so the four-environment run uses four experts, four
gates and four towers.

Adaptations recorded rather than hidden: the author's Wheat599 template feeds
250 PCs (wheat599_pc95.pkl); this run feeds the raw 1279 markers. Folds are the
canonical manifest, not the author's 90/10 split. Seeds 42 and 2026: inner
early stopping per seed, selection scores per-environment inner PCC, each seed
refits for the median of its own three inner best epochs and the two test
predictions are averaged. Independent fits run in joblib processes, each
seeding itself.
"""
import csv
import importlib.metadata
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from joblib import Parallel, delayed
from torch_rechub.basic.features import DenseFeature
from torch_rechub.models.multi_task import MMOE

from genopred.data.wheat599 import load, sha256_of
from genopred.evaluation.metrics import pcc, rmse, mae

STATE = {}


def build(n_features, spec, n_tasks=4):
    features = [DenseFeature(str(i)) for i in range(n_features)]
    model = MMOE(features, ['regression'] * n_tasks, int(spec['experts']),
                 {'dims': list(spec['expert_dims']), 'dropout': float(spec['dropout'])},
                 [{'dims': list(spec['tower_dims'])}] * n_tasks)
    return model, [feature.name for feature in features]


def fit_predict(x, y, train, target, spec, cfg, seed, epochs, select):
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device(cfg['device'])
    model, names = build(x.shape[1], spec)
    model.to(device)
    xt = torch.as_tensor(x, dtype=torch.float32, device=device)
    yt = torch.as_tensor(y, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg['training']['learning_rate']),
                                 weight_decay=float(cfg['training']['weight_decay']))
    generator = torch.Generator().manual_seed(seed)
    best_pcc, best_epoch, best_state = -np.inf, 0, None
    batch_size = int(cfg['training']['batch_size'])
    for epoch in range(1, int(epochs) + 1):
        model.train()
        order = torch.randperm(len(train), generator=generator)
        for start in range(0, len(train), batch_size):
            batch = order[start:start + batch_size]
            rows = torch.as_tensor(train)[batch].to(device)
            inputs = {name: xt[rows, i].unsqueeze(1) for i, name in enumerate(names)}
            loss = torch.nn.functional.mse_loss(model(inputs), yt[rows])
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward(); optimizer.step()
        if select:
            model.eval()
            with torch.no_grad():
                rows = torch.as_tensor(target).to(device)
                pred = model({name: xt[rows, i].unsqueeze(1) for i, name in enumerate(names)})
            scores = [pcc(y[target, e], pred[:, e].cpu().numpy()) for e in range(y.shape[1])]
            score = float(np.mean(scores)) if all(np.isfinite(scores)) else -np.inf
            if score > best_pcc:
                best_pcc, best_epoch = score, epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            elif epoch - best_epoch >= int(cfg['training']['patience']):
                break
    if select:
        if best_state is None:
            raise ValueError('No finite validation PCC')
        model.load_state_dict(best_state)
        model.eval()
    with torch.no_grad():
        rows = torch.as_tensor(target).to(device)
        pred = model({name: xt[rows, i].unsqueeze(1) for i, name in enumerate(names)})
    return pred.cpu().numpy(), best_epoch, sum(p.numel() for p in model.parameters())


def init_worker(x, y, cfg):
    torch.set_num_threads(int(cfg['cpu_threads']))
    device = torch.device(cfg['device'])
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(float(cfg['gpu_memory_fraction']))
    STATE.update(x=x, y=y, cfg=cfg)


def dev_job(train_idx, val_idx, seed):
    try:
        cfg = STATE['cfg']
        pred, best_epoch, n_params = fit_predict(STATE['x'], STATE['y'], train_idx, val_idx,
                                                 cfg['architecture'], cfg, seed,
                                                 cfg['training']['epochs'], True)
        return {'pred': pred, 'best_epoch': best_epoch, 'n_params': n_params}
    except Exception as exc:  # noqa: BLE001
        return {'error': f'{type(exc).__name__}: {exc}'}


def outer_job(dev_idx, test_idx, seed, epochs):
    try:
        cfg = STATE['cfg']
        pred, _, _ = fit_predict(STATE['x'], STATE['y'], dev_idx, test_idx,
                                 cfg['architecture'], cfg, seed, epochs, False)
        return {'pred': pred}
    except Exception as exc:  # noqa: BLE001
        return {'error': f'{type(exc).__name__}: {exc}'}


def write(path, fields, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)


def run(cfg, args, runner):
    data = load(Path(cfg['data']), cfg['data_sha256'])
    folds = json.loads(Path(cfg['splits']).read_text())['folds']
    if len(folds) != 10 or any(len(f['inner']) != 3 for f in folds):
        raise ValueError('Unexpected splits')
    seeds = [int(s) for s in cfg['seeds']]
    jobs_n = int(cfg.get('jobs', 6))
    device = torch.device(cfg['device'])
    torch.set_num_threads(int(cfg['cpu_threads']))
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(float(cfg['gpu_memory_fraction']))
    args.out.mkdir(parents=True, exist_ok=False)
    versions = {'torch': torch.__version__, 'torch_rechub': importlib.metadata.version('torch-rechub')}
    phase = 'development' if args.selection is None else 'outer'
    selection = json.loads(args.selection.read_text()) if args.selection else None
    if selection and (selection['splits_sha256'] != sha256_of(Path(cfg['splits']))
                      or selection['config_sha256'] != sha256_of(args.config)):
        raise ValueError('Frozen selection does not match config or splits')
    (args.out/'provenance.json').write_text(json.dumps({
        'config': cfg, 'phase': phase, 'seeds': seeds, 'jobs': jobs_n, 'versions': versions,
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'data_sha256': cfg['data_sha256'], 'splits_sha256': sha256_of(Path(cfg['splits'])),
        'source': {'url': 'https://github.com/chaodian12/mtcro', 'upstream': 'torch_rechub MMOE'},
        'adaptations': ['raw 1279 markers instead of the author 250-PC template',
                        'canonical manifest folds instead of the author 90/10 split',
                        'four environments as four regression tasks',
                        'early stopping on inner macro PCC, patience 30'],
        'rule': 'inner macro PCC; per-seed median best epoch; full dev refit; seed-averaged test prediction',
        'parallel': 'joblib loky processes; per-fit seeding',
    }, indent=2)+'\n')
    failures = []
    start = time.perf_counter()
    x = np.asarray(data.X); y = np.asarray(data.Y)
    initargs = (x, y, cfg)
    if phase == 'development':
        jobs, keys = [], []
        for fold in folds:
            f = fold['fold']
            for j, split in enumerate(fold['inner']):
                tr, val = np.asarray(split['train_idx']), np.asarray(split['val_idx'])
                if set(tr) | set(val) != set(fold['dev_idx']) or (set(tr) | set(val)) & set(fold['test_idx']):
                    raise ValueError('Split leakage')
                for seed in seeds:
                    jobs.append((tr, val, seed)); keys.append((f, j, seed))
        results = Parallel(n_jobs=jobs_n, initializer=init_worker, initargs=initargs)(
            delayed(dev_job)(*job) for job in jobs)
        records = []
        for key, res in zip(keys, results):
            if 'error' in res:
                failures.append({'key': str(key), 'reason': res['error']})
                raise RuntimeError(f'development fit failed: {res["error"]}')
            records.append({'fold': key[0], 'inner': key[1], 'seed': key[2],
                            'best_epoch': res['best_epoch'], 'n_params': res['n_params'],
                            'inner_pcc': float(np.mean([pcc(y[np.asarray(folds[key[0]]['inner'][key[1]]['val_idx']), e],
                                                             res['pred'][:, e]) for e in range(y.shape[1])]))})
        for fold in folds:
            f = fold['fold']
            epochs = {}
            for seed in seeds:
                epochs[str(seed)] = sorted(r['best_epoch'] for r in records
                                           if r['fold'] == f and r['seed'] == seed)[1]
            (args.out/f'selection_{f}.json').write_text(json.dumps(
                {'fold': f, 'epochs': epochs,
                 'inner': [r for r in records if r['fold'] == f]}, indent=2))
            print(f'fold {f} selected epochs {epochs}', flush=True)
        (args.out/'selection.json').write_text(json.dumps({
            'folds': [json.loads((args.out/f"selection_{fold['fold']}.json").read_text()) for fold in folds],
            'splits_sha256': sha256_of(Path(cfg['splits'])),
            'config_sha256': sha256_of(args.config)}, indent=2))
        (args.out/'trajectories.json').write_text(json.dumps(records, indent=2))
        (args.out/'failures.json').write_text(json.dumps(failures, indent=2))
        (args.out/'timing.json').write_text(json.dumps({'seconds': time.perf_counter() - start,
                                                        'jobs': jobs_n}, indent=2))
        return
    chosen = {entry['fold']: entry for entry in selection['folds']}
    jobs, keys = [], []
    for fold in folds:
        f = fold['fold']; dev = np.asarray(fold['dev_idx']); test = np.asarray(fold['test_idx'])
        for seed in seeds:
            jobs.append((dev, test, seed, chosen[f]['epochs'][str(seed)])); keys.append((f, seed))
    results = Parallel(n_jobs=jobs_n, initializer=init_worker, initargs=initargs)(
        delayed(outer_job)(*job) for job in jobs)
    preds = {}
    for key, res in zip(keys, results):
        if 'error' in res:
            failures.append({'key': str(key), 'reason': res['error']})
            raise RuntimeError(f'outer fit failed: {res["error"]}')
        preds[key] = res['pred']
    rows, seedrows, oof, seedoof, undefined = [], [], [], [], []
    for fold in folds:
        f = fold['fold']; test = np.asarray(fold['test_idx'])
        for e, name in enumerate(data.environments):
            for seed in seeds:
                pred = preds[(f, seed)][:, e]
                seedrows.append({'fold': f, 'environment': name, 'seed': seed,
                                 'epochs': chosen[f]['epochs'][str(seed)],
                                 'pcc': pcc(y[test, e], pred), 'rmse': rmse(y[test, e], pred),
                                 'mae': mae(y[test, e], pred)})
                seedoof.extend({'fold': f, 'environment': name, 'seed': seed,
                                'material_index': int(i), 'y_true': float(y[i, e]), 'y_pred': float(p)}
                               for i, p in zip(test, pred))
            mean_pred = np.mean([preds[(f, s)][:, e] for s in seeds], axis=0)
            value = {'pcc': pcc(y[test, e], mean_pred), 'rmse': rmse(y[test, e], mean_pred),
                     'mae': mae(y[test, e], mean_pred)}
            if not np.isfinite(value['pcc']):
                undefined.append({'fold': f, 'environment': name, 'reason': 'constant prediction'})
            rows.append({'fold': f, 'environment': name, **value})
            oof.extend({'fold': f, 'environment': name, 'material_index': int(i),
                        'y_true': float(y[i, e]), 'y_pred': float(p)} for i, p in zip(test, mean_pred))
    write(args.out/'per_fold.csv', list(rows[0]), rows)
    write(args.out/'per_fold_seed.csv', list(seedrows[0]), seedrows)
    write(args.out/'oof.csv', ['fold', 'environment', 'material_index', 'y_true', 'y_pred'], oof)
    write(args.out/'oof_seed.csv', ['fold', 'environment', 'seed', 'material_index', 'y_true', 'y_pred'], seedoof)
    (args.out/'undefined_pcc.json').write_text(json.dumps(undefined, indent=2)+'\n')
    per_environment = {}
    for name in data.environments:
        rs = [r for r in rows if r['environment'] == name]
        entry = {'undefined_pcc': sum(1 for r in rs if not np.isfinite(r['pcc']))}
        for metric in ('pcc', 'rmse', 'mae'):
            values = [r[metric] for r in rs if np.isfinite(r[metric])]
            entry[metric] = float(np.mean(values)) if values else None
        per_environment[name] = entry
    macro = {'pcc': None if undefined else float(np.mean([per_environment[n]['pcc'] for n in data.environments])),
             'rmse': float(np.mean([per_environment[n]['rmse'] for n in data.environments])),
             'mae': float(np.mean([per_environment[n]['mae'] for n in data.environments]))}
    (args.out/'summary.json').write_text(json.dumps({
        'n_runs': len(rows), 'failures': [], 'seeds': seeds, 'jobs': jobs_n,
        'per_environment': per_environment, 'macro': macro,
        'undefined_pcc_count': len(undefined)}, indent=2)+'\n')
    (args.out/'failures.json').write_text(json.dumps(failures, indent=2))
    (args.out/'timing.json').write_text(json.dumps({'seconds': time.perf_counter() - start,
                                                    'jobs': jobs_n}, indent=2))
    print(json.dumps({'macro': macro, 'undefined': len(undefined)}), flush=True)
