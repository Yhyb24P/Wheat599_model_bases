"""WheatGP official network under the unified protocol with seeds 42 and 2026.

The public network and its published defaults are kept. Selection scores the
seed-averaged inner validation prediction per candidate; each seed then refits
on the full development set for the median of its own three inner best epochs.
Constant predictions keep an undefined PCC: rows are retained, the macro PCC is
reported as null, and the author's constant-as-zero convention is descriptive
only. Ported from the frozen wheatgp_nested runner; the paired quantile/huber
comparison is not part of this run. Independent fits run in joblib processes;
each fit seeds itself, so the per-fit results match a serial run.
"""
import csv
import importlib.util
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from joblib import Parallel, delayed
from sklearn.decomposition import PCA

from wheat599.data import load, sha256_of
from wheat599.metrics import pcc, rmse, mae

STATE = {}


def prepare(x, train, query, kind, seed, cfg):
    if set(train) & set(query):
        raise ValueError('Overlapping fit and query materials')
    if kind == 'raw':
        return x[train].astype('float32'), x[query].astype('float32')
    pca = PCA(n_components=cfg['pca_components'], svd_solver='randomized', random_state=seed)
    return pca.fit_transform(x[train]).astype('float32'), pca.transform(x[query]).astype('float32')


def segments(x):
    width = x.shape[1] // 5
    if width < 5:
        raise ValueError('Insufficient feature count')
    return torch.split(x, [width] * 4 + [x.shape[1] - 4 * width], dim=1)


def validation_pcc(y, prediction):
    if not np.isfinite(prediction).all():
        raise ValueError('Nonfinite prediction')
    if np.std(y) == 0 or np.std(prediction) == 0:
        return 0.0  # The public WheatGP training script defines this case as zero.
    return pcc(y, prediction)


def _seed(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def fit(model_type, xf, yf, xq, yq, lr, epochs, cfg, device, seed, select=True):
    _seed(seed)
    model = model_type(8 * (xf.shape[1] - 20)).to(device)
    x = torch.as_tensor(xf, device=device); q = torch.as_tensor(xq, device=device)
    center, scale = float(yf.mean()), float(yf.std())
    if scale <= 0: raise ValueError('Constant training target')
    y = torch.as_tensor((yf - center)[:, None] / scale, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=90, gamma=0.1)
    generator = torch.Generator(device='cpu').manual_seed(seed)
    best, best_epoch, best_pred = -np.inf, 0, None
    for epoch in range(1, epochs + 1):
        model.train()
        for idx in torch.randperm(len(x), generator=generator).split(cfg['batch_size']):
            idx = idx.to(device); optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(*segments(x[idx])), y[idx])
            if not torch.isfinite(loss): raise ValueError('Nonfinite training loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        scheduler.step()
        if select:
            model.eval()
            with torch.no_grad(): pred = model(*segments(q)).flatten().cpu().numpy() * scale + center
            score = validation_pcc(yq, pred)
            if np.isfinite(score) and score > best:
                best, best_epoch, best_pred = float(score), epoch, pred.copy()
            elif epoch - best_epoch >= cfg['patience']: break
    if not select:
        model.eval()
        with torch.no_grad(): best_pred = model(*segments(q)).flatten().cpu().numpy() * scale + center
        best_epoch = epochs
    if best_pred is None or not np.isfinite(best_pred).all(): raise ValueError('Invalid prediction')
    return best_pred, best_epoch, epoch


def init_worker(x, y, cfg):
    torch.set_num_threads(int(cfg['cpu_threads']))
    spec = importlib.util.spec_from_file_location('wheatgp_official', cfg['source'])
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    STATE.update(x=x, y=y, cfg=cfg, model_type=module.wheatGP_base)


def dev_job(train_idx, val_idx, env, candidate, seed):
    try:
        cfg = STATE['cfg']
        kind, lr = cfg['_candidates'][candidate]
        xf, xq = prepare(STATE['x'], train_idx, val_idx, kind, seed, cfg)
        pred, best, epochs = fit(STATE['model_type'], xf, STATE['y'][train_idx, env], xq,
                                 STATE['y'][val_idx, env], lr, cfg['max_epochs'], cfg,
                                 torch.device(cfg['device']), seed)
        return {'pred': pred, 'best_epoch': best, 'epochs_run': epochs}
    except Exception as exc:  # noqa: BLE001
        return {'error': f'{type(exc).__name__}: {exc}'}


def outer_job(dev_idx, test_idx, env, candidate, seed, epochs):
    try:
        cfg = STATE['cfg']
        kind, lr = cfg['_candidates'][candidate]
        xf, xq = prepare(STATE['x'], dev_idx, test_idx, kind, seed, cfg)
        pred, _, _ = fit(STATE['model_type'], xf, STATE['y'][dev_idx, env], xq, None, lr,
                         epochs, cfg, torch.device(cfg['device']), seed, select=False)
        return {'pred': pred}
    except Exception as exc:  # noqa: BLE001
        return {'error': f'{type(exc).__name__}: {exc}'}


def write(path, fields, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)


def run(cfg, args, runner):
    if sha256_of(Path(cfg['source'])) != cfg['source_sha256']:
        raise ValueError('Official source mismatch')
    data = load(Path(cfg['data']), cfg['data_sha256'])
    folds = json.loads(Path(cfg['splits']).read_text())['folds']
    if len(folds) != 10 or any(len(f['inner']) != 3 for f in folds): raise ValueError('Unexpected splits')
    seeds = [int(s) for s in cfg['seeds']]
    candidates = [(kind, lr) for kind in ('raw', 'pca') for lr in cfg['learning_rates']]
    jobs_n = int(cfg.get('jobs', 8))
    worker_cfg = {**cfg, '_candidates': candidates}
    device = torch.device(cfg['device'])
    torch.set_num_threads(int(cfg['cpu_threads']))
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(float(cfg['gpu_memory_fraction']))
    args.out.mkdir(parents=True, exist_ok=False)
    here = Path(__file__)
    (args.out/'provenance.json').write_text(json.dumps({
        'config': cfg,
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'sha256': {str(p): sha256_of(p) for p in (Path(cfg['source']), Path(cfg['splits']),
                                                  args.config, here, runner)},
        'seeds': seeds, 'jobs': jobs_n, 'parallel': 'joblib loky processes; per-fit seeding',
        'rule': 'mean inner PCC of seed-averaged predictions; per-seed median best epoch; '
                'full dev refit; constant PCC undefined and retained',
    }, indent=2)+'\n')
    start = time.perf_counter(); failures = []
    phase = 'development' if args.selection is None else 'outer'
    x = np.asarray(data.X); y = np.asarray(data.Y)
    initargs = (x, y, worker_cfg)
    if phase == 'development':
        records = []
        for fold in folds:
            f = fold['fold']
            jobs, keys = [], []
            for j, split in enumerate(fold['inner']):
                tr, val = np.asarray(split['train_idx']), np.asarray(split['val_idx'])
                if set(tr) | set(val) != set(fold['dev_idx']) or (set(tr) | set(val)) & set(fold['test_idx']):
                    raise ValueError('Split leakage')
                for e in range(len(data.environments)):
                    for c in range(len(candidates)):
                        for seed in seeds:
                            jobs.append((tr, val, e, c, seed)); keys.append((j, e, c, seed))
            results = Parallel(n_jobs=jobs_n, initializer=init_worker, initargs=initargs)(
                delayed(dev_job)(*job) for job in jobs)
            preds = {}
            for key, res in zip(keys, results):
                if 'error' in res:
                    failures.append({'fold': f, 'key': str(key), 'reason': res['error']})
                    raise RuntimeError(f'development fit failed: {res["error"]}')
                preds[key] = res
                records.append({'fold': f, 'inner': key[0], 'environment': key[1],
                                'candidate': key[2], 'seed': key[3],
                                'best_epoch': res['best_epoch'], 'epochs_run': res['epochs_run']})
            selected = []
            for e in range(len(data.environments)):
                scores = []
                for c in range(len(candidates)):
                    per_inner = []
                    for j, split in enumerate(fold['inner']):
                        val = np.asarray(split['val_idx'])
                        mean_pred = np.mean([preds[(j, e, c, s)]['pred'] for s in seeds], axis=0)
                        per_inner.append(validation_pcc(y[val, e], mean_pred))
                    scores.append(float(np.mean(per_inner)))
                c = int(np.argmax(scores))
                epochs = {}
                for seed in seeds:
                    epochs[str(seed)] = sorted(
                        r['best_epoch'] for r in records if r['fold'] == f
                        and r['environment'] == e and r['candidate'] == c and r['seed'] == seed)[1]
                selected.append({'environment': e, 'candidate': c, 'kind': candidates[c][0],
                                 'learning_rate': candidates[c][1], 'epochs': epochs, 'scores': scores})
            (args.out/f'selection_{f}.json').write_text(json.dumps(selected, indent=2))
            (args.out/'trajectories.json').write_text(json.dumps(records, indent=2))
            print(f'fold {f} selected', flush=True)
        (args.out/'selection.json').write_text(json.dumps({
            'folds': [json.loads((args.out/f"selection_{fold['fold']}.json").read_text()) for fold in folds],
            'source_sha256': cfg['source_sha256'],
            'splits_sha256': sha256_of(Path(cfg['splits'])),
            'config_sha256': sha256_of(args.config)}, indent=2))
        (args.out/'failures.json').write_text(json.dumps(failures, indent=2))
        (args.out/'timing.json').write_text(json.dumps({'seconds': time.perf_counter() - start,
                                                        'jobs': jobs_n}, indent=2))
        return
    selection = json.loads(args.selection.read_text())
    if (selection['source_sha256'] != cfg['source_sha256']
            or selection['splits_sha256'] != sha256_of(Path(cfg['splits']))
            or selection['config_sha256'] != sha256_of(args.config)):
        raise ValueError('Selection provenance mismatch')
    rows, seedrows, oof, seedoof, undefined = [], [], [], [], []
    for fold in folds:
        f = fold['fold']; dev = np.asarray(fold['dev_idx']); test = np.asarray(fold['test_idx'])
        selected = selection['folds'][f]
        jobs, keys = [], []
        for entry in selected:
            e = entry['environment']
            for seed in seeds:
                jobs.append((dev, test, e, entry['candidate'], seed, entry['epochs'][str(seed)]))
                keys.append((e, seed))
        results = Parallel(n_jobs=jobs_n, initializer=init_worker, initargs=initargs)(
            delayed(outer_job)(*job) for job in jobs)
        preds = {}
        for key, res in zip(keys, results):
            if 'error' in res:
                failures.append({'fold': f, 'key': str(key), 'reason': res['error']})
                raise RuntimeError(f'outer fit failed: {res["error"]}')
            preds[key] = res['pred']
        for entry in selected:
            e = entry['environment']
            for seed in seeds:
                pred = preds[(e, seed)]
                seedrows.append({'fold': f, 'environment': data.environments[e], 'seed': seed,
                                 'epochs': entry['epochs'][str(seed)],
                                 'pcc': pcc(y[test, e], pred), 'rmse': rmse(y[test, e], pred),
                                 'mae': mae(y[test, e], pred)})
                seedoof.extend({'fold': f, 'environment': data.environments[e], 'seed': seed,
                                'material_index': int(i), 'y_true': float(y[i, e]), 'y_pred': float(p)}
                               for i, p in zip(test, pred))
            mean_pred = np.mean([preds[(e, s)] for s in seeds], axis=0)
            value = {'pcc': pcc(y[test, e], mean_pred), 'rmse': rmse(y[test, e], mean_pred),
                     'mae': mae(y[test, e], mean_pred)}
            if not np.isfinite(value['pcc']):
                undefined.append({'fold': f, 'environment': data.environments[e],
                                  'reason': 'constant prediction'})
            rows.append({'fold': f, 'environment': data.environments[e], **value})
            oof.extend({'fold': f, 'environment': data.environments[e], 'material_index': int(i),
                        'y_true': float(y[i, e]), 'y_pred': float(p)}
                       for i, p in zip(test, mean_pred))
            print(f'[outer f{f} e{e}] pcc={value["pcc"]:.4f}', flush=True)
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
    defined = [r['pcc'] for r in rows if np.isfinite(r['pcc'])]
    (args.out/'summary.json').write_text(json.dumps({
        'n_runs': len(rows), 'failures': [], 'seeds': seeds, 'jobs': jobs_n,
        'per_environment': per_environment, 'macro': macro,
        'undefined_pcc_count': len(undefined),
        'pcc_defined_only_mean': float(np.mean(defined)) if defined else None,
        'pcc_constant_as_zero': float(np.mean([r['pcc'] if np.isfinite(r['pcc']) else 0.0 for r in rows])),
        'note': 'macro pcc is null when any outer fold prediction is constant; '
                'pcc_constant_as_zero is the author training convention, not the standard Pearson macro',
    }, indent=2)+'\n')
    (args.out/'failures.json').write_text(json.dumps(failures, indent=2))
    (args.out/'timing.json').write_text(json.dumps({'seconds': time.perf_counter() - start,
                                                    'jobs': jobs_n}, indent=2))
    print(json.dumps({'macro': macro, 'undefined': len(undefined)}), flush=True)
