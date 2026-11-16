"""GSformer CSA/NSA under the unified protocol, marker-only.

The author repository ships the two architectures and their training rules
(Adam lr 5e-5, weight decay 1e-3, MSE, batch 48, 25 epochs, early stop when the
validation loss improves by less than 1e-6 for five epochs) but no Wheat599
training files, so this runner supplies the project splits and keeps the
author's model classes untouched from a pinned checkout.

Adaptations recorded rather than hidden: the paper's inputs are DNNGP-style
SNP/PC features; this run feeds raw 1279 markers zero-padded to the shape each
architecture requires (CSA 1280 = 32x40 chunks, NSA 1440 = 20x72 tokens). The
target is the published wheat.Y four-environment table; no GY1-GY4 to column
mapping is assumed. Seeds 42 and 2026; per-seed median best epoch refit;
seed-averaged test prediction. Independent fits run in joblib processes, each
seeding itself.
"""
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from joblib import Parallel, delayed

from genopred.data.wheat599 import load, sha256_of
from genopred.evaluation.metrics import pcc, rmse, mae

STATE = {}


def _stub_matplotlib():
    try:
        import matplotlib.pyplot  # noqa: F401  (imported but unused by NSAformer)
    except ModuleNotFoundError:
        import types
        matplotlib = types.ModuleType('matplotlib')
        pyplot = types.ModuleType('matplotlib.pyplot')
        matplotlib.pyplot = pyplot
        sys.modules['matplotlib'] = matplotlib
        sys.modules['matplotlib.pyplot'] = pyplot


def load_classes(source_dir, commit):
    source = Path(source_dir)
    actual = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != commit:
        raise ValueError('GSformer source commit mismatch')
    if subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain', '--untracked-files=no'],
                               text=True).strip():
        raise ValueError('GSformer source checkout has tracked modifications')
    sys.path.insert(0, str(source / 'code'))
    sys.dont_write_bytecode = True
    _stub_matplotlib()
    from CSAformer import CSAformer
    from NSAformer import Transformer
    return CSAformer, Transformer


def build(variant, n_features, cfg, classes):
    CSAformer, Transformer = classes
    if variant == 'csa':
        a = cfg['architecture']
        return CSAformer(before_length=n_features, seq_length=int(a['seq_length']),
                         seq_num=int(a['seq_num']), head=int(a['heads']),
                         d_model=int(a['d_model']), d_ff=int(a['d_ff']),
                         dropout=float(a['dropout']), N=1)
    if variant == 'nsa':
        a = cfg['architecture']
        sparse = {'compress_block_size': int(a['compress_block_size']),
                  'selection_block_size': int(a['selection_block_size']),
                  'num_selected_blocks': int(a['num_selected_blocks'])}
        return Transformer(num_tokens=n_features, add_num=0, dim=int(a['dim']), depth=1,
                           dim_head=int(a['dim_head']), heads=int(a['heads']),
                           dropout=float(a['dropout']), num_length=n_features,
                           ff_expansion_factor=2, sparse_attn_kwargs=sparse)
    raise ValueError(f'unknown variant {variant!r}')


def _seed(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def fit(variant, classes, x, y, train, val, cfg, seed, epochs, select):
    _seed(seed)
    device = torch.device(cfg['device'])
    model = build(variant, x.shape[1], cfg, classes).to(device)
    xt = torch.as_tensor(x, dtype=torch.float32, device=device)
    yt = torch.as_tensor(y, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg['training']['learning_rate']),
                                 weight_decay=float(cfg['training']['weight_decay']))
    batch_size = int(cfg['training']['batch_size'])
    best_pcc, best_epoch, best_state = -np.inf, 0, None
    previous, stale = np.inf, 0
    for epoch in range(1, int(epochs) + 1):
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
            rows = torch.as_tensor(val).to(device)
            pred = model(xt[rows]).ravel()
        val_loss = float(torch.nn.functional.mse_loss(pred, yt[rows]))
        score = pcc(y[val], pred.cpu().numpy())
        if np.isfinite(score) and score > best_pcc:
            best_pcc, best_epoch = float(score), epoch
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
        rows = torch.as_tensor(val).to(device)
        pred = model(xt[rows]).ravel().cpu().numpy()
    return pred, best_epoch, epoch, int(sum(p.numel() for p in model.parameters()))


def init_worker(x, y, cfg):
    torch.set_num_threads(int(cfg['cpu_threads']))
    device = torch.device(cfg['device'])
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(float(cfg['gpu_memory_fraction']))
    STATE.update(x=x, y=y, cfg=cfg, classes=load_classes(cfg['source_dir'], cfg['source_commit']),
                 variant=cfg['variant'], device=device)


def dev_job(train_idx, val_idx, env, seed):
    try:
        cfg = STATE['cfg']
        pred, best_epoch, epochs_run, n_params = fit(STATE['variant'], STATE['classes'],
                                                     STATE['x'], STATE['y'][:, env],
                                                     train_idx, val_idx, cfg, seed,
                                                     cfg['training']['epochs'], True)
        return {'pred': pred, 'best_epoch': best_epoch, 'epochs_run': epochs_run, 'n_params': n_params}
    except Exception as exc:  # noqa: BLE001
        return {'error': f'{type(exc).__name__}: {exc}'}


def outer_job(dev_idx, test_idx, env, seed, epochs):
    try:
        cfg = STATE['cfg']
        pred, _, _, _ = fit(STATE['variant'], STATE['classes'], STATE['x'], STATE['y'][:, env],
                            dev_idx, test_idx, cfg, seed, epochs, False)
        return {'pred': pred}
    except Exception as exc:  # noqa: BLE001
        return {'error': f'{type(exc).__name__}: {exc}'}


def write(path, fields, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)


def run(cfg, args, runner):
    variant = cfg['variant']
    classes = load_classes(cfg['source_dir'], cfg['source_commit'])
    data = load(Path(cfg['data']), cfg['data_sha256'])
    folds = json.loads(Path(cfg['splits']).read_text())['folds']
    if len(folds) != 10 or any(len(f['inner']) != 3 for f in folds):
        raise ValueError('Unexpected splits')
    pad = int(cfg['pad_to'])
    if pad < data.p:
        raise ValueError('pad_to smaller than the marker count')
    x = np.zeros((data.n, pad), dtype=np.float32)
    x[:, :data.p] = data.X
    seeds = [int(s) for s in cfg['seeds']]
    jobs_n = int(cfg.get('jobs', 8))
    device = torch.device(cfg['device'])
    torch.set_num_threads(int(cfg['cpu_threads']))
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(float(cfg['gpu_memory_fraction']))
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out/'provenance.json').write_text(json.dumps({
        'config': cfg, 'variant': variant, 'seeds': seeds, 'jobs': jobs_n,
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'data_sha256': cfg['data_sha256'], 'splits_sha256': sha256_of(Path(cfg['splits'])),
        'source': {'url': 'https://github.com/hajudien/GSformer', 'commit': cfg['source_commit'],
                   'files': {name: sha256_of(Path(cfg['source_dir'])/'code'/name)
                             for name in ('CSAformer.py', 'NSAformer.py', 'native_sparse_attention.py')}},
        'adaptations': ['raw markers zero-padded to the architecture shape instead of SNP/PC inputs',
                        'published wheat.Y four-environment target; GY1-GY4 mapping not assumed',
                        'canonical manifest folds instead of an undisclosed author split',
                        'early stop on validation loss (author rule); refit epoch = median best-PCC epoch',
                        'matplotlib import is stubbed when absent because NSAformer imports it without using it; no numerical path affected'],
        'rule': 'per-seed early stop; median best epoch; full dev refit; seed-averaged test prediction',
        'parallel': 'joblib loky processes; per-fit seeding',
    }, indent=2)+'\n')
    phase = 'development' if args.selection is None else 'outer'
    failures = []
    start = time.perf_counter()
    y = np.asarray(data.Y)
    initargs = (x, y, cfg)
    if phase == 'development':
        jobs, keys = [], []
        for fold in folds:
            f = fold['fold']
            for j, split in enumerate(fold['inner']):
                tr, val = np.asarray(split['train_idx']), np.asarray(split['val_idx'])
                if set(tr) | set(val) != set(fold['dev_idx']) or (set(tr) | set(val)) & set(fold['test_idx']):
                    raise ValueError('Split leakage')
                for e in range(len(data.environments)):
                    for seed in seeds:
                        jobs.append((tr, val, e, seed)); keys.append((f, j, e, seed))
        results = Parallel(n_jobs=jobs_n, initializer=init_worker, initargs=initargs)(
            delayed(dev_job)(*job) for job in jobs)
        records = []
        for key, res in zip(keys, results):
            if 'error' in res:
                failures.append({'key': str(key), 'reason': res['error']})
                raise RuntimeError(f'development fit failed: {res["error"]}')
            records.append({'fold': key[0], 'inner': key[1], 'environment': key[2],
                            'seed': key[3], 'best_epoch': res['best_epoch'],
                            'epochs_run': res['epochs_run']})
        selected = []
        for fold in folds:
            f = fold['fold']
            epochs = {}
            for seed in seeds:
                epochs[str(seed)] = {
                    env: sorted(r['best_epoch'] for r in records
                                if r['fold'] == f and r['seed'] == seed
                                and r['environment'] == e)[1]
                    for e, env in enumerate(data.environments)}
            selected.append({'fold': f, 'epochs': epochs})
            (args.out/f'selection_{f}.json').write_text(json.dumps(selected[-1], indent=2))
            print(f'fold {f} epochs {epochs}', flush=True)
        (args.out/'selection.json').write_text(json.dumps({
            'folds': selected, 'splits_sha256': sha256_of(Path(cfg['splits'])),
            'config_sha256': sha256_of(args.config)}, indent=2))
        (args.out/'trajectories.json').write_text(json.dumps(records, indent=2))
        (args.out/'failures.json').write_text(json.dumps(failures, indent=2))
        (args.out/'timing.json').write_text(json.dumps({'seconds': time.perf_counter() - start,
                                                        'jobs': jobs_n}, indent=2))
        return
    selection = json.loads(args.selection.read_text())
    if (selection['splits_sha256'] != sha256_of(Path(cfg['splits']))
            or selection['config_sha256'] != sha256_of(args.config)):
        raise ValueError('Frozen selection does not match config or splits')
    chosen = {entry['fold']: entry for entry in selection['folds']}
    jobs, keys = [], []
    for fold in folds:
        f = fold['fold']; dev = np.asarray(fold['dev_idx']); test = np.asarray(fold['test_idx'])
        for e, env in enumerate(data.environments):
            for seed in seeds:
                jobs.append((dev, test, e, seed, chosen[f]['epochs'][str(seed)][env]))
                keys.append((f, env, seed))
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
        for e, env in enumerate(data.environments):
            for seed in seeds:
                pred = preds[(f, env, seed)]
                seedrows.append({'fold': f, 'environment': env, 'seed': seed,
                                 'epochs': chosen[f]['epochs'][str(seed)][env],
                                 'pcc': pcc(y[test, e], pred), 'rmse': rmse(y[test, e], pred),
                                 'mae': mae(y[test, e], pred)})
                seedoof.extend({'fold': f, 'environment': env, 'seed': seed,
                                'material_index': int(i), 'y_true': float(y[i, e]), 'y_pred': float(p)}
                               for i, p in zip(test, pred))
            mean_pred = np.mean([preds[(f, env, s)] for s in seeds], axis=0)
            value = {'pcc': pcc(y[test, e], mean_pred), 'rmse': rmse(y[test, e], mean_pred),
                     'mae': mae(y[test, e], mean_pred)}
            if not np.isfinite(value['pcc']):
                undefined.append({'fold': f, 'environment': env, 'reason': 'constant prediction'})
            rows.append({'fold': f, 'environment': env, **value})
            oof.extend({'fold': f, 'environment': env, 'material_index': int(i),
                        'y_true': float(y[i, e]), 'y_pred': float(p)} for i, p in zip(test, mean_pred))
    write(args.out/'per_fold.csv', list(rows[0]), rows)
    write(args.out/'per_fold_seed.csv', list(seedrows[0]), seedrows)
    write(args.out/'oof.csv', ['fold', 'environment', 'material_index', 'y_true', 'y_pred'], oof)
    write(args.out/'oof_seed.csv', ['fold', 'environment', 'seed', 'material_index', 'y_true', 'y_pred'], seedoof)
    (args.out/'undefined_pcc.json').write_text(json.dumps(undefined, indent=2)+'\n')
    per_environment = {}
    for env in data.environments:
        rs = [r for r in rows if r['environment'] == env]
        entry = {'undefined_pcc': sum(1 for r in rs if not np.isfinite(r['pcc']))}
        for metric in ('pcc', 'rmse', 'mae'):
            values = [r[metric] for r in rs if np.isfinite(r[metric])]
            entry[metric] = float(np.mean(values)) if values else None
        per_environment[env] = entry
    macro = {'pcc': None if undefined else float(np.mean([per_environment[e]['pcc'] for e in data.environments])),
             'rmse': float(np.mean([per_environment[e]['rmse'] for e in data.environments])),
             'mae': float(np.mean([per_environment[e]['mae'] for e in data.environments]))}
    (args.out/'summary.json').write_text(json.dumps({
        'n_runs': len(rows), 'failures': [], 'seeds': seeds, 'pad_to': pad, 'jobs': jobs_n,
        'per_environment': per_environment, 'macro': macro,
        'undefined_pcc_count': len(undefined)}, indent=2)+'\n')
    (args.out/'failures.json').write_text(json.dumps(failures, indent=2))
    (args.out/'timing.json').write_text(json.dumps({'seconds': time.perf_counter() - start,
                                                    'jobs': jobs_n}, indent=2))
    print(json.dumps({'macro': macro, 'undefined': len(undefined)}), flush=True)
