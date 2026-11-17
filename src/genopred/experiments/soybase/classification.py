"""SoyBase qualitative-trait classification under the nested protocol.

Ready-made classifiers (XGBoost, regularized multinomial Logistic) and the
existing TabM backbone with a classification head on SoySNP50K dosage. One
trait per run. Marker-only; SNP call-rate/MAF filtering, missing imputation
and scaling are fitted inside the current outer development set (fit-only).
Random models use protocol seeds 42/2026: inner selection scores the balanced
accuracy of seed-averaged validation probabilities, the outer test prediction
is the seed-averaged probability argmax. Plain cross-entropy, no class
weighting. Classical fits spread over joblib workers with single-threaded
estimators; TabM runs sequentially on one GPU, per-member CE averaged in
training and per-member softmax probabilities averaged at inference (official
TabM classification usage), early stopping on the inner validation BA only.
"""
import importlib.metadata
import json
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler

from genopred.data import soybase
from genopred.data.wheat2000 import file_sha256
from genopred.evaluation.metrics import compute_classification
from genopred.models.classical.classical_baselines import candidates, write

RANDOM_MODELS = ('xgb', 'logistic')
NEURAL_MODELS = ('tabm', 'ising', 'chr_resmlp')


def snp_filter(X, dev, min_call_rate, min_maf):
    """Fit-only SNP mask: call-rate and MAF computed on the development rows."""
    sub = X[dev]
    call_rate = np.mean(~np.isnan(sub), axis=0)
    filled = np.where(np.isnan(sub), 0.0, sub)
    alt_freq = filled.sum(axis=0) / (2.0 * np.maximum((~np.isnan(sub)).sum(axis=0), 1))
    maf = np.minimum(alt_freq, 1.0 - alt_freq)
    return (call_rate >= min_call_rate) & (maf >= min_maf)


def impute(X, dev):
    """Fill missing dosage with the development-set per-SNP mean (fit-only)."""
    sub = X[dev]
    means = np.nanmean(sub, axis=0)
    means = np.where(np.isnan(means), 0.0, means)
    out = X.copy()
    missing = np.isnan(out)
    out[missing] = np.take(means, np.where(missing)[1])
    return out


def fit_predict_proba(model, spec, X, y, train, target, seed, n_classes):
    if model == 'logistic':
        scaler = StandardScaler().fit(X[train])
        from sklearn.linear_model import LogisticRegression
        est = LogisticRegression(C=spec['C'], max_iter=spec['max_iter'],
                                 solver='lbfgs', random_state=seed, n_jobs=1)
        est.fit(scaler.transform(X[train]), y[train])
        return est.predict_proba(scaler.transform(X[target]))
    if model == 'xgb':
        from xgboost import XGBClassifier
        est = XGBClassifier(
            n_estimators=spec['n_estimators'], max_depth=spec['max_depth'],
            learning_rate=spec['learning_rate'], subsample=spec['subsample'],
            colsample_bytree=spec['colsample_bytree'],
            objective='multi:softprob', num_class=n_classes, eval_metric='mlogloss',
            random_state=seed, tree_method='hist', n_jobs=1, verbosity=0)
        est.fit(X[train], y[train])
        return est.predict_proba(X[target])
    raise ValueError(f'unknown model {model!r}')


def _inner_score(job):
    model, spec, seeds, X, y, train, valid, n_classes = job
    start = time.time()
    proba = np.mean([fit_predict_proba(model, spec, X, y, train, valid, s, n_classes)
                     for s in seeds], axis=0)
    pred = proba.argmax(axis=1)
    from sklearn.metrics import balanced_accuracy_score
    return float(balanced_accuracy_score(y[valid], pred)), time.time() - start


def _outer_predict(job):
    model, spec, seeds, X, y, dev, test, n_classes = job
    start = time.time()
    proba = {s: fit_predict_proba(model, spec, X, y, dev, test, s, n_classes) for s in seeds}
    return proba, time.time() - start


def fit_neural_proba(model_name, spec, train_cfg, X, y, train, target, seed, n_classes,
                     device, select_best=True, chrom=None):
    """Neural classification with fit-only scaling and CE.

    TabM: per-member CE mean in training, per-member softmax probability mean
    at inference (official classification usage). Ising: single conditional
    energy model, exact enumeration, logits = E[h|x] @ readout.

    select_best=True (inner): early stop on target validation BA, return the
    best-epoch state. select_best=False (outer refit): train exactly
    max_epochs epochs and return the final state; target labels are never read.
    """
    import torch
    import torch.nn.functional as F
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    x_train = torch.as_tensor(X[train], dtype=torch.float32)
    labels = torch.as_tensor(y[train], dtype=torch.long)
    if model_name == 'tabm':
        from genopred.models.tabm.backbone import GenomicTabM
        # GenomicTabM standardizes x on the fit rows; a constant label column
        # keeps the regression y-buffers inert (classification never uses them).
        model = GenomicTabM(x_train, torch.zeros(len(train), 1), variant='raw',
                            n_blocks=int(spec['n_blocks']), d_block=int(spec['d_block']),
                            dropout=float(spec['dropout']), k=int(spec['k'])).to(device)
        model.backbone.output = torch.nn.Linear(int(spec['d_block']), n_classes).to(device)
    elif model_name == 'ising':
        from genopred.models.structured.supervised_ising import SupervisedIsing
        model = SupervisedIsing(x_train, None, spins=int(spec['spins']),
                                coupled=True, n_classes=n_classes).to(device)
    elif model_name == 'chr_resmlp':
        from genopred.models.structured.chr_resmlp import ChrResMLP
        model = ChrResMLP(x_train, chrom, n_classes, d_group=int(spec['d_group']),
                          d_fuse=int(spec['d_fuse'])).to(device)
    else:
        raise ValueError(f'unknown neural model {model_name!r}')
    x_train, labels = x_train.to(device), labels.to(device)
    x_target = torch.as_tensor(X[target], dtype=torch.float32).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(spec['learning_rate']),
                                  weight_decay=float(spec['weight_decay']))
    generator = torch.Generator().manual_seed(seed)
    batch_size = int(train_cfg['batch_size'])
    best_ba, best_state, best_epoch, stale = -1.0, None, 0, 0
    epochs_run = 0

    def member_proba(inputs):
        model.eval()
        chunks = []
        with torch.no_grad():
            for start in range(0, len(inputs), int(train_cfg['eval_batch_size'])):
                logits = model(inputs[start:start + int(train_cfg['eval_batch_size'])])
                proba = logits.softmax(dim=-1)
                if logits.ndim == 3:
                    proba = proba.mean(dim=1)
                chunks.append(proba.cpu().numpy())
        return np.concatenate(chunks)

    for epoch in range(1, int(train_cfg['max_epochs']) + 1):
        epochs_run = epoch
        model.train()
        order = torch.randperm(len(train), generator=generator)
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_train[batch])
            if logits.ndim == 3:
                loss = F.cross_entropy(logits.reshape(-1, n_classes),
                                       labels[batch].unsqueeze(1).expand(-1, logits.shape[1]).reshape(-1))
            else:
                loss = F.cross_entropy(logits, labels[batch])
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite {model_name} loss at epoch {epoch}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg['gradient_clip']))
            optimizer.step()
        if select_best:
            from sklearn.metrics import balanced_accuracy_score
            val_ba = float(balanced_accuracy_score(y[target], member_proba(x_target).argmax(axis=1)))
            if val_ba > best_ba:
                best_ba, best_epoch, stale = val_ba, epoch, 0
                best_state = {key: value.detach().cpu().clone()
                              for key, value in model.state_dict().items()}
            else:
                stale += 1
                if stale >= int(train_cfg['patience']):
                    break
    if select_best:
        model.load_state_dict(best_state)
        model.to(device)
    else:
        best_epoch, best_ba = epochs_run, float('nan')
    return member_proba(x_target), model, {'best_epoch': best_epoch, 'best_val_ba': best_ba,
                                           'epochs_run': epochs_run}


def _run_neural(cfg, args, runner_path, data, trait, n_classes, labeled, X_all, y_all,
                folds, split_hash, models, space, seeds, frozen):
    import torch
    from sklearn.metrics import balanced_accuracy_score
    device = torch.device(cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    torch.set_num_threads(int(cfg.get('cpu_threads', 2)))
    pos = {int(g): i for i, g in enumerate(labeled)}
    args.out.mkdir(parents=True, exist_ok=False)
    weights_dir = args.out / 'weights'
    versions = {n: importlib.metadata.version(n) for n in ('numpy', 'scikit-learn', 'torch', 'tabm')}
    models_dir = Path(__file__).resolve().parents[2] / 'models'
    backbone_paths = {'tabm': models_dir / 'tabm' / 'backbone.py',
                      'ising': models_dir / 'structured' / 'supervised_ising.py',
                      'chr_resmlp': models_dir / 'structured' / 'chr_resmlp.py'}
    chrom_all = None
    if 'chr_resmlp' in models:
        marker_map = Path(cfg['marker_map'])
        chrom_all = soybase.load_chromosomes(marker_map, data.snp_ids)
    (args.out / 'provenance.json').write_text(json.dumps({
        'config': cfg, 'trait': trait, 'splits_sha256': split_hash,
        'cache_sha256': file_sha256(Path(cfg['cache'])),
        'marker_map_sha256': file_sha256(Path(cfg['marker_map'])) if chrom_all is not None else None,
        'runner_sha256': file_sha256(Path(runner_path)), 'model_sha256': file_sha256(Path(__file__)),
        'backbone_sha256': {m: file_sha256(backbone_paths[m]) for m in models},
        'selection_sha256': file_sha256(Path(args.selection)) if args.selection else None,
        'versions': versions, 'device': str(device),
        'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        'n_labeled': int(len(labeled)), 'n_classes': n_classes,
        'class_map': soybase.CLASS_MAPS[trait],
        'rule': 'mean inner BA of seed-averaged member-mean probabilities; first candidate on tie; '
                'early stop on inner validation BA (patience), best-epoch state; full dev refit with '
                'the selected epoch count; fit-only SNP filter/impute/scale; per-member CE mean, '
                'per-member softmax mean',
    }, indent=2))
    selected, valrows, rows, oof, failures, timing = [], [], [], [], [], []
    for fold in folds:
        k = fold['fold']
        dev = np.asarray([pos[g] for g in fold['dev_idx']])
        test = np.asarray([pos[g] for g in fold['test_idx']])
        inner = [(np.asarray([pos[g] for g in v['train_idx']]),
                  np.asarray([pos[g] for g in v['val_idx']])) for v in fold['inner']]
        mask = snp_filter(X_all, dev, cfg['min_call_rate'], cfg['min_maf'])
        X = impute(X_all[:, mask], dev)
        chrom = chrom_all[mask] if chrom_all is not None else None
        y = y_all
        print(f'fold {k}: dev={len(dev)} test={len(test)} snps={int(mask.sum())}', flush=True)
        start = time.time()
        try:
            for model in models:
                if frozen is None:
                    scores = np.zeros((len(space[model]), len(inner)))
                    details = {}
                    for i, spec in enumerate(space[model]):
                        for j, (train, valid) in enumerate(inner):
                            probas, infos = [], []
                            for seed in seeds:
                                fit_start = time.time()
                                proba, fitted, info = fit_neural_proba(model, spec, cfg['train'], X, y, train, valid,
                                                                       seed, n_classes, device, chrom=chrom)
                                del fitted
                                if device.type == 'cuda':
                                    torch.cuda.empty_cache()
                                probas.append(proba)
                                infos.append(info)
                                valrows.append({'model': model, 'fold': k, 'inner_fold': j,
                                                'candidate': i, 'seed': seed,
                                                'best_epoch': info['best_epoch'],
                                                'epochs_run': info['epochs_run'],
                                                'ba': float(info['best_val_ba']),
                                                'seconds': round(time.time() - fit_start, 1)})
                            mean_proba = np.mean(probas, axis=0)
                            scores[i, j] = balanced_accuracy_score(y[valid], mean_proba.argmax(axis=1))
                            details[(i, j)] = infos
                        write(args.out / 'validation.csv',
                              ['model', 'fold', 'inner_fold', 'candidate', 'seed', 'best_epoch', 'epochs_run', 'ba', 'seconds'],
                              valrows)
                        print(f'  candidate {i} {space[model][i]}: '
                              f'inner BA per fold {[round(float(v), 4) for v in scores[i]]} '
                              f'mean {scores[i].mean():.4f}', flush=True)
                    i = int(np.argmax(scores.mean(1)))
                    chosen_epochs = int(np.median([info['best_epoch']
                                                   for j in range(len(inner)) for info in details[(i, j)]]))
                    selected.append({'model': model, 'fold': k, 'candidate': i,
                                     'spec': space[model][i], 'inner_ba': float(scores[i].mean()),
                                     'refit_epochs': chosen_epochs})
                    (args.out / 'selection.json').write_text(json.dumps(
                        {'config': cfg, 'splits_sha256': split_hash, 'selected': selected}, indent=2))
                else:
                    entry = next(v for v in frozen['selected'] if (v['model'], v['fold']) == (model, k))
                    spec = space[model][entry['candidate']]
                    if spec != entry['spec']:
                        raise ValueError('Frozen spec does not match candidate order')
                    train_cfg = dict(cfg['train'])
                    train_cfg['max_epochs'] = int(entry['refit_epochs'])
                    probas = {}
                    fit_start = time.time()
                    for seed in seeds:
                        proba, fitted, info = fit_neural_proba(model, spec, train_cfg, X, y, dev, test,
                                                               seed, n_classes, device,
                                                               select_best=False, chrom=chrom)
                        probas[seed] = proba
                        weights_dir.mkdir(exist_ok=True)
                        state_path = weights_dir / f'f{k}_{model}_s{seed}.pt'
                        torch.save({key: value.detach().cpu()
                                    for key, value in fitted.state_dict().items()}, state_path)
                        del fitted
                        if device.type == 'cuda':
                            torch.cuda.empty_cache()
                    mean_proba = np.mean(list(probas.values()), axis=0)
                    pred = mean_proba.argmax(axis=1)
                    met = compute_classification(y[test], pred, n_classes)
                    rows.append({'model': model, 'fold': k, 'candidate': entry['candidate'],
                                 'inner_ba': entry['inner_ba'],
                                 'fit_seconds': round(time.time() - fit_start, 2),
                                 'balanced_accuracy': met['balanced_accuracy'], 'macro_f1': met['macro_f1'],
                                 'per_class': met['per_class'], 'confusion_matrix': met['confusion_matrix']})
                    oof.extend({'model': model, 'fold': k, 'material_index': int(labeled[idx]),
                                'y_true': int(y[idx]), 'y_pred': int(p)}
                               for idx, p in zip(test, pred))
            phase = 'selection' if frozen is None else 'outer'
            timing.append({'fold': k, 'phase': phase, 'seconds': round(time.time() - start, 1)})
            print(f'  {phase} complete in {time.time()-start:.1f}s', flush=True)
        except Exception as exc:
            failures.append({'fold': k, 'reason': repr(exc)})
            print(f'FAILED {k}: {exc}', flush=True)
        finally:
            write(args.out / 'validation.csv',
                  ['model', 'fold', 'inner_fold', 'candidate', 'seed', 'best_epoch', 'epochs_run', 'ba', 'seconds'],
                  valrows)
            write(args.out / 'per_fold.csv',
                  ['model', 'fold', 'balanced_accuracy', 'macro_f1', 'candidate', 'inner_ba', 'fit_seconds'],
                  [{kk: r[kk] for kk in ('model', 'fold', 'balanced_accuracy', 'macro_f1', 'candidate', 'inner_ba', 'fit_seconds')} for r in rows])
            write(args.out / 'oof.csv', ['model', 'fold', 'material_index', 'y_true', 'y_pred'], oof)
            write(args.out / 'failures.csv', ['fold', 'reason'], failures)
            (args.out / 'per_fold_detail.json').write_text(json.dumps(rows, indent=2))
            (args.out / 'timing.json').write_text(json.dumps({'device': str(device), 'folds': timing}, indent=2))
    expected = len(models) * len(folds)
    if failures or (len(selected) != expected if frozen is None else len(rows) != expected):
        raise RuntimeError('Incomplete run')
    if frozen is None:
        print('Development selection complete; no test predictions made.', flush=True)
        return
    summary = {'trait': trait, 'n_classes': n_classes, 'n_runs': len(rows), 'failures': [], 'models': {}}
    for model in models:
        mrows = [r for r in rows if r['model'] == model]
        per_class = {}
        for c in range(n_classes):
            per_class[str(c)] = {
                'precision': float(np.mean([r['per_class'][str(c)]['precision'] for r in mrows])),
                'recall': float(np.mean([r['per_class'][str(c)]['recall'] for r in mrows])),
                'f1': float(np.mean([r['per_class'][str(c)]['f1'] for r in mrows]))}
        confusion = np.sum([np.asarray(r['confusion_matrix']) for r in mrows], axis=0).tolist()
        summary['models'][model] = {
            'macro_balanced_accuracy': float(np.mean([r['balanced_accuracy'] for r in mrows])),
            'macro_f1': float(np.mean([r['macro_f1'] for r in mrows])),
            'per_class_mean': per_class, 'confusion_matrix_sum': confusion,
            'seeds': list(seeds)}
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


def run(cfg, args, runner_path):
    src = json.loads(Path(cfg['source']).read_text())
    cache = Path(cfg['cache'])
    data = soybase.load_soybase(src['hapmap']['path'], src['hapmap']['sha256'],
                                src['grin_observations']['path'], src['grin_observations']['sha256'],
                                cache, file_sha256(cache))
    trait = cfg['trait']
    j = soybase.TRAITS.index(trait)
    n_classes = len(soybase.CLASS_MAPS[trait])
    labeled = np.flatnonzero(data.Y[:, j] >= 0)
    X_all = data.X[labeled]
    y_all = data.Y[labeled, j]
    splits = json.loads(Path(cfg['splits']).read_text())
    folds = splits['folds']
    split_hash = file_sha256(Path(cfg['splits']))
    # splits index global rows; remap to the labeled subset position
    pos = {int(g): i for i, g in enumerate(labeled)}
    models = tuple(cfg['search'])
    space = {m: candidates(cfg, m) for m in models}
    seeds = tuple(cfg['seeds'])
    frozen = json.loads(args.selection.read_text()) if args.selection else None
    if frozen and (frozen['config'] != cfg or frozen['splits_sha256'] != split_hash):
        raise ValueError('Frozen selection does not match config or splits')
    if any(m in NEURAL_MODELS for m in models):
        if not all(m in NEURAL_MODELS for m in models):
            raise ValueError('Neural and classical models must run in separate configs')
        return _run_neural(cfg, args, runner_path, data, trait, n_classes, labeled, X_all, y_all,
                           folds, split_hash, models, space, seeds, frozen)
    args.out.mkdir(parents=True, exist_ok=False)
    versions = {n: importlib.metadata.version(n) for n in ('numpy', 'scikit-learn', 'xgboost')}
    (args.out / 'provenance.json').write_text(json.dumps({
        'config': cfg, 'trait': trait, 'splits_sha256': split_hash,
        'cache_sha256': file_sha256(cache),
        'runner_sha256': file_sha256(Path(runner_path)), 'model_sha256': file_sha256(Path(__file__)),
        'selection_sha256': file_sha256(Path(args.selection)) if args.selection else None,
        'versions': versions, 'jobs': cfg['jobs'],
        'n_labeled': int(len(labeled)), 'n_classes': n_classes,
        'class_map': soybase.CLASS_MAPS[trait],
        'rule': 'mean inner BA of seed-averaged probabilities; first candidate on tie; full dev refit; fit-only SNP filter/impute/scale',
    }, indent=2))
    selected, valrows, rows, oof, failures, timing = [], [], [], [], [], []
    for fold in folds:
        k = fold['fold']
        dev = np.asarray([pos[g] for g in fold['dev_idx']])
        test = np.asarray([pos[g] for g in fold['test_idx']])
        inner = [(np.asarray([pos[g] for g in v['train_idx']]),
                  np.asarray([pos[g] for g in v['val_idx']])) for v in fold['inner']]
        mask = snp_filter(X_all, dev, cfg['min_call_rate'], cfg['min_maf'])
        X = impute(X_all[:, mask], dev)
        y = y_all
        print(f'fold {k}: dev={len(dev)} test={len(test)} snps={int(mask.sum())}', flush=True)
        start = time.time()
        try:
            if frozen is None:
                jobs, keys = [], []
                for model in models:
                    for jinner, (train, valid) in enumerate(inner):
                        for i, spec in enumerate(space[model]):
                            jobs.append((model, spec, seeds, X, y, train, valid, n_classes))
                            keys.append((model, jinner, i))
                results = Parallel(n_jobs=cfg['jobs'], backend='threading')(
                    delayed(_inner_score)(job) for job in jobs)
                scores = {}
                for (model, jinner, i), (score, seconds) in zip(keys, results):
                    scores.setdefault(model, np.zeros((len(space[model]), len(inner))))[i, jinner] = score
                    valrows.append({'model': model, 'fold': k, 'inner_fold': jinner,
                                    'candidate': i, 'ba': float(score), 'seconds': round(seconds, 3)})
                for model in models:
                    i = int(np.argmax(scores[model].mean(1)))
                    selected.append({'model': model, 'fold': k, 'candidate': i,
                                     'spec': space[model][i], 'inner_ba': float(scores[model][i].mean())})
                (args.out / 'selection.json').write_text(json.dumps(
                    {'config': cfg, 'splits_sha256': split_hash, 'selected': selected}, indent=2))
                timing.append({'fold': k, 'phase': 'selection', 'seconds': round(time.time() - start, 1)})
                print(f'  selection complete in {time.time()-start:.1f}s', flush=True)
            else:
                entries = {}
                for model in models:
                    entry = next(v for v in frozen['selected'] if (v['model'], v['fold']) == (model, k))
                    spec = space[model][entry['candidate']]
                    if spec != entry['spec']:
                        raise ValueError('Frozen spec does not match candidate order')
                    entries[model] = (spec, entry)
                jobs = [(model, spec, seeds, X, y, dev, test, n_classes)
                        for model, (spec, _) in entries.items()]
                results = Parallel(n_jobs=cfg['jobs'], backend='threading')(
                    delayed(_outer_predict)(job) for job in jobs)
                for (model, (spec, entry)), (proba, seconds) in zip(entries.items(), results):
                    mean_proba = np.mean(list(proba.values()), axis=0)
                    pred = mean_proba.argmax(axis=1)
                    met = compute_classification(y[test], pred, n_classes)
                    rows.append({'model': model, 'fold': k, 'candidate': entry['candidate'],
                                 'inner_ba': entry['inner_ba'], 'fit_seconds': round(seconds, 2),
                                 'balanced_accuracy': met['balanced_accuracy'], 'macro_f1': met['macro_f1'],
                                 'per_class': met['per_class'], 'confusion_matrix': met['confusion_matrix']})
                    oof.extend({'model': model, 'fold': k, 'material_index': int(labeled[idx]),
                                'y_true': int(y[idx]), 'y_pred': int(p)}
                               for idx, p in zip(test, pred))
                timing.append({'fold': k, 'phase': 'outer', 'seconds': round(time.time() - start, 1)})
                print(f'  outer evaluation complete in {time.time()-start:.1f}s', flush=True)
        except Exception as exc:
            failures.append({'fold': k, 'reason': repr(exc)})
            print(f'FAILED {k}: {exc}', flush=True)
        finally:
            write(args.out / 'validation.csv', ['model', 'fold', 'inner_fold', 'candidate', 'ba', 'seconds'], valrows)
            write(args.out / 'per_fold.csv', ['model', 'fold', 'balanced_accuracy', 'macro_f1', 'candidate', 'inner_ba', 'fit_seconds'],
                  [{kk: r[kk] for kk in ('model', 'fold', 'balanced_accuracy', 'macro_f1', 'candidate', 'inner_ba', 'fit_seconds')} for r in rows])
            write(args.out / 'oof.csv', ['model', 'fold', 'material_index', 'y_true', 'y_pred'], oof)
            write(args.out / 'failures.csv', ['fold', 'reason'], failures)
            (args.out / 'per_fold_detail.json').write_text(json.dumps(rows, indent=2))
            (args.out / 'timing.json').write_text(json.dumps({'jobs': cfg['jobs'], 'folds': timing}, indent=2))
    expected = len(models) * len(folds)
    if failures or (len(selected) != expected if frozen is None else len(rows) != expected):
        raise RuntimeError('Incomplete run')
    if frozen is None:
        print('Development selection complete; no test predictions made.', flush=True)
        return
    summary = {'trait': trait, 'n_classes': n_classes, 'n_runs': len(rows), 'failures': [], 'models': {}}
    for model in models:
        mrows = [r for r in rows if r['model'] == model]
        per_class = {}
        for c in range(n_classes):
            per_class[str(c)] = {
                'precision': float(np.mean([r['per_class'][str(c)]['precision'] for r in mrows])),
                'recall': float(np.mean([r['per_class'][str(c)]['recall'] for r in mrows])),
                'f1': float(np.mean([r['per_class'][str(c)]['f1'] for r in mrows]))}
        confusion = np.sum([np.asarray(r['confusion_matrix']) for r in mrows], axis=0).tolist()
        summary['models'][model] = {
            'macro_balanced_accuracy': float(np.mean([r['balanced_accuracy'] for r in mrows])),
            'macro_f1': float(np.mean([r['macro_f1'] for r in mrows])),
            'per_class_mean': per_class, 'confusion_matrix_sum': confusion,
            'seeds': list(seeds)}
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)
