"""SoyBase qualitative-trait classification under the nested protocol.

Ready-made classifiers (XGBoost, regularized multinomial Logistic) on SoySNP50K
dosage. One trait per run. Marker-only; SNP call-rate/MAF filtering, missing
imputation and scaling are fitted inside the current outer development set
(fit-only). Random models use protocol seeds 42/2026: inner selection scores the
balanced accuracy of seed-averaged validation probabilities, the outer test
prediction is the seed-averaged probability argmax. Plain cross-entropy, no
class weighting (baseline first). Fits spread over joblib loky workers;
estimators run single-threaded.
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