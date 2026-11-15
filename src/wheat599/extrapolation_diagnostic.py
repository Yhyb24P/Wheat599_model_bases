"""Describe saved OOF distance/error patterns and existing kernel shrinkage."""
import csv
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import eigvalsh
from scipy.stats import spearmanr

from wheat599.data import load, sha256_of
from wheat599.kernel_baselines import kernels


def spectral_summary(material, signal, residual):
    """Conditional Gaussian smoother EDF, excluding fitted-intercept correction."""
    material = np.maximum(np.asarray(material), 0)
    tasks = eigvalsh(signal, residual)
    if tasks.min() < -1e-8:
        raise ValueError('Signal covariance is not positive semidefinite')
    tasks = np.maximum(tasks, 0)
    snr = material[:, None] * tasks
    leverage = snr / (1 + snr)
    small = material <= 1e-12
    return dict(effective_df=float(leverage.sum()), modes_signal_above_noise=int(np.sum(snr > 1)),
                observations=int(snr.size), material_numerical_rank=int(np.sum(~small)),
                effective_df_in_material_modes_at_most_1e_minus12=float(leverage[small].sum()),
                task_generalized_eigenvalues=tasks.tolist(),
                material_eigenvalue_at_snr_one=[float(1 / t) if t > 0 else None for t in tasks])


def quartiles(distance):
    """Fold-local bins, keeping equal distances together even if sizes differ."""
    return np.searchsorted(np.quantile(distance, [.25, .5, .75]), distance, side='right')


def aggregate(rows):
    truth = np.array([r['truth_standardized'] for r in rows])
    pred = np.array([r['prediction_standardized'] for r in rows])
    distance = np.array([r['nearest_hamming_fraction'] for r in rows])
    rms_truth = float(np.sqrt(np.mean(truth**2)))
    return dict(n=len(rows), distance_mean=float(distance.mean()), distance_min=float(distance.min()),
                distance_max=float(distance.max()), rmse=float(np.sqrt(np.mean((pred - truth)**2))),
                mae=float(np.mean(np.abs(pred - truth))),
                prediction_rms=float(np.sqrt(np.mean(pred**2))), truth_rms=rms_truth,
                amplitude_ratio=float(np.sqrt(np.mean(pred**2)) / rms_truth) if rms_truth else None)


def run(cfg, args, runner):
    if args.selection is not None:
        raise ValueError('Diagnostic only; no training or selection')
    start = time.perf_counter()
    torch.set_num_threads(1)
    read = lambda path: json.loads(Path(path).read_text())
    data = load(Path(cfg['data']), cfg['data_sha256'])
    folds = read(cfg['splits'])['folds']
    if sorted(i for f in folds for i in f['test_idx']) != list(range(data.n)):
        raise ValueError('Incomplete test material coverage')
    split_by_id = {f['fold']: f for f in folds}
    paths = [Path(args.config), Path(cfg['splits']), Path(runner), Path(__file__),
             Path(__file__).with_name('kernel_baselines.py')]
    for source in (cfg['raw_source'], cfg['kernel_source'], cfg['reml_source']):
        info = read(Path(source) / 'provenance.json')
        if info['config']['data_sha256'] != cfg['data_sha256']:
            raise ValueError('Source data mismatch')
        old = read(info['config']['splits'])['folds']
        for f in old:
            current = split_by_id[f['fold']]
            dev = f['dev_idx'] if 'dev_idx' in f else f['train_idx'] + f['val_idx']
            if set(dev) != set(current['dev_idx']) or set(f['test_idx']) != set(current['test_idx']):
                raise ValueError('Source outer materials differ')
        paths.extend((Path(source) / 'provenance.json', Path(info['config']['splits'])))
    predictions = {}
    for source, fields in ((cfg['raw_source'], {'quantile': 'tabm', 'huber': 'huber', 'quantile_huber': 'fusion'}),
                           (cfg['kernel_source'], None)):
        path = Path(source) / 'oof.csv'
        paths.append(path)
        with path.open() as handle:
            for row in csv.DictReader(handle):
                f, env, material = int(row['fold']), row['environment'], int(row['material_index'])
                if material not in split_by_id[f]['test_idx'] or env not in data.environments:
                    raise ValueError('Invalid OOF identity')
                if abs(float(row['y_true']) - data.Y[material, data.environments.index(env)]) > 1e-6:
                    raise ValueError('OOF truth mismatch')
                for field, model in (fields.items() if fields else [('y_pred', row['model'])]):
                    key = model, f, env, material
                    if key in predictions or not np.isfinite(float(row[field])):
                        raise ValueError('Duplicate or nonfinite prediction')
                    predictions[key] = float(row[field])
    models = ['gaussian', 'additive_rbf', 'tabm', 'huber', 'fusion']
    expected = {(m, f['fold'], e, i) for m in models for f in folds
                for e in data.environments for i in f['test_idx']}
    if set(predictions) != expected:
        raise ValueError('Incomplete OOF prediction coverage')
    calibration = read(Path(cfg['reml_source']) / 'calibration.json')
    choices = read(cfg['huber_selection'])['selected']
    paths.extend((Path(cfg['reml_source']) / 'calibration.json', Path(cfg['huber_selection'])))
    rows, spectra = [], []
    for fold in folds:
        f, dev, test = fold['fold'], np.array(fold['dev_idx']), np.array(fold['test_idx'])
        if set(dev) & set(test) or set(dev) | set(test) != set(range(data.n)):
            raise ValueError('Invalid outer split')
        x, query = data.X[dev], data.X[test]
        distances = query.sum(1)[:, None] + x.sum(1)[None] - 2 * query @ x.T
        nearest = distances.min(1) / data.p
        bins = quartiles(nearest)
        means, scales = data.Y[dev].mean(0), data.Y[dev].std(0)
        for e, env in enumerate(data.environments):
            for model in models:
                for j, material in enumerate(test):
                    rows.append(dict(model=model, fold=f, environment=env, material_index=int(material),
                        distance_quartile=int(bins[j]), nearest_hamming_fraction=float(nearest[j]),
                        truth_standardized=float((data.Y[material, e] - means[e]) / scales[e]),
                        prediction_standardized=float((predictions[model, f, env, material] - means[e]) / scales[e])))
        record = next(r for r in calibration if r['fold'] == f)
        a, _, r, _ = kernels(data.X, dev, test, record['bandwidth'], 'cpu')
        eigenvalues = np.linalg.eigvalsh(record['fraction'] * a.numpy() + (1 - record['fraction']) * r.numpy())
        b, noise = np.asarray(record['signal_covariance']), np.asarray(record['residual_covariance'])
        spectra.append(dict(fold=f, model='reml_full', **spectral_summary(eigenvalues, b, noise)))
        spectra.append(dict(fold=f, model='reml_diagonal_counterfactual',
                            **spectral_summary(eigenvalues, np.diag(np.diag(b)), np.diag(np.diag(noise)))))
        choice = next(c for c in choices if c['fold'] == f)
        base, huber = choice['base'], choice['huber']
        a, _, r, _ = kernels(data.X, dev, test, base['bandwidth'], 'cpu')
        eig = np.linalg.eigvalsh(base['fraction'] * a.numpy() + (1 - base['fraction']) * r.numpy())
        task = (1 - huber['eta']) * np.eye(4) + huber['eta'] * np.corrcoef(data.Y[dev].T)
        ridge = base['ridge'][0] * huber['multiplier']
        spectra.append(dict(fold=f, model='huber_squared_geometry_proxy', ridge=ridge,
                            **spectral_summary(eig, task, ridge * np.eye(4))))
    grouped, correlations = [], []
    for model in models:
        for env in data.environments:
            per = [r for r in rows if r['model'] == model and r['environment'] == env]
            for q in range(4):
                subset = [r for r in per if r['distance_quartile'] == q]
                if subset:
                    grouped.append(dict(model=model, environment=env, quartile=q, **aggregate(subset)))
            rho = spearmanr([r['nearest_hamming_fraction'] for r in per],
                            [(r['prediction_standardized'] - r['truth_standardized'])**2 for r in per]).statistic
            correlations.append(dict(model=model, environment=env,
                                     spearman_distance_squared_error=float(rho) if np.isfinite(rho) else None))
    args.out.mkdir(parents=True, exist_ok=False)
    for name, values in (('by_material.csv', rows), ('distance_bins.csv', grouped)):
        with (args.out / name).open('w') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    summary = dict(scope='Descriptive saved outer-OOF diagnostic; no model fits, no configuration or cutoff selection',
        binning='Within-fold nearest-dev Hamming distance quartiles, ties stay together; fold-fit phenotype centering/scaling',
        amplitude='RMS prediction deviation from fit mean / RMS truth deviation from that same fit mean',
        distance_correlations=correlations, spectra=spectra,
        limitations=['Distance association does not identify a causal error mechanism; weak association does not prove estimation noise dominance.',
            'Effective df is a conditional Gaussian smoother trace, not effective sample size, excluding fitted-intercept correction.',
            'REML matrices are the saved fold-specific fitted estimates; the diagonal counterpart is not separately refitted.',
            'Huber spectrum is its squared-loss geometry proxy, not the nonlinear Huber estimator hat matrix.',
            'The 1e-12 cutoff is numerical; SNR=1 is a descriptive crossover, not a chosen statistical truncation rule.',
            'Repeatedly used outer folds support descriptive diagnostics only, never new configuration choices.'])
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    (args.out / 'provenance.json').write_text(json.dumps(dict(config=cfg,
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        seconds=time.perf_counter() - start, sha256={str(p): sha256_of(p) for p in paths}), indent=2) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(13, 6), constrained_layout=True)
    for col, env in enumerate(data.environments):
        for model in models:
            part = [g for g in grouped if g['model'] == model and g['environment'] == env]
            for row, key in enumerate(('rmse', 'amplitude_ratio')):
                axes[row, col].plot([g['distance_mean'] for g in part], [g[key] for g in part], marker='o', label=model)
        axes[0, col].set_title('Env ' + env)
        axes[1, col].set_xlabel('Nearest-fit Hamming fraction')
    axes[0, 0].set_ylabel('Fit-standardized RMSE')
    axes[1, 0].set_ylabel('RMS prediction / truth')
    axes[0, -1].legend(fontsize=8)
    fig.savefig(args.out / 'distance_error_amplitude.png', dpi=180)
    plt.close(fig)
