"""Finite-panel Gaussian simulation references, not real-data headroom estimates."""
import csv
import json
import subprocess
import time
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from genopred.data.wheat599 import load, sha256_of
from genopred.adapters.wheat2000_gblup import predict_grid, raw_marker_kernel


def draw(eigenvalues, vectors, h2, seeds, environment):
    """Independent Gaussian traits; h2 specifies expected average panel variance."""
    if not 0 < h2 < 1:
        raise ValueError('Simulation h2 must lie strictly between zero and one')
    genetic, phenotypes = [], []
    root = vectors * np.sqrt(np.maximum(eigenvalues, 0) * h2)
    for seed in seeds:
        rng = np.random.default_rng(np.random.SeedSequence([seed, environment]))
        g = root @ rng.standard_normal(len(eigenvalues))
        genetic.append(g)
        phenotypes.append(g + np.sqrt(1 - h2) * rng.standard_normal(len(g)))
    return np.column_stack(genetic), np.column_stack(phenotypes)


def conditional_mean(kernel, y, fit, query, delta):
    """Known zero mean and covariance ratio; query phenotype values are unused."""
    factor = cho_factor(kernel[np.ix_(fit, fit)] + delta * np.eye(len(fit)))
    return kernel[np.ix_(query, fit)] @ cho_solve(factor, y[fit])


def correlations(truth, predictions):
    a, b = truth - truth.mean(0), predictions - predictions.mean(0)
    denominator = np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0)
    if np.any(denominator == 0):
        raise ValueError('Undefined simulation PCC from constant values')
    result = np.sum(a * b, axis=0) / denominator
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite simulation PCC')
    return result


def run(cfg, args, runner):
    if args.selection is not None:
        raise ValueError('Simulation diagnostic has no model-selection phase')
    start = time.perf_counter()
    read = lambda p: json.loads(Path(p).read_text())
    data = load(Path(cfg['data']), cfg['data_sha256'])
    folds = read(cfg['splits'])['folds']
    if len(folds) != 10 or sorted(i for f in folds for i in f['test_idx']) != list(range(data.n)):
        raise ValueError('Simulation requires existing complete ten-fold material coverage')
    for f in folds:
        if set(f['dev_idx']) & set(f['test_idx']) or set(f['dev_idx']) | set(f['test_idx']) != set(range(data.n)):
            raise ValueError('Invalid simulation fit/query split')
    source = read(cfg['parameter_source'])
    h2s = source['h2_per_env']
    seeds = cfg['simulation_seeds']
    if len(set(seeds)) != len(seeds) or not seeds:
        raise ValueError('Simulation repetitions require distinct explicit seeds')
    centered = data.X - data.X.mean(0)
    kernel = raw_marker_kernel(centered)
    kernel /= np.diag(kernel).mean()
    eigenvalues, vectors = np.linalg.eigh(kernel)
    if eigenvalues.min() < -1e-8:
        raise ValueError('Non-PSD simulation covariance')
    args.out.mkdir(parents=True, exist_ok=False)

    def save(name, value):
        (args.out / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

    paths = [Path(__file__), Path(runner), Path(args.config), Path(cfg['splits']),
             Path(cfg['parameter_source']), Path(__file__).with_name('wheat2000_gblup.py')]
    save('provenance.json', dict(config=cfg, git_commit=subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], text=True).strip(), sha256={str(p): sha256_of(p) for p in paths},
        generation='All-panel genotype-only centered marker GRM with mean diagonal one; known joint Gaussian simulation prior',
        parameters='Assumed generating h2 scenarios taken from the specified diagnostic source; no new estimation',
        real_phenotypes='Loaded for data validation only; never used for prediction, parameter estimation, or scoring here',
        intercepts='conditional_mean: known zero mean; centered_gblup: fit phenotype mean and fit-centered kernel via existing predict_grid'))
    state = dict(status='running', completed=0, expected=len(data.environments) * len(folds))
    failures, rows, arrays, summary = [], [], {}, {}
    save('execution_status.json', state)
    try:
        for e, env in enumerate(data.environments):
            h2 = h2s[env]
            g, y = draw(eigenvalues, vectors, h2, seeds, e)
            arrays[env + '_g'], arrays[env + '_y'] = g, y
            predictions = {m: np.empty_like(y) for m in ('conditional_mean', 'centered_gblup')}
            scores = {m: [] for m in (*predictions, 'known_genetic_value')}
            for f in folds:
                fit, query = np.asarray(f['dev_idx']), np.asarray(f['test_idx'])
                delta = (1 - h2) / h2
                pred = dict(conditional_mean=conditional_mean(kernel, y, fit, query, delta),
                    centered_gblup=predict_grid(kernel, y, fit, query, [delta])[0], known_genetic_value=g[query])
                for model, values in pred.items():
                    if values.shape != (len(query), len(seeds)) or not np.isfinite(values).all():
                        raise ValueError('Invalid simulated query prediction')
                    if model in predictions:
                        predictions[model][query] = values
                    pccs = correlations(y[query], values)
                    scores[model].append(pccs)
                    for j, seed in enumerate(seeds):
                        rows.append(dict(environment=env, fold=f['fold'], seed=seed, model=model,
                            pcc=float(pccs[j]), rmse=float(np.sqrt(np.mean((y[query, j] - values[:, j])**2))),
                            latent_g_rmse=float(np.sqrt(np.mean((g[query, j] - values[:, j])**2)))))
                state['completed'] += 1
            result = dict(assumed_h2=h2, sqrt_h2_reference=float(np.sqrt(h2)), repetitions=len(seeds), models={})
            for model, values in scores.items():
                per_repetition = np.mean(values, axis=0)
                pooled = correlations(y, g if model == 'known_genetic_value' else predictions[model])
                result['models'][model] = dict(fold_mean_pcc=float(per_repetition.mean()),
                    repetition_sd=float(per_repetition.std(ddof=1)),
                    monte_carlo_se=float(per_repetition.std(ddof=1) / np.sqrt(len(seeds))),
                    pooled_oof_pcc_mean=float(pooled.mean()), per_repetition_fold_mean_pcc=per_repetition.tolist())
                if model in predictions:
                    arrays[env + '_' + model] = predictions[model]
            result['sqrt_h2_minus_conditional_pcc'] = result['sqrt_h2_reference'] - result['models']['conditional_mean']['fold_mean_pcc']
            result['sampled_known_g_minus_conditional_pcc'] = result['models']['known_genetic_value']['fold_mean_pcc'] - result['models']['conditional_mean']['fold_mean_pcc']
            summary[env] = result
            save('execution_status.json', state)
        macro_repetitions = {m: np.mean([summary[e]['models'][m]['per_repetition_fold_mean_pcc']
            for e in data.environments], axis=0)
            for m in ('conditional_mean', 'centered_gblup', 'known_genetic_value')}
        save('summary.json', dict(scope='Model-conditional simulation references only, not estimates of real reducible/irreducible error',
            environments=summary,
            macro={m: dict(fold_mean_pcc=float(v.mean()), repetition_sd=float(v.std(ddof=1)),
                           monte_carlo_se=float(v.std(ddof=1) / np.sqrt(len(seeds))))
                   for m, v in macro_repetitions.items()},
            caveats=['Full-panel X defines the simulation population; this is not an inductive preprocessing recipe for a new real predictor.',
                'Known covariance/variance ratio gives privileged simulation information; these are not estimated from each fit dataset.',
                'Four environments are separate independent Gaussian simulation scenarios, not a correlated multi-environment phenotype model.',
                'The parameter source uses real labels for an exploratory diagnostic; simulation Monte Carlo error does not include parameter uncertainty.',
                'Finite-panel correlations with correlated materials need not equal sqrt(h2), even for known genetic values.',
                'Deviation from the assumed Gaussian model can raise or lower real predictive accuracy.',
                'Simulation PCC minus an unmatched real fusion score does not identify real reducible headroom.',
                'Monte Carlo SE is over independently simulated repetitions, never treating overlapping folds as independent samples.']))
        state['status'] = 'complete'
    except BaseException as error:
        state['status'] = 'failed'
        failures.append(dict(completed=state['completed'], error=repr(error)))
        raise
    finally:
        np.savez_compressed(args.out / 'simulation_arrays.npz', **arrays)
        with (args.out / 'per_repetition_fold.csv').open('w') as handle:
            writer = csv.DictWriter(handle, fieldnames=['environment', 'fold', 'seed', 'model', 'pcc', 'rmse', 'latent_g_rmse'])
            writer.writeheader()
            writer.writerows(rows)
        save('execution_status.json', state)
        save('failures.json', failures)
        save('timing.json', dict(seconds=time.perf_counter() - start))
