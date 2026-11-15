"""H1 ceiling diagnostic: marker-GRM REML h2 per environment + marker saturation.

Estimand: h_X^2 = sigma_X^2 / (sigma_X^2 + sigma_eps^2) for the additive marker
GRM built from all 599 materials (variance-component diagnostic, not a
prediction protocol). r_max ~= sqrt(h_X^2) is the additive prediction ceiling
of THIS marker model; no extra capture fraction is applied (the GRM already
represents the markers' explained share). Not an absolute ceiling: non-additive
signal and optimization/regularization error are outside this estimate.

Saturation check (corroborating only): GBLUP OOF PCC on 10%/25%/100% random
marker subsets under the frozen splits. LD redundancy and small-sample
estimation confound saturation; it cannot alone prove a data ceiling.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from wheat599.data import load, sha256_of
from wheat599.metrics import pcc
from wheat599.wheat2000_gblup import predict_grid

DATA_SHA = '8710523389007dd8b7c88a5deb74ee16aea0bae158496bc11f0626fb43b5ea21'
SPLITS = ROOT / 'results/get_icm_protocol_20261103/splits_manifest.json'
FUSION_REF = {'1': 0.587756, '2': 0.506038, '4': 0.470057, '5': 0.542173}
OUT = Path(__file__).resolve().parent


def reml_h2(eigenvalues, ut_y, ut_one):
    # Profile both intercept and sigma_g^2 (free two-component REML). With
    # W = K + delta I, the profiled restricted objective replaces the quadratic
    # Q by (n-1) log(Q/(n-1)); sigma_g^2 = Q/(n-1), sigma_e^2 = delta sigma_g^2.
    n = len(eigenvalues)

    def terms(log_delta):
        cov = eigenvalues + np.exp(log_delta)
        info = (ut_one ** 2 / cov).sum()
        b = (ut_one * ut_y / cov).sum() / info
        resid = ut_y - ut_one * b
        q = (resid ** 2 / cov).sum()
        return cov, info, q

    def nll(log_delta):
        cov, info, q = terms(log_delta)
        return 0.5 * (np.log(cov).sum() + np.log(info) + (n - 1) * np.log(q / (n - 1)))

    res = minimize_scalar(nll, bounds=(-14, 14), method='bounded')
    delta = float(np.exp(res.x))
    _, _, q = terms(res.x)
    sigma_g2 = float(q / (n - 1))
    sigma_e2 = delta * sigma_g2
    h2 = sigma_g2 / (sigma_g2 + sigma_e2)
    return h2, delta, sigma_g2, sigma_e2


def main():
    data = load(ROOT / 'data/raw/wheat.RData', DATA_SHA)
    X, Y = np.asarray(data.X, dtype=np.float64), np.asarray(data.Y)
    n, p = X.shape
    Z = X - X.mean(0)
    K = Z @ Z.T / p
    K /= np.diag(K).mean()
    w, U = np.linalg.eigh(K)
    w = np.maximum(w, 0.0)
    ut_y = U.T @ Y
    ut_one = U.T @ np.ones(n)
    h2, deltas, sigma_g2, sigma_e2 = {}, {}, {}, {}
    for e, env in enumerate(data.environments):
        h2[env], deltas[env], sigma_g2[env], sigma_e2[env] = reml_h2(w, ut_y[:, e], ut_one)
    r_max = {env: float(np.sqrt(h2[env])) for env in h2}
    pheno_corr = np.corrcoef(Y.T).tolist()

    splits = json.loads(SPLITS.read_text())
    folds = splits['folds']
    lambdas = np.logspace(-6, 3, 19)
    rng = np.random.default_rng(42)
    saturation = {}
    for frac in (0.1, 0.25, 1.0):
        cols = np.sort(rng.choice(p, size=int(round(p * frac)), replace=False)) if frac < 1.0 else np.arange(p)
        Xs = X[:, cols]
        raw = Xs @ Xs.T / len(cols)
        scores = []
        for fold in folds:
            test = np.asarray(fold['test_idx'])
            dev = np.asarray(fold['dev_idx'])
            inner = [(np.asarray(v['train_idx']), np.asarray(v['val_idx'])) for v in fold['inner']]
            for e, env in enumerate(data.environments):
                y = Y[:, e]
                inner_pcc = np.zeros(len(lambdas))
                for train, valid in inner:
                    pred = predict_grid(raw, y, train, valid, lambdas)[:, :, 0]
                    inner_pcc += [pcc(y[valid], pr) for pr in pred]
                best = int(np.argmax(inner_pcc / len(inner)))
                pred = predict_grid(raw, y, dev, test, lambdas[best:best + 1])[0, :, 0]
                scores.append(pcc(y[test], pred))
        saturation[str(frac)] = {'macro_pcc': float(np.mean(scores)), 'n_markers': len(cols)}
        print(f'saturation {frac}: {np.mean(scores):.6f} ({len(cols)} markers)', flush=True)

    summary = {
        'scope': 'H1 diagnostic, not a protocol comparison; GRM uses all materials for variance components',
        'estimand': 'h_X^2 = sigma_X^2/(sigma_X^2+sigma_eps^2) of the additive marker GRM, concentrated REML profiling both intercept and sigma_g^2; r_max=sqrt(h_X^2), no extra capture fraction',
        'revision': 'corrects b59bd26 which fixed sigma_g^2=1 (linear Q); now profiles sigma_g^2 with (n-1)log(Q/(n-1)). Still exploratory/oracle-reference, not an identifiable real ceiling.',
        'h2_per_env': h2, 'reml_delta_per_env': deltas,
        'reml_sigma_g2_per_env': sigma_g2, 'reml_sigma_e2_per_env': sigma_e2,
        'r_max_per_env': r_max,
        'fusion_outer_pcc_reference': FUSION_REF,
        'gap_r_max_minus_fusion': {e: r_max[e] - FUSION_REF[e] for e in r_max},
        'phenotype_correlation': {'envs': list(data.environments), 'matrix': pheno_corr},
        'marker_saturation_gblup': saturation,
        'caveats': [
            'r_max is the additive ceiling of this marker model, not an absolute ceiling; non-additive signal is outside it',
            'multi-environment borrowing can push per-env PCC above sqrt(h2_single)',
            'small-n optimization/regularization error means attainable PCC may sit below r_max',
            'subset saturation is corroborating only: LD redundancy and small-sample estimation confound it',
        ],
    }
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2))
    (OUT / 'provenance.json').write_text(json.dumps({
        'data_sha256': DATA_SHA, 'splits_sha256': sha256_of(SPLITS),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'seed': 42, 'lambda_grid': 'logspace(-6,3,19)', 'inner_folds': 3,
        'fusion_reference_source': 'docs/RESULTS.md frozen outer (2026-11-06)',
    }, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
