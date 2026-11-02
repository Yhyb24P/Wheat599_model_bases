#!/usr/bin/env python
"""Background-conditioned additive ridge with conditional analytic LOO gates."""
import argparse
import csv
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
from wheat599.data import load, sha256_of
from wheat599.metrics import pcc, rmse, mae


def kernel(g, h, gram, global_penalty, background_penalty):
    return gram * (1.0 / global_penalty + (g @ h.T) / background_penalty)


def solve(k, y):
    """Generalized ridge solve with an unpenalized intercept."""
    n = y.shape[0]
    chol = torch.linalg.cholesky(k + torch.eye(n, device=y.device, dtype=y.dtype))
    inv = torch.cholesky_inverse(chol)
    ones = torch.ones_like(y)
    u = inv @ ones
    denom = ones.T @ u
    proj = inv - (u @ u.T) / denom
    alpha = proj @ y
    intercept = (u.T @ y) / denom
    return alpha, intercept, torch.diagonal(proj).reshape(-1, 1)


def gates(x, w):
    return torch.softmax(x @ w, dim=1)


def fit_candidate(x, gram, y, gl, bl, cfg):
    w = torch.nn.Parameter(torch.randn(x.shape[1], cfg['gates'], device=x.device, dtype=x.dtype) * cfg['gate_init_std'])
    opt = torch.optim.Adam([w], lr=cfg['gate_lr'])
    best_loss, best_w = float('inf'), None
    for step in range(cfg['gate_steps'] + 1):
        opt.zero_grad(set_to_none=True)
        q = gates(x, w)
        alpha, _, diag = solve(kernel(q, q, gram, gl, bl), y)
        loss = torch.mean((alpha / diag)**2)
        if not torch.isfinite(loss):
            raise ValueError('Non-finite conditional LOO loss')
        if loss.item() < best_loss:
            best_loss, best_w = loss.item(), w.detach().clone()
        if step < cfg['gate_steps']:
            loss.backward()
            opt.step()
    with torch.no_grad():
        q = gates(x, best_w)
        alpha, intercept, _ = solve(kernel(q, q, gram, gl, bl), y)
    return best_w, alpha, intercept, best_loss


def write_csv(path, fields, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/background_ridge.yaml')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    data = load(Path(cfg['data']), cfg['data_sha256'])
    folds = json.loads(Path(cfg['splits']).read_text())['folds']
    if len(folds) != 10:
        raise ValueError('Expected ten fixed folds')
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'provenance.json').write_text(json.dumps({'config': cfg, 'splits_sha256': sha256_of(Path(cfg['splits'])), 'selection': 'minimum validation RMSE; ties use first configured pair', 'loo': 'analytic leave-one-out conditional on training PCA and gate values; gates are not refit per held-out sample', 'kernel': 'centered X X.T / p * (1/global_penalty + q q.T/background_penalty), unit noise, unpenalized intercept', 'test': 'once after validation selection'}, indent=2))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(0.35)
    rows, oof, failures, choices = [], [], [], []
    fields = ['environment','fold','pcc','rmse','mae','global_penalty','background_penalty','val_rmse','loo_mse','seconds','failed','reason']
    for f in folds:
        tr, va, te = (np.asarray(f[k], dtype=int) for k in ('train_idx','val_idx','test_idx'))
        if len(set(tr) | set(va) | set(te)) != data.n or set(tr) & set(va) or set(tr) & set(te) or set(va) & set(te):
            raise ValueError('Invalid split')
        centered = data.X - data.X[tr].mean(axis=0)
        pca = PCA(n_components=cfg['pca_components'], svd_solver='randomized', random_state=cfg['seed']).fit(centered[tr])
        scores = pca.transform(centered)
        scale = scores[tr].std(axis=0)
        if np.any(scale == 0):
            raise ValueError('Zero PCA scale')
        scores = (scores - scores[tr].mean(axis=0)) / scale
        x = torch.tensor(np.column_stack([np.ones(data.n), scores]), dtype=torch.float64, device=device)
        z = torch.tensor(centered, dtype=torch.float64, device=device)
        gram = z @ z[tr].T / data.p
        for j, env in enumerate(data.environments):
            t0 = time.time()
            try:
                torch.manual_seed(cfg['seed'])
                y = data.Y[:, j]
                ym, ys = y[tr].mean(), y[tr].std()
                if ys == 0:
                    raise ValueError('Constant training target')
                yt = torch.tensor(((y[tr] - ym) / ys)[:, None], dtype=torch.float64, device=device)
                selected = None
                for gl, bl in itertools.product(cfg['global_penalty'], cfg['background_penalty']):
                    torch.manual_seed(cfg['seed'])
                    w, alpha, intercept, loo = fit_candidate(x[tr], gram[tr], yt, gl, bl, cfg)
                    with torch.no_grad():
                        predval = (intercept + kernel(gates(x[va], w), gates(x[tr], w), gram[va], gl, bl) @ alpha).cpu().numpy().ravel() * ys + ym
                    vr = rmse(y[va], predval)
                    if not np.isfinite(vr):
                        raise ValueError('Non-finite validation metric')
                    choices.append({'environment': env, 'fold': f['fold'], 'global_penalty': gl, 'background_penalty': bl, 'val_rmse': vr, 'loo_mse': loo})
                    if selected is None or vr < selected[0]:
                        selected = (vr, gl, bl, w, alpha, intercept, loo)
                vr, gl, bl, w, alpha, intercept, loo = selected
                with torch.no_grad():
                    pred = (intercept + kernel(gates(x[te], w), gates(x[tr], w), gram[te], gl, bl) @ alpha).cpu().numpy().ravel() * ys + ym
                metrics = {'pcc': pcc(y[te], pred), 'rmse': rmse(y[te], pred), 'mae': mae(y[te], pred)}
                if not all(np.isfinite(v) for v in metrics.values()):
                    raise ValueError('Non-finite test metric')
                rows.append({'environment':env, 'fold':f['fold'], **metrics, 'global_penalty':gl, 'background_penalty':bl, 'val_rmse':vr, 'loo_mse':loo, 'seconds':round(time.time()-t0,2), 'failed':False, 'reason':''})
                oof.extend({'environment':env,'fold':f['fold'],'material_index':int(idx),'y_true':float(y[idx]),'y_pred':float(p)} for idx,p in zip(te,pred))
                print(f"env={env} fold={f['fold']} PCC={metrics['pcc']:.4f} {time.time()-t0:.1f}s", flush=True)
            except Exception as exc:
                failure = {'environment':env, 'fold':f['fold'], 'reason':repr(exc)}
                failures.append(failure)
                rows.append({'environment':env, 'fold':f['fold'], 'failed':True, 'reason':repr(exc)})
                print(f'FAILED {failure}', flush=True)
                if isinstance(exc, torch.cuda.OutOfMemoryError):
                    raise
            finally:
                write_csv(args.out/'per_fold.csv', fields, rows)
                write_csv(args.out/'oof.csv', ['environment','fold','material_index','y_true','y_pred'], oof)
                write_csv(args.out/'failures.csv', ['environment','fold','reason'], failures)
                write_csv(args.out/'validation.csv', ['environment','fold','global_penalty','background_penalty','val_rmse','loo_mse'], choices)
    summary = {'n_runs':len(rows), 'n_failures':len(failures)}
    if len(rows) == 40 and not failures:
        summary['per_environment'] = {e:{m:{'mean':float(np.mean([r[m] for r in rows if r['environment']==e])), 'sd':float(np.std([r[m] for r in rows if r['environment']==e],ddof=1))} for m in ('pcc','rmse','mae')} for e in data.environments}
        summary['macro'] = {m:float(np.mean([summary['per_environment'][e][m]['mean'] for e in data.environments])) for m in ('pcc','rmse','mae')}
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
