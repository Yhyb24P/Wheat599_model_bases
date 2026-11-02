#!/usr/bin/env python
"""Evaluate independently initialized official MoEGP experts on fixed folds."""
import argparse
import copy
import csv
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
from wheat599.data import load, sha256_of
from wheat599.metrics import pcc, rmse, mae


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/moegp.yaml')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    source = Path(cfg['source'])
    commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != cfg['source_commit'] or subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True).strip():
        raise RuntimeError('Official MoEGP source must match the configured clean commit')
    sys.path.insert(0, str(source))
    sys.dont_write_bytecode = True
    from utils.moe import Gating, creat_mlp_layer

    folds = json.loads(Path(cfg['splits']).read_text())['folds']
    data = load(Path(cfg['data']), cfg['data_sha256'])
    if len(folds) != 10:
        raise ValueError('Expected ten fixed folds')
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'provenance.json').write_text(json.dumps({'config': cfg, 'source_commit': commit, 'splits_sha256': sha256_of(Path(cfg['splits'])), 'independent_experts': True, 'x_scaling': 'none', 'y_scaling': 'train-only', 'checkpoint': 'minimum validation RMSE'}, indent=2))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(0.45)
    rows, predictions, failures = [], [], []
    for fold in folds:
        tr, va, te = (np.array(fold[k], dtype=int) for k in ('train_idx', 'val_idx', 'test_idx'))
        if len(set(tr) & set(va) | set(tr) & set(te) | set(va) & set(te)) or len(set(tr) | set(va) | set(te)) != data.n:
            raise ValueError('Invalid split')
        for j, env in enumerate(data.environments):
            start = time.time()
            model = None
            try:
                random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
                def expert():
                    return creat_mlp_layer(data.p, 1, cfg['hidden'], dropout=cfg['dropout'], batch_norm=True)
                model = Gating(data.p, cfg['experts'], expert(), top_k=cfg['top_k'], loss_coef=cfg['loss_coef'])
                model.experts = torch.nn.ModuleList([expert() for _ in range(cfg['experts'])])
                model.to(device)
                y = data.Y[:, j]
                mu, sd = y[tr].mean(), y[tr].std()
                if sd == 0:
                    raise ValueError('Constant training target')
                X = torch.as_tensor(data.X, dtype=torch.float32, device=device)
                Y = torch.as_tensor((y-mu)/sd, dtype=torch.float32, device=device).view(-1, 1)
                loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X[tr], Y[tr]), batch_size=cfg['batch_size'], shuffle=True)
                opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
                sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min')
                best, state, epoch_best = float('inf'), None, None
                for epoch in range(cfg['epochs']):
                    model.train()
                    for xb, yb in loader:
                        opt.zero_grad(set_to_none=True)
                        pred, aux = model(xb)
                        loss = torch.nn.functional.mse_loss(pred, yb) + aux
                        if not torch.isfinite(loss):
                            raise ValueError('Non-finite training loss')
                        loss.backward(); opt.step()
                    model.eval()
                    with torch.no_grad():
                        pv, aux = model(X[va])
                        val_mse = torch.nn.functional.mse_loss(pv, Y[va]).item()
                        sched.step(val_mse + aux.item())
                    if not np.isfinite(val_mse):
                        raise ValueError('Non-finite validation loss')
                    if val_mse < best:
                        best, state, epoch_best = val_mse, copy.deepcopy(model.state_dict()), epoch
                model.load_state_dict(state); model.eval()
                with torch.no_grad():
                    pred = model(X[te])[0].cpu().numpy().ravel() * sd + mu
                vals = {'pcc': pcc(y[te], pred), 'rmse': rmse(y[te], pred), 'mae': mae(y[te], pred)}
                if not all(np.isfinite(v) for v in vals.values()):
                    raise ValueError('Non-finite test metric')
                rows.append({'environment': env, 'fold': fold['fold'], **vals, 'best_epoch': epoch_best, 'val_rmse': best**0.5*sd, 'seconds': round(time.time()-start, 2), 'failed': False, 'reason': ''})
                predictions.extend({'environment': env, 'fold': fold['fold'], 'material_index': int(idx), 'y_true': float(y[idx]), 'y_pred': float(p)} for idx, p in zip(te, pred))
                print(f"env={env} fold={fold['fold']} PCC={vals['pcc']:.4f} {time.time()-start:.1f}s", flush=True)
            except Exception as exc:
                failures.append({'environment': env, 'fold': fold['fold'], 'reason': repr(exc)})
                rows.append({'environment': env, 'fold': fold['fold'], 'pcc': '', 'rmse': '', 'mae': '', 'best_epoch': '', 'val_rmse': '', 'seconds': round(time.time()-start, 2), 'failed': True, 'reason': repr(exc)})
                print(f'FAILED env={env} fold={fold["fold"]}: {exc}', flush=True)
                if isinstance(exc, torch.cuda.OutOfMemoryError):
                    raise
            finally:
                del model
    def write(name, fields, items):
        with (args.out / name).open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields, lineterminator='\n'); w.writeheader(); w.writerows(items)
    write('per_fold.csv', ['environment','fold','pcc','rmse','mae','best_epoch','val_rmse','seconds','failed','reason'], rows)
    write('oof.csv', ['environment','fold','material_index','y_true','y_pred'], predictions)
    write('failures.csv', ['environment','fold','reason'], failures)
    summary = {'n_runs': len(rows), 'failures': len(failures), 'per_environment': {}}
    if not failures and len(rows) == 40:
        for env in data.environments:
            summary['per_environment'][env] = {m: {'mean': float(np.mean([r[m] for r in rows if r['environment'] == env])), 'sd': float(np.std([r[m] for r in rows if r['environment'] == env], ddof=1))} for m in ('pcc','rmse','mae')}
        summary['macro'] = {m: float(np.mean([summary['per_environment'][e][m]['mean'] for e in data.environments])) for m in ('pcc','rmse','mae')}
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
