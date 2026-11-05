#!/usr/bin/env python
"""Nested PCC selection and full-development refit for conditioned ridge and RR."""
import argparse
import csv
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
from wheat599.data import load, sha256_of
from wheat599.metrics import pcc, rmse, mae
from run_background_ridge import fit_candidate, gates, kernel, write_csv
from wheat599.background_rbf import inputs as hybrid_inputs, fit_predict as hybrid_predict


def design(cfg):
    keys = ('pca_components','gates','global_penalty','background_penalty','gate_steps','gate_lr')
    combinations = list(itertools.product(*(cfg[k] for k in keys)))
    rng = np.random.default_rng(cfg['search_seed'])
    chosen = rng.choice(len(combinations), size=cfg['budget'], replace=False)
    return [dict(zip(keys, combinations[i])) for i in chosen]


def features(X, train, target, n_pc, seed, device):
    centered = X - X[train].mean(axis=0)
    pca = PCA(n_components=n_pc, svd_solver='randomized', random_state=seed).fit(centered[train])
    scores = pca.transform(centered[np.r_[train,target]])
    mu, sd = scores[:len(train)].mean(0), scores[:len(train)].std(0)
    if np.any(sd == 0):
        raise ValueError('Constant PCA component')
    x = torch.tensor(np.column_stack([np.ones(len(scores)), (scores-mu)/sd]), dtype=torch.float64, device=device)
    ztr = torch.tensor(centered[train], dtype=torch.float64, device=device)
    ztarget = torch.tensor(centered[target], dtype=torch.float64, device=device)
    return x[:len(train)], x[len(train):], ztr @ ztr.T / X.shape[1], ztarget @ ztr.T / X.shape[1]


def model_predict(x, xt, gram, cross, ytrain, c, cfg):
    ym, ys = ytrain.mean(), ytrain.std()
    if ys == 0:
        raise ValueError('Constant target')
    y = torch.tensor(((ytrain-ym)/ys)[:,None], dtype=torch.float64, device=x.device)
    torch.manual_seed(cfg['seed'])
    local = {'gates':c['gates'], 'gate_steps':c['gate_steps'], 'gate_lr':c['gate_lr'], 'gate_init_std':cfg['gate_init_std']}
    w, alpha, intercept, _ = fit_candidate(x, gram, y, c['global_penalty'], c['background_penalty'], local)
    with torch.no_grad():
        pred = intercept + kernel(gates(xt,w), gates(x,w), cross, c['global_penalty'], c['background_penalty']) @ alpha
    return pred.cpu().numpy().ravel()*ys+ym


def rr_predict(X, y, train, target, alpha):
    z = X - X[train].mean(axis=0)
    mu = y[train].mean()
    gram = z[train] @ z[train].T
    weights = np.linalg.solve(gram + alpha*np.eye(len(train)), y[train]-mu)
    return mu + (z[target] @ z[train].T) @ weights


def run_hybrid(cfg, args):
    data = load(Path(cfg['data']), cfg['data_sha256'])
    folds = json.loads(Path(cfg['splits']).read_text())['folds']
    if len(folds) != 10:
        raise ValueError('Expected ten outer folds')
    split_hash = sha256_of(Path(cfg['splits']))
    frozen = json.loads(args.selection.read_text()) if args.selection else None
    if frozen is not None:
        if frozen['config'] != cfg or frozen['splits_sha256'] != split_hash:
            raise ValueError('Frozen selections do not match configuration or splits')
        keys = [(v['environment'], v['fold']) for v in frozen['selected']]
        expected = {(e, f['fold']) for e in data.environments for f in folds}
        if len(keys) != len(expected) or set(keys) != expected:
            raise ValueError('Frozen selection incomplete or duplicated')
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'provenance.json').write_text(json.dumps({
        'config': cfg, 'splits_sha256': split_hash,
        'runner_sha256': sha256_of(Path(__file__)),
        'model_sha256': sha256_of(ROOT/'src/wheat599/background_rbf.py'),
        'solver_sha256': sha256_of(ROOT/'src/wheat599/kernel_ridge.py'),
        'selection_sha256': sha256_of(args.selection) if args.selection else None,
        'selection': 'mean inner-fold PCC; ties use first bandwidth',
        'refit': 'entire outer development set',
    }, indent=2))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(0.35)
    selections, validation, rows, oof, failures = [], [], [], [], []
    for fold in folds:
        test = np.asarray(fold['test_idx'], dtype=int)
        dev = np.sort(np.r_[fold['train_idx'], fold['val_idx']])
        if len(set(test) & set(dev)) or len(set(test) | set(dev)) != data.n:
            raise ValueError('Invalid outer split')
        inner = [(dev[a], dev[b]) for a, b in KFold(n_splits=cfg['inner_folds'], shuffle=True, random_state=cfg['seed']+fold['fold']).split(dev)]
        print(f"outer fold {fold['fold']}: dev={len(dev)} test={len(test)}", flush=True)
        for j, env in enumerate(data.environments):
            start = time.time(); y = data.Y[:, j]
            try:
                if frozen is None:
                    scores = np.empty((len(cfg['bandwidths']), len(inner)))
                    for k, (train, valid) in enumerate(inner):
                        feat = features(data.X, train, valid, cfg['pca_components'], cfg['seed'], device)
                        for i, h in enumerate(cfg['bandwidths']):
                            prepared = hybrid_inputs(data.X, train, valid, feat, h, cfg.get('redundancy_shrinkage'))
                            pred, weights, loo = hybrid_predict(prepared, y[train], cfg, cfg['seed'])
                            scores[i, k] = pcc(y[valid], pred)
                            if not np.isfinite(scores[i, k]):
                                raise ValueError('Non-finite inner PCC')
                            validation.append({'environment':env, 'fold':fold['fold'], 'inner_fold':k, 'bandwidth':h, 'pcc':float(scores[i,k]), 'loo_mse':loo, 'weights':json.dumps(weights)})
                    winner = int(np.argmax(scores.mean(axis=1)))
                    selections.append({'environment':env, 'fold':fold['fold'], 'bandwidth':cfg['bandwidths'][winner], 'inner_pcc':float(scores[winner].mean())})
                    (args.out/'selection.json').write_text(json.dumps({'config':cfg, 'splits_sha256':split_hash, 'selected':selections}, indent=2))
                    print(f'  env={env} selected in {time.time()-start:.1f}s', flush=True)
                else:
                    entry = next(v for v in frozen['selected'] if v['environment']==env and v['fold']==fold['fold'])
                    if entry['bandwidth'] not in cfg['bandwidths']:
                        raise ValueError('Selected bandwidth is not configured')
                    feat = features(data.X, dev, test, cfg['pca_components'], cfg['seed'], device)
                    prepared = hybrid_inputs(data.X, dev, test, feat, entry['bandwidth'], cfg.get('redundancy_shrinkage'))
                    pred, weights, loo = hybrid_predict(prepared, y[dev], cfg, cfg['seed'])
                    metrics = {'pcc':pcc(y[test],pred), 'rmse':rmse(y[test],pred), 'mae':mae(y[test],pred)}
                    if not all(np.isfinite(v) for v in metrics.values()):
                        raise ValueError('Non-finite test metric')
                    rows.append({'environment':env, 'fold':fold['fold'], **metrics, 'bandwidth':entry['bandwidth'], 'inner_pcc':entry['inner_pcc'], 'additive_weight':weights[0], 'background_weight':weights[1], 'rbf_weight':weights[2], 'loo_mse':loo, 'seconds':round(time.time()-start,2)})
                    oof.extend({'environment':env, 'fold':fold['fold'], 'material_index':int(idx), 'y_true':float(y[idx]), 'y_pred':float(v)} for idx,v in zip(test,pred))
                    print(f'  env={env} evaluated in {time.time()-start:.1f}s', flush=True)
            except Exception as exc:
                failures.append({'environment':env, 'fold':fold['fold'], 'reason':repr(exc)})
                print(f'FAILED {env} {fold["fold"]}: {exc}', flush=True)
                if isinstance(exc, torch.cuda.OutOfMemoryError):
                    raise
            finally:
                write_csv(args.out/'validation.csv', ['environment','fold','inner_fold','bandwidth','pcc','loo_mse','weights'], validation)
                write_csv(args.out/'per_fold.csv', ['environment','fold','pcc','rmse','mae','bandwidth','inner_pcc','additive_weight','background_weight','rbf_weight','loo_mse','seconds'], rows)
                write_csv(args.out/'oof.csv', ['environment','fold','material_index','y_true','y_pred'], oof)
                write_csv(args.out/'failures.csv', ['environment','fold','reason'], failures)
    if failures or (len(selections) != 40 if frozen is None else len(rows) != 40):
        raise RuntimeError('Incomplete run; inspect failures')
    if frozen is None:
        print('Development selection complete; no test predictions made.', flush=True)
        return
    summary = {'n_runs':len(rows), 'n_failures':0, 'per_environment':{e:{m:{'mean':float(np.mean([r[m] for r in rows if r['environment']==e])), 'sd':float(np.std([r[m] for r in rows if r['environment']==e], ddof=1))} for m in ('pcc','rmse','mae')} for e in data.environments}}
    summary['macro'] = {m:float(np.mean([summary['per_environment'][e][m]['mean'] for e in data.environments])) for m in ('pcc','rmse','mae')}
    (args.out/'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, default=ROOT/'configs/background_ridge_nested.yaml')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--selection', type=Path, default=None, help='Frozen development-only selections; omit for selection phase')
    p.add_argument('--resume-summary', action='store_true',
                   help='Independent TabM only: finalize saved fits and predictions without training')
    args = p.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    if args.resume_summary and cfg.get('model') != 'independent_tabm_final':
        p.error('--resume-summary requires model independent_tabm_final')
    if cfg.get('model') == 'robust_icm':
        from wheat599.robust_icm import run
        return run(cfg, args, Path(__file__))
    if cfg.get('model') == 'non_gaussian_icm':
        from wheat599.non_gaussian_icm import run
        return run(cfg, args, Path(__file__))
    if cfg.get('model') == 'non_gaussian_control':
        from wheat599.posterior_control import run
        return run(cfg, args, Path(__file__))
    if cfg.get('model') == 'lcm_gp':
        from wheat599.lcm_gp import run
        return run(cfg, args, Path(__file__))
    if cfg.get('model') == 'dnawhisper':
        from wheat599.dnawhisper.run import run
        return run(cfg, args, Path(__file__))
    if cfg.get('model') == 'locus_context':
        from wheat599.locus_context_run import run
        return run(cfg, args, Path(__file__))
    if cfg.get('model') == 'genomic_tabm':
        from wheat599.genomic_tabm_run import run
        return run(cfg, args, Path(__file__))
    if cfg.get('model') == 'retrieval_tabm':
        from wheat599.retrieval_run import run
        return run(cfg, args, Path(__file__))
    if cfg.get('model') == 'retrieval_fusion':
        from wheat599.retrieval_fusion import run
        return run(cfg, args, Path(__file__))
    if cfg.get('model') == 'ragp':
        from wheat599.ragp_run import run
        return run(cfg, args, Path(__file__))
    if cfg.get('model') == 'environment_tabm':
        from wheat599.environment_run import run
        return run(cfg, args, Path(__file__))
    if cfg.get('model') == 'phenotype_rotation':
        from wheat599.rotation_run import run
        return run(cfg, args, Path(__file__))
    if cfg.get('model') == 'environment_selection':
        from wheat599.environment_selection import run
        return run(cfg, args, Path(__file__))
    if cfg.get('model') == 'late_transfer':
        from wheat599.late_transfer_run import run
        return run(cfg, args, Path(__file__))
    if cfg.get('model') == 'independent_tabm_freeze':
        from wheat599.independent_freeze import run
        return run(cfg, args, Path(__file__))
    if cfg.get('model') == 'independent_tabm_final':
        from wheat599.independent_final import run
        return run(cfg, args, Path(__file__))
    if cfg.get('model') == 'vector_diagnostic':
        from wheat599.vector_diagnostic import run
        return run(cfg, args, Path(__file__))
    if cfg.get('model') == 'joint_inference':
        from wheat599.joint_training import run
        return run(cfg, args, Path(__file__))
    if cfg.get('model') == 'matrix_inference':
        from wheat599.matrix_training import run
        return run(cfg, args, Path(__file__))
    if cfg.get('model') == 'matrix_kernel_reference':
        from wheat599.matrix_kernel_reference import run
        return run(cfg, args, Path(__file__))
    if cfg.get('model') == 'sparse_interaction_diagnostic':
        from wheat599.sparse_diagnostic import run
        return run(cfg, args, Path(__file__))
    if cfg.get('model') == 'heterogeneous_icm':
        from wheat599.heterogeneous_icm import run
        return run(cfg, args, Path(__file__))
    if cfg.get('model') == 'kernel_averaging':
        from wheat599.kernel_averaging import run
        return run(cfg, args, Path(__file__))
    if cfg.get('model') == 'target_icm':
        from wheat599.joint_kernel_selection import run_target
        return run_target(cfg, args, Path(__file__))
    if cfg.get('model') == 'reml_icm':
        from wheat599.reml_icm import run
        return run(cfg, args, Path(__file__))
    if cfg.get('model') == 'get_icm_joint':
        from wheat599.joint_kernel_selection import run
        return run(cfg, args, Path(__file__))
    if cfg.get('model') == 'get_icm':
        from wheat599.get_icm import run
        return run(cfg, args, Path(__file__))
    if cfg.get('model') == 'bh_lmc':
        from wheat599.bh_lmc import run
        return run(cfg, args, Path(__file__))
    if cfg.get('model') == 'spectral_icm':
        from wheat599.spectral_icm import run
        return run(cfg, args, Path(__file__))
    if cfg.get('model') == 'multienv_kernel':
        from wheat599.multienv_kernel import run
        return run(cfg, args, Path(__file__))
    if cfg.get('model') == 'kernel_baselines':
        from wheat599.kernel_baselines import run
        return run(cfg, args, Path(__file__))
    if cfg.get('model') == 'background_rbf':
        return run_hybrid(cfg, args)
    data = load(Path(cfg['data']), cfg['data_sha256'])
    folds = json.loads(Path(cfg['splits']).read_text())['folds']
    if len(folds) != 10:
        raise ValueError('Expected ten outer folds')
    candidates = design(cfg)
    frozen = json.loads(args.selection.read_text()) if args.selection else None
    if frozen is not None and (frozen['config'] != cfg or frozen['candidates'] != candidates or frozen['splits_sha256'] != sha256_of(Path(cfg['splits']))):
        raise ValueError('Frozen selections do not match configuration or splits')
    if frozen is not None:
        keys = [(v['environment'], v['fold']) for v in frozen['selected']]
        expected = {(e, f['fold']) for e in data.environments for f in folds}
        if len(keys) != len(expected) or set(keys) != expected:
            raise ValueError('Frozen selection is incomplete or duplicated')
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out/'provenance.json').write_text(json.dumps({'config':cfg,'splits_sha256':sha256_of(Path(cfg['splits'])),'runner_sha256':sha256_of(Path(__file__)),'core_sha256':sha256_of(ROOT/'scripts/run_background_ridge.py'),'selection_sha256':sha256_of(args.selection) if args.selection else None,'selection':'mean inner-fold PCC, ties use first candidate; RR uses same PCC rule','refit':'entire outer development set; no test feedback','candidates':candidates},indent=2))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(0.35)
    rows, oof, validation, failures, selected_rows = [], [], [], [], []
    field = ['model','environment','fold','pcc','rmse','mae','selected','inner_pcc','seconds','failed','reason']
    for fold in folds:
        test = np.asarray(fold['test_idx'], dtype=int)
        dev = np.sort(np.r_[fold['train_idx'],fold['val_idx']])
        if len(set(test)&set(dev)) or len(set(test)|set(dev)) != data.n:
            raise ValueError('Invalid outer split')
        inner = [(dev[a],dev[b]) for a,b in KFold(n_splits=cfg['inner_folds'],shuffle=True,random_state=cfg['seed']+fold['fold']).split(dev)]
        print(f"outer fold {fold['fold']}: dev={len(dev)} test={len(test)}",flush=True)
        for j,env in enumerate(data.environments):
            start=time.time(); y=data.Y[:,j]
            try:
                if frozen is not None:
                    match=[v for v in frozen['selected'] if v['environment']==env and v['fold']==fold['fold']]
                    if len(match)!=1:
                        raise ValueError('Missing or duplicate frozen selection')
                    entry=match[0]; i=entry['candidate']; alpha=entry['rr_alpha']; chosen=candidates[i]
                    x,xt,gram,cross=features(data.X,dev,test,chosen['pca_components'],cfg['seed'],device)
                    predictions={'background':model_predict(x,xt,gram,cross,y[dev],chosen,cfg), 'rrblup':rr_predict(data.X,y,dev,test,alpha)}
                    for model,pred in predictions.items():
                        metrics={'pcc':pcc(y[test],pred),'rmse':rmse(y[test],pred),'mae':mae(y[test],pred)}
                        if not all(np.isfinite(v) for v in metrics.values()):
                            raise ValueError('Non-finite test metric')
                        rows.append({'model':model,'environment':env,'fold':fold['fold'],**metrics,'selected':i if model=='background' else alpha,'inner_pcc':entry['background_pcc'] if model=='background' else entry['rr_pcc'],'seconds':round(time.time()-start,2),'failed':False,'reason':''})
                        oof.extend({'model':model,'environment':env,'fold':fold['fold'],'material_index':int(idx),'y_true':float(y[idx]),'y_pred':float(v)} for idx,v in zip(test,pred))
                    print(f"  env={env} evaluated in {time.time()-start:.1f}s",flush=True)
                    continue
                scores=np.empty((len(candidates),len(inner)))
                rr_scores=np.empty((len(cfg['rr_alphas']),len(inner)))
                for k,(train,valid) in enumerate(inner):
                    cache={d:features(data.X,train,valid,d,cfg['seed'],device) for d in cfg['pca_components']}
                    for i,c in enumerate(candidates):
                        pred=model_predict(*cache[c['pca_components']],y[train],c,cfg)
                        scores[i,k]=pcc(y[valid],pred)
                        if not np.isfinite(scores[i,k]):
                            raise ValueError('Non-finite inner PCC')
                    for i,alpha in enumerate(cfg['rr_alphas']):
                        rr_scores[i,k]=pcc(y[valid],rr_predict(data.X,y,train,valid,alpha))
                        if not np.isfinite(rr_scores[i,k]):
                            raise ValueError('Non-finite RR inner PCC')
                for model,vals in (('background',scores),('rrblup',rr_scores)):
                    validation.extend({'model':model,'environment':env,'fold':fold['fold'],'candidate':i,'inner_fold':k,'pcc':float(v)} for i,row in enumerate(vals) for k,v in enumerate(row))
                i=int(np.argmax(scores.mean(axis=1))); r=int(np.argmax(rr_scores.mean(axis=1)))
                selected_rows.append({'environment':env,'fold':fold['fold'],'candidate':i,'rr_alpha':cfg['rr_alphas'][r],'background_pcc':float(scores[i].mean()),'rr_pcc':float(rr_scores[r].mean())})
                (args.out/'selection.json').write_text(json.dumps({'config':cfg,'candidates':candidates,'splits_sha256':sha256_of(Path(cfg['splits'])),'selected':selected_rows},indent=2))
                print(f"  env={env} selected in {time.time()-start:.1f}s",flush=True)
            except Exception as exc:
                failures.append({'environment':env,'fold':fold['fold'],'reason':repr(exc)})
                print(f'FAILED {env} {fold["fold"]}: {exc}',flush=True)
                if isinstance(exc,torch.cuda.OutOfMemoryError):
                    raise
            finally:
                write_csv(args.out/'per_fold.csv',field,rows)
                write_csv(args.out/'oof.csv',['model','environment','fold','material_index','y_true','y_pred'],oof)
                write_csv(args.out/'validation.csv',['model','environment','fold','candidate','inner_fold','pcc'],validation)
                write_csv(args.out/'failures.csv',['environment','fold','reason'],failures)
    if frozen is None:
        if len(selected_rows)!=40 or failures:
            raise RuntimeError('Development selection incomplete; no test evaluation permitted')
        print('Development selection complete; no test predictions made.',flush=True)
        return
    summary={'n_runs':len(rows),'failures':failures}
    if len(rows)==80 and not failures:
        summary['models']={m:{e:{metric:{'mean':float(np.mean([r[metric] for r in rows if r['model']==m and r['environment']==e])),'sd':float(np.std([r[metric] for r in rows if r['model']==m and r['environment']==e],ddof=1))} for metric in ('pcc','rmse','mae')} for e in data.environments} for m in ('background','rrblup')}
        for m in summary['models']:
            summary['models'][m]['macro']={metric:float(np.mean([summary['models'][m][e][metric]['mean'] for e in data.environments])) for metric in ('pcc','rmse','mae')}
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
