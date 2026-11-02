"""Multi-environment additive and Gaussian kernels for unseen materials."""
import numpy as np
import torch
import csv
import json
from pathlib import Path
from sklearn.model_selection import KFold

from wheat599.kernel_baselines import kernels, predict
from wheat599.metrics import pcc
from wheat599.data import load, sha256_of
from wheat599.metrics import rmse, mae


def reliable_correlation(predictions, targets):
    """PSD task correlation gated by training-only OOF predictive correlation."""
    p = np.asarray(predictions, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if p.shape != y.shape or p.ndim != 2 or not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError('Invalid OOF predictions or targets')
    pc = p - p.mean(axis=0)
    yc = y - y.mean(axis=0)
    ps = np.linalg.norm(pc, axis=0)
    ys = np.linalg.norm(yc, axis=0)
    good = (ps > 1e-12) & (ys > 1e-12)
    z = pc / np.where(ps > 1e-12, ps, 1)
    corr = z.T @ z
    corr[np.diag_indices_from(corr)] = 1
    r = np.zeros(p.shape[1])
    r[good] = np.clip(np.sum(pc[:, good] * yc[:, good], axis=0) / (ps[good] * ys[good]), 0, 1)
    root = np.sqrt(r)
    return root[:, None] * corr * root[None, :] + np.diag(1 - r), r


def fit_predict(kernels, cross_kernels, targets, task_matrices, ridge):
    """Joint KRR with independent unpenalized intercepts; row-major tasks."""
    y = torch.as_tensor(targets, dtype=kernels[0].dtype, device=kernels[0].device)
    n, q = y.shape
    mean, scale = y.mean(0), y.std(0, unbiased=False)
    if torch.any(scale <= 0):
        raise ValueError('Constant training target')
    yn = (y - mean) / scale
    dtype, device = y.dtype, y.device
    noise = torch.as_tensor(ridge, dtype=dtype, device=device)
    cov = torch.zeros((n*q, n*q), dtype=dtype, device=device)
    for kernel, task in zip(kernels, task_matrices):
        cov += torch.kron(kernel.contiguous(), torch.as_tensor(task, dtype=dtype, device=device).contiguous())
    cov.diagonal().add_(noise.repeat(n))
    design = torch.eye(q, dtype=dtype, device=device).repeat(n, 1)
    rhs = torch.cat((yn.reshape(-1, 1), design), dim=1)
    solved = torch.cholesky_solve(rhs, torch.linalg.cholesky(cov))
    intercept = torch.linalg.solve(design.T @ solved[:, 1:], design.T @ solved[:, :1])
    coefficients = (solved[:, :1] - solved[:, 1:] @ intercept).reshape(n, q)
    prediction = intercept.reshape(1, q).expand(cross_kernels[0].shape[0], q).clone()
    for cross, task in zip(cross_kernels, task_matrices):
        prediction += cross @ coefficients @ torch.as_tensor(task, dtype=dtype, device=device)
    return (prediction * scale + mean).cpu().numpy()


def choose_base(X, Y, cfg, seed, device):
    """Choose independent environment kernels or a single shared ICM kernel."""
    candidates = [(h, lam, frac) for h in cfg['bandwidths'] for lam in cfg['ridge'] for frac in cfg['additive_fractions']]
    scores = np.empty((len(candidates), cfg['base_folds'], Y.shape[1]))
    splits = KFold(n_splits=cfg['base_folds'], shuffle=True, random_state=seed)
    for k, (tr, va) in enumerate(splits.split(X)):
        for h in cfg['bandwidths']:
            mats = kernels(X, tr, va, h, device)
            for i, (hh, lam, frac) in enumerate(candidates):
                if hh != h:
                    continue
                for e in range(Y.shape[1]):
                    scores[i, k, e] = pcc(Y[va, e], predict(*mats, Y[tr, e], frac, lam))
    if not np.isfinite(scores).all():
        raise ValueError('Non-finite base selection PCC')
    means = scores.mean(axis=1)
    independent = [candidates[int(np.argmax(means[:, e]))] for e in range(Y.shape[1])]
    common = candidates[int(np.argmax(means.mean(axis=1)))]
    return independent, common


def crossfit_reliability(X, Y, cfg, device):
    """OOF predictions use fixed single-kernel hyperparameters within training data."""
    pa, pr = np.empty_like(Y), np.empty_like(Y)
    splits = KFold(n_splits=cfg['reliability_folds'], shuffle=True, random_state=cfg['reliability_seed'])
    for tr, va in splits.split(X):
        mats = kernels(X, tr, va, cfg['reliability_bandwidth'], device)
        for e in range(Y.shape[1]):
            pa[va, e] = predict(*mats, Y[tr, e], 1.0, cfg['reliability_ridge'])
            pr[va, e] = predict(*mats, Y[tr, e], 0.0, cfg['reliability_ridge'])
    return reliable_correlation(pa, Y)[0], reliable_correlation(pr, Y)[0]


def task_correlation(Y):
    centered = Y - Y.mean(0)
    norm = np.linalg.norm(centered, axis=0)
    if np.any(norm <= 1e-12):
        raise ValueError('Constant training phenotype')
    z = centered / norm
    return z.T @ z


def prepare(X, Y, target, params, cfg, device):
    """Kernel and task statistics depend exclusively on the fit subset."""
    idx = np.arange(len(X))
    allx = np.concatenate([X, target])
    target_idx = np.arange(len(X), len(allx))
    hvals = sorted(set(p[0] for p in params))
    mats = {h: kernels(allx, idx, target_idx, h, device) for h in hvals}
    ca, cr = crossfit_reliability(X, Y, cfg, device)
    return mats, ca, cr, task_correlation(Y)


def prediction(prepared, Y, params, eta_a, eta_r, mode):
    mats, ca, cr, phenotype_corr = prepared
    q = Y.shape[1]
    identity = np.eye(q)
    hvals = sorted(mats)
    frac = np.array([p[2] for p in params])
    lam = np.array([p[1] for p in params])
    if mode == 'icm':
        if len(set(params)) != 1 or eta_a != eta_r:
            raise ValueError('ICM requires shared kernel, ridge and task matrix')
        ca = cr = phenotype_corr
    sa = (1-eta_a)*identity + eta_a*ca
    sr = (1-eta_r)*identity + eta_r*cr
    root = np.sqrt(frac)
    components = [mats[hvals[0]][0]]
    cross = [mats[hvals[0]][1]]
    tasks = [root[:,None]*sa*root[None,:]]
    for h in hvals:
        root = np.sqrt(np.array([(1-p[2]) if p[0] == h else 0 for p in params]))
        components.append(mats[h][2]); cross.append(mats[h][3])
        tasks.append(root[:,None]*sr*root[None,:])
    return fit_predict(components, cross, Y, tasks, lam)


def run(cfg, args, runner_path):
    data = load(Path(cfg['data']), cfg['data_sha256'])
    folds = json.loads(Path(cfg['splits']).read_text())['folds']
    split_hash = sha256_of(Path(cfg['splits']))
    frozen = json.loads(args.selection.read_text()) if args.selection else None
    if frozen and (frozen['config'] != cfg or frozen['splits_sha256'] != split_hash):
        raise ValueError('Selection configuration or split mismatch')
    if frozen and ({x['fold'] for x in frozen['selected']} != {f['fold'] for f in folds} or len(frozen['selected']) != len(folds)):
        raise ValueError('Missing or duplicate frozen selection')
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out/'provenance.json').write_text(json.dumps({'config':cfg, 'splits_sha256':split_hash, 'model_sha256':sha256_of(Path(__file__)), 'runner_sha256':sha256_of(runner_path), 'selection_sha256':sha256_of(args.selection) if args.selection else None, 'rule':'macro PCC over environments and inner folds; earliest grid tie', 'protocol':'CV1; full development refit'}, indent=2))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(0.35)
    selected, valrows, rows, oof, failures = [], [], [], [], []
    combos = [(a,r) for a in cfg['sharing'] for r in cfg['sharing']]
    for fold in folds:
        k = fold['fold']; test = np.asarray(fold['test_idx']); dev = np.sort(np.r_[fold['train_idx'], fold['val_idx']])
        if len(set(dev)&set(test)) or len(set(dev)|set(test)) != data.n:
            raise ValueError('Invalid outer split')
        print(f'fold {k}', flush=True)
        try:
            if frozen is None:
                icm_scores = np.empty((len(cfg['sharing']),cfg['inner_folds'],4))
                selective_scores = np.empty((len(combos),cfg['inner_folds'],4))
                splits = KFold(n_splits=cfg['inner_folds'], shuffle=True, random_state=cfg['seed']+k)
                for j,(itr,iva) in enumerate(splits.split(dev)):
                    tr,va = dev[itr],dev[iva]
                    independent,common = choose_base(data.X[tr],data.Y[tr],cfg,cfg['seed']+1000+k*10+j,device)
                    prep_ind = prepare(data.X[tr],data.Y[tr],data.X[va],independent,cfg,device)
                    for i,(ea,er) in enumerate(combos):
                        pred = prediction(prep_ind,data.Y[tr],independent,ea,er,'selective')
                        selective_scores[i,j] = [pcc(data.Y[va,e],pred[:,e]) for e in range(4)]
                    prep_icm = prepare(data.X[tr],data.Y[tr],data.X[va],[common]*4,cfg,device)
                    for i,eta in enumerate(cfg['sharing']):
                        pred = prediction(prep_icm,data.Y[tr],[common]*4,eta,eta,'icm')
                        icm_scores[i,j] = [pcc(data.Y[va,e],pred[:,e]) for e in range(4)]
                    print(f'  inner {j} complete', flush=True)
                if not np.isfinite(icm_scores).all() or not np.isfinite(selective_scores).all():
                    raise ValueError('Non-finite validation PCC')
                im = icm_scores.mean(axis=(1,2)); sm = selective_scores.mean(axis=(1,2))
                ii,si = int(np.argmax(im)),int(np.argmax(sm))
                selected.append({'fold':k,'icm_eta':cfg['sharing'][ii],'icm_pcc':float(im[ii]),'selective_eta_a':combos[si][0],'selective_eta_r':combos[si][1],'selective_pcc':float(sm[si])})
                for model, grid, scores in [('icm',[(e,e) for e in cfg['sharing']],icm_scores),('selective',combos,selective_scores)]:
                    for i,(ea,er) in enumerate(grid):
                        for j in range(cfg['inner_folds']):
                            for e,env in enumerate(data.environments):
                                valrows.append({'model':model,'fold':k,'inner_fold':j,'environment':env,'eta_a':ea,'eta_r':er,'pcc':float(scores[i,j,e])})
                (args.out/'selection.json').write_text(json.dumps({'config':cfg,'splits_sha256':split_hash,'selected':selected},indent=2))
            else:
                entry = next(v for v in frozen['selected'] if v['fold']==k)
                independent,common = choose_base(data.X[dev],data.Y[dev],cfg,cfg['seed']+k,device)
                for model,params,ea,er in [('icm',[common]*4,entry['icm_eta'],entry['icm_eta']),('selective',independent,entry['selective_eta_a'],entry['selective_eta_r'])]:
                    if ea not in cfg['sharing'] or er not in cfg['sharing']:
                        raise ValueError('Invalid sharing strength')
                    prep = prepare(data.X[dev],data.Y[dev],data.X[test],params,cfg,device)
                    pred = prediction(prep,data.Y[dev],params,ea,er,model)
                    for e,env in enumerate(data.environments):
                        metrics = {'pcc':pcc(data.Y[test,e],pred[:,e]),'rmse':rmse(data.Y[test,e],pred[:,e]),'mae':mae(data.Y[test,e],pred[:,e])}
                        if not all(np.isfinite(v) for v in metrics.values()):
                            raise ValueError('Non-finite test metric')
                        rows.append({'model':model,'environment':env,'fold':k,**metrics,'eta_a':ea,'eta_r':er,'bandwidth':params[e][0],'ridge':params[e][1],'additive_fraction':params[e][2]})
                        oof.extend({'model':model,'environment':env,'fold':k,'material_index':int(idx),'y_true':float(data.Y[idx,e]),'y_pred':float(v)} for idx,v in zip(test,pred[:,e]))
        except Exception as exc:
            failures.append({'fold':k,'reason':repr(exc)}); print(f'FAILED {k}: {exc}',flush=True)
            if isinstance(exc,torch.cuda.OutOfMemoryError): raise
        finally:
            for name,fields,items in [('validation.csv',['model','fold','inner_fold','environment','eta_a','eta_r','pcc'],valrows),('per_fold.csv',['model','environment','fold','pcc','rmse','mae','eta_a','eta_r','bandwidth','ridge','additive_fraction'],rows),('oof.csv',['model','environment','fold','material_index','y_true','y_pred'],oof),('failures.csv',['fold','reason'],failures)]:
                with (args.out/name).open('w',newline='') as f:
                    writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n'); writer.writeheader(); writer.writerows(items)
    if failures or (len(selected)!=10 if frozen is None else len(rows)!=80):
        raise RuntimeError('Incomplete run; inspect failures')
    if frozen is None:
        print('Development selection complete; no test predictions made.',flush=True); return
    summary={'n_runs':len(rows),'n_failures':0,'models':{}}
    for model in ('icm','selective'):
        per={e:{m:float(np.mean([r[m] for r in rows if r['model']==model and r['environment']==e])) for m in ('pcc','rmse','mae')} for e in data.environments}
        summary['models'][model]={'per_environment':per,'macro':{m:float(np.mean([per[e][m] for e in data.environments])) for m in ('pcc','rmse','mae')}}
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary),flush=True)
