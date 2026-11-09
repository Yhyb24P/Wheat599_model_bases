"""Cross-fitted heterogeneous-teacher distillation into unchanged quantile TabM."""
import copy
import itertools
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.model_selection import KFold

from wheat599 import quantile_tabm as qt
from wheat599.data import load, sha256_of
from wheat599.heterogeneous_icm import choose
from wheat599.robust_icm import features, fit_predict
from wheat599.metrics import pcc, rmse, mae


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def splits(n, seed):
    return [(a, b) for a, b in KFold(3, shuffle=True, random_state=seed).split(np.arange(n))]


def distillation_loss(output, target, teacher, quantiles, strength):
    supervised = qt.quantile_loss(output, target, quantiles)
    # Distil each member's quantile mean; do not penalise inter-quantile spread.
    distil = output.shape[-1] * (output.mean(-1) - teacher.reshape(-1, 1)).square().mean()
    return supervised + strength * distil, supervised, distil


def predict(model, x, device):
    model.eval()
    with torch.no_grad():
        xt = torch.as_tensor(x, dtype=torch.float32, device=device)
        return model.inverse_y(model(xt).mean((1, 2))[:, None])[:, 0].cpu().numpy()


def fixed_fit(x, y, config, seed, index, epochs, cfg, device):
    model = qt.build_model(x, y, config, seed, cfg, device)
    xt = torch.as_tensor(x, dtype=torch.float32, device=device)
    yt = model.standardize_y(torch.as_tensor(y[:, None], dtype=torch.float32, device=device))
    q = torch.linspace(cfg['quantile_min'], cfg['quantile_max'], cfg['n_quantiles'], device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    generator = torch.Generator().manual_seed(int(seed) * 1000 + index)
    for _ in range(int(epochs)):
        qt.update_epoch(model, xt, yt, q, optimizer, generator, cfg['training'], device)
    return model


def prepare_query(x, y, query, base):
    # Query phenotypes are unavailable, not placeholders containing held-out truth.
    nx = len(x)
    all_x = np.concatenate((x, query))
    all_y = np.concatenate((y, np.full((len(query), y.shape[1]), np.nan)))
    return features(all_x, all_y, np.arange(nx), np.arange(nx, len(all_x)), base)


def fit_teacher(x, y, query, configs, cfg, kernel_cfg, split_seed, device, record, weight_dir):
    """Only teacher-training labels enter this API; held-out and student-val labels cannot enter selection."""
    inner = splits(len(x), split_seed)
    q_predictions, q_epochs = {}, {}
    for j, (tr, va) in enumerate(inner):
        for e in range(y.shape[1]):
            for c, config in enumerate(configs):
                for seed in cfg['seeds']:
                    pred, detail = qt.train(x[tr], y[tr, e], x[va], y[va, e], config, seed, c, cfg, device)
                    q_predictions[j, e, c, seed] = pred
                    q_epochs[j, e, c, seed] = detail['best_epoch']
                    record('teacher_selection', dict(teacher_inner=j, environment=e, config_index=c, seed=seed, **detail))
    q_selected, q_target, refit_info = [], [], []
    weight_dir.mkdir(parents=True, exist_ok=False)
    for e in range(y.shape[1]):
        scores = [float(np.mean([pcc(y[va, e], np.mean([q_predictions[j, e, c, seed] for seed in cfg['seeds']], axis=0))
                               for j, (_, va) in enumerate(inner)])) for c in range(len(configs))]
        c = int(np.argmax(scores)); q_selected.append(c)
        by_seed = []
        for seed in cfg['seeds']:
            epochs = int(np.median([q_epochs[j, e, c, seed] for j in range(3)]))
            model = fixed_fit(x, y[:, e], configs[c], seed, c, epochs, cfg, device)
            pred = predict(model, query, device)
            if not np.isfinite(pred).all():
                raise ValueError('Nonfinite teacher prediction')
            by_seed.append(pred)
            path = weight_dir / f'e{e}_s{seed}.pt'
            torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, path)
            entry = dict(environment=e, config_index=c, seed=seed, epochs=epochs, weight_sha256=sha256_of(path))
            refit_info.append(entry)
            record('teacher_refit', entry)
        q_target.append(np.mean(by_seed, axis=0))
    huber_candidates = list(itertools.product(kernel_cfg['etas'], kernel_cfg['huber_deltas'], kernel_cfg['ridge_multipliers']))
    huber_scores = np.empty((3, len(huber_candidates)))
    kernel_bases = []
    for j, (tr, va) in enumerate(inner):
        sub = [(tr[a], tr[b]) for a, b in splits(len(tr), split_seed + 100 + j)]
        _, base, _ = choose(x, y, sub, kernel_cfg, 'cpu')
        kernel_bases.append(base)
        prepared = features(x, y, tr, va, base)
        for c, (eta, delta, multiplier) in enumerate(huber_candidates):
            pred, detail = fit_predict(prepared, base['ridge'][0] * multiplier, eta, delta, kernel_cfg)
            score = np.mean([pcc(y[va, e], pred[:, e]) for e in range(y.shape[1])])
            if not np.isfinite(score):
                raise ValueError('Nonfinite Huber teacher selection score')
            huber_scores[j, c] = score
            record('huber_selection', dict(teacher_inner=j, eta=eta, delta=delta, multiplier=multiplier, pcc=float(score), **detail))
    means = huber_scores.mean(0)
    best = next(i for i, score in enumerate(means) if max(means) - score <= kernel_cfg['tie_tolerance'])
    eta, delta, multiplier = huber_candidates[best]
    _, base, _ = choose(x, y, inner, kernel_cfg, 'cpu')
    prepared = prepare_query(x, y, query, base)
    hp, detail = fit_predict(prepared, base['ridge'][0] * multiplier, eta, delta, kernel_cfg)
    record('huber_refit', dict(eta=eta, delta=delta, multiplier=multiplier, **detail))
    qp = np.column_stack(q_target)
    if not np.isfinite(qp).all() or not np.isfinite(hp).all():
        raise ValueError('Nonfinite teacher target')
    return qp, hp, dict(quantile_selected=q_selected, refits=refit_info, kernel_bases=kernel_bases,
        huber_base=base, huber_selected=dict(eta=eta, delta=delta, multiplier=multiplier),
        inner=[dict(train=a.tolist(), valid=b.tolist()) for a, b in inner])


def crossfit_teacher(x, y, fit_ids, configs, cfg, kernel_cfg, seed, device, record, out):
    qt_target, huber_target = np.empty_like(y), np.empty_like(y)
    coverage = np.zeros(len(y), dtype=int)
    partitions = []
    for k, (tr, held) in enumerate(splits(len(y), seed)):
        if set(tr) & set(held) or set(tr) | set(held) != set(range(len(y))):
            raise ValueError('Invalid teacher cross-fitting split')
        def log(kind, detail):
            record(kind, dict(teacher_fold=k, **detail))
        # Deliberately slice Y before calling teacher, excluding all held-out traits.
        qp, hp, detail = fit_teacher(x[tr], y[tr], x[held], configs, cfg, kernel_cfg,
                                     seed + 1000 + k, device, log, out / f'teacher_{k}')
        qt_target[held], huber_target[held] = qp, hp
        coverage[held] += 1
        partitions.append(dict(teacher_fold=k, train_ids=np.asarray(fit_ids)[tr].tolist(),
                               heldout_ids=np.asarray(fit_ids)[held].tolist(), **detail))
    if not np.all(coverage == 1):
        raise ValueError('Teacher targets must cover every fitting material exactly once')
    np.savez_compressed(out / 'teacher_targets.npz', fit_ids=fit_ids, quantile=qt_target,
                        huber=huber_target, fusion=.5 * (qt_target + huber_target))
    write(out / 'teacher_partitions.json', partitions)
    fusion = .5 * (qt_target + huber_target)
    write(out / 'teacher_quality.json', {name: [[fn(y[:, e], pred[:, e]) for fn in (pcc, rmse, mae)]
        for e in range(y.shape[1])] for name, pred in (('quantile', qt_target), ('huber', huber_target), ('fusion', fusion))})
    return fusion


def finetune(model, x_fit, y_fit, teacher, x_val, y_val, strength, config, seed, index, cfg, device):
    start = time.perf_counter()
    torch.manual_seed(int(seed) + 900000)
    x = torch.as_tensor(x_fit, dtype=torch.float32, device=device)
    y = model.standardize_y(torch.as_tensor(y_fit[:, None], dtype=torch.float32, device=device))
    t = model.standardize_y(torch.as_tensor(teacher[:, None], dtype=torch.float32, device=device))
    q = torch.linspace(cfg['quantile_min'], cfg['quantile_max'], cfg['n_quantiles'], device=device)
    spec = cfg['finetuning']
    opt = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'] * spec['learning_rate_multiplier'],
                            weight_decay=config['weight_decay'])
    generator = torch.Generator().manual_seed(int(seed) * 1000 + index + 900000)
    best_prediction = predict(model, x_val, device)
    best_pcc, best_epoch = pcc(y_val, best_prediction), 0
    if not np.isfinite(best_pcc):
        raise ValueError('Nonfinite warm-start PCC')
    warm_prediction = best_prediction.copy()
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    history = []
    for epoch in range(1, spec['max_epochs'] + 1):
        model.train(); total_pin, total_kd, seen = 0., 0., 0
        for batch in torch.randperm(len(x), generator=generator).to(device).split(cfg['training']['batch_size']):
            opt.zero_grad(set_to_none=True)
            loss, pin, kd = distillation_loss(model(x[batch]), y[batch], t[batch], q, strength)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite distillation loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['training']['gradient_clip'])
            opt.step()
            total_pin += pin.item() * len(batch); total_kd += kd.item() * len(batch); seen += len(batch)
        pred = predict(model, x_val, device); score = pcc(y_val, pred)
        if not np.isfinite(score):
            raise ValueError('Nonfinite student validation PCC')
        history.append(dict(epoch=epoch, pinball=total_pin / seen, teacher_mse_times_q=total_kd / seen, val_pcc=score))
        if score > best_pcc + cfg['training']['tie_tolerance']:
            best_pcc, best_epoch, best_prediction = score, epoch, pred.copy()
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= spec['patience']:
            break
    model.load_state_dict(best_state)
    return best_prediction, warm_prediction, dict(strength=strength, best_pcc=best_pcc, best_epoch=best_epoch,
        epochs_run=epoch, seconds=time.perf_counter() - start, history=history)


def run(cfg, args, runner):
    if args.selection is not None:
        raise ValueError('Development only')
    torch.set_num_threads(cfg['cpu_threads']); device = torch.device(cfg['device'])
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(cfg['gpu_memory_fraction'])
    data = load(Path(cfg['data']), cfg['data_sha256']); folds = read(cfg['splits'])['folds']
    configs = yaml.safe_load(Path(cfg['architecture_config']).read_text())['configurations']
    kernel_cfg = yaml.safe_load(Path(cfg['huber_config']).read_text())
    source = Path(cfg['baseline_source']); old = read(source / 'summary.json')
    baseline_rows = read(source / 'trajectories.json')
    old_index = {(v['fold'], v['inner'], v['environment'], v['config_index'], v['seed']):v for v in baseline_rows}
    if len(old_index) != 1920 or len(folds) != 10 or any(len(f['inner']) != 3 for f in folds) or len(configs) != 8 or len(cfg['seeds']) != 2 or cfg['strengths'] != [0., .01, .1, 1.]:
        raise ValueError('Unexpected frozen budget or baseline coverage')
    if read(source / 'execution_status.json') != dict(status='complete', completed=1920, expected=1920) or read(source / 'failures.json'):
        raise ValueError('Incomplete baseline')
    baseline_prov = read(source / 'provenance.json')
    for key in ('data_sha256', 'seeds', 'n_quantiles', 'quantile_min', 'quantile_max', 'training'):
        if baseline_prov['config'][key] != cfg[key]:
            raise ValueError(f'Original training recipe changed: {key}')
    if baseline_prov['sha256'][cfg['splits']] != sha256_of(Path(cfg['splits'])):
        raise ValueError('Baseline split mismatch')
    audit = read(cfg['baseline_audit'])['sha256']
    for f in folds:
        for j in range(3):
            path = source / f"predictions_{f['fold']}_{j}.npz"
            if sha256_of(path) != audit[str(path)]:
                raise ValueError('Baseline predictions changed')
    args.out.mkdir(parents=True, exist_ok=False)
    source_paths = [Path(__file__), Path(qt.__file__), Path(runner), Path(args.config), Path(cfg['splits']),
        Path(cfg['architecture_config']), Path(cfg['huber_config']), Path(cfg['baseline_audit']), source / 'summary.json',
        source / 'trajectories.json', source / 'provenance.json', Path(cfg['fusion_reference']) / 'summary.json']
    source_paths += [source / f"predictions_{f['fold']}_{j}.npz" for f in folds for j in range(3)]
    source_paths += [Path(__file__).with_name(n+'.py') for n in ('robust_icm','heterogeneous_icm','reml_icm','kernel_baselines','genomic_tabm','data')]
    write(args.out / 'provenance.json', dict(config=cfg, git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        torch_version=str(torch.__version__), sha256={str(p):sha256_of(p) for p in source_paths}))
    expected = dict(teacher_selection=17280, teacher_refit=720, huber_selection=6480, huber_refit=90,
                    student_initialization=240, student_finetune=960)
    state = dict(status='running', completed={k:0 for k in expected}, expected=expected)
    records, failures, start = [], [], time.perf_counter()
    context = {}
    def record(kind, detail):
        records.append(dict(kind=kind, **context, **detail)); state['completed'][kind] += 1
        write(args.out / 'execution_status.json', state)
    try:
        for f in folds:
            fold = f['fold']; choices = next(r['selected'] for r in old['folds'] if r['fold']==fold)
            for j, split in enumerate(f['inner']):
                tr, va = np.array(split['train_idx']), np.array(split['val_idx'])
                if set(tr)&set(va) or set(tr)|set(va)!=set(f['dev_idx']) or (set(tr)|set(va))&set(f['test_idx']):
                    raise ValueError('Invalid student split')
                context = dict(fold=fold, inner=j)
                unit = args.out / f'unit_{fold}_{j}'; unit.mkdir()
                teacher = crossfit_teacher(data.X[tr], data.Y[tr], tr, configs, cfg, kernel_cfg,
                                            cfg['split_seed'] + fold * 10 + j, device, record, unit)
                predictions, student_details = {}, []
                original = np.load(source / f'predictions_{fold}_{j}.npz')
                for e in range(4):
                    c = choices[e]; config = configs[c]
                    for seed in cfg['seeds']:
                        epochs = old_index[(fold,j,e,c,seed)]['best_epoch']
                        warm = fixed_fit(data.X[tr],data.Y[tr,e],config,seed,c,epochs,cfg,device)
                        warm_pred = predict(warm,data.X[va],device)
                        reference = original[f'e{e}_c{c}_s{seed}']
                        difference = float(np.max(abs(warm_pred-reference)))
                        if difference > cfg['warm_prediction_tolerance']:
                            raise ValueError(f'Warm start does not reproduce original prediction: {difference}')
                        record('student_initialization', dict(environment=e,config_index=c,seed=seed,epochs=epochs,max_difference=difference))
                        predictions[f'e{e}_s{seed}_original']=warm_pred
                        for strength in cfg['strengths']:
                            model = copy.deepcopy(warm)
                            pred, _, detail = finetune(model,data.X[tr],data.Y[tr,e],teacher[:,e],data.X[va],data.Y[va,e],
                                                       strength,config,seed,c,cfg,device)
                            predictions[f'e{e}_s{seed}_l{strength}']=pred
                            weight_path = unit / f'student_e{e}_s{seed}_l{strength}.pt'
                            torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},weight_path)
                            detail.update(environment=e,config_index=c,seed=seed,weight_sha256=sha256_of(weight_path))
                            student_details.append(detail)
                            record('student_finetune',{k:v for k,v in detail.items() if k!='history'})
                np.savez_compressed(args.out / f'predictions_{fold}_{j}.npz',**predictions)
                write(unit / 'student_training.json',student_details)
                write(args.out / 'trajectories.json',records)
                print(f'completed student unit {fold}/{j}',flush=True)
        if state['completed'] != expected:
            raise ValueError('Incomplete distillation budget')
        summaries=[]
        for f in folds:
            fold=f['fold']; arrays=[np.load(args.out/f'predictions_{fold}_{j}.npz') for j in range(3)]
            scores=np.empty((4,len(cfg['strengths'])))
            for e in range(4):
                for l,strength in enumerate(cfg['strengths']):
                    scores[e,l]=np.mean([pcc(data.Y[s['val_idx'],e],np.mean([arrays[j][f'e{e}_s{seed}_l{strength}'] for seed in cfg['seeds']],axis=0)) for j,s in enumerate(f['inner'])])
            chosen=[cfg['strengths'][int(np.argmax(scores[e]))] for e in range(4)]
            methods={'student':[], 'extra_training_lambda0':[], 'original_quantile':[]}
            for method in methods:
                for j,s in enumerate(f['inner']):
                    matrix=[]
                    for e in range(4):
                        suffix='original' if method=='original_quantile' else 'l0.0' if method=='extra_training_lambda0' else f'l{chosen[e]}'
                        pred=np.mean([arrays[j][f'e{e}_s{seed}_{suffix}'] for seed in cfg['seeds']],axis=0)
                        matrix.append([fn(data.Y[s['val_idx'],e],pred) for fn in (pcc,rmse,mae)])
                    methods[method].append(matrix)
            summaries.append(dict(fold=fold,selected_strengths=chosen,metrics={k:np.mean(v,axis=0).tolist() for k,v in methods.items()}))
        methods=list(summaries[0]['metrics'])
        macro={m:dict(zip(('pcc','rmse','mae'),np.mean([r['metrics'][m] for r in summaries],axis=(0,1)).tolist())) for m in methods}
        fusion_source=Path(cfg['fusion_reference'])
        fusion=read(fusion_source/'summary.json')
        for row in summaries:
            row['metrics']['original_fusion']=next(r['metrics']['original_fusion'] for r in fusion['folds'] if r['fold']==row['fold'])
        macro['original_fusion']=dict(zip(('pcc','rmse','mae'),np.mean([r['metrics']['original_fusion'] for r in summaries],axis=(0,1)).tolist()))
        student, original = macro['student'], macro['original_quantile']
        gate = (student['pcc'] > max(macro[m]['pcc'] for m in ('original_quantile','original_fusion','extra_training_lambda0'))
                and student['rmse'] <= original['rmse'] and student['mae'] <= original['mae'])
        write(args.out/'summary.json',dict(macro=macro,folds=summaries,outer_gate_passed=bool(gate)))
        state['status']='complete'
    except BaseException as exc:
        state['status']='failed';failures.append(dict(context=context,error=repr(exc)))
        raise
    finally:
        write(args.out/'execution_status.json',state);write(args.out/'failures.json',failures)
        write(args.out/'trajectories.json',records);write(args.out/'timing.json',dict(seconds=time.perf_counter()-start))
