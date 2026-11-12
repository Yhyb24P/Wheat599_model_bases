"""Complete Mul-PheG2P under the Wheat599 unified ten-fold protocol.

Official source: wjhhh2928/Mul-PheG2P @ e20ca6847c7dec6ce80386e94df4378ad2d1fbec
(MIT). This module keeps the full author mechanism and lets the TARGET encoder
participate as a fusion source (target self-attention), which the frozen-source
re-evaluation in ``mulphe.py`` excluded. The target encoder follows the paper's
two stages: independent pretraining, then target adaptation fine-tuned from the
pretrained weights.

Fusion candidates, labelled separately and never chosen from outer test:

* ``scalar_with_target`` — the paper-confirmable prediction-space fusion. Each
  frozen encoder contributes its trained scalar phenotype prediction, including
  the target encoder's adapted prediction,
  fused by ``MetaTraitFusion`` attention and trained with MSE only. Differences
  from the public ``fine_tune_predict.py`` are recorded, not hidden: the public
  script adds an extra 1->64->1 head after the fusion, a +0.2*Pearson loss term
  and gradient clipping; the paper's fusion is MSE-only, so those three public
  additions are NOT used here. The Adam + ReduceLROnPlateau schedule is retained
  from the public code. This is an adaptation, not a claim of paper reproduction.
* ``hidden_with_target`` — the public ``train.py`` default embedding fusion
  (``EmbeddingAttentionFusion``) over source+target embeddings, MinMax target
  scaling fit on train only, with the public validation linear calibration. The
  raw (uncalibrated) prediction is also kept as ``hidden_with_target_uncalibrated``.

Calibration boundary: the full-dev refit has no separate validation split, so the
linear calibration coefficients (a, b) are fit per target/config/seed on the OOF
validation predictions and labels of the SAME outer fold's three inner splits,
frozen before any test prediction is produced, and applied to the refit's test
output. No test label is used to fit or choose calibration.

Stages and public default caps (``train.py`` @ e20ca68): source/target pretrain
300, target adaptation (finetune) 300, fusion 200; lr .001, batch 32, width 128.
The paper reports 500/300/200; both are recorded and the code defaults are
authoritative for the "public default recipe".

Fitting boundary: every fit uses only the current train/val rows; no fitting
function accepts a test phenotype; scaling and imputation are fit on train only.
``run`` selects epochs/config on the three inner folds' validation rows, refits
on the full development set at per-stage median inner epochs, and reads test
labels once for scoring after every fit has finished.
"""
from __future__ import annotations

import csv
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from wheat599 import mulphe
from wheat599.data import load, sha256_of
from wheat599.metrics import mae, pcc, rmse

ARMS = ('scalar_with_target', 'hidden_with_target')
SYSTEMS = ('scalar_with_target', 'hidden_with_target_uncalibrated', 'hidden_with_target')
STAGES = ('pretrain', 'finetune', 'fusion')
METRICS = ('pcc', 'rmse', 'mae')
METRIC_FUNCTIONS = (pcc, rmse, mae)
OFFICIAL_REPO = mulphe.OFFICIAL_REPO
OFFICIAL_COMMIT = mulphe.OFFICIAL_COMMIT


# -- target adaptation (paper stage 2, fine-tuned from the pretrained target) --
def _adapt_target(x_train, y_train, x_val, y_val, config, seed, device,
                  init_model, init_head, fixed_epochs=None):
    mulphe._seed(seed)
    model = mulphe._encoder_for(config, x_train.shape[1]).to(device)
    head = nn.Linear(int(config['hidden_dim']), 1).to(device)
    model.load_state_dict(init_model)
    head.load_state_dict(init_head)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(head.parameters()),
                                 lr=float(config['lr']))
    loader = mulphe._loader(x_train, y_train, config['batch_size'], True, seed)
    best = {'score': np.inf, 'epoch': 0, 'model': None, 'head': None}
    history = []
    epochs = int(fixed_epochs) if fixed_epochs is not None else int(config['finetune_epochs'])
    for epoch in range(1, epochs + 1):
        model.train()
        head.train()
        total = 0.0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = head(model(batch_x)).squeeze(1)
            loss = F.mse_loss(prediction, batch_y) + 0.2 * mulphe.pearson_loss(prediction, batch_y)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite target-adaptation loss')
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(batch_x)
        if fixed_epochs is not None:
            history.append({'epoch': epoch, 'train_loss': total / len(x_train)})
            continue
        score, val_mse = mulphe.batch_validation_loss(
            model, head, x_val, y_val, device, config['batch_size'], combined=True)
        history.append({'epoch': epoch, 'train_loss': total / len(x_train),
                        'val_score': score, 'val_mse': val_mse})
        if score < best['score']:
            best = {'score': score, 'epoch': epoch,
                    'model': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    'head': {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}}
    if fixed_epochs is not None:
        best = {'score': None, 'epoch': epochs,
                'model': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                'head': {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}}
    model.load_state_dict(best['model'])
    head.load_state_dict(best['head'])
    return model, head, best, history


# -- paper-confirmable prediction-space fusion (MSE only) ------------------
def _scalar_fusion_arm(encoders, x_train, y_train, x_val, y_val, x_predict,
                       config, seed, device, fixed_epoch=None):
    started = time.perf_counter()
    num_traits = len(encoders)
    scalar_train = mulphe._source_predictions(encoders, x_train, device)
    scalar_val = None if x_val is None else mulphe._source_predictions(encoders, x_val, device)
    scalar_predict = mulphe._source_predictions(encoders, x_predict, device)
    scaler = mulphe._TargetScaler('none', y_train)
    y_train_s = scaler.transform(y_train)
    y_val_s = None if y_val is None else scaler.transform(y_val)
    mulphe._seed(seed)
    fusion = mulphe.MetaTraitFusion(num_traits, int(config['fusion_hidden'])).to(device)
    optimizer = torch.optim.Adam(fusion.parameters(), lr=float(config['lr']))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-5)
    best = {'loss': np.inf, 'epoch': 0, 'fusion': None}
    history = []
    epochs = int(fixed_epoch) if fixed_epoch is not None else int(config['fusion_epochs'])

    def _forward(scalars):
        tensor = torch.from_numpy(np.asarray(scalars, dtype=np.float32)).to(device)
        return fusion(tensor)[0]

    for epoch in range(1, epochs + 1):
        fusion.train()
        total = 0.0
        for batch_x, batch_y in mulphe._loader(scalar_train, y_train_s,
                                               config['batch_size'], True, seed):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            prediction = fusion(batch_x)[0]
            loss = F.mse_loss(prediction, batch_y)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite scalar-fusion training loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(batch_x)
        fusion.eval()
        if y_val_s is None:
            history.append({'epoch': epoch, 'train_loss': total / len(scalar_train)})
            continue
        with torch.no_grad():
            val_values = []
            for batch_x, batch_y in mulphe._loader(scalar_val, y_val_s,
                                                   config['batch_size'], False, seed):
                prediction = fusion(batch_x.to(device))[0]
                val_values.append(float(F.mse_loss(prediction, batch_y.to(device))))
            val_loss = float(np.mean(val_values))
        scheduler.step(val_loss)
        history.append({'epoch': epoch, 'train_loss': total / len(scalar_train),
                        'val_loss': val_loss})
        if val_loss < best['loss']:
            best = {'loss': val_loss, 'epoch': epoch,
                    'fusion': {k: v.detach().cpu().clone() for k, v in fusion.state_dict().items()}}
    if y_val_s is None:
        best = {'loss': None, 'epoch': epochs,
                'fusion': {k: v.detach().cpu().clone() for k, v in fusion.state_dict().items()}}
    fusion.load_state_dict(best['fusion'])
    fusion.eval()
    with torch.no_grad():
        val_prediction = None if scalar_val is None else scaler.inverse(
            _forward(scalar_val).cpu().numpy())
        prediction = scaler.inverse(_forward(scalar_predict).cpu().numpy())
    return {'arm': 'scalar_with_target', 'prediction': prediction,
            'val_prediction': val_prediction,
            'val_metrics': None if y_val is None else mulphe._metrics(y_val, val_prediction),
            'best_epoch': best['epoch'], 'epochs_run': len(history), 'history': history,
            'scaling': scaler.info(), 'fusion_input': 'source_predictions_with_target_mse',
            'num_traits': num_traits,
            'state_sha256': mulphe._state_sha256(
                *[entry['model'].state_dict() for entry in encoders], best['fusion']),
            'parameters': mulphe._parameter_count(fusion),
            'total_parameters': mulphe._parameter_count(
                *[entry['model'] for entry in encoders], fusion),
            'train_seconds': time.perf_counter() - started,
            'amp': False, 'device': str(device), 'seed': int(seed), 'dtype': 'float32',
            'mode': 'fixed_full_dev' if y_val is None else 'validation'}


def fit_candidate(x_train, y_train, x_val, y_val, x_predict, *, target_name,
                  source_names, source_train, source_val, config, seed, device,
                  fixed_epochs=None):
    """Fit the complete candidate for one fold/target without test phenotypes.

    ``x_predict`` are the rows to output predictions for: validation rows in the
    development pass, test rows in the fixed full-dev refit. Stages: source and
    target independent pretrain, target adaptation (fine-tuned from the pretrained
    target), then fusion. ``fixed_epochs`` (refit only) is
    ``{'pretrain': {name: epoch}, 'finetune': epoch, 'fusion': epoch}`` where the
    pretrain dict covers every source name and the target pretrain name.
    """
    config = {**mulphe.DEFAULT_CONFIG, **(config or {}), 'target_name': str(target_name)}
    source_names = [str(name) for name in source_names]
    if str(target_name) in source_names:
        raise ValueError('Target environment cannot be a source')
    val_mode = x_val is not None
    if val_mode != (y_val is not None):
        raise ValueError('x_val and y_val must both be present or both absent')
    if not val_mode and fixed_epochs is None:
        raise ValueError('Fixed full-dev refit requires fixed_epochs')
    if val_mode and fixed_epochs is not None:
        raise ValueError('fixed epochs are only valid in no-validation refit mode')
    x_train = np.asarray(x_train, dtype=np.float64)
    x_predict = np.asarray(x_predict, dtype=np.float64)
    medians = mulphe._fit_impute(x_train)
    x_train = mulphe._apply_impute(x_train, medians)
    x_predict = mulphe._apply_impute(x_predict, medians)
    x_val = None if not val_mode else mulphe._apply_impute(np.asarray(x_val, dtype=np.float64), medians)
    y_train_arr = np.asarray(y_train, dtype=np.float64)
    train_median = float(np.median(y_train_arr[np.isfinite(y_train_arr)]))
    y_train = mulphe._target_fill(y_train, train_median)
    y_val = None if not val_mode else mulphe._target_fill(y_val, train_median)
    source_train = np.asarray(source_train, dtype=np.float64)
    source_val = None if not val_mode else np.asarray(source_val, dtype=np.float64)
    target_pretrain_name = f'{target_name}__pretrain'

    source_started = time.perf_counter()
    sources = mulphe.train_source_encoders(
        x_train, source_train, x_val, source_val, source_names, config,
        int(seed) + 1000, device,
        fixed_epochs=None if val_mode else fixed_epochs['pretrain'])
    target_pretrain_epoch = None if val_mode else int(fixed_epochs['pretrain'][target_pretrain_name])
    pre_model, pre_head, pre_best, pre_history = mulphe._train_predictor(
        x_train, y_train, x_val, y_val, config, int(seed), device,
        'linear', 'pretrain_epochs', 'mse_pearson', 'combined',
        fixed_epochs=target_pretrain_epoch)
    source_seconds = time.perf_counter() - source_started
    adapt_model, adapt_head, adapt_best, adapt_history = _adapt_target(
        x_train, y_train, x_val, y_val, config, int(seed), device,
        pre_model.state_dict(), pre_head.state_dict(),
        fixed_epochs=None if val_mode else int(fixed_epochs['finetune']))
    target_entry = {'name': f'{target_name}__target', 'model': adapt_model,
                    'head': adapt_head, 'best_epoch': adapt_best['epoch'],
                    'history': adapt_history}
    encoders = sources + [target_entry]
    encoder_names = [entry['name'] for entry in encoders]
    fusion_fixed = None if val_mode else int(fixed_epochs['fusion'])
    hidden = mulphe.run_arm(
        'embedding', x_train, y_train, x_val, y_val, x_predict, config, int(seed),
        device, source_models=encoders, source_names=encoder_names,
        fixed_epoch=fusion_fixed)
    hidden['arm'] = 'hidden_with_target'
    hidden['fusion_input'] = 'embedding_with_target'
    scalar = _scalar_fusion_arm(
        encoders, x_train, y_train, x_val, y_val, x_predict, config, int(seed),
        device, fixed_epoch=fusion_fixed)
    arms = {'scalar_with_target': scalar, 'hidden_with_target': hidden}
    pretrain_epochs = {entry['name']: int(entry['best_epoch']) for entry in sources}
    pretrain_epochs[target_pretrain_name] = int(pre_best['epoch'])
    stage_epochs = {'pretrain': pretrain_epochs, 'finetune': int(adapt_best['epoch']),
                    'fusion': {arm: int(arms[arm]['best_epoch']) for arm in ARMS}}
    return {'arms': arms, 'stage_epochs': stage_epochs,
            'mode': 'validation' if val_mode else 'fixed_full_dev',
            'source_names': source_names, 'target_name': str(target_name),
            'encoder_names': encoder_names, 'seed': int(seed),
            'source_pretraining_seconds': source_seconds, 'impute_scope': 'train',
            'official': {'repo': OFFICIAL_REPO, 'commit': OFFICIAL_COMMIT, 'license': 'MIT'}}


# -- runner helpers --------------------------------------------------------
def _read(path):
    return json.loads(Path(path).read_text())


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def _write_csv(path, fields, rows):
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def _git(*arguments):
    return subprocess.check_output(['git', *arguments], text=True).strip()


def _macro(truth, prediction):
    values = {name: float(function(truth, prediction))
              for name, function in zip(METRICS, METRIC_FUNCTIONS)}
    if not all(np.isfinite(value) for value in values.values()):
        raise FloatingPointError(f'Non-finite score: {values}')
    return values


def _mean(entries):
    return {name: float(np.mean([entry[name] for entry in entries])) for name in METRICS}


def _check_split(fold, split):
    tr = np.asarray(split['train_idx'], dtype=np.int64)
    val = np.asarray(split['val_idx'], dtype=np.int64)
    dev = set(int(i) for i in fold['dev_idx'])
    test = set(int(i) for i in fold['test_idx'])
    if set(tr.tolist()) & set(val.tolist()):
        raise ValueError('Inner train and validation overlap')
    if set(tr.tolist()) | set(val.tolist()) != dev:
        raise ValueError('Inner train plus validation must equal the development set')
    if (set(tr.tolist()) | set(val.tolist())) & test:
        raise ValueError('Inner rows overlap the outer test set')
    return tr, val


def _median_stage_epochs(inner_epochs, arm):
    """Per-stage median of the three inner best epochs for one seed/config/arm."""
    pretrain = {name: int(np.median([e['pretrain'][name] for e in inner_epochs]))
                for name in inner_epochs[0]['pretrain']}
    finetune = int(np.median([e['finetune'] for e in inner_epochs]))
    fusion = int(np.median([e['fusion'][arm] for e in inner_epochs]))
    return {'pretrain': pretrain, 'finetune': finetune, 'fusion': fusion}


def _fit_calibration(pred, truth):
    """Public validation linear calibration; identity fallback if degenerate."""
    pred = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if not np.isfinite(pred).all() or not np.isfinite(truth).all():
        raise ValueError('Nonfinite calibration input')
    if np.ptp(pred) <= np.finfo(np.float64).eps * max(1., np.max(np.abs(pred))):
        return 1.0, 0.0
    try:
        a, b = np.polyfit(pred, truth, 1)
        if np.isfinite(a) and np.isfinite(b):
            return float(a), float(b)
    except (np.linalg.LinAlgError, ValueError):
        pass
    return 1.0, 0.0


# -- unified ten-fold protocol ---------------------------------------------
def run(cfg, args, runner):
    if args.selection is not None:
        raise ValueError('Mul-PheG2P selects on inner validation; there is no selection input')
    seeds = [int(seed) for seed in cfg['seeds']]
    configurations = list(cfg['configurations'])
    author_index = int(cfg['author_default_index'])
    torch.set_num_threads(int(cfg['cpu_threads']))
    device = torch.device(cfg['device'])
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(float(cfg['gpu_memory_fraction']))
    data = load(Path(cfg['data']), cfg['data_sha256'])
    folds = _read(cfg['splits'])['folds']
    environments = [str(name) for name in data.environments]
    if len(folds) != 10 or any(len(f['inner']) != 3 for f in folds):
        raise ValueError('Expected 10 outer folds with 3 inner splits each')
    subset = getattr(args, 'outer_folds', None)
    if subset is not None:
        if len(subset) != len(set(subset)) or not set(subset).issubset({f['fold'] for f in folds}):
            raise ValueError('Invalid outer fold subset')
        folds = [f for f in folds if f['fold'] in subset]
    covered = np.sort(np.concatenate([np.asarray(f['test_idx'], dtype=int) for f in folds]))
    if len(configurations) != 8 or not 0 <= author_index < len(configurations):
        raise ValueError('Expected 8 configurations and a valid author_default_index')
    if len(set(seeds)) != len(seeds) or not seeds:
        raise ValueError('Seeds must be a non-empty distinct list')
    model_config = cfg.get('model_config') or {}
    x_all = np.asarray(data.X, dtype=np.float64)
    truth = np.asarray(data.Y, dtype=np.float64)
    args.out.mkdir(parents=True, exist_ok=False)

    sources = [Path(cfg['splits']), Path(cfg['data']), Path(__file__),
               Path(mulphe.__file__), Path(runner), Path(args.config)]
    _write_json(args.out / 'provenance.json', {
        'config': cfg, 'git_commit': _git('rev-parse', 'HEAD'),
        'git_dirty': _git('status', '--porcelain') != '',
        'torch_version': str(torch.__version__),
        'sha256': {str(path): sha256_of(path) for path in sources},
        'environment_order': environments,
        'official': {'repo': OFFICIAL_REPO, 'commit': OFFICIAL_COMMIT, 'license': 'MIT'},
        'model_arms': list(ARMS), 'output_systems': list(SYSTEMS), 'stages': list(STAGES),
        'author_default_index': author_index,
        'recipe': {'public_default_caps': {'pretrain': 300, 'finetune': 300, 'fusion': 200},
                   'paper_reported_caps': {'pretrain': 500, 'finetune': 300, 'fusion': 200},
                   'public_default_lr': 0.001, 'public_default_batch': 32,
                   'public_default_hidden': 128,
                   'scalar_space': 'trained phenotype prediction per encoder, no additional standardisation',
                   'scalar_loss': 'MSE only (paper); public extra head / +0.2 Pearson / grad clip not used',
                   'target_encoder': 'independent pretrain then adaptation, included as a fusion source',
                   'hidden_calibration': 'linear (a,b) fit on the 3 inner OOF val preds/labels, frozen before test'},
        'deviations': ['train-only scaling and imputation',
                       'no test phenotype enters any fit',
                       'source environments exclude the target; target encoder added as a source',
                       'scalar fusion uses source and adapted-target prediction heads without additional standardisation',
                       'scalar fusion is MSE-only; public fine_tune_predict extra head, +0.2 Pearson and grad clip omitted',
                       'target adaptation fine-tunes a pretrained target encoder; public fine_tune.train instead adapts a mean-source-feature head',
                       'hidden calibration coefficients come from inner OOF validation, not a refit validation split'],
        'scope': 'inner-validation selection; full-dev refit at per-stage median inner epochs; '
                 'test labels read once after every fit; no test-side branch/config/calibration choice'})

    expected_dev = len(folds) * 3 * len(environments) * len(configurations) * len(seeds)
    state = {'status': 'running', 'phase': 'development', 'completed': 0, 'expected': expected_dev}
    records, failures = [], []
    started = time.perf_counter()
    epochs = {int(f['fold']): {e: {c: {s: [] for s in seeds} for c in range(len(configurations))}
                               for e in range(len(environments))} for f in folds}
    try:
        for fold in folds:
            number = int(fold['fold'])
            for inner, split in enumerate(fold['inner']):
                tr, val = _check_split(fold, split)
                predictions = {}
                for environment in range(len(environments)):
                    sources_pos = [p for p in range(len(environments)) if p != environment]
                    for c, configuration in enumerate(configurations):
                        merged = {**model_config, **configuration}
                        for seed in seeds:
                            result = fit_candidate(
                                x_all[tr], truth[tr, environment],
                                x_all[val], truth[val, environment], x_all[val],
                                target_name=environments[environment],
                                source_names=[environments[p] for p in sources_pos],
                                source_train=truth[np.ix_(tr, sources_pos)],
                                source_val=truth[np.ix_(val, sources_pos)],
                                config=merged, seed=seed, device=device)
                            epochs[number][environment][c][seed].append(result['stage_epochs'])
                            for arm in ARMS:
                                entry = result['arms'][arm]
                                value = np.asarray(entry['val_prediction'], dtype=np.float64)
                                if value.shape != (len(val),) or not np.isfinite(value).all():
                                    raise FloatingPointError('Non-finite validation prediction')
                                predictions[f'e{environment}_c{c}_s{seed}_{arm}'] = value
                                records.append({
                                    'fold': number, 'inner': inner, 'environment': environment,
                                    'config_index': c, 'seed': seed, 'arm': arm,
                                    'best_epoch': entry['best_epoch'],
                                    'stage_epochs': result['stage_epochs'],
                                    'val_metrics': entry['val_metrics'],
                                    'train_seconds': entry['train_seconds']})
                            state['completed'] += 1
                            _write_json(args.out / 'execution_status.json', state)
                np.savez_compressed(args.out / f'development_{number}_{inner}.npz', **predictions)
        if state['completed'] != expected_dev:
            raise ValueError('Incomplete development pass')

        # hidden-fusion calibration from the three inner OOF validation pairs
        calibration = {}
        for fold in folds:
            number = int(fold['fold'])
            arrays = [np.load(args.out / f'development_{number}_{j}.npz') for j in range(3)]
            for environment in range(len(environments)):
                for c in range(len(configurations)):
                    for seed in seeds:
                        preds, labels = [], []
                        for j, split in enumerate(fold['inner']):
                            val = np.asarray(split['val_idx'], dtype=np.int64)
                            preds.append(arrays[j][f'e{environment}_c{c}_s{seed}_hidden_with_target'])
                            labels.append(truth[val, environment])
                        a, b = _fit_calibration(np.concatenate(preds), np.concatenate(labels))
                        calibration[f'{number}|{environment}|{c}|{seed}'] = {'a': a, 'b': b}
        _write_json(args.out / 'calibration.json', calibration)

        # selection on inner validation only (seed-averaged predictions)
        selection = []
        for fold in folds:
            number = int(fold['fold'])
            arrays = [np.load(args.out / f'development_{number}_{j}.npz') for j in range(3)]
            per_env = {}
            for environment in range(len(environments)):
                per_arm = {}
                for arm in ARMS:
                    scores = []
                    for c in range(len(configurations)):
                        inner_scores = []
                        for j, split in enumerate(fold['inner']):
                            val = np.asarray(split['val_idx'], dtype=np.int64)
                            seed_mean = np.mean([arrays[j][f'e{environment}_c{c}_s{seed}_{arm}']
                                                 for seed in seeds], axis=0)
                            inner_scores.append(float(pcc(truth[val, environment], seed_mean)))
                        scores.append(float(np.mean(inner_scores)))
                    per_arm[arm] = int(np.argmax(scores))
                per_env[environment] = per_arm
            selection.append({'fold': number, 'selected': per_env})
        _write_json(args.out / 'selection.json', selection)

        # full-dev refit at per-stage median inner epochs; test scored once
        state['phase'] = 'refit'
        test_rows, refit_records = [], []
        variants = ('selected', 'author_default')
        for fold in folds:
            number = int(fold['fold'])
            dev = np.asarray(fold['dev_idx'], dtype=np.int64)
            test = np.asarray(fold['test_idx'], dtype=np.int64)
            chosen_env = next(s['selected'] for s in selection if s['fold'] == number)
            for environment in range(len(environments)):
                sources_pos = [p for p in range(len(environments)) if p != environment]
                source_names = [environments[p] for p in sources_pos]
                for arm in ARMS:
                    config_indices = {'selected': chosen_env[environment][arm],
                                      'author_default': author_index}
                    for variant in variants:
                        c = config_indices[variant]
                        merged = {**model_config, **configurations[c]}
                        for seed in seeds:
                            inner_epochs = epochs[number][environment][c][seed]
                            if len(inner_epochs) != 3:
                                raise ValueError('Expected three inner stage-epoch records')
                            fixed = _median_stage_epochs(inner_epochs, arm)
                            result = fit_candidate(
                                x_all[dev], truth[dev, environment], None, None, x_all[test],
                                target_name=environments[environment],
                                source_names=source_names,
                                source_train=truth[np.ix_(dev, sources_pos)],
                                source_val=None, config=merged, seed=seed, device=device,
                                fixed_epochs=fixed)
                            entry = result['arms'][arm]
                            raw = np.asarray(entry['prediction'], dtype=np.float64)
                            if raw.shape != (len(test),) or not np.isfinite(raw).all():
                                raise FloatingPointError('Non-finite test prediction')
                            coef = calibration[f'{number}|{environment}|{c}|{seed}']
                            if arm == 'scalar_with_target':
                                outputs = {'scalar_with_target': raw}
                            else:
                                outputs = {'hidden_with_target_uncalibrated': raw,
                                           'hidden_with_target': coef['a'] * raw + coef['b']}
                            for system, prediction in outputs.items():
                                if not np.isfinite(prediction).all():
                                    raise FloatingPointError('Non-finite calibrated prediction')
                                test_rows.extend([
                                    {'variant': variant, 'arm': system, 'fold': number,
                                     'environment': environments[environment], 'config_index': c,
                                     'seed': seed, 'material_index': int(material),
                                     'y_pred': f'{value:.10g}'}
                                    for material, value in zip(test, prediction)])
                            refit_records.append({
                                'variant': variant, 'arm': arm, 'fold': number,
                                'environment': environment, 'config_index': c, 'seed': seed,
                                'fixed_epochs': fixed, 'calibration': coef,
                                'state_sha256': entry['state_sha256'],
                                'parameters': entry['parameters'],
                                'total_parameters': entry['total_parameters'],
                                'train_seconds': entry['train_seconds']})
            _write_json(args.out / 'execution_status.json', state)
        _write_csv(args.out / 'predictions.csv',
                   ['variant', 'arm', 'fold', 'environment', 'config_index', 'seed',
                    'material_index', 'y_pred'],
                   sorted(test_rows, key=lambda row: (row['variant'], row['arm'], row['fold'],
                                                      row['environment'], row['seed'],
                                                      row['material_index'])))
        _write_json(args.out / 'refit.json', refit_records)

        # scoring: fold means and pooled OOF kept separate
        grouped = {}
        for row in test_rows:
            key = (row['variant'], row['arm'], int(row['fold']), row['environment'], int(row['seed']))
            grouped.setdefault(key, {})[int(row['material_index'])] = float(row['y_pred'])
        summary = {}
        for variant in variants:
            for system in SYSTEMS:
                grid, oof = [], np.full((len(environments), len(seeds), data.n), np.nan)
                for e_index, environment in enumerate(environments):
                    for fold in folds:
                        number = int(fold['fold'])
                        test = np.asarray(fold['test_idx'], dtype=np.int64)
                        for s_index, seed in enumerate(seeds):
                            values = grouped[(variant, system, number, environment, seed)]
                            if set(values) != set(int(m) for m in test):
                                raise ValueError('Test predictions differ from the frozen test set')
                            prediction = np.array([values[int(m)] for m in test], dtype=np.float64)
                            scores = _macro(truth[test, e_index], prediction)
                            grid.append({'variant': variant, 'arm': system, 'fold': number,
                                         'environment': environment, 'seed': seed,
                                         'materials': len(test), **scores})
                            oof[e_index, s_index, test] = prediction
                if not np.isfinite(oof[:,:,covered]).all():
                    raise ValueError('Selected outer folds leave materials uncovered')
                seed_mean_environment = {
                    environment: _mean([g for g in grid if g['environment'] == environment])
                    for environment in environments}
                ensemble_grid = []
                for e_index, environment in enumerate(environments):
                    for fold in folds:
                        test = np.asarray(fold['test_idx'], dtype=np.int64)
                        scores = _macro(truth[test, e_index], oof[e_index][:,test].mean(axis=0))
                        ensemble_grid.append({'fold':int(fold['fold']), 'environment':environment, **scores})
                per_environment = {e: _mean([g for g in ensemble_grid if g['environment']==e])
                                   for e in environments}
                pooled_environment = {
                    environment: _mean([_macro(truth[covered, e_index], oof[e_index, s_index, covered])
                                        for s_index in range(len(seeds))])
                    for e_index, environment in enumerate(environments)}
                summary[f'{variant}/{system}'] = {
                    'per_fold_macro': _mean([per_environment[e] for e in environments]),
                    'per_environment': per_environment,
                    'seed_metric_mean': _mean([seed_mean_environment[e] for e in environments]),
                    'ensemble_grid': ensemble_grid,
                    'pooled_oof_macro': _mean([pooled_environment[e] for e in environments]),
                    'pooled_oof_per_environment': pooled_environment,
                    'seed_ensemble_pooled_macro': _mean([
                        _macro(truth[covered, e_index], oof[e_index][:,covered].mean(axis=0))
                        for e_index in range(len(environments))]),
                    'grid': grid}
        _write_csv(args.out / 'per_fold.csv',
                   ['variant', 'arm', 'fold', 'environment', 'seed', 'materials', 'pcc', 'rmse', 'mae'],
                   [{key: (f'{row[key]:.12g}' if key in METRICS else row[key])
                     for key in ('variant', 'arm', 'fold', 'environment', 'seed',
                                 'materials', 'pcc', 'rmse', 'mae')}
                    for variant in variants for system in SYSTEMS
                    for row in summary[f'{variant}/{system}']['grid']])
        _write_json(args.out / 'summary.json', {
            'systems': {name: {k: value for k, value in entry.items() if k != 'grid'}
                        for name, entry in summary.items()},
            'metric_order': list(METRICS), 'model_arms': list(ARMS),
            'output_systems': list(SYSTEMS), 'variants': list(variants),
            'folds': [int(f['fold']) for f in folds], 'seeds': seeds,
            'author_default_index': author_index,
            'note': 'per_fold_macro averages seed predictions before scoring environment-folds; seed_metric_mean averages per-seed scores; pooled_oof_macro '
                    'evaluates one OOF vector per environment and is a different reading; '
                    'seed_ensemble averages seed predictions before scoring; hidden_with_target is '
                    'inner-OOF-calibrated, hidden_with_target_uncalibrated is the raw fusion output'})
        _write_json(args.out / 'trajectories.json', records)
        _write_json(args.out / 'timing.json', {
            'total_seconds': round(time.perf_counter() - started, 3),
            'device': str(device), 'amp': False, 'dtype': 'float32',
            'development_fits': expected_dev, 'refit_fits': len(refit_records),
            'cuda': None if device.type != 'cuda' else {
                'name': torch.cuda.get_device_name(device),
                'peak_allocated_megabytes': float(torch.cuda.max_memory_allocated(device)) / 2 ** 20,
                'peak_reserved_megabytes': float(torch.cuda.max_memory_reserved(device)) / 2 ** 20}})
        _write_json(args.out / 'failures.json', [])
        state.update(status='complete', phase='summary_written')
        _write_json(args.out / 'execution_status.json', state)
    except BaseException as error:
        state['status'] = 'failed'
        failures.append({'completed': state['completed'], 'phase': state['phase'],
                         'type': type(error).__name__, 'message': str(error),
                         'at': time.strftime('%Y-%m-%dT%H:%M:%S%z')})
        _write_json(args.out / 'failures.json', failures)
        _write_json(args.out / 'execution_status.json', state)
        raise
    finally:
        artifacts = {path.name: {'bytes': path.stat().st_size, 'sha256': sha256_of(path)}
                     for path in sorted(args.out.iterdir())
                     if not path.is_dir() and path.name != 'artifact_manifest.json'}
        _write_json(args.out / 'artifact_manifest.json', {
            'files': artifacts, 'weights_stored': False,
            'note': 'state hashes only; no checkpoint files are written'})
    return summary