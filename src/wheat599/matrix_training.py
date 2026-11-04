"""Matched synthetic training; deliberately contains no real-data evaluation."""

import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

import torch
from torch.nn import functional as F

from wheat599.matrix_inference import MatrixInference
from wheat599.matrix_tasks import sample_task
from wheat599.posterior_direct import DirectPredictor


def build_model(config, variant, seed):
    torch.manual_seed(seed)
    network = config['network']
    if variant == 'matrix':
        return MatrixInference(config['tasks']['environments'], **network)
    if variant == 'direct':
        return DirectPredictor(markers=max(config['tasks']['marker_counts']),
                               environments=config['tasks']['environments'],
                               width=network['width'], heads=4,
                               hidden=network['psi_hidden'], layers=network['layers'])
    raise ValueError(f'Unknown inference variant: {variant}')


def predict(model, data, variant):
    if variant == 'matrix':
        return model(data.x, data.y, data.query)
    padding = model.markers - data.x.shape[1]
    if padding < 0:
        raise ValueError('Direct context control cannot exceed its marker capacity')
    # This existing control has three prior-parameter inputs. They are fixed here,
    # never filled with hidden generating parameters. Covariance is context-only.
    covariance = data.y.T @ data.y / len(data.y)
    return model(F.pad(data.x, (0, padding)), data.y,
                 F.pad(data.query, (0, padding)), covariance, 1., 1., 1.)


def task_panel(config, families, seed, count, *, design=None):
    return [sample_task(config, seed + 10000 * index + j, family=family, design=design)
            for index, family in enumerate(families) for j in range(count)]


@torch.no_grad()
def evaluate(model, panel, variant, device):
    model.eval()
    records = []
    for task in panel:
        data, target = task.prepare()
        data.x, data.y, data.query = (x.to(device) for x in (data.x, data.y, data.query))
        prediction = predict(model, data, variant)
        loss = float((prediction - target.to(device)).square().mean())
        if not torch.isfinite(prediction).all() or not math.isfinite(loss):
            raise FloatingPointError('Nonfinite synthetic evaluation')
        records.append({**task.metadata, 'mse': loss,
                        'context_mean_mse': float(target.square().mean())})
    if not records:
        raise ValueError('Empty evaluation panel')
    return sum(r['mse'] for r in records) / len(records), records


def train_one(config, variant, seed, validation, device):
    model = build_model(config, variant, seed).to(device)
    training = config['training']
    optimizer = torch.optim.AdamW(model.parameters(), lr=training['learning_rate'],
                                  weight_decay=training['weight_decay'])
    history, best = [], None
    start = time.perf_counter()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    for step in range(training['steps'] + 1):
        if step:
            # Identical task IDs for both architectures at a given model seed.
            task = sample_task(config['tasks'], training['task_seed'] + seed * 100000 + step,
                               device=device)
            data, target = task.prepare()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = (predict(model, data, variant) - target).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite training loss at step {step}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), training['gradient_clip'],
                                           error_if_nonfinite=True)
            optimizer.step()
        if step % training['evaluate_every'] == 0 or step == training['steps']:
            score, records = evaluate(model, validation, variant, device)
            history.append({'step': step, 'training_mse': float(loss.detach()) if step else None,
                            'validation_mse': score})
            if best is None or score < best['validation_mse']:
                best = {'step': step, 'validation_mse': score, 'validation_tasks': records,
                        'state': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    model.load_state_dict(best.pop('state'))
    return model, {'seed': seed, 'variant': variant, 'parameters': sum(p.numel() for p in model.parameters()),
                   'seconds': time.perf_counter() - start,
                   'peak_megabytes': (torch.cuda.max_memory_allocated(device) / 1024**2
                                      if device.type == 'cuda' else None),
                   'history': history, **best}


def run(config, args, runner):
    if args.selection is not None:
        raise ValueError('Synthetic preparation has no outer-test or selection-input path')
    if config['device'] == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Requested CUDA is unavailable; choose CPU explicitly')
    device = torch.device(config['device'])
    torch.set_num_threads(config.get('cpu_threads', 2))
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(config['gpu_memory_fraction'])
    args.out.mkdir(parents=True, exist_ok=False)
    here = Path(__file__).parent
    paths = [here / name for name in ('matrix_training.py', 'matrix_tasks.py',
                                      'matrix_inference.py', 'posterior_direct.py')]
    paths.append(Path(runner))
    provenance = {'config': config,
                  'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                  'torch_version': torch.__version__, 'device': str(device),
                  'scope': 'Synthetic tasks only; no Wheat599 genotype or phenotype read',
                  'selection': 'Lowest synthetic validation MSE; earliest checkpoint wins ties; '
                               'held-out admixture panel scored once after selection'}
    (args.out / 'provenance.json').write_text(json.dumps(provenance, indent=2))
    training = config['training']
    validation = task_panel(config['tasks'], config['tasks']['families'],
                            training['validation_seed'], training['tasks_per_family'])
    held_out = task_panel(config['tasks'], config['held_out']['families'],
                          config['held_out']['seed'], config['held_out']['tasks_per_family'],
                          design='admixture')
    results, failures = [], []
    for variant in config['variants']:
        for seed in config['seeds']:
            try:
                model, record = train_one(config, variant, seed, validation, device)
                _, record['held_out_tasks'] = evaluate(model, held_out, variant, device)
                path = args.out / f'{variant}_{seed}.pt'
                torch.save(model.cpu().state_dict(), path)
                record['checkpoint_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
                results.append(record)
                print(json.dumps({key: record[key] for key in
                                  ('variant', 'seed', 'step', 'validation_mse', 'seconds')}), flush=True)
                del model
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            except Exception as exc:
                failures.append({'variant': variant, 'seed': seed, 'reason': repr(exc)})
                raise
            finally:
                (args.out / 'training.json').write_text(json.dumps(results, indent=2))
                (args.out / 'failures.json').write_text(json.dumps(failures, indent=2))
    (args.out / 'selection.json').write_text(json.dumps(
        {'config': config, 'selected': [{key: r[key] for key in
          ('variant', 'seed', 'step', 'validation_mse', 'checkpoint_sha256')} for r in results]}, indent=2))
