"""One bounded development comparison of original and orthogonally rotated targets.

The original 1920 independent trajectories are reused at their saved checkpoints.
Four direction predictors retain the same training code, seeds, configurations and
early stopping rule. Configuration selection and all reported scores use the four
original environments after inverse transformation; outer test rows are never read.
"""
import csv
import json
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from wheat599.data import load, sha256_of
from wheat599.environment_run import Trajectory, train_trajectory
from wheat599.environment_selection import metric_matrix, read_predictions, select_indices
from wheat599.genomic_tabm_run import model_source_sha, verify_versions
from wheat599.retrieval_run import check_split

METRICS = ('pcc', 'rmse', 'mae')


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def control_bank(config):
    source = Path(config['original_source'])
    provenance = json.loads((source / 'provenance.json').read_text())
    status = json.loads((source / 'execution_status.json').read_text())
    if status['status'] != 'complete' or json.loads((source / 'failures.json').read_text()):
        raise ValueError('Original development source is incomplete or failed')
    if provenance['data_sha256'] != config['data_sha256'] or \
            provenance['splits_sha256'] != sha256_of(Path(config['splits'])):
        raise ValueError('Original source data/splits differ')
    for key in ('seeds', 'configurations', 'training', 'dtype'):
        if provenance['config'][key] != config[key]:
            raise ValueError(f'Original training protocol differs: {key}')
    for name, expected in config['original_artifact_sha256'].items():
        if sha256_of(source / name) != expected:
            raise ValueError(f'Original source hash differs: {name}')
    for name in ('environment_run.py', 'environment_tabm.py', 'genomic_tabm.py'):
        pinned = [value for key, value in provenance['source_sha256'].items()
                  if Path(key).name == name]
        if pinned != [sha256_of(Path(__file__).with_name(name))]:
            raise ValueError(f'Original training/model source changed: {name}')
    return read_predictions(source / 'predictions.csv', 'independent',
                            ['1', '2', '4', '5'])


def original_units(bank, data, folds, config):
    """Dense config x seed x validation-material x environment units, no test access."""
    units, expected = {}, set()
    for fold in folds:
        for inner in range(len(fold['inner'])):
            _, ids = check_split(fold, inner)
            cube = []
            for c in range(len(config['configurations'])):
                members = []
                for seed in config['seeds']:
                    keys = [[(int(fold['fold']), inner, c, int(seed), int(i), e)
                             for e in range(data.n_env)] for i in ids]
                    expected.update(key for row in keys for key in row)
                    members.append([[bank[key] for key in row] for row in keys])
                cube.append(members)
            units[int(fold['fold']), inner] = np.asarray(cube, dtype=np.float64)
    if set(bank) != expected:
        raise ValueError('Original prediction key set is incomplete or duplicated')
    return units


def select_and_score(units, data, folds, policy, tolerance):
    """Same saved-checkpoint selection freedom for common, strong original separate."""
    rows, selections = [], []
    for fold in folds:
        f = int(fold['fold'])
        truth = [data.Y[np.asarray(split['val_idx'])] for split in fold['inner']]
        scores = np.asarray([[metric_matrix(p.mean(0), truth[j])[:, 0]
                              for p in units[f, j]] for j in range(len(truth))])
        common, separate = select_indices(scores, tolerance)
        indices = common if policy == 'common' else separate
        ensemble, per_seed = [], []
        for j in range(len(truth)):
            cube = units[f, j]
            prediction = np.column_stack([cube[c, :, :, e].mean(0)
                                          for e, c in enumerate(indices)])
            ensemble.append(metric_matrix(prediction, truth[j]))
            per_seed.append(np.mean([metric_matrix(np.column_stack(
                [cube[c, si, :, e] for e, c in enumerate(indices)]), truth[j])
                for si in range(cube.shape[1])], axis=0))
        matrix = np.mean(ensemble, axis=0)
        rows.append({'fold': f, 'config_indices': indices,
                     'ensemble': dict(zip(METRICS, matrix.mean(0).tolist())),
                     'seed_mean': dict(zip(METRICS, np.mean(per_seed, axis=0).mean(0).tolist())),
                     'per_environment': matrix.tolist()})
        selections.append({'fold': f, 'config_indices': indices,
                           'scores_config_environment': scores.mean(0).tolist()})
    return {
        'ensemble': {m: float(np.mean([r['ensemble'][m] for r in rows])) for m in METRICS},
        'seed_mean': {m: float(np.mean([r['seed_mean'][m] for r in rows])) for m in METRICS},
        'per_environment': np.mean([r['per_environment'] for r in rows], axis=0).tolist(),
        'per_fold': rows, 'selection': selections,
    }


def compare(candidate, reference):
    ref = {r['fold']: r for r in reference['per_fold']}
    paired = [{'fold': row['fold'], **{
        m: row['ensemble'][m] - ref[row['fold']]['ensemble'][m] for m in METRICS}}
        for row in candidate['per_fold']]
    return {'mean_difference': {m: float(np.mean([r[m] for r in paired])) for m in METRICS},
            'pcc_wins': sum(row['pcc'] > 0 for row in paired), 'paired': paired}


def run(config, args, runner):
    from wheat599.phenotype_rotation import PhenotypeRotation
    if args.selection is not None:
        raise ValueError('Rotation budget is development only; no external selections')
    verify_versions(config)
    torch.set_num_threads(int(config['cpu_threads']))
    device = torch.device(config['device'])
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(float(config['gpu_memory_fraction']))
    data = load(Path(config['data']), config['data_sha256'])
    if data.n_env != 4:
        raise ValueError('This fixed rotation experiment requires four environments')
    folds = json.loads(Path(config['splits']).read_text())['folds']
    original = original_units(control_bank(config), data, folds, config)
    references = {name: select_and_score(original, data, folds, policy,
                                        config['training']['tie_tolerance'])
                  for name, policy in [('original_common', 'common'),
                                       ('original_strong', 'per_environment')]}
    for name, expected in config['expected_control_pcc'].items():
        if abs(references[name]['ensemble']['pcc'] - expected) > 1e-8:
            raise ValueError(f'Original source corner did not reproduce: {name}')
    expected = len(folds) * 3 * len(config['seeds']) * len(config['configurations']) * 4
    if expected != int(config['expected_fits']):
        raise ValueError('Configuration does not match frozen fit budget')
    args.out.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), Path(runner), Path(args.config)] + [
        Path(__file__).with_name(n) for n in ('phenotype_rotation.py', 'environment_run.py',
                                             'environment_tabm.py', 'genomic_tabm.py',
                                             'environment_selection.py', 'data.py', 'metrics.py')]
    dump(args.out / 'provenance.json', {
        'config': config, 'git_commit': subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'data_sha256': config['data_sha256'],
        'splits_sha256': sha256_of(Path(config['splits'])),
        'source_sha256': {str(p): sha256_of(p) for p in sources},
        'model_source': model_source_sha(), 'torch_version': torch.__version__,
        'scope': 'Inner development only; four directions retained; no outer evaluation',
        'checkpoint': 'Per-direction validation PCC; unchanged independent training code',
        'selection': 'Original-environment seed-ensemble macro PCC averaged over three inner folds',
    })
    state = {'status': 'running', 'completed_trajectories': 0, 'expected': expected,
             'folds_complete': []}
    records, transforms, failures, rotated = [], [], [], {}
    started = time.perf_counter()
    try:
        with (args.out / 'predictions.csv').open('w', newline='') as handle:
            writer = csv.writer(handle, lineterminator='\n')
            writer.writerow(['fold', 'inner', 'config_index', 'seed', 'material_index',
                             'y_pred_0', 'y_pred_1', 'y_pred_2', 'y_pred_3'])
            for fold in folds:
                for inner in range(3):
                    fit, val = check_split(fold, inner)
                    transform = PhenotypeRotation.fit(data.Y[fit])
                    transforms.append({'fold': int(fold['fold']), 'inner': inner,
                                       'fit_ids': fit.tolist(), **transform.metadata()})
                    y = np.full_like(data.Y, np.nan, dtype=np.float64)
                    y[fit] = transform.transform(data.Y[fit])
                    y[val] = transform.transform(data.Y[val])
                    proxy = SimpleNamespace(X=data.X, Y=y, environments=('PC1', 'PC2', 'PC3', 'PC4'))
                    cube = []
                    for c, settings in enumerate(config['configurations']):
                        seed_predictions = []
                        for seed in config['seeds']:
                            components = []
                            for direction in range(4):
                                trajectory = Trajectory(int(fold['fold']), inner, int(seed),
                                                        'independent', c, settings, direction)
                                record = train_trajectory(proxy, fold, inner, trajectory, config, device)
                                components.append(record.pop('prediction')[:, 0])
                                record.pop('state')
                                records.append(record)
                                state['completed_trajectories'] += 1
                            prediction = transform.inverse_transform(np.column_stack(components))
                            if not np.isfinite(prediction).all():
                                raise FloatingPointError('Nonfinite inverse predictions')
                            seed_predictions.append(prediction)
                            for material, values in zip(val, prediction):
                                writer.writerow([fold['fold'], inner, c, seed, int(material), *values])
                        cube.append(seed_predictions)
                    rotated[int(fold['fold']), inner] = np.asarray(cube, dtype=np.float64)
                    handle.flush()
                    dump(args.out / 'trajectories.json', records)
                    dump(args.out / 'transforms.json', transforms)
                    dump(args.out / 'execution_status.json', state)
                state['folds_complete'].append(int(fold['fold']))
                dump(args.out / 'execution_status.json', state)
        candidate = select_and_score(rotated, data, folds, 'common',
                                     config['training']['tie_tolerance'])
        comparisons = {name: compare(candidate, ref) for name, ref in references.items()}
        rule = config['continuation_rule']
        advance = all(c['mean_difference']['pcc'] >= float(rule['minimum_macro_pcc_gain'])
                      and c['pcc_wins'] >= int(rule['minimum_fold_wins'])
                      for c in comparisons.values())
        summary = {**references, 'rotated_common': candidate, 'comparisons': comparisons,
                   'continuation_rule': rule, 'continue_rotation': advance,
                   'expected_fits': expected, 'completed_fits': len(records), 'fit_failures': 0,
                   'cost': {k: float(sum(r[k] for r in records)) for k in
                            ('train_seconds', 'preprocess_seconds', 'inference_seconds')},
                   'scope': 'Conditional development comparison; no test use or SOTA claim'}
        dump(args.out / 'summary.json', summary)
        dump(args.out / 'selection.json', candidate['selection'])
        dump(args.out / 'paired_differences.json', comparisons)
        dump(args.out / 'timings.json', {'total_seconds': time.perf_counter() - started,
                                       'fits': len(records)})
        state['status'] = 'complete'
        dump(args.out / 'execution_status.json', state)
        dump(args.out / 'failures.json', failures)
        paths = [p for p in args.out.iterdir() if p.is_file()]
        dump(args.out / 'artifact_manifest.json', {
            'files': {p.name: {'sha256': sha256_of(p), 'bytes': p.stat().st_size} for p in paths},
            'large_files_remote_only': ['predictions.csv', 'trajectories.json'],
            'checkpoint_storage': 'State SHA only; no persistent weights'})
        return summary
    except Exception as error:
        failures.append({'stage': 'development', 'completed_fits': len(records),
                         'reason': repr(error)})
        state['status'] = 'failed'
        dump(args.out / 'failures.json', failures)
        dump(args.out / 'execution_status.json', state)
        raise
