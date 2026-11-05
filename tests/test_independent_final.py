"""Synthetic checks for the final independent TabM evaluation."""

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

TABM_DEPENDENCIES = '/tmp/wheat599_tabm_deps'
if TABM_DEPENDENCIES not in sys.path:
    sys.path.insert(0, TABM_DEPENDENCIES)

from wheat599 import independent_final as runner  # noqa: E402

torch.set_num_threads(1)

TINY = {'n_blocks': 1, 'd_block': 4, 'dropout': 0.0, 'k': 2,
        'learning_rate': 0.001, 'weight_decay': 0.0}


def make_data(seed=5, materials=40, markers=8, environments=4):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(materials, markers, generator=generator, dtype=torch.float64)
    y = torch.randn(materials, environments, generator=generator, dtype=torch.float64)
    return SimpleNamespace(X=x.numpy(), Y=y.numpy(), n=materials, p=markers,
                           n_env=environments, environments=('1', '2', '4', '5'))


def write_csv(path, rows, fields):
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2))


def make_fixture(tmp_path, monkeypatch):
    data = make_data()
    n, n_env = data.n, data.n_env
    parts = np.array_split(np.arange(n), 10)
    folds = [{'fold': fold, 'dev_idx': sorted(set(range(n)) - set(part.tolist())),
              'test_idx': part.tolist()} for fold, part in enumerate(parts)]
    splits = {'folds': folds}
    splits_path = tmp_path / 'splits.json'
    write_json(splits_path, splits)
    historical_path = tmp_path / 'historical.json'
    write_json(historical_path, splits)

    source_run = tmp_path / 'source_run'
    selection_source = tmp_path / 'selection_source'
    predictions_path = source_run / 'predictions.csv'
    trajectories_path = source_run / 'trajectories.json'
    select_source_path = source_run / 'selection.json'
    selection_list_path = selection_source / 'selection.json'
    artifact_manifest_path = source_run / 'artifact_manifest.json'
    write_csv(predictions_path, [{'placeholder': 1}], ['placeholder'])
    write_json(trajectories_path, {})
    selection_list = [{'fold': fold, 'training': 'independent',
                       'selection': 'per_environment', 'config_indices': [0, 0, 0, 0]}
                      for fold in range(10)]
    write_json(selection_list_path, selection_list)
    environment_selection = {
        str(fold): {'independent': {str(environment): {'config_index': 0}
                                    for environment in range(n_env)}}
        for fold in range(10)}
    write_json(select_source_path, environment_selection)
    write_json(artifact_manifest_path, {'git_commit': 'testcommit'})

    prediction_sha = runner.sha256_of(predictions_path)
    trajectory_sha = runner.sha256_of(trajectories_path)
    selection_sha = runner.sha256_of(selection_list_path)
    data_sha = '0' * 64
    splits_sha = runner.sha256_of(splits_path)
    rows = []
    for fold in folds:
        for environment_index, environment in enumerate(data.environments):
            for seed in (42, 2026):
                special = fold['fold'] == 0 and environment_index == 0 and seed == 42
                inner = [0, 0, 0] if special else [1, 1, 1]
                epochs = 0 if special else 1
                test_ids = fold['test_idx']
                rows.append({
                    'fold': fold['fold'], 'environment_index': environment_index,
                    'environment': environment, 'seed': seed, 'config_index': 0,
                    'configuration': dict(TINY),
                    **{name: TINY[name] for name in ('n_blocks', 'd_block', 'dropout', 'k',
                                                     'learning_rate', 'weight_decay')},
                    'inner_best_epochs': inner, 'median_epochs': epochs,
                    'fit_ids': fold['dev_idx'], 'test_ids': test_ids,
                    'fit_count': len(fold['dev_idx']), 'test_count': len(test_ids),
                    'fit_ids_sha256': runner.ids_sha256(fold['dev_idx']),
                    'test_ids_sha256': runner.ids_sha256(test_ids),
                    'data_sha256': data_sha, 'splits_sha256': splits_sha,
                    'prediction_source_sha256': prediction_sha,
                    'trajectory_source_sha256': trajectory_sha,
                    'selection_source_sha256': selection_sha,
                    'source_git_commit': 'testcommit',
                })
    manifest_document = {'schema_version': 1, 'rows': rows, 'row_count': len(rows),
                         'rows_sha256': runner.rows_sha256(rows),
                         'verification': {'synthetic': True}}
    manifest_path = tmp_path / 'manifest.json'
    write_json(manifest_path, manifest_document)
    freeze_path = tmp_path / 'freeze.yaml'
    freeze = {'configurations': [dict(TINY)], 'source_run': str(source_run),
              'selection_source': str(selection_source),
              'historical_splits': str(historical_path),
              'huber_source': str(tmp_path / 'huber'),
              'gls_source': str(tmp_path / 'huber'),
              'bimsgp_source': str(tmp_path / 'bimsgp'),
              'seeds': [42, 2026], 'predictions_sha256': prediction_sha,
              'trajectories_sha256': trajectory_sha, 'selection_sha256': selection_sha}
    freeze_path.write_text(yaml.safe_dump(freeze))

    huber_oof = []
    bimsgp_oof = []
    for fold in folds:
        for environment_index, environment in enumerate(data.environments):
            truth = data.Y[fold['test_idx'], environment_index]
            for material, value in zip(fold['test_idx'], truth):
                huber_oof.append({'environment': environment, 'fold': fold['fold'],
                                  'material_index': int(material), 'y_true': float(value),
                                  'y_pred': float(0.8 * value + 0.1)})
                bimsgp_oof.append({'environment': environment, 'fold': fold['fold'],
                                   'material_index': int(material), 'y_true': float(value),
                                   'y_pred': float(0.6 * value - 0.1)})
    huber_dir = tmp_path / 'huber'
    bimsgp_dir = tmp_path / 'bimsgp'
    write_csv(huber_dir / 'oof.csv', huber_oof,
              ['environment', 'fold', 'material_index', 'y_true', 'y_pred'])
    write_csv(bimsgp_dir / 'oof.csv', bimsgp_oof,
              ['environment', 'fold', 'material_index', 'y_true', 'y_pred'])
    huber_per_fold = []
    differences = []
    for fold in folds:
        for environment in data.environments:
            subset = [row for row in huber_oof if row['environment'] == environment
                      and row['fold'] == fold['fold']]
            truth = np.array([row['y_true'] for row in subset])
            prediction = np.array([row['y_pred'] for row in subset])
            metrics = {'pcc': float(np.corrcoef(truth, prediction)[0, 1]),
                       'rmse': float(np.sqrt(np.mean((truth - prediction) ** 2))),
                       'mae': float(np.mean(np.abs(truth - prediction)))}
            huber_per_fold.append({'model': 'huber', 'environment': environment,
                                   'fold': fold['fold'], **metrics})
            differences.append({'environment': environment, 'fold': fold['fold'],
                                'delta_pcc': 0.01, 'delta_rmse': 0.005,
                                'delta_mae': 0.003})
    write_csv(huber_dir / 'per_fold.csv', huber_per_fold,
              ['model', 'environment', 'fold', 'pcc', 'rmse', 'mae'])
    write_csv(huber_dir / 'paired_differences.csv', differences,
              ['environment', 'fold', 'delta_pcc', 'delta_rmse', 'delta_mae'])
    write_csv(bimsgp_dir / 'per_fold.csv',
              [{key: row[key] for key in ('environment', 'fold', 'pcc', 'rmse', 'mae')}
               for row in huber_per_fold],
              ['environment', 'fold', 'pcc', 'rmse', 'mae'])
    config = {
        'model': 'independent_tabm_final', 'data': 'unused', 'data_sha256': data_sha,
        'splits': str(splits_path), 'historical_splits': str(historical_path),
        'manifest': str(manifest_path), 'manifest_sha256': runner.sha256_of(manifest_path),
        'rows_sha256': manifest_document['rows_sha256'],
        'source_run': str(source_run), 'selection_source': str(selection_source),
        'freeze_config': str(freeze_path), 'huber_source': str(huber_dir),
        'gls_source': str(huber_dir), 'bimsgp_source': str(bimsgp_dir),
        'seeds': [42, 2026], 'device': 'cpu', 'dtype': 'float32', 'cpu_threads': 1,
        'gpu_memory_fraction': 0.5, 'batch_size': 8, 'gradient_clip': 5.0,
        'expected_fits': 80, 'save_weights': True,
        'versions': {'tabm': '0.0.3', 'rtdl_num_embeddings': '0.0.12'},
        'require_committed_manifest': False,
    }
    config_path = tmp_path / 'config.yaml'
    config_path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(runner, 'load', lambda path, sha: data)
    return data, config, config_path, manifest_document


def test_eighty_row_manifest_runs_all_fits(tmp_path, monkeypatch):
    data, config, config_path, manifest = make_fixture(tmp_path, monkeypatch)
    args = SimpleNamespace(out=tmp_path / 'out', selection=None, config=config_path)
    runner.run(config, args, Path(__file__))
    status = json.loads((args.out / 'execution_status.json').read_text())
    assert status['status'] == 'complete'
    assert status['completed_fits'] == status['expected_fits'] == 80
    assert json.loads((args.out / 'failures.json').read_text()) == []
    validation = json.loads((args.out / 'manifest_validation.json').read_text())
    assert validation['row_count'] == 80
    assert validation['rows_sha256'] == manifest['rows_sha256']
    seed_rows = pd.read_csv(args.out / 'seed_predictions.csv')
    oof_rows = pd.read_csv(args.out / 'oof.csv')
    assert len(seed_rows) == data.n * data.n_env * 2
    assert len(oof_rows) == data.n * data.n_env
    assert sorted(seed_rows.seed.unique()) == [42, 2026]
    assert oof_rows.groupby(['environment', 'fold']).size().eq(4).all()
    weights = list((args.out / 'weights').glob('*.pt'))
    assert len(weights) == 80
    weight_lines = (args.out / 'weights_sha256.txt').read_text().strip().splitlines()
    assert len(weight_lines) == 81
    per_fold = pd.read_csv(args.out / 'per_fold.csv')
    fold_metrics = pd.read_csv(args.out / 'fold_metrics.csv')
    assert len(per_fold) == 40
    assert len(fold_metrics) == 10
    summary = json.loads((args.out / 'summary.json').read_text())
    assert summary['n_runs'] == 80 and summary['n_failures'] == 0
    assert set(summary['primary']) == {'pcc', 'rmse', 'mae'}
    assert summary['baselines']['gls']['derived'] == 'huber_minus_paired_difference'
    assert summary['baselines']['gls']['sources']['per_fold']['sha256']
    comparison = pd.read_csv(args.out / 'comparison.csv')
    assert len(comparison) == 40
    assert {'candidate_pcc', 'huber_pcc', 'gls_pcc', 'bimsgp_pcc'} <= set(comparison)
    assert {'huber_delta_pcc', 'gls_delta_pcc', 'bimsgp_delta_rmse'} <= set(comparison)
    fold_comparison = pd.read_csv(args.out / 'fold_comparison.csv')
    assert len(fold_comparison) == 10
    assert {'candidate_pcc', 'huber_delta_pcc', 'gls_delta_rmse'} <= set(fold_comparison)
    comparison_summary = json.loads((args.out / 'comparison_summary.json').read_text())
    assert 'primary_delta' in comparison_summary and 'fold_macro' in comparison_summary
    provenance = json.loads((args.out / 'provenance.json').read_text())
    assert any(key.endswith('genomic_tabm.py') for key in provenance['source_sha256'])
    assert 'tabm' in provenance['model_source']
    fits = json.loads((args.out / 'fits.json').read_text())
    assert min(row['epochs'] for row in fits) == 0
    assert max(row['epochs'] for row in fits) == 1


def test_manifest_sha_mismatch_is_rejected_before_fits(tmp_path, monkeypatch):
    _, config, config_path, _ = make_fixture(tmp_path, monkeypatch)
    bad = copy.deepcopy(config)
    bad['manifest_sha256'] = '0' * 64
    args = SimpleNamespace(out=tmp_path / 'bad_out', selection=None,
                           config=config_path)
    with pytest.raises(ValueError, match='Manifest file sha256'):
        runner.run(bad, args, Path(__file__))
    assert not args.out.exists()


def test_baseline_test_id_order_is_checked():
    rows = [{'material_index': 1, 'y_true': 0.0, 'y_pred': 0.0},
            {'material_index': 2, 'y_true': 1.0, 'y_pred': 1.0}]
    baseline = {('1', 0): rows}
    candidate = {('1', 0): {'ids': [1, 2], 'truth': np.array([0.0, 1.0])}}
    runner.verify_baseline_groups(baseline, candidate)
    wrong = {('1', 0): list(reversed(rows))}
    with pytest.raises(ValueError, match='test ID order'):
        runner.verify_baseline_groups(wrong, candidate)


def test_pinned_versions_are_checked_before_fits(tmp_path, monkeypatch):
    _, config, config_path, _ = make_fixture(tmp_path, monkeypatch)
    bad = copy.deepcopy(config)
    bad['versions'] = {'tabm': '0.0.0', 'rtdl_num_embeddings': '0.0.12'}
    args = SimpleNamespace(out=tmp_path / 'version_out', selection=None,
                           config=config_path)
    with pytest.raises(RuntimeError, match='Pinned package'):
        runner.run(bad, args, Path(__file__))
    assert not args.out.exists()


def test_resume_summary_reuses_completed_fits_without_retraining(tmp_path, monkeypatch):
    data, config, config_path, _ = make_fixture(tmp_path, monkeypatch)
    normal_out = tmp_path / 'normal_out'
    runner.run(config, SimpleNamespace(out=normal_out, selection=None,
                                       config=config_path), Path(__file__))
    normal = json.loads((normal_out / 'summary.json').read_text())

    huber_oof = Path(config['huber_source']) / 'oof.csv'
    huber_bytes = huber_oof.read_bytes()
    failed_out = tmp_path / 'failed_out'
    huber_oof.unlink()
    with pytest.raises(FileNotFoundError):
        runner.run(config, SimpleNamespace(out=failed_out, selection=None,
                                           config=config_path), Path(__file__))
    huber_oof.write_bytes(huber_bytes)
    status_failed = json.loads((failed_out / 'execution_status.json').read_text())
    assert status_failed['status'] == 'failed'
    failures_before = (failed_out / 'failures.json').read_bytes()
    seed_path = failed_out / 'seed_predictions.csv'
    seed_hash = runner.sha256_of(seed_path)
    fits_hash = runner.sha256_of(failed_out / 'fits.json')
    provenance_before = json.loads((failed_out / 'provenance.json').read_text())

    def forbidden(*arguments, **keywords):
        raise AssertionError('fit_one must not run during summary recovery')

    monkeypatch.setattr(runner, 'fit_one', forbidden)
    runner.run(config, SimpleNamespace(out=failed_out, selection=None,
                                       config=config_path, resume_summary=True),
               Path(__file__))
    resumed = json.loads((failed_out / 'summary.json').read_text())
    assert resumed['primary'] == pytest.approx(normal['primary'])
    assert resumed['fold_mean'] == pytest.approx(normal['fold_mean'])
    assert resumed['pooled_oof'] == pytest.approx(normal['pooled_oof'])
    for environment in normal['per_environment']:
        for metric in normal['per_environment'][environment]:
            assert resumed['per_environment'][environment][metric] == pytest.approx(
                normal['per_environment'][environment][metric])
    status = json.loads((failed_out / 'execution_status.json').read_text())
    assert status['status'] == 'complete' and status['resumed'] is True
    assert status['fit_failures'] == 0
    assert status['original_summary_failures'] == 1
    assert status['summary_recovery'] == 'success'
    assert (failed_out / 'failures.json').read_bytes() == failures_before
    assert runner.sha256_of(seed_path) == seed_hash
    assert runner.sha256_of(failed_out / 'fits.json') == fits_hash
    provenance = json.loads((failed_out / 'provenance.json').read_text())
    assert provenance['source_sha256'] == provenance_before['source_sha256']
    assert provenance['resume']['initial_git_commit'] == provenance_before['git_commit']
    assert provenance['resume']['fit_failures'] == 0
    assert provenance['resume']['original_summary_failures'] == 1

    lines = seed_path.read_text().splitlines()
    seed_path.write_text('\n'.join(lines[:-1]) + '\n')
    with pytest.raises(ValueError, match='seed prediction count'):
        runner.run(config, SimpleNamespace(out=failed_out, selection=None,
                                           config=config_path, resume_summary=True),
                   Path(__file__))
