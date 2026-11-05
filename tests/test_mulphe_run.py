"""Synthetic coverage for the Mul-PheG2P runner: manifest rules, leak boundaries, scoring scopes."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from wheat599 import mulphe, mulphe_run

ARMS = ['direct', 'target_pretrain', 'frozen_sources', 'scalar_prediction', 'embedding']
ENVIRONMENTS = ('1', '2', '4', '5')


def synthetic_data(rows=40, markers=25, seed=0):
    rng = np.random.default_rng(seed)
    x = (rng.random((rows, markers)) < 0.5).astype(np.float64)
    signal = x[:, :4].mean(axis=1)
    y = np.stack([signal + 0.2 * rng.normal(size=rows) for _ in ENVIRONMENTS], axis=1)

    class Data:
        X, Y = x, y
        environments = ENVIRONMENTS
        n, n_env, p = rows, len(ENVIRONMENTS), markers

    return Data()


def manifest_dict(folds):
    return {'folds': folds, 'rule': 'synthetic', 'numpy': '0', 'sklearn': '0',
            'description': 'test'}


def make_folds(n_materials=40, folds=2, seed=1):
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_materials)
    size = n_materials // folds
    parts = []
    for index in range(folds):
        test = order[index * size:(index + 1) * size] if index < folds - 1 else order[size:]
        dev = np.setdiff1d(order, test)
        train, val = dev[:int(0.8 * len(dev))], dev[int(0.8 * len(dev)):]
        parts.append({'fold': index, 'dev_idx': dev.tolist(), 'train_idx': train.tolist(),
                      'val_idx': val.tolist(), 'test_idx': test.tolist()})
    return parts


def write_manifest(directory, folds):
    path = Path(directory) / 'manifest.json'
    path.write_text(json.dumps(manifest_dict(folds)))
    return path, mulphe_run.sha256_of(path)


def fake_arm_result(arm, n_test, n_val, seed=42):
    rng = np.random.default_rng(abs(hash((arm, n_test, seed))) % 2 ** 32)
    return {'arm': arm, 'prediction': rng.normal(size=n_test), 'val_prediction': rng.normal(size=n_val),
            'val_metrics': {'pcc': 0.5, 'rmse': 0.4, 'mae': 0.3}, 'best_epoch': 2, 'epochs_run': 3,
            'history': [{'epoch': 1}, {'epoch': 2}, {'epoch': 3}],
            'scaling': {'mode': 'minmax', 'scope': 'train', 'low': 0.0, 'high': 1.0},
            'state_sha256': 'a' * 64, 'parameters': 100, 'total_parameters': 400,
            'train_seconds': 0.1, 'source_names': ['x'], 'source_pretraining': [{'name': 'x', 'best_epoch': 1}],
            'fusion_input': 'embedding',
            'source_scalars': {'train': np.zeros((5, 3)), 'val': np.zeros((2, 3))}
            if arm == 'scalar_prediction' else None}


@pytest.fixture
def recorder(monkeypatch):
    """Capture every array the runner hands to the model and keep training synthetic."""
    calls = []

    def run_arms(x_train, y_train, x_val, y_val, x_test, **kwargs):
        calls.append({'x_train': x_train, 'y_train': y_train, 'x_val': x_val, 'y_val': y_val,
                      'x_test': x_test, 'kwargs': kwargs})
        return {'arms': {arm: fake_arm_result(arm, len(x_test), len(x_val)) for arm in kwargs['arms']},
                'source_names': kwargs['source_names'], 'source_pretraining': [],
                'source_pretraining_seconds': 0.0, 'impute_scope': 'train',
                'official': {'repo': mulphe.OFFICIAL_REPO, 'commit': mulphe.OFFICIAL_COMMIT,
                             'license': 'MIT'}}

    monkeypatch.setattr(mulphe, 'run_arms', run_arms)
    monkeypatch.setattr(mulphe, 'train_source_encoders',
                        lambda x_train, y_train, x_val, y_val, source_names, config, seed, device:
                        [{'name': str(name), 'best_epoch': 1, 'model': None} for name in source_names])
    return calls


# -- frozen manifest -------------------------------------------------------
def test_frozen_five_fold_manifest_is_accepted():
    path = Path('results/mulphe_splits_20261105/manifest.json')
    if not path.is_file():
        pytest.skip('frozen manifest is not present in this checkout')
    manifest, folds, digest = mulphe_run.verify_manifest(
        path, '0b959a3a5b8879c0537c9da669e037b5f991081beee5e522c7d7c7281d3df1ba', 599, 4)
    assert len(folds) == 5 and digest == manifest.get('splits_sha256', digest)
    assert [int(fold['fold']) for fold in folds] == [0, 1, 2, 3, 4]


def test_manifest_rejects_leaky_or_incomplete_partitions(tmp_path):
    folds = make_folds()
    path, digest = write_manifest(tmp_path, folds)
    mulphe_run.verify_manifest(path, digest, 40, 4)
    duplicated = [dict(fold) for fold in folds]
    material = duplicated[1]['test_idx'][0]
    for field in ('dev_idx', 'train_idx', 'val_idx'):
        duplicated[0][field] = [m for m in duplicated[0][field] if m != material]
    duplicated[0]['test_idx'] = duplicated[0]['test_idx'] + [material]
    path, digest = write_manifest(tmp_path, duplicated)
    with pytest.raises(ValueError, match='more than once'):
        mulphe_run.verify_manifest(path, digest, 40, 4)

    folds = make_folds()
    folds[0]['dev_idx'] = folds[0]['train_idx']  # dev is not train plus val
    path, digest = write_manifest(tmp_path, folds)
    with pytest.raises(ValueError, match='dev_idx'):
        mulphe_run.verify_manifest(path, digest, 40, 4)

    folds = make_folds()
    folds[0]['test_idx'] = folds[0]['test_idx'] + folds[0]['test_idx'][:1]
    path, digest = write_manifest(tmp_path, folds)
    with pytest.raises(ValueError, match='duplicate indices'):
        mulphe_run.verify_manifest(path, digest, 40, 4)

    folds = make_folds()
    folds[0]['test_idx'] = folds[0]['test_idx'] + folds[0]['train_idx'][:1]
    path, digest = write_manifest(tmp_path, folds)
    with pytest.raises(ValueError, match='test overlaps the development set'):
        mulphe_run.verify_manifest(path, digest, 40, 4)
    with pytest.raises(ValueError, match='SHA mismatch'):
        mulphe_run.verify_manifest(path, '0' * 64, 40, 4)


def test_job_sequence_counts_folds_environments_and_seeds():
    folds = make_folds()
    jobs = mulphe_run.job_sequence(folds, 4, [42, 43])
    assert len(jobs) == len(folds) * 4 * 2 == 16
    assert {job[0] for job in jobs} == {0, 1}
    assert {job[1] for job in jobs} == {0, 1, 2, 3}
    assert {job[2] for job in jobs} == {42, 43}


# -- fitting ---------------------------------------------------------------
def test_fit_never_passes_test_phenotypes(tmp_path, recorder):
    data = synthetic_data()
    folds = make_folds()
    config = {'arms': ARMS, 'seeds': [42], 'model_config': {}, 'status_every': 1}
    rows, records, validation, seconds = mulphe_run.fit(
        config, data, folds, 'cpu', len(folds) * 4,
        tmp_path / 'execution_status.json', tmp_path / 'failures.json')
    assert seconds >= 0 and len(recorder) == len(folds) * 4
    train_sizes = {len(fold['train_idx']) for fold in folds}
    val_sizes = {len(fold['val_idx']) for fold in folds}
    for call in recorder:
        assert 'y_test' not in call['kwargs']
        assert call['kwargs']['source_models'] is not None  # shared across the fold's targets
        assert len(call['y_train']) in train_sizes and len(call['y_val']) in val_sizes
        assert len(call['kwargs']['source_train']) == len(call['y_train'])
        assert call['x_test'].ndim == 2 and call['x_test'].shape[1] == data.p
        assert all(name not in call['kwargs']['source_names']
                   for name in [call['kwargs']['target_name']])
    assert len(rows) == sum(len(fold['test_idx']) for fold in folds) * len(ARMS) * len(ENVIRONMENTS)
    assert len(records) == len(folds) * 4 * len(ARMS)
    assert json.loads((tmp_path / 'failures.json').read_text()) == []
    status = json.loads((tmp_path / 'execution_status.json').read_text())
    assert status['status'] == 'complete' and status['completed_fits'] == len(folds) * 4
    assert all(entry['source_encoder_reuse'] for entry in validation)


def test_fit_can_train_source_encoders_per_target(tmp_path, recorder):
    data = synthetic_data()
    folds = make_folds()
    config = {'arms': ARMS, 'seeds': [42], 'model_config': {}, 'share_source_encoders': False}
    mulphe_run.fit(config, data, folds, 'cpu', len(folds) * 4,
                   tmp_path / 'execution_status.json', tmp_path / 'failures.json')
    assert all(call['kwargs']['source_models'] is None for call in recorder)


def test_fit_records_are_json_safe_and_seeded(tmp_path, recorder):
    data = synthetic_data()
    folds = make_folds()
    config = {'arms': ['direct'], 'seeds': [7], 'model_config': {}}
    rows, records, validation, _ = mulphe_run.fit(
        config, data, folds, 'cpu', len(folds) * 1,
        tmp_path / 'execution_status.json', tmp_path / 'failures.json')
    json.dumps(records)
    assert records[0]['state_sha256'] == 'a' * 64
    assert records[0]['epochs_recorded'] == 3
    assert all(row['y_pred'] == format(float(row['y_pred']), '.10g') for row in rows)
    assert all('source_scalars' not in record and 'prediction' not in record for record in records)


def test_fit_writes_a_failure_after_the_first_crash(tmp_path, monkeypatch):
    data = synthetic_data()
    folds = make_folds()

    def explode(*args, **kwargs):
        raise RuntimeError('synthetic crash')

    monkeypatch.setattr(mulphe, 'run_arms', explode)
    with pytest.raises(RuntimeError):
        mulphe_run.fit({'arms': ['direct'], 'seeds': [42], 'model_config': {}}, data, folds, 'cpu', 8,
                       tmp_path / 'execution_status.json', tmp_path / 'failures.json')
    failures = json.loads((tmp_path / 'failures.json').read_text())
    assert failures[0]['type'] == 'RuntimeError' and failures[0]['completed_fits'] == 0
    assert json.loads((tmp_path / 'execution_status.json').read_text())['status'] == 'failed'


# -- scoring ---------------------------------------------------------------
def prediction_grid(config, data, folds, value=0.5):
    rows = []
    for fold in folds:
        for environment in data.environments:
            for seed in config['seeds']:
                for arm in config['arms']:
                    for material in fold['test_idx']:
                        rows.append({'arm': arm, 'fold': fold['fold'], 'target_environment': environment,
                                     'seed': seed, 'material_index': material,
                                     'y_pred': value + 0.01 * material})
    return rows


def test_score_separates_fold_mean_from_pooled_oof():
    data, folds = synthetic_data(), make_folds()
    config = {'arms': ['direct'], 'seeds': [42]}
    rows = prediction_grid(config, data, folds)
    per_fold, pooled, comparison = mulphe_run.score(config, data, folds, rows, {})
    truth = np.asarray(data.Y)
    materials = np.array(sorted({row['material_index'] for row in rows}))
    assert comparison == {} and per_fold['direct']['scope'].startswith('mean of per-fold')
    prediction = 0.5 + 0.01 * materials
    expected_pcc = mulphe_run.pcc(truth[:, 0], prediction)
    assert pooled['direct']['per_environment']['1']['pcc'] == pytest.approx(expected_pcc)
    assert set(pooled['direct']['per_seed_macro']) == {'42'}
    assert pooled['direct']['seed_ensemble_macro'] == pooled['direct']['per_seed_macro']['42']
    assert len(per_fold['direct']['fold_pcc']) == len(folds)


def test_score_refuses_missing_or_extra_materials():
    data, folds = synthetic_data(), make_folds()
    config = {'arms': ['direct'], 'seeds': [42]}
    rows = prediction_grid(config, data, folds)
    with pytest.raises(ValueError, match='frozen test set'):
        mulphe_run.score(config, data, folds, rows[:-1], {})
    with pytest.raises(ValueError, match='Empty prediction group'):
        mulphe_run.score(config, data, folds, [], {})
    extra = rows + [{'arm': 'direct', 'fold': folds[0]['fold'], 'target_environment': '1',
                     'seed': 42, 'material_index': folds[1]['test_idx'][0], 'y_pred': 0.1}]
    with pytest.raises(ValueError, match='frozen test set'):
        mulphe_run.score(config, data, folds, extra, {})
    unexpected = rows + [{'arm': 'invented', 'fold': folds[0]['fold'], 'target_environment': '1',
                          'seed': 42, 'material_index': folds[0]['test_idx'][0], 'y_pred': 0.1}]
    with pytest.raises(ValueError, match='Unexpected prediction key'):
        mulphe_run.score(config, data, folds, unexpected, {})


def write_reference(path, data, folds, scale=0.9, fold_shift=0):
    """Stored OOF in the run's own fold structure (or shifted away from it)."""
    with path.open('w') as handle:
        handle.write('environment,fold,material_index,y_true,y_pred\n')
        for fold in folds:
            for e_index, environment in enumerate(data.environments):
                for material in fold['test_idx']:
                    truth = float(data.Y[material, e_index])
                    handle.write(f'{environment},{int(fold["fold"]) + fold_shift},{material},'
                                 f'{truth:.12g},{truth * scale:.12g}\n')
    return path


def load(tmp_path, data, name='oof.csv', **kwargs):
    return mulphe_run.load_reference(tmp_path / name, list(data.environments))


def test_reference_pairs_the_same_fold_environment_material(tmp_path):
    data, folds = synthetic_data(), make_folds()
    config = {'arms': ['direct'], 'seeds': [42]}
    rows = prediction_grid(config, data, folds)
    write_reference(tmp_path / 'oof.csv', data, folds)
    _, _, comparison = mulphe_run.score(
        config, data, folds, rows, {'huber': load(tmp_path, data)})
    entry = comparison['huber']['direct']
    assert entry['protocol']['same_fold_structure'] is True
    assert entry['protocol']['paired_units'] == len(folds) * len(ENVIRONMENTS)
    assert entry['unpaired'] == []
    units = {(unit['fold'], unit['environment']) for unit in entry['per_fold']}
    assert units == {(fold['fold'], environment) for fold in folds for environment in ENVIRONMENTS}
    for unit in entry['per_fold']:
        assert unit['materials'] == len(unit['test']) if 'test' in unit else unit['materials'] > 0
        for metric in mulphe_run.METRICS:
            assert unit['model_minus_reference'][metric] == pytest.approx(
                unit['model'][metric] - unit['reference'][metric])
    fold_mean = entry['fold_mean']['per_environment']
    assert set(fold_mean) == set(ENVIRONMENTS)
    assert fold_mean['1']['folds'] == [int(fold['fold']) for fold in folds]
    assert entry['fold_mean']['macro']['delta'] == pytest.approx(
        {metric: np.mean([fold_mean[environment]['fold_mean_delta'][metric]
                          for environment in ENVIRONMENTS]) for metric in mulphe_run.METRICS})
    assert 0 <= entry['fold_mean']['per_environment']['1']['pcc_wins_of_folds'] <= len(folds)


def test_foreign_reference_folds_are_not_reported_as_the_same_protocol(tmp_path):
    data, folds = synthetic_data(), make_folds()
    config = {'arms': ['direct'], 'seeds': [42]}
    rows = prediction_grid(config, data, folds)
    write_reference(tmp_path / 'oof.csv', data, folds, fold_shift=5)
    _, _, comparison = mulphe_run.score(
        config, data, folds, rows, {'huber': load(tmp_path, data)})
    entry = comparison['huber']['direct']
    assert entry['protocol']['same_fold_structure'] is False
    assert entry['protocol']['foreign_reference_folds'] == [5, 6]
    assert entry['protocol']['paired_units'] == 0
    assert entry['fold_mean']['macro'] == {}
    assert entry['pooled_shared_materials']['1']['materials'] == data.n
    assert entry['pooled_shared_materials']['1']['reference']['pcc'] == pytest.approx(1.0, abs=1e-9)


def test_reference_rows_outside_the_manifest_fold_are_refused(tmp_path):
    data, folds = synthetic_data(), make_folds()
    config = {'arms': ['direct'], 'seeds': [42]}
    rows = prediction_grid(config, data, folds)
    path = write_reference(tmp_path / 'oof.csv', data, folds)
    lines = path.read_text().splitlines()
    fields = lines[1].split(',')
    material = int(folds[1]['test_idx'][0])            # a material belonging to the other fold
    fields[2] = str(material)
    fields[3] = f'{float(data.Y[material, 0]):.12g}'   # keeps y_true consistent
    path.write_text('\n'.join([lines[0], ','.join(fields)] + lines[2:]) + '\n')
    with pytest.raises(ValueError, match='outside fold'):
        mulphe_run.score(config, data, folds, rows, {'huber': load(tmp_path, data)})


def test_reference_comparison_asserts_y_true_against_the_data(tmp_path):
    data, folds = synthetic_data(), make_folds()
    config = {'arms': ['direct'], 'seeds': [42]}
    rows = prediction_grid(config, data, folds)
    path = write_reference(tmp_path / 'oof.csv', data, folds)
    lines = path.read_text().splitlines()
    fields = lines[1].split(',')
    fields[3] = str(float(fields[3]) + 1.0)
    path.write_text('\n'.join([lines[0], ','.join(fields)] + lines[2:]) + '\n')
    with pytest.raises(ValueError, match='y_true disagrees'):
        mulphe_run.score(config, data, folds, rows, {'huber': load(tmp_path, data)})


def test_load_reference_rejects_unknown_environment_and_duplicates(tmp_path):
    data, folds = synthetic_data(), make_folds()
    path = write_reference(tmp_path / 'oof.csv', data, folds)
    mulphe_run.load_reference(path, list(data.environments))
    broken = tmp_path / 'broken.csv'
    broken.write_text(path.read_text().replace('1,0,', '9,0,', 1))
    with pytest.raises(ValueError, match='not in the data'):
        mulphe_run.load_reference(broken, list(data.environments))
    duplicate = tmp_path / 'duplicate.csv'
    duplicate.write_text(path.read_text() + path.read_text().splitlines()[1] + '\n')
    with pytest.raises(ValueError, match='Duplicate reference row'):
        mulphe_run.load_reference(duplicate, list(data.environments))


def test_pooled_shared_rejects_a_material_predicted_twice(tmp_path):
    data, folds = synthetic_data(), make_folds()
    config = {'arms': ['direct'], 'seeds': [42]}
    rows = prediction_grid(config, data, folds)
    path = write_reference(tmp_path / 'oof.csv', data, folds)
    extra = path.read_text().splitlines()[1].split(',')
    extra[1] = '9'                                    # same material under a foreign fold
    extra[4] = str(float(extra[4]) * 0.5)             # with a different prediction
    path.write_text(path.read_text() + ','.join(extra) + '\n')
    with pytest.raises(ValueError, match='twice with different values'):
        mulphe_run.score(config, data, folds, rows, {'huber': load(tmp_path, data)})
    duplicate = tmp_path / 'duplicate.csv'
    duplicate.write_text(path.read_text() + path.read_text().splitlines()[1] + '\n')
    with pytest.raises(ValueError, match='Duplicate reference row'):
        mulphe_run.load_reference(duplicate, list(data.environments))


# -- driver ----------------------------------------------------------------
def test_run_writes_every_artifact(tmp_path, monkeypatch, recorder):
    data = synthetic_data()
    folds = make_folds()
    path, digest = write_manifest(tmp_path, folds)
    config_path = tmp_path / 'config.json'
    config = {'data': 'unused', 'data_sha256': None, 'splits': str(path), 'splits_sha256': digest,
              'device': 'cpu', 'cpu_threads': 1, 'arms': ['direct'], 'seeds': [42],
              'model_config': {}, 'status_every': 1}
    config_path.write_text(json.dumps(config))
    monkeypatch.setattr(mulphe_run, 'load', lambda *args, **kwargs: data)
    args = SimpleNamespace(out=tmp_path / 'run', config=config_path, selection=None)
    out = mulphe_run.run(config, args, str(config_path))
    assert out is not None
    names = {entry.name for entry in (tmp_path / 'run').iterdir()}
    assert {'predictions.csv', 'fits.json', 'summary.json', 'provenance.json', 'manifest_validation.json',
            'execution_status.json', 'failures.json', 'validation.json', 'timings.json',
            'artifact_manifest.json'} <= names
    manifest = json.loads((tmp_path / 'run' / 'artifact_manifest.json').read_text())
    for name, entry in manifest['files'].items():
        assert mulphe_run.sha256_of(tmp_path / 'run' / name) == entry['sha256']
    summary = json.loads((tmp_path / 'run' / 'summary.json').read_text())
    assert summary['pooled_oof']['direct']['scope'].startswith('one pooled OOF')
    assert summary['completed_fits'] == len(folds) * 4
    assert json.loads((tmp_path / 'run' / 'provenance.json').read_text())['official']['commit'] \
        == mulphe.OFFICIAL_COMMIT


def test_run_refuses_selection_and_unknown_arms(tmp_path, monkeypatch):
    data = synthetic_data()
    folds = make_folds()
    path, digest = write_manifest(tmp_path, folds)
    config_path = tmp_path / 'config.json'
    config = {'data': 'unused', 'data_sha256': None, 'splits': str(path), 'splits_sha256': digest,
              'device': 'cpu', 'arms': ['invented'], 'seeds': [42], 'model_config': {}}
    config_path.write_text(json.dumps(config))
    monkeypatch.setattr(mulphe_run, 'load', lambda *args, **kwargs: data)
    args = SimpleNamespace(out=tmp_path / 'one', config=config_path, selection=None)
    with pytest.raises(ValueError, match='non-empty distinct subset'):
        mulphe_run.run(config, args, str(config_path))
    config['arms'] = ['direct']
    args = SimpleNamespace(out=tmp_path / 'two', config=config_path, selection=Path('x.json'))
    with pytest.raises(ValueError, match='no selection input'):
        mulphe_run.run(config, args, str(config_path))


def test_yaml_dispatch_key_stays_out_of_model_params(tmp_path, monkeypatch, recorder):
    """`model: mulphe` selects the runner; only `model_config` reaches the model."""
    data = synthetic_data()
    folds = make_folds()
    path, digest = write_manifest(tmp_path, folds)
    config_path = tmp_path / 'config.yaml'
    config_path.write_text(yaml.safe_dump({
        'model': 'mulphe', 'model_config': {'direct_epochs': 3}, 'arms': ['direct'], 'seeds': [42],
        'data': 'unused', 'data_sha256': None, 'splits': str(path), 'splits_sha256': digest,
        'device': 'cpu', 'cpu_threads': 1, 'status_every': 1}))
    loaded = yaml.safe_load(config_path.read_text())
    monkeypatch.setattr(mulphe_run, 'load', lambda *args, **kwargs: data)
    mulphe_run.run(loaded, SimpleNamespace(out=tmp_path / 'run', config=config_path, selection=None),
                   str(config_path))
    assert recorder[0]['kwargs']['config'] == {'direct_epochs': 3}
    provenance = json.loads((tmp_path / 'run' / 'provenance.json').read_text())
    assert provenance['config']['model'] == 'mulphe'
    assert 'model' not in str(recorder[0]['kwargs']['config'])


@pytest.mark.parametrize('name', ['mulphe_fivefold', 'mulphe_tenfold'])
def test_frozen_configs_are_read_by_the_runner(name):
    path = Path('configs') / f'{name}.yaml'
    if not path.is_file():
        pytest.skip(f'{name} is not frozen yet')
    config = yaml.safe_load(path.read_text())
    assert config['model'] == 'mulphe' and isinstance(config['model_config'], dict)
    assert set(config['arms']) <= set(mulphe.ARMS) and 'scalar_embedding' not in config['arms']
    manifest, folds, _ = mulphe_run.verify_manifest(
        Path(config['splits']), config['splits_sha256'], 599, 4)
    assert len(folds) == len({int(fold['fold']) for fold in folds})
    assert config['share_source_encoders'] is True
    for reference_name, entry in (config.get('references') or {}).items():
        loaded = mulphe_run.load_reference(entry['oof'], ['1', '2', '4', '5'])
        fold_test = {int(fold['fold']): {int(m) for m in fold['test_idx']} for fold in folds}
        assert sorted({f for values in loaded['columns'].values() for (f, _) in values}
                      - set(fold_test)) == []
        assert not [(e, f, m) for e, values in loaded['columns'].items() for (f, m) in values
                    if m not in fold_test.get(f, set())]
        if name == 'mulphe_tenfold':
            assert {(f, e) for e, values in loaded['columns'].items() for (f, _) in values} == \
                   {(f, e) for f in fold_test for e in loaded['columns']}
