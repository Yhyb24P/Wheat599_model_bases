"""Budget, original-environment selection and outer-label isolation checks."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from wheat599 import rotation_run as run
from wheat599.phenotype_rotation import PhenotypeRotation


def fixture():
    rng = np.random.default_rng(71)
    data = SimpleNamespace(X=rng.normal(size=(48, 8)), Y=rng.normal(size=(48, 4)),
                           environments=('1', '2', '4', '5'), n_env=4)
    folds = []
    for f, test in enumerate(np.array_split(np.arange(48), 2)):
        dev = np.setdiff1d(np.arange(48), test)
        inner = [{'train_idx': np.setdiff1d(dev, v).tolist(), 'val_idx': v.tolist()}
                 for v in np.array_split(dev, 3)]
        folds.append({'fold': f, 'test_idx': test.tolist(), 'dev_idx': dev.tolist(),
                      'inner': inner})
    return data, folds


def test_same_fixed_linear_operator_is_invariant_to_full_rotation():
    data, _ = fixture()
    rotation = PhenotypeRotation.fit(data.Y)
    operator = np.random.default_rng(17).normal(size=(7, 48))
    original = (operator @ ((data.Y - rotation.mean_) / rotation.std_)) \
        * rotation.std_ + rotation.mean_
    recovered = rotation.inverse_transform(operator @ rotation.transform(data.Y))
    np.testing.assert_allclose(recovered, original, atol=1e-12, rtol=0)


def test_common_selection_uses_all_original_environments():
    data, folds = fixture()
    units = {}
    for fold in folds:
        for j, split in enumerate(fold['inner']):
            y = data.Y[split['val_idx']]
            good = y + 0.15 * data.X[split['val_idx'], :4]
            env2_only = -y.copy()
            env2_only[:, 1] = y[:, 1]
            units[fold['fold'], j] = np.asarray([[good, good], [env2_only, env2_only]])
    common = run.select_and_score(units, data, folds, 'common', 1e-12)
    separate = run.select_and_score(units, data, folds, 'per_environment', 1e-12)
    assert all(r['config_indices'] == [0, 0, 0, 0] for r in common['selection'])
    assert all(r['config_indices'] == [0, 1, 0, 0] for r in separate['selection'])


def test_full_development_driver_keeps_outer_labels_hidden(tmp_path, monkeypatch):
    data, folds = fixture()
    splits = tmp_path / 'splits.json'
    splits.write_text(json.dumps({'folds': folds}))
    config_path = tmp_path / 'config.json'
    config_path.write_text('{}')
    bank = {}
    for fold in folds:
        for j, split in enumerate(fold['inner']):
            for seed in [42, 2026]:
                for material in split['val_idx']:
                    for e in range(4):
                        bank[fold['fold'], j, 0, seed, material, e] = float(
                            0.8 * data.Y[material, e] + 0.05 * data.X[material, e])
    settings = {'n_blocks': 1, 'd_block': 8, 'dropout': 0.0, 'k': 2,
                'learning_rate': 0.001, 'weight_decay': 0.0}
    config = {'data': 'unused', 'data_sha256': 'synthetic', 'splits': str(splits),
              'device': 'cpu', 'cpu_threads': 1, 'seeds': [42, 2026],
              'configurations': [settings], 'expected_control_pcc': {},
              'expected_fits': 48, 'training': {'tie_tolerance': 1e-12},
              'continuation_rule': {'minimum_macro_pcc_gain': 0.001,
                                    'minimum_fold_wins': 2}}
    calls = []

    def train(proxy, fold, inner, trajectory, config, device):
        assert np.isnan(proxy.Y[fold['test_idx']]).all()
        fit = fold['inner'][inner]['train_idx']
        val = fold['inner'][inner]['val_idx']
        assert np.isfinite(proxy.Y[fit]).all() and np.isfinite(proxy.Y[val]).all()
        calls.append((fold['fold'], inner, trajectory.seed, trajectory.environment))
        # This fake trajectory verifies plumbing, not model performance or learning.
        values = 0.8 * proxy.Y[val, trajectory.environment] + 0.02 * proxy.X[val, 0]
        return {'prediction': values[:, None], 'state': {},
                'trajectory': trajectory.__dict__, 'best_epoch': 0,
                'train_seconds': 0.0, 'preprocess_seconds': 0.0, 'inference_seconds': 0.0}

    monkeypatch.setattr(run, 'train_trajectory', train)
    monkeypatch.setattr(run, 'load', lambda *a: data)
    monkeypatch.setattr(run, 'control_bank', lambda c: bank)
    monkeypatch.setattr(run, 'verify_versions', lambda c: None)
    monkeypatch.setattr(run, 'model_source_sha', lambda: {})
    out = tmp_path / 'out'
    summary = run.run(config, SimpleNamespace(out=out, config=config_path, selection=None),
                      Path(__file__))
    assert len(calls) == len(set(calls)) == 48
    assert summary['completed_fits'] == 48 and summary['fit_failures'] == 0
    assert json.loads((out / 'execution_status.json').read_text())['status'] == 'complete'
    assert len(json.loads((out / 'transforms.json').read_text())) == 6
    assert len(json.loads((out / 'trajectories.json').read_text())) == 48
    assert set(summary['comparisons']) == {'original_common', 'original_strong'}
    assert all(len(set(r['config_indices'])) == 1
               for r in summary['rotated_common']['selection'])
