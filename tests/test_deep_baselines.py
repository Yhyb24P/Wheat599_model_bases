"""WS2 deep baselines: frozen-budget rules, encoding and the nested driver."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from genopred.models.comparators import deep_baselines as db


def test_frozen_lr_median_and_padding():
    sequences = [[0.1, 0.1, 0.08], [0.1, 0.08], [0.1, 0.1, 0.1, 0.05]]
    assert db.frozen_lr(sequences, 3) == [0.1, 0.1, 0.08]
    # epoch 4: short runs pad with their last value -> median(0.08, 0.08, 0.05)
    assert db.frozen_lr(sequences, 4) == [0.1, 0.1, 0.08, 0.08]
    with pytest.raises(ValueError):
        db.frozen_lr([[0.0]], 1)


def test_deepgs_structure_and_shapes():
    import torch
    model = db.DeepGS(1279)
    # official Wheat599 panel: (1279-18+1-4)//4+1 = 315 pooled positions
    assert model.pooled == 315
    assert model.fc1.in_features == 8 * 315
    out = model(torch.zeros(3, 1279))
    assert out.shape == (3,)
    with pytest.raises(ValueError):
        db.DeepGS(20)


def _fake_run(tmp_path, monkeypatch, model='deepgs'):
    """Ten folds x two envs, tiny budgets, deterministic stub trainer."""
    n, p, n_env = 60, 40, 2
    rng = np.random.default_rng(0)
    X = rng.integers(0, 2, size=(n, p)).astype(np.float64)
    Y = rng.normal(size=(n, n_env))
    folds = []
    order = rng.permutation(n)
    for f in range(10):
        test = order[f * 6:(f + 1) * 6]
        dev = np.sort(np.setdiff1d(order, test))
        inner = []
        for k in range(3):
            val = dev[k * 18:(k + 1) * 18]
            fit = np.setdiff1d(dev, val)
            inner.append({'train_idx': fit.tolist(), 'val_idx': val.tolist()})
        folds.append({'fold': f, 'dev_idx': dev.tolist(), 'test_idx': test.tolist(), 'inner': inner})
    splits = tmp_path / 'splits.json'
    splits.write_text(json.dumps({'folds': folds}))
    cfg = {'model': model, 'data': str(tmp_path / 'wheat.RData'), 'data_sha256': 'x',
           'splits': str(splits), 'seeds': [42, 2026], 'device': 'cpu', 'cpu_threads': 1,
           'gpu_memory_fraction': 0.1,
           'configurations': [{'tag': 'a'}, {'tag': 'b'}],
           'training': {'max_epochs': 2}}
    config_path = tmp_path / 'config.yaml'
    config_path.write_text(json.dumps(cfg))
    data = SimpleNamespace(X=X, Y=Y, n=n, n_env=n_env, environments=('1', '2'))
    monkeypatch.setattr(db, 'load', lambda *a: data)
    monkeypatch.setattr(db, '_validate', lambda cfg: None)
    monkeypatch.setattr(db, 'sha256_of', lambda path: 'hash')

    def fake_inner(cfg, data, encoded, client, fit, val, env, spec, seed, device):
        rng = np.random.default_rng(seed + env + (0 if spec['tag'] == 'a' else 100))
        return {'prediction': data.Y[val, env] * 0.5 + rng.normal(0, .1, len(val)),
                'best_epoch': 1 + (seed % 2), 'best_pcc': 0.5,
                'lr_sequence': [0.01, 0.01], 'seconds': 0.0}

    def fake_refit(cfg, data, encoded, client, dev, test, env, spec, seed, epochs, lr_sequence, device):
        assert epochs in (1, 2) and len(lr_sequence) == epochs
        return data.Y[test, env] * 0.5

    monkeypatch.setattr(db, '_inner_fit', fake_inner)
    monkeypatch.setattr(db, '_refit', fake_refit)
    return cfg, config_path, folds


def test_nested_driver_selection_then_single_scoring(tmp_path, monkeypatch):
    cfg, config_path, folds = _fake_run(tmp_path, monkeypatch)
    dev_out = tmp_path / 'dev'
    db.run(cfg, SimpleNamespace(out=dev_out, selection=None, config=config_path), config_path)
    selection = json.loads((dev_out / 'selection.json').read_text())
    assert len(selection['selected']) == 20  # 10 folds x 2 envs
    assert not (dev_out / 'summary.json').exists()
    for entry in selection['selected']:
        for seed in ('42', '2026'):
            block = entry['seeds'][seed]
            assert block['median_epoch'] == sorted(block['per_inner_best_epochs'])[1]
    final_out = tmp_path / 'final'
    db.run(cfg, SimpleNamespace(out=final_out, selection=dev_out / 'selection.json',
                                config=config_path), config_path)
    summary = json.loads((final_out / 'summary.json').read_text())
    assert summary['n_runs'] == 20 and summary['n_failures'] == 0
    import csv
    oof = list(csv.DictReader((final_out / 'oof.csv').open()))
    assert len(oof) == 2 * 60
    covered = sorted(int(r['material_index']) for r in oof if r['environment'] == '1')
    assert covered == list(range(60))


def test_soydngp_encoding_matches_upstream_rules():
    checkout = Path('/home/hsy/overlays/ws2_upstreams/SoyDNGPNext')
    if not checkout.is_dir():
        pytest.skip('official SoyDNGPNext checkout only exists on omen')
    X = np.array([[0, 1, 0, 1], [1, 1, 0, 0]], dtype=np.float64)
    encoded = db.soydngp_encode(X, checkout)
    assert encoded.shape == (2, 3, 206, 206)
    # marker 0 (DArT 0 -> code 3): channels (0,1,1); marker 1 (code 1): (1,1,0)
    assert encoded[0, 0, 0, 0] == 0 and encoded[0, 1, 0, 0] == 1 and encoded[0, 2, 0, 0] == 1
    assert encoded[0, 0, 0, 1] == 1 and encoded[0, 1, 0, 1] == 1 and encoded[0, 2, 0, 1] == 0
