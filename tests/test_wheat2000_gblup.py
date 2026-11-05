import numpy as np

from wheat599.wheat2000_gblup import predict_grid, raw_marker_kernel


def test_centered_kernel_matches_primal_marker_ridge():
    rng = np.random.default_rng(21)
    x = rng.integers(0, 2, size=(30, 7)).astype(float)
    y = rng.normal(size=(30, 3))
    fit, query = np.arange(23), np.arange(23, 30)
    lambdas = [0.001, 0.1, 10.]
    pred = predict_grid(raw_marker_kernel(x), y, fit, query, lambdas)
    z = x - x[fit].mean(0)
    for j, ridge in enumerate(lambdas):
        beta = np.linalg.solve(z[fit].T @ z[fit] + ridge * x.shape[1] * np.eye(x.shape[1]),
                               z[fit].T @ (y[fit] - y[fit].mean(0)))
        expected = z[query] @ beta + y[fit].mean(0)
        np.testing.assert_allclose(pred[j], expected, atol=1e-11, rtol=1e-11)


def test_query_phenotypes_are_never_read():
    rng = np.random.default_rng(22)
    x, y = rng.normal(size=(20, 6)), rng.normal(size=(20, 2))
    fit, query = np.arange(15), np.arange(15, 20)
    kernel = raw_marker_kernel(x)
    before = predict_grid(kernel, y, fit, query, [0.1])
    y[query] = np.nan
    after = predict_grid(kernel, y, fit, query, [0.1])
    np.testing.assert_array_equal(before, after)


def test_development_then_frozen_final_all_eight_traits(tmp_path, monkeypatch):
    import json
    from pathlib import Path
    from types import SimpleNamespace
    from wheat599 import wheat2000_data, wheat2000_gblup, independent_final
    from wheat599.data import sha256_of
    rng = np.random.default_rng(25)
    traits = ('TKW', 'TW', 'GL', 'GW', 'GH', 'GP', 'SDS', 'PHT')
    data = wheat2000_data.Wheat2000Data(
        rng.integers(0, 2, size=(40, 7)).astype(np.float32), rng.normal(size=(40, 8)),
        traits, np.arange(40), np.asarray([f'line{i}' for i in range(40)]),
        np.asarray([str(i * 2) for i in range(40)]))
    splits = tmp_path / 'splits.json'
    splits.write_text(json.dumps({'trait_names': list(traits),
                                 'folds': wheat2000_data.make_splits(data, n_splits=2)}))
    config = {'data': 'unused', 'data_sha256': 'synthetic', 'cache': 'unused',
              'cache_sha256': 'synthetic', 'splits': str(splits),
              'splits_sha256': sha256_of(splits), 'traits': list(traits),
              'ridge_logspace': {'start': -4, 'stop': 2, 'num': 5},
              'cpu_threads': 1, 'tie_tolerance': 1e-12}
    cfg_path = tmp_path / 'config.json'
    cfg_path.write_text(json.dumps(config))
    monkeypatch.setattr(wheat2000_data, 'load_wheat2000', lambda *a: data)
    # Temporary test fixtures cannot be committed; production final always requires it.
    monkeypatch.setattr(independent_final, 'ensure_frozen_manifest', lambda p: None)
    dev = tmp_path / 'development'
    wheat2000_gblup.run(config, SimpleNamespace(out=dev, config=cfg_path, selection=None),
                       Path(__file__))
    assert not (dev / 'oof.csv').exists()
    assert len(json.loads((dev / 'selection.json').read_text())['selected']) == 16
    final = tmp_path / 'final'
    wheat2000_gblup.run(config, SimpleNamespace(out=final, config=cfg_path,
                                              selection=dev / 'selection.json'), Path(__file__))
    summary = json.loads((final / 'summary.json').read_text())
    assert summary['n_trait_folds'] == 16 and summary['oof_rows'] == 320
    assert set(summary['per_trait']) == set(traits)
    assert json.loads((final / 'execution_status.json').read_text())['status'] == 'complete'
    assert json.loads((final / 'failures.json').read_text()) == []
