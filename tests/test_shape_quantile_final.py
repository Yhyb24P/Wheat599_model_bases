import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from genopred.data.wheat599 import sha256_of
from genopred.models.neural.shape_quantile_final import manifest, reference


def test_manifest_uses_each_seed_median_for_selected_config():
    folds = [{'fold': f} for f in range(10)]
    seeds, configs = [42, 2026], [{} for _ in range(4)]
    summary = {'folds': [{'fold': f, 'selected': [0, 1, 2, 3]} for f in range(10)]}
    records = [dict(fold=f, inner=j, environment=e, config_index=c, seed=s,
                    best_epoch=[2, 9, 5][j] + (s == 2026))
               for f in range(10) for j in range(3) for e in range(4)
               for c in range(4) for s in seeds]
    rows = manifest(folds, seeds, configs, summary, records, max_epochs=20)
    assert len(rows) == 80
    assert rows[:2] == [(0, 0, 0, 42, 5), (0, 0, 0, 2026, 6)]
    with pytest.raises(ValueError, match='Incomplete development'):
        manifest(folds, seeds, configs, summary, records[:-1], max_epochs=20)


def test_reference_maps_models_and_rejects_duplicate_material(tmp_path):
    path = tmp_path / 'oof.csv'
    row = dict(fold=0, environment='4', material_index=6, y_true=.2,
               quantile=.3, quantile_huber=.4, huber=.5, mse_tabm=.6)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=row); writer.writeheader(); writer.writerow(row)
    original, huber, mse = reference(path)
    assert original[(0, '4', 6)] == (.2, .3, .4)
    assert huber[(0, '4', 6)] == (.2, .5) and mse[(0, '4', 6)] == (.2, .6)
    with path.open('a', newline='') as handle:
        csv.DictWriter(handle, fieldnames=row).writerow(row)
    with pytest.raises(ValueError, match='Duplicate'):
        reference(path)


def test_outer_run_scores_only_after_all_fixed_fits(tmp_path, monkeypatch):
    from genopred.models.neural import shape_quantile_final as final
    n = 599
    x = np.arange(n, dtype=float)[:, None]
    y = np.column_stack([x[:, 0] / 100 + e for e in range(4)])
    names = ('1', '2', '4', '5')
    folds = []
    for f, test in enumerate(np.array_split(np.arange(n), 10)):
        dev = np.setdiff1d(np.arange(n), test)
        inner = [dict(val_idx=dev[j::3].tolist(), train_idx=np.setdiff1d(dev, dev[j::3]).tolist()) for j in range(3)]
        folds.append(dict(fold=f, dev_idx=dev.tolist(), test_idx=test.tolist(), inner=inner))
    splits, arch, config_file = (tmp_path/name for name in ('splits.json', 'arch.yaml', 'config.yaml'))
    splits.write_text(json.dumps({'folds': folds}))
    arch.write_text(yaml.safe_dump({'configurations': [{}]}))
    cfg = dict(data='unused', data_sha256='unused', splits=str(splits), architecture_config=str(arch),
               optimizer_variant='shape_midpoint', seeds=[42, 2026], workers=1, device='cpu', cpu_threads=1,
               training={'max_epochs':10})
    config_file.write_text(yaml.safe_dump(cfg))
    source = tmp_path/'dev'; source.mkdir()
    records = [dict(fold=f, inner=j, environment=e, config_index=0, seed=s, best_epoch=[1,3,2][j])
               for f in range(10) for j in range(3) for e in range(4) for s in cfg['seeds']]
    (source/'trajectories.json').write_text(json.dumps(records))
    (source/'summary.json').write_text(json.dumps({'folds':[{'fold':f,'selected':[0]*4} for f in range(10)]}))
    (source/'execution_status.json').write_text(json.dumps(dict(status='complete', completed=240, expected=240)))
    (source/'failures.json').write_text('[]')
    paths = [config_file, arch, splits, Path('src/genopred/models/neural/quantile_tabm.py')]
    (source/'provenance.json').write_text(json.dumps({'config':cfg,'sha256':{str(p.resolve() if str(p).startswith('src/') else p):sha256_of(p) for p in paths}}))
    data = SimpleNamespace(X=x, Y=y, environments=names)
    monkeypatch.setattr(final, 'load', lambda *_: data)
    called = []
    def fake_fit(task):
        f, e, _, seed, epochs = task
        assert epochs == 2
        called.append(task)
        pred = x[folds[f]['test_idx'], 0] / 100 + e + (.1 if seed == 42 else -.1)
        return task, pred, {'best_pcc': None, '_distribution': np.repeat(pred[:, None, None], 3, axis=2)}
    class Pool:
        def __init__(self, **_): pass
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def map(self, fn, tasks, **_): return map(fn, tasks)
    def fake_reference(_):
        assert len(called) == 80
        original, huber, mse = {}, {}, {}
        for fold in folds:
            for e, name in enumerate(names):
                for i in fold['test_idx']:
                    key, truth = (fold['fold'], name, i), y[i, e]
                    original[key] = (truth, truth, truth)
                    huber[key] = mse[key] = (truth, truth)
        return original, huber, mse
    monkeypatch.setattr(final, 'ProcessPoolExecutor', Pool)
    monkeypatch.setattr(final, '_fit_outer', fake_fit)
    monkeypatch.setattr(final, 'reference', fake_reference)
    real_hash = final.sha256_of
    monkeypatch.setattr(final, 'sha256_of', lambda p: '0'*64 if Path(p) == Path('results/wheat599/neural/quantile_tabm_final_20261106/oof.csv') else real_hash(Path(p)))
    out = tmp_path/'out'
    final.run(cfg, SimpleNamespace(selection=source/'summary.json', out=out, config=config_file), __file__)
    result = json.loads((out/'summary.json').read_text())
    assert result['macro']['quantile']['pcc'] == pytest.approx(1)
    assert result['macro']['quantile']['rmse'] == pytest.approx(0, abs=1e-12)
    assert result['probabilistic']['macro'][1] == pytest.approx(1)
    rows = list(csv.DictReader((out/'oof.csv').open()))
    assert len(rows) == 599*4
    for row in rows:
        assert float(row['upper90']) - float(row['lower90']) == pytest.approx(float(row['width90']))
        assert float(row['lower90']) <= float(row['y_true']) <= float(row['upper90'])
