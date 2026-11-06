from types import SimpleNamespace
import numpy as np
import pytest
import torch

pytest.importorskip('tabm')
from wheat599.quantile_tabm import train
from wheat599.quantile_tabm_final import fit_one, summarize


def test_outer_zero_epoch_reuses_development_initialization(tmp_path):
    rng = np.random.default_rng(7)
    x, y = rng.normal(size=(32, 8)), rng.normal(size=32)
    c = dict(n_blocks=2, d_block=16, dropout=.1, k=2, learning_rate=.001, weight_decay=.001)
    spec = dict(n_quantiles=5, quantile_min=.001, quantile_max=.999,
        training=dict(max_epochs=0, patience=2, batch_size=8, gradient_clip=5., tie_tolerance=1e-12))
    row = dict(fold=0, environment_index=0, seed=42, config_index=0, epochs=0)
    expected, _ = train(x[:24], y[:24], x[24:], y[24:], c, 42, 0, spec, 'cpu')
    actual, info = fit_one(x[:24], y[:24], x[24:], row, c, spec, 'cpu', tmp_path)
    np.testing.assert_array_equal(expected, actual)
    assert len(info['weight_sha256']) == 64


def test_outer_summary_uses_fixed_equal_weight_and_checks_coverage():
    rng = np.random.default_rng(11)
    data = SimpleNamespace(Y=rng.normal(size=(599,4)), environments=('1','2','4','5'))
    folds = [{'fold':j, 'test_idx':a.tolist()} for j,a in enumerate(np.array_split(np.arange(599),10))]
    huber, mse, preds = {}, {}, {}
    for fold in folds:
        f, ids = fold['fold'], fold['test_idx']
        for e, name in enumerate(data.environments):
            truth = data.Y[ids,e]
            for seed in (42,2026):
                preds[f'f{f}_e{e}_s{seed}'] = truth + .2 + .01*(seed==42)
            for i, material in enumerate(ids):
                huber[f,name,material] = (truth[i], truth[i]-.4)
                mse[f,name,material] = (truth[i], truth[i]+.1)
    result, rows = summarize(data, folds, preds, [42,2026], huber, mse)
    assert len(rows)==2396 and len(result['per_fold'])==40
    assert rows[0]['quantile_huber'] == pytest.approx(.5*(rows[0]['quantile']+rows[0]['huber']))
    del huber[next(iter(huber))]
    with pytest.raises(ValueError, match='Baseline'):
        summarize(data, folds, preds, [42,2026], huber, mse)
