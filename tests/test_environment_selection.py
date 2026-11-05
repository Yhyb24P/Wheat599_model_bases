from types import SimpleNamespace

import numpy as np
import pytest

from wheat599.environment_selection import select_indices, compute, read_predictions


def test_selects_across_inner_not_separately_with_environment_inequality():
    scores = np.array([[[1., 0.], [0., 1.]],
                       [[1., 0.], [0., 1.]],
                       [[0., .8], [1., 0.]]])
    common, separate = select_indices(scores)
    assert common == [1, 1]
    assert separate == [0, 1]
    means = scores.mean(0)
    assert np.mean([means[c, e] for e, c in enumerate(separate)]) > means[1].mean()
    with pytest.raises(ValueError):
        select_indices(np.full((3, 2, 2), np.nan))


def test_complete_frozen_grid_and_missing_prediction():
    y = np.random.default_rng(42).normal(size=(10, 2))
    folds = [{'fold': 0, 'dev_idx': list(range(9)), 'test_idx': [9],
              'inner': [{'val_idx': list(range(j*3, (j+1)*3)),
                         'train_idx': [i for i in range(9) if i//3 != j]}
                        for j in range(3)]}]
    bank = {(0, j, c, seed, i, e): y[i, e]
            for j in range(3) for c in range(2) for seed in [42, 2026]
            for i in folds[0]['inner'][j]['val_idx'] for e in range(2)}
    summary, rows, choices = compute(SimpleNamespace(Y=y), folds,
                                      {'shared': bank, 'independent': dict(bank)},
                                      [42, 2026], 2)
    assert len(summary) == len(rows) == len(choices) == 4
    for value in summary.values():
        assert value['ensemble']['pcc'] == pytest.approx(1.)
        assert value['ensemble']['rmse'] == 0
    bank.pop(next(iter(bank)))
    with pytest.raises(ValueError, match='Missing'):
        compute(SimpleNamespace(Y=y), folds, {'shared': bank}, [42, 2026], 2)


def test_long_csv_identity_and_duplicate_rejection(tmp_path):
    path = tmp_path / 'predictions.csv'
    header = 'variant,fold,inner,config_index,seed,material_index,environment,y_pred\n'
    row = 'independent,0,1,2,42,17,4,.5\n'
    path.write_text(header + row)
    assert read_predictions(path, 'independent', ['1', '2', '4', '5']) == {(0, 1, 2, 42, 17, 2): .5}
    path.write_text(header + row + row)
    with pytest.raises(ValueError, match='Duplicate'):
        read_predictions(path, 'independent', ['1', '2', '4', '5'])
