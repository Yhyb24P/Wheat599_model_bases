import pytest

from wheat599.grouped_kernel_run import summarize


def test_random_metrics_and_paired_gate_require_a_complete_grid():
    seeds = [101, 202, 303, 404, 505, 606]
    rows = []
    for f in range(10):
        for j in range(3):
            for arm, pcc in [('original', .48), ('correlation', .484)] + [
                    (f'random_{s}', .48 + i*.001) for i, s in enumerate(seeds)]:
                rows.append(dict(fold=f, inner=j, arm=arm,
                                 per_environment=[dict(pcc=pcc, rmse=.9, mae=.7)]*4))
    result = summarize(rows, seeds)
    assert result['stage2_eligible']
    assert result['metrics']['random_mean']['macro']['pcc'] == pytest.approx(.4825)
    assert result['comparisons']['random_mean']['pcc_gain'] == pytest.approx(.0015)
    for row in rows:
        if row['arm'] == 'correlation' and row['fold'] >= 6:
            row['per_environment'] = [dict(pcc=.47, rmse=.9, mae=.7)]*4
    assert not summarize(rows, seeds)['stage2_eligible']
    with pytest.raises(ValueError, match='grid'):
        summarize(rows[:-1], seeds)
    with pytest.raises(ValueError, match='grid'):
        summarize(rows + rows[:1], seeds)
