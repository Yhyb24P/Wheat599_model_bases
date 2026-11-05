import pytest

from wheat599.grouped_weights_run import summarize


def test_neural_gain_requires_both_controls_and_strong_reference():
    rows = [dict(fold=f, inner=j, arm=arm, metrics=[[score, .9, .7]]*4)
            for f in range(10) for j in range(3)
            for arm, score in [('identity', .483), ('scalar', .484), ('network', .487)]]
    result = summarize(rows)
    assert result['mechanism_pass'] and not result['outer_eligible']
    for r in rows:
        if r['arm'] == 'network':
            r['metrics'] = [[.494, .9, .7]]*4
    assert summarize(rows)['outer_eligible']
    for r in rows:
        if r['arm'] == 'scalar':
            r['metrics'] = [[.495, .9, .7]]*4
    assert not summarize(rows)['mechanism_pass']
    with pytest.raises(ValueError, match='comparison'):
        summarize(rows[:-1])
