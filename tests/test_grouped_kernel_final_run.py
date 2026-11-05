import argparse
from pathlib import Path

import pytest

from wheat599 import grouped_kernel_final_run as runner


def args(selection=None):
    return argparse.Namespace(selection=selection, out=Path('/tmp/never'),
                              config=Path('/tmp/never-config.yaml'))


def test_selection_is_rejected():
    with pytest.raises(ValueError, match='frozen outer evaluation'):
        runner.run({'tau': .25, 'groups': 64}, args(selection=Path('/tmp/sel.json')),
                   Path('/tmp/runner.py'))


def test_configuration_drift_is_rejected_before_data_access():
    with pytest.raises(ValueError, match='fixed grouped protocol'):
        runner.run({'tau': .5, 'groups': 64}, args(), Path('/tmp/runner.py'))
    with pytest.raises(ValueError, match='fixed grouped protocol'):
        runner.run({'tau': .25, 'groups': 63}, args(), Path('/tmp/runner.py'))


def test_summarize_aggregates_folds_and_environments():
    rows = []
    for f in range(10):
        for arm in ('original', 'grouped'):
            shift = .1 if arm == 'grouped' else 0.
            rows.append({'fold': f, 'arm': arm,
                         'per_environment': [dict(pcc=shift + .01 * f + .001 * e,
                                                  rmse=1. + e, mae=.5 + e) for e in range(4)]})
    summary = runner.summarize(rows, expected_original_macro=.0465)
    assert summary['completed_fits'] == 20
    assert summary['metrics']['original']['macro']['pcc'] == pytest.approx(.0465)
    assert summary['metrics']['grouped']['macro']['pcc'] == pytest.approx(.1465)
    assert summary['metrics']['original']['per_environment'][2][0] == pytest.approx(.047)
    assert summary['metrics']['original']['fold_macro_pcc'][7] == pytest.approx(.0715)
    assert summary['comparison']['pcc_gain'] == pytest.approx(.1)
    assert summary['comparison']['fold_wins'] == 10
