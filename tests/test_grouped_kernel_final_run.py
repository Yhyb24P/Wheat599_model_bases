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
