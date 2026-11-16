"""Organizational tree must keep every folder <=6 immediate subfolders."""
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _check_layout():
    spec = importlib.util.spec_from_file_location('check_layout', ROOT / 'scripts' / 'check_layout.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LayoutTest(unittest.TestCase):
    def test_fanout_within_limit(self):
        module = _check_layout()
        found = module.violations(ROOT)
        self.assertEqual(found, [], f"fan-out >{module.MAX_FANOUT}: {found}")

    def test_run_dirs_are_opaque_leaves(self):
        module = _check_layout()
        # a frozen run with >6 internal fold dirs must not be flagged (interior exempt)
        run = ROOT / 'results/wheat599/baselines/comparators/mulphe/mulphe_unified_parallel_20261112'
        if run.exists():
            self.assertTrue(module.is_run_dir(run))


if __name__ == '__main__':
    unittest.main()