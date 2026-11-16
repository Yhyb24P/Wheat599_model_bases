#!/usr/bin/env python3
"""Development-only CLI for the quantile TabM module."""
import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from genopred.models.neural import quantile_tabm

parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True)
parser.add_argument('--out', required=True, type=Path)
args = parser.parse_args()
args.selection = None
quantile_tabm.run(yaml.safe_load(Path(args.config).read_text()), args, Path(__file__))
