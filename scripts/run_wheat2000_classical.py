#!/usr/bin/env python3
"""Run Wheat2000 classical baselines: development selection or frozen final."""
import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from genopred.models.classical.wheat2000_classical import run

parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True)
parser.add_argument('--out', required=True, type=Path)
parser.add_argument('--selection', type=Path)
args = parser.parse_args()
run(yaml.safe_load(Path(args.config).read_text()), args, Path(__file__))
