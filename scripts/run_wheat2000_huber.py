#!/usr/bin/env python
"""Drive the Wheat2000 Huber ICM phases."""
import argparse
import sys
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
from genopred.models.kernels.wheat2000_huber import run  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--phase', choices=['dev', 'final', 'report'], required=True)
    parser.add_argument('--selection', type=Path)
    parser.add_argument('--folds', nargs='*', type=int)
    parser.add_argument('--traits', nargs='*')
    args = parser.parse_args()
    return run(yaml.safe_load(args.config.read_text()), args, Path(__file__))


if __name__ == '__main__':
    main()
