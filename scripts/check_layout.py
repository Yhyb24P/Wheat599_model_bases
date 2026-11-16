"""Organizational-tree fan-out check: every folder <=6 immediate subfolders.

Exemptions (technical/layout necessities, not preferences):
- machine-generated dirs (.git/.state/.cache/.pytest_cache/.qoder/.yi/node_modules/
  __pycache__) regenerate automatically and cannot be constrained;
- the repo root (standard top-level layout) and local attachment areas
  (workspace/, local/);
- run-directory interiors: a dir holding provenance/execution_status/manifest/
  summary is an opaque evidence leaf (its internal layout is frozen bytes).
Run dirs themselves still count toward their parent's fan-out.
"""
import os
import sys
from pathlib import Path

MAX_FANOUT = 6
EXEMPT_NAMES = {'.git', '.state', '.cache', '.pytest_cache', '.qoder', '.yi',
                'node_modules', '__pycache__', 'workspace', 'local'}
RUN_MARKERS = ('provenance.json', 'execution_status.json', 'manifest.json', 'summary.json')


def is_run_dir(path):
    return any((path / m).is_file() for m in RUN_MARKERS)


def violations(root):
    root = Path(root).resolve()
    found = []
    for dirpath, dirnames, _ in os.walk(root):
        p = Path(dirpath)
        dirnames[:] = sorted(d for d in dirnames
                             if d not in EXEMPT_NAMES and not d.startswith('.'))
        if is_run_dir(p):
            dirnames[:] = []  # opaque evidence leaf: do not descend
            continue
        if p == root:
            continue  # repo root exempt from the count
        if len(dirnames) > MAX_FANOUT:
            found.append((str(p.relative_to(root)), len(dirnames)))
    return found


def main(argv=None):
    root = Path(argv[1]) if argv and len(argv) > 1 else Path(__file__).resolve().parents[1]
    found = violations(root)
    if found:
        print(f"fan-out >{MAX_FANOUT} violations:")
        for rel, n in found:
            print(f"  {rel}: {n} subfolders")
        return 1
    print(f"layout OK: every organizational folder <={MAX_FANOUT} subfolders")
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))