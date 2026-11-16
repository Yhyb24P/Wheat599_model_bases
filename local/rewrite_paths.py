#!/usr/bin/env python3
"""Rewrite path references in a file using the canonical migration map.

Handles: results dir prefixes (results/<run> -> results/<fam>/<run>), exact src
and config paths, and dotted wheat599 imports -> genopred. Longest-match first to
avoid partial clobber. Used for WP-C (build_report.py, tests) after WP-A/B merge.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
M = json.loads((ROOT / 'docs/PATH_MIGRATION.json').read_text())


def build_replacements():
    reps = {}
    # results dir prefixes (longest first); match results/<run> boundary
    for old, new in M['results'].items():
        reps[old] = new
    # exact src + config paths
    for old, new in M['src'].items():
        reps[old] = new
    for old, new in M['configs'].items():
        reps[old] = new
    # site moves
    for old, new in M['site'].items():
        reps[old] = new
    return reps


def rewrite_text(text):
    reps = build_replacements()
    # path replacements: longest old first, require non-path-char boundary after
    for old in sorted(reps, key=len, reverse=True):
        new = reps[old]
        # replace old when followed by / or quote/space/paren/end (not a longer path segment)
        text = re.sub(re.escape(old) + r'(?=[/\s"\'\)\],;]|$)', new, text)
    # dotted imports: wheat599.<mod> -> genopred.<...>, longest first
    for old in sorted(M['imports'], key=len, reverse=True):
        new = M['imports'][old]
        text = re.sub(re.escape(old) + r'(?=[\s.\'\"]|$)', new, text)
    # bare package references in strings like 'wheat599.' + module dispatch handled manually
    return text


if __name__ == '__main__':
    for arg in sys.argv[1:]:
        p = Path(arg)
        original = p.read_text()
        updated = rewrite_text(original)
        if updated != original:
            p.write_text(updated)
            print(f'rewrote {arg}')
        else:
            print(f'no change {arg}')