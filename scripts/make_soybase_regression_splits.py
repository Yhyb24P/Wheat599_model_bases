"""Freeze the material-level joint holdout splits for SoyBase regression.

Run: PYTHONPATH=src python scripts/make_soybase_regression_splits.py
"""
import json
from pathlib import Path

from genopred.data import soybase
from genopred.data.wheat2000 import file_sha256

RAW = Path('data/raw/soybase')
OUT = Path('protocols/soybase/regression_splits.json')


def main():
    meta = json.loads((RAW / 'soybase_regression_cache.npz.metadata.json').read_text())
    data = soybase.load_soybase_regression(RAW / 'soybase_regression_cache.npz',
                                           meta['cache']['sha256'])
    folds, info = soybase.make_regression_splits(data, n_splits=10, seed=42, inner_splits=3)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        raise FileExistsError(OUT)
    document = {'source': 'SoyBase SoySNP50K + GRIN quantitative observations',
                'cache_sha256': meta['cache']['sha256'],
                'trait_names': list(soybase.REG_TRAITS), **info, 'folds': folds}
    OUT.write_text(json.dumps(document, indent=2))
    print(json.dumps({'splits': str(OUT), 'sha256': file_sha256(OUT), **info}, indent=2))


if __name__ == '__main__':
    main()
