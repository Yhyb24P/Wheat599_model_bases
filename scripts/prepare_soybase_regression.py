"""Build the SoyBase seven-trait regression cache from GRIN quantitative obs.

Reuses the classification cache genotype (X/sample_ids/snp_ids) and parses the
regression observations into a float64 Y (n,7) with NaN unavailable. Writes a
verified NPZ + metadata SHA. Run: python scripts/prepare_soybase_regression.py.
"""
import json
from pathlib import Path

import numpy as np

from genopred.data import soybase
from genopred.data.wheat2000 import file_sha256, _array_sha

RAW = Path('data/raw/soybase')


def main():
    cls_cache = RAW / 'soybase_cache.npz'
    reg_obs = RAW / 'grin_regression_observations.json'
    src = json.loads((RAW / 'source_regression.json').read_text())
    if file_sha256(reg_obs) != src['grin_regression_observations']['sha256']:
        raise ValueError('regression observations SHA mismatch')
    with np.load(cls_cache, allow_pickle=False) as a:
        X = np.asarray(a['X'], dtype=np.float32)
        sample_ids = np.asarray(a['sample_ids']).astype(str)
        snp_ids = np.asarray(a['snp_ids']).astype(str)
    Y, stats = soybase.parse_regression_phenotype(reg_obs, sample_ids)
    out = RAW / 'soybase_regression_cache.npz'
    np.savez_compressed(out, X=X, Y=Y, trait_names=np.asarray(soybase.REG_TRAITS, dtype=np.str_),
                        sample_ids=sample_ids.astype(np.str_), snp_ids=snp_ids.astype(np.str_))
    meta = {
        'schema_version': 1,
        'task': 'regression',
        'source': {'classification_cache': {'path': str(cls_cache), 'sha256': file_sha256(cls_cache)},
                   'grin_regression_observations': src['grin_regression_observations']},
        'cache': {'path': str(out), 'sha256': file_sha256(out), 'bytes': out.stat().st_size},
        'schema': {'accessions': int(X.shape[0]), 'snps': int(X.shape[1]),
                   'traits': list(soybase.REG_TRAITS), 'reg_grin_name': soybase.REG_GRIN_NAME,
                   'label_unavailable': 'NaN', 'replicates': 'averaged per accession/trait'},
        'arrays': {'X': {'shape': list(X.shape), 'dtype': 'float32', 'sha256': _array_sha(X)},
                   'Y': {'shape': list(Y.shape), 'dtype': 'float64', 'sha256': _array_sha(Y)}},
        'phenotype_stats': stats,
    }
    (RAW / 'soybase_regression_cache.npz.metadata.json').write_text(json.dumps(meta, indent=2))
    print(json.dumps({'cache_sha256': meta['cache']['sha256'], 'phenotype_stats': stats}, indent=2))


if __name__ == '__main__':
    main()