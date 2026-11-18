"""Fetch GRIN quantitative observations for the SoyBase seven-trait regression.

Reuses the SoySNP50K genotype already downloaded by fetch_soybase.py. Queries
the GRIN observations API for the seven regression traits and records SHA-256.
Raw data lives at /mnt/e/WSL-data/soybase (data/raw/soybase symlink), not in Git.
Run: python scripts/fetch_soybase_regression.py [--out DIR].
"""
import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

# GRIN trait ids / db ids for the seven quantitative regression traits.
REG_TRAITS = {'Hgt': 1, 'R1': 2, 'R8': 3, 'SdWgt': 6, 'Yield': 9, 'oil': 8, 'protein': 7}
REG_DBIDS = {'Hgt': 51009, 'R1': 51010, 'R8': 51011, 'SdWgt': 51015,
             'Yield': 51020, 'oil': 51019, 'protein': 51016}


def sha256(path, chunk=8 * 2 ** 20):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='/mnt/e/WSL-data/soybase')
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dest = out / 'grin_regression_observations.json'
    if dest.exists() and dest.stat().st_size > 0:
        print(f'exists {dest.name}')
    else:
        body = json.dumps({'traitIds': list(REG_TRAITS.values()),
                           'traitDbIds': list(REG_DBIDS.values()),
                           'cultivarNames': [], 'filters': {}}).encode()
        request = urllib.request.Request('https://grin.soybase.org/observations',
                                         data=body, headers={'Content-Type': 'application/json',
                                                             'User-Agent': 'Mozilla/5.0 (research)'})
        print('query GRIN regression observations API')
        with urllib.request.urlopen(request) as response:
            dest.write_bytes(response.read())
    source = {
        'grin_regression_observations': {'path': str(dest), 'sha256': sha256(dest),
                                         'bytes': dest.stat().st_size},
        'reg_traits': REG_TRAITS, 'reg_dbids': REG_DBIDS,
    }
    (out / 'source_regression.json').write_text(json.dumps(source, indent=2))
    print(json.dumps(source, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())