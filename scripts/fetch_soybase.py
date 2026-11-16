"""Download SoySNP50K genotype + GRIN phenotype to E-drive cold storage.

Idempotent: skips files already present. Verifies the SoyBase MD5 checksums for
genotype files and records SHA-256 of the HapMap and GRIN observations for the
loader. Raw data lives at /mnt/e/WSL-data/soybase (data/raw/soybase symlink),
not tracked by Git. Run: python scripts/fetch_soybase.py [--out DIR].
"""
import argparse
import hashlib
import json
import subprocess
import urllib.request
from pathlib import Path

BASE = 'https://data.soybase.org/Glycine/max/diversity/Wm82.gnm2.div.Song_Hyten_2015'
PRE = 'glyma.Wm82.gnm2.div.Song_Hyten_2015'
GENO_FILES = [f'{PRE}.hmp.gz', f'{PRE}.PIconversion.txt.gz', f'{PRE}.markerMap.txt.gz',
              f'CHECKSUM.Wm82.gnm2.div.Song_Hyten_2015.md5',
              f'MANIFEST.Wm82.gnm2.div.Song_Hyten_2015.yml',
              f'README.Wm82.gnm2.div.Song_Hyten_2015.yml']
GRIN_TRAITS = {'FC': 25, 'P_DENS': 23, 'POD': 24, 'ST': 17}
GRIN_DBIDS = {'FC': 51058, 'P_DENS': 51056, 'POD': 51057, 'ST': 51050}


def sha256(path, chunk=8 * 2 ** 20):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def download(url, dest):
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        print(f'exists {dest.name}')
        return
    print(f'download {dest.name}')
    urllib.request.urlretrieve(url, dest)


def fetch_grin_observations(dest):
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        print(f'exists {dest.name}')
        return
    body = json.dumps({'traitIds': list(GRIN_TRAITS.values()),
                       'traitDbIds': list(GRIN_DBIDS.values()),
                       'cultivarNames': [], 'filters': {}}).encode()
    request = urllib.request.Request('https://grin.soybase.org/observations',
                                     data=body, headers={'Content-Type': 'application/json'})
    print('query GRIN observations API')
    with urllib.request.urlopen(request) as response:
        data = response.read()
    dest.write_bytes(data)


def verify_md5(out):
    checksum = out / f'CHECKSUM.Wm82.gnm2.div.Song_Hyten_2015.md5'
    if not checksum.exists():
        return
    for line in checksum.read_text().splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        digest, name = parts[0], parts[-1].lstrip('*')
        target = out / name
        if not target.exists():
            continue  # only verify files we downloaded
        actual = hashlib.md5(target.read_bytes()).hexdigest()
        if actual != digest:
            raise ValueError(f'MD5 mismatch: {name}')
        print(f'md5 OK {name}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='/mnt/e/WSL-data/soybase')
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in GENO_FILES:
        download(f'{BASE}/{name}', out / name)
    verify_md5(out)
    fetch_grin_observations(out / 'grin_observations.json')
    hmp = out / f'{PRE}.hmp.gz'
    obs = out / 'grin_observations.json'
    source = {
        'hapmap': {'path': str(hmp), 'sha256': sha256(hmp), 'bytes': hmp.stat().st_size},
        'grin_observations': {'path': str(obs), 'sha256': sha256(obs), 'bytes': obs.stat().st_size},
        'reference_assembly': 'Wm82.gnm2', 'dataset': 'Song_Hyten_2015',
        'grin_traits': GRIN_TRAITS, 'grin_dbids': GRIN_DBIDS,
    }
    (out / 'source.json').write_text(json.dumps(source, indent=2))
    print(json.dumps(source, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())