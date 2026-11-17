"""SoyBase (SoySNP50K USDA germplasm) classification loader and cache preparation.

Genotype: SoyBase Wm82.gnm2.div.Song_Hyten_2015 HapMap (hmp.gz), 20087 accessions
(PI IDs) x SNPs; allele calls are homozygous single letters, IUPAC heterozygote
codes, or N (missing). Dosage = count of the second allele (0/1/2), NaN missing.

Phenotype: GRIN Global observations (grin.soybase.org) for four qualitative
traits, matched to genotype accessions by normalized PI ID. GRIN has more
categories than the benchmark; we filter to the benchmark classes and DROP
accessions whose records conflict within the benchmark classes (documented,
not blind keep-first). Per-trait labels are int-encoded; -1 = unavailable.

Raw data and cache live under data/raw/soybase (E-drive cold storage symlink),
not tracked by Git. SNP call-rate/MAF filtering is fit-set-only and happens in
the experiment, not here; the cache keeps the full matched genotype.
"""
import hashlib
import json
import re
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np

from genopred.data.wheat2000 import file_sha256, _array_sha

# IUPAC ambiguity codes -> the two alleles they represent (heterozygous).
IUPAC_HET = {'R': {'A', 'G'}, 'Y': {'C', 'T'}, 'K': {'G', 'T'},
             'M': {'A', 'C'}, 'S': {'C', 'G'}, 'W': {'A', 'T'}}

TRAITS = ('FC', 'P_DENS', 'POD', 'ST')
TRAIT_NAME = {'FC': 'Flower color', 'P_DENS': 'Pubescence density',
              'POD': 'Pod color', 'ST': 'Stem termination type'}
# Benchmark class labels (BiMSGP soybase_classification.json); GRIN raw values
# outside these sets are excluded from the task.
CLASS_MAPS = {'FC': {'W': 0, 'P': 1},
              'P_DENS': {'Ssp': 0, 'N': 1},
              'POD': {'Tn': 0, 'Br': 1, 'Bl': 2},
              'ST': {'N': 0, 'S': 1, 'D': 2}}


class SoyBaseData(NamedTuple):
    X: np.ndarray            # (n_accessions, n_snps) float32 dosage, NaN missing
    Y: np.ndarray            # (n_accessions, 4) int64 labels, -1 unavailable
    trait_names: tuple
    class_maps: dict
    sample_ids: np.ndarray   # normalized PI IDs
    snp_ids: np.ndarray


def _norm_pi(name):
    return re.sub(r'\s+', '', str(name)).upper()


def parse_hapmap(path):
    """Return (accession_ids, snp_ids, X float32 dosage matrix).

    Vectorized per-SNP encoding via pandas; monomorphic single-allele SNPs
    (no '/' in the alleles column) are dropped as non-informative.
    """
    import pandas as pd
    het_by_pair = {frozenset(v): k for k, v in IUPAC_HET.items()}
    frame = pd.read_csv(path, sep='\t', compression='gzip', dtype=str)
    accession_ids = [_norm_pi(c) for c in frame.columns[11:]]
    keep = frame['alleles'].str.contains('/', na=False).to_numpy()
    snp_ids = frame.loc[keep, 'rs#'].tolist()
    alleles = frame.loc[keep, 'alleles'].to_numpy()
    calls = frame.iloc[:, 11:].to_numpy(dtype=str)[keep]
    if calls.shape[1] != len(accession_ids):
        raise ValueError('HapMap accession column count mismatch')
    X = np.full(calls.shape, np.nan, dtype=np.float32)
    for i, al in enumerate(alleles):
        a1, a2 = al.split('/')
        row = calls[i]
        X[i, row == a1] = 0.0
        X[i, row == a2] = 2.0
        het = het_by_pair.get(frozenset((a1, a2)))
        if het is not None:
            X[i, row == het] = 1.0
    return accession_ids, snp_ids, np.ascontiguousarray(X.T)


def parse_phenotype(obs_path, accession_ids):
    """Match GRIN observations to accessions; filter benchmark classes; drop conflicts.

    Returns (Y int64 (n,4) with -1 unavailable, stats dict).
    """
    observations = json.loads(Path(obs_path).read_text())
    index = {pi: i for i, pi in enumerate(accession_ids)}
    # collect benchmark-class values per accession per trait
    values = {t: {} for t in TRAITS}
    raw_counts = {t: 0 for t in TRAITS}
    for rec in observations:
        trait = next((t for t in TRAITS if TRAIT_NAME[t] == rec.get('trait_name')), None)
        if trait is None:
            continue
        pi = _norm_pi(rec.get('cultivar_name', ''))
        if pi not in index:
            continue
        raw_counts[trait] += 1
        val = rec.get('value')
        if val in CLASS_MAPS[trait]:
            values[trait].setdefault(pi, set()).add(val)
    n = len(accession_ids)
    Y = np.full((n, len(TRAITS)), -1, dtype=np.int64)
    stats = {}
    for j, trait in enumerate(TRAITS):
        kept = conflict = 0
        for pi, vs in values[trait].items():
            i = index[pi]
            if len(vs) == 1:
                Y[i, j] = CLASS_MAPS[trait][next(iter(vs))]
                kept += 1
            else:
                conflict += 1  # conflicting benchmark records -> drop (-1)
        stats[trait] = {'raw_matched_records': raw_counts[trait],
                        'labeled_accessions': kept, 'conflict_dropped': conflict,
                        'class_counts': {c: int((Y[:, j] == k).sum())
                                         for c, k in CLASS_MAPS[trait].items()}}
    return Y, stats


def prepare(hmp_path, hmp_sha, obs_path, obs_sha, out_npz, *, metadata_path=None,
            overwrite=False):
    """Build verified NPZ cache (full matched genotype + per-trait labels)."""
    hmp_path, obs_path, out_npz = Path(hmp_path), Path(obs_path), Path(out_npz)
    metadata_path = Path(metadata_path) if metadata_path is not None else \
        out_npz.parent / (out_npz.name + '.metadata.json')
    if not hmp_path.is_file() or not obs_path.is_file():
        raise FileNotFoundError('missing SoyBase raw input')
    if file_sha256(hmp_path) != str(hmp_sha):
        raise ValueError('SoyBase HapMap sha256 mismatch')
    if file_sha256(obs_path) != str(obs_sha):
        raise ValueError('SoyBase GRIN observations sha256 mismatch')
    if out_npz.exists() and not overwrite:
        raise FileExistsError(out_npz)
    if metadata_path.exists() and not overwrite:
        raise FileExistsError(metadata_path)
    accession_ids, snp_ids, X = parse_hapmap(hmp_path)
    if len(set(accession_ids)) != len(accession_ids):
        raise ValueError('duplicate genotype accession IDs')
    Y, pheno_stats = parse_phenotype(obs_path, accession_ids)
    if X.shape[0] != len(accession_ids) or Y.shape[0] != len(accession_ids):
        raise ValueError('genotype/phenotype accession alignment mismatch')
    if not np.isin(X[~np.isnan(X)], (0.0, 1.0, 2.0)).all():
        raise ValueError('dosage not in {0,1,2}')
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    sample_ids = np.asarray(accession_ids, dtype=np.str_)
    snp_arr = np.asarray(snp_ids, dtype=np.str_)
    trait_arr = np.asarray(TRAITS, dtype=np.str_)
    np.savez_compressed(out_npz, X=X, Y=Y, trait_names=trait_arr,
                        sample_ids=sample_ids, snp_ids=snp_arr)
    metadata = {
        'schema_version': 1,
        'source': {
            'hapmap': {'path': str(hmp_path), 'sha256': str(hmp_sha),
                       'bytes': hmp_path.stat().st_size},
            'grin_observations': {'path': str(obs_path), 'sha256': str(obs_sha),
                                  'bytes': obs_path.stat().st_size},
            'reference_assembly': 'Wm82.gnm2', 'dataset': 'Song_Hyten_2015',
            'genotype_doi': '10.1534/g3.115.019000',
            'phenotype_source': 'https://grin.soybase.org (GRIN Global)',
        },
        'cache': {'path': str(out_npz), 'sha256': file_sha256(out_npz),
                  'bytes': out_npz.stat().st_size},
        'schema': {'accessions': len(accession_ids), 'snps': len(snp_ids),
                   'traits': list(TRAITS), 'class_maps': CLASS_MAPS,
                   'dosage': 'count of second allele; NaN missing',
                   'label_unavailable': -1,
                   'conflict_policy': 'drop accessions with conflicting benchmark-class records per trait'},
        'arrays': {
            'X': {'shape': list(X.shape), 'dtype': 'float32', 'sha256': _array_sha(X)},
            'Y': {'shape': list(Y.shape), 'dtype': 'int64', 'sha256': _array_sha(Y)},
        },
        'phenotype_stats': pheno_stats,
        'snp_filtering': 'fit-set-only call-rate/MAF applied in experiment, not in cache',
        'created_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))
    return metadata


def load_soybase(hmp_path, hmp_sha, obs_path, obs_sha, cache_path, cache_sha):
    """Load and verify the prepared cache; all SHAs required."""
    for path, sha in ((hmp_path, hmp_sha), (obs_path, obs_sha), (cache_path, cache_sha)):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        if file_sha256(path) != str(sha):
            raise ValueError(f'SoyBase sha256 mismatch: {path}')
    with np.load(cache_path, allow_pickle=False) as archive:
        required = {'X', 'Y', 'trait_names', 'sample_ids', 'snp_ids'}
        if set(archive.files) != required:
            raise ValueError('SoyBase cache has unexpected arrays')
        X = np.asarray(archive['X'], dtype=np.float32)
        Y = np.asarray(archive['Y'], dtype=np.int64)
        trait_names = tuple(str(v) for v in archive['trait_names'])
        sample_ids = np.asarray(archive['sample_ids']).astype(str)
        snp_ids = np.asarray(archive['snp_ids']).astype(str)
    if X.ndim != 2 or Y.ndim != 2 or X.shape[0] != Y.shape[0]:
        raise ValueError('SoyBase cache X/Y shapes invalid')
    if trait_names != TRAITS:
        raise ValueError('SoyBase cache trait names mismatch')
    if X.shape[0] != sample_ids.size or X.shape[1] != snp_ids.size:
        raise ValueError('SoyBase cache ID lengths disagree')
    if not np.isin(X[~np.isnan(X)], (0, 1, 2)).all():
        raise ValueError('SoyBase dosage not in {0,1,2}')
    if not np.isin(Y, (-1, 0, 1, 2)).all():
        raise ValueError('SoyBase labels invalid')
    return SoyBaseData(X, Y, trait_names, CLASS_MAPS, sample_ids, snp_ids)


def load_chromosomes(marker_map_path, snp_ids):
    """Per-SNP chromosome labels from the official markerMap (verified order).

    Gm01..Gm20 keep their names; every scaffold_* maps to one 'scaffold' group
    (unplaced markers, ~1% of SNPs). Raises if any cached SNP is unmapped.
    """
    import gzip
    chrom = {}
    with gzip.open(marker_map_path, 'rt') as handle:
        for line in handle:
            parts = line.split()
            chrom[parts[0]] = parts[1]
    labels = []
    for snp in snp_ids:
        if snp not in chrom:
            raise ValueError(f'SNP {snp} missing from markerMap')
        name = chrom[snp]
        labels.append(name if name.startswith('Gm') else 'scaffold')
    return np.asarray(labels)


def make_stratified_splits(data, trait, n_splits=10, seed=42, inner_splits=3):
    """StratifiedKFold outer/inner splits over accessions with a valid label for trait.

    Returns (folds, info). Indices are global rows into data.X/Y. Group integrity
    is preserved; per-fold class support is recorded (minority classes may be
    unevenly supported across folds, reported not hidden).
    """
    from sklearn.model_selection import StratifiedKFold
    if trait not in TRAITS:
        raise ValueError(f'unknown trait {trait}')
    j = TRAITS.index(trait)
    labels = data.Y[:, j]
    idx = np.flatnonzero(labels >= 0)
    y = labels[idx]
    n_classes = len(CLASS_MAPS[trait])
    if len(idx) < n_splits * n_classes:
        raise ValueError('too few labeled accessions for stratified splits')
    folds = []
    support = []
    outer = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (dev_local, test_local) in enumerate(outer.split(idx, y)):
        dev_idx = idx[dev_local]
        test_idx = idx[test_local]
        inner = []
        inner_skf = StratifiedKFold(n_splits=inner_splits, shuffle=True,
                                    random_state=seed + fold)
        for train_local, val_local in inner_skf.split(dev_idx, labels[dev_idx]):
            inner.append({'train_idx': dev_idx[train_local].tolist(),
                          'val_idx': dev_idx[val_local].tolist()})
        folds.append({'fold': fold, 'dev_idx': dev_idx.tolist(),
                      'test_idx': test_idx.tolist(), 'inner': inner})
        support.append({str(c): int((labels[test_idx] == c).sum()) for c in range(n_classes)})
    covered = sorted(v for f in folds for v in f['test_idx'])
    if covered != idx.tolist():
        raise ValueError('outer test folds do not cover labeled accessions once')
    info = {'trait': trait, 'n_labeled': int(len(idx)), 'n_classes': n_classes,
            'class_maps': CLASS_MAPS[trait], 'test_support_per_fold': support}
    return folds, info
