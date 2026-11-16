"""Wheat2000 (DeepGS public RData) loader, cache preparation and split helper.

The public RData has two row-aligned but separately identified matrices: ``X`` with
line-ID rownames and ``pheno`` with a different original row-index set. They are kept
positional and both ID vectors are preserved; no join or set intersection is used.

The cache is produced by an R export in little-endian column-major order. R also
writes a row-major copy of both matrices; Python reconstructs both and compares
every element, and compares R/Python file digests before writing the NPZ. The raw
RData and cache stay under ``data/raw`` and are not tracked by Git.
"""
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np
from sklearn.model_selection import KFold

R_SCRIPT = r"""
raw <- Sys.getenv("WHEAT2000_RAW")
out <- Sys.getenv("WHEAT2000_TMP")
load(raw)
stopifnot(exists("X"), exists("pheno"))
X <- as.matrix(X)
pheno <- as.matrix(pheno)
storage.mode(X) <- "double"
storage.mode(pheno) <- "double"
if (is.null(rownames(X)) || is.null(rownames(pheno)) ||
    is.null(colnames(pheno)) || anyDuplicated(rownames(X)) != 0L ||
    anyDuplicated(rownames(pheno)) != 0L) {
  stop("Wheat2000 X/pheno must have unique rownames and pheno colnames")
}
writeLines(as.character(c(nrow(X), ncol(X), nrow(pheno), ncol(pheno))),
           file.path(out, "dims.txt"))
writeLines(colnames(pheno), file.path(out, "traits.txt"))
writeLines(rownames(X), file.path(out, "genotype_ids.txt"))
writeLines(rownames(pheno), file.path(out, "phenotype_ids.txt"))
writeBin(as.double(X), file.path(out, "X_col.f64"), size = 8, endian = "little")
writeBin(as.double(t(X)), file.path(out, "X_row.f64"), size = 8, endian = "little")
writeBin(as.double(pheno), file.path(out, "Y_col.f64"), size = 8, endian = "little")
writeBin(as.double(t(pheno)), file.path(out, "Y_row.f64"), size = 8,
         endian = "little")
files <- file.path(out, c("X_col.f64", "X_row.f64", "Y_col.f64", "Y_row.f64"))
md5 <- tools::md5sum(files)
writeLines(paste(basename(names(md5)), as.character(md5), sep = "\t"),
           file.path(out, "md5.txt"))
cat("R_EXPORT_OK\n")
"""


class Wheat2000Data(NamedTuple):
    X: np.ndarray
    Y: np.ndarray
    trait_names: tuple
    sample_ids: np.ndarray
    genotype_ids: np.ndarray
    phenotype_ids: np.ndarray


def file_sha256(path, chunk=8 * 2 ** 20):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def file_md5(path, chunk=8 * 2 ** 20):
    digest = hashlib.md5()
    with Path(path).open('rb') as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _read_lines(path):
    return Path(path).read_text().splitlines()


def _read_checksums(path):
    values = {}
    for line in _read_lines(path):
        name, digest = line.split('\t')
        values[name] = digest
    return values


def _array_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def prepare(raw_path, raw_sha, out_npz, *, metadata_path=None, overwrite=False,
            rscript='Rscript'):
    """Export the RData to a verified NPZ and return JSON-safe provenance."""
    raw_path = Path(raw_path)
    out_npz = Path(out_npz)
    metadata_path = Path(metadata_path) if metadata_path is not None else \
        out_npz.parent / (out_npz.name + '.metadata.json')
    if not raw_path.is_file():
        raise FileNotFoundError(raw_path)
    if file_sha256(raw_path) != str(raw_sha):
        raise ValueError('Wheat2000 raw RData sha256 mismatch')
    if out_npz.exists() and not overwrite:
        raise FileExistsError(out_npz)
    if metadata_path.exists() and not overwrite:
        raise FileExistsError(metadata_path)
    executable = shutil.which(rscript)
    if executable is None:
        raise FileNotFoundError(f'Rscript executable not found: {rscript}')
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='wheat2000_prepare_', dir=out_npz.parent))
    try:
        script_path = temporary / 'export.R'
        script_path.write_text(R_SCRIPT)
        environment = os.environ.copy()
        environment.update({'WHEAT2000_RAW': str(raw_path.resolve()),
                            'WHEAT2000_TMP': str(temporary.resolve())})
        completed = subprocess.run([executable, str(script_path)], env=environment,
                                   capture_output=True, text=True)
        if completed.returncode != 0:
            raise RuntimeError(f'R export failed: {completed.stderr.strip()}')
        version = subprocess.run([executable, '--version'], capture_output=True,
                                 text=True).stdout.strip()
        dims = [int(value) for value in _read_lines(temporary / 'dims.txt')]
        if len(dims) != 4:
            raise ValueError('R export dims.txt is malformed')
        rows, markers, pheno_rows, traits_count = dims
        if rows != pheno_rows:
            raise ValueError('X and pheno row counts differ')
        traits = _read_lines(temporary / 'traits.txt')
        genotype_ids = _read_lines(temporary / 'genotype_ids.txt')
        phenotype_ids = _read_lines(temporary / 'phenotype_ids.txt')
        if len(genotype_ids) != rows or len(phenotype_ids) != rows:
            raise ValueError('ID lengths disagree with matrix rows')
        if len(traits) != traits_count:
            raise ValueError('Trait count disagrees with pheno columns')
        if len(set(genotype_ids)) != rows or len(set(phenotype_ids)) != rows:
            raise ValueError('IDs must be unique')
        if len(set(traits)) != len(traits):
            raise ValueError('Trait names must be unique')

        x_col = np.fromfile(temporary / 'X_col.f64', dtype='<f8')
        x_row = np.fromfile(temporary / 'X_row.f64', dtype='<f8')
        y_col = np.fromfile(temporary / 'Y_col.f64', dtype='<f8')
        y_row = np.fromfile(temporary / 'Y_row.f64', dtype='<f8')
        if x_col.size != rows * markers or x_row.size != rows * markers \
                or y_col.size != rows * traits_count or y_row.size != rows * traits_count:
            raise ValueError('R export sizes disagree with dims')
        x_col = x_col.reshape((rows, markers), order='F')
        x_row = x_row.reshape((rows, markers), order='C')
        y_col = y_col.reshape((rows, traits_count), order='F')
        y_row = y_row.reshape((rows, traits_count), order='C')
        if not np.array_equal(x_col, x_row) or not np.array_equal(y_col, y_row):
            raise ValueError('R column-major and row-major exports disagree')
        if not np.isfinite(x_col).all() or not np.isfinite(y_col).all():
            raise ValueError('Wheat2000 X/Y contain non-finite values')
        if not np.isin(x_col, (0.0, 1.0)).all():
            raise ValueError('Wheat2000 X is not binary 0/1')
        checksums = _read_checksums(temporary / 'md5.txt')
        for name in ('X_col.f64', 'X_row.f64', 'Y_col.f64', 'Y_row.f64'):
            if checksums.get(name) != file_md5(temporary / name):
                raise ValueError(f'R/Python md5 mismatch for {name}')
        x32 = x_col.astype(np.float32)
        if not np.array_equal(x32.astype(np.float64), x_col):
            raise ValueError('float32 X conversion is not exact')
        y64 = y_col.astype(np.float64)
        sample_ids = np.arange(rows, dtype=np.int64)
        np.savez_compressed(
            out_npz, X=x32, Y=y64, trait_names=np.asarray(traits, dtype=np.str_),
            sample_ids=sample_ids,
            genotype_ids=np.asarray(genotype_ids, dtype=np.str_),
            phenotype_ids=np.asarray(phenotype_ids, dtype=np.str_),
        )
        cache_sha = file_sha256(out_npz)
        metadata = {
            'schema_version': 1,
            'source': {'path': str(raw_path), 'sha256': str(raw_sha),
                       'bytes': raw_path.stat().st_size},
            'cache': {'path': str(out_npz), 'sha256': cache_sha,
                      'bytes': out_npz.stat().st_size},
            'schema': {'rows': rows, 'markers': markers,
                       'traits': traits, 'trait_count': traits_count,
                       'sample_ids': '0..rows-1'},
            'arrays': {
                'X': {'shape': [rows, markers], 'dtype': 'float32',
                      'sha256': _array_sha(x32), 'bytes': int(x32.nbytes)},
                'Y': {'shape': [rows, traits_count], 'dtype': 'float64',
                      'sha256': _array_sha(y64), 'bytes': int(y64.nbytes)},
            },
            'ids': {
                'genotype': {'count': len(genotype_ids),
                             'sha256': hashlib.sha256(
                                 '\n'.join(genotype_ids).encode()).hexdigest()},
                'phenotype': {'count': len(phenotype_ids),
                              'sha256': hashlib.sha256(
                                  '\n'.join(phenotype_ids).encode()).hexdigest()},
                'relation': 'positional; no join or intersection',
            },
            'r_python': {
                'x_column_major_md5': checksums['X_col.f64'],
                'x_row_major_md5': checksums['X_row.f64'],
                'x_column_major_sha256': file_sha256(temporary / 'X_col.f64'),
                'x_row_major_sha256': file_sha256(temporary / 'X_row.f64'),
                'y_column_major_md5': checksums['Y_col.f64'],
                'y_row_major_md5': checksums['Y_row.f64'],
                'y_column_major_sha256': file_sha256(temporary / 'Y_col.f64'),
                'y_row_major_sha256': file_sha256(temporary / 'Y_row.f64'),
                'x_values_equal': True, 'y_values_equal': True,
                'column_major': True,
            },
            'rscript': {'executable': executable, 'version': version,
                        'script_sha256': hashlib.sha256(R_SCRIPT.encode()).hexdigest()},
            'created_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        }
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(metadata, indent=2))
        return metadata
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def load_wheat2000(raw_path, raw_sha, cache_path, cache_sha):
    """Load and verify the prepared cache; raw and cache SHA are both required."""
    raw_path, cache_path = Path(raw_path), Path(cache_path)
    if not raw_path.is_file():
        raise FileNotFoundError(raw_path)
    if not cache_path.is_file():
        raise FileNotFoundError(cache_path)
    if file_sha256(raw_path) != str(raw_sha):
        raise ValueError('Wheat2000 raw RData sha256 mismatch')
    if file_sha256(cache_path) != str(cache_sha):
        raise ValueError('Wheat2000 cache sha256 mismatch')
    with np.load(cache_path, allow_pickle=False) as archive:
        required = {'X', 'Y', 'trait_names', 'sample_ids',
                    'genotype_ids', 'phenotype_ids'}
        if set(archive.files) != required:
            raise ValueError('Wheat2000 cache has unexpected arrays')
        X = np.asarray(archive['X'], dtype=np.float32)
        Y = np.asarray(archive['Y'], dtype=np.float64)
        trait_names = tuple(str(value) for value in archive['trait_names'])
        sample_ids = np.asarray(archive['sample_ids'], dtype=np.int64)
        genotype_ids = np.asarray(archive['genotype_ids']).astype(str)
        phenotype_ids = np.asarray(archive['phenotype_ids']).astype(str)
    if X.ndim != 2 or Y.ndim != 2 or X.shape[0] != Y.shape[0]:
        raise ValueError('Wheat2000 cache X/Y shapes are invalid')
    if X.shape[0] != sample_ids.size or sample_ids.tolist() != list(range(X.shape[0])):
        raise ValueError('Wheat2000 cache sample_ids must be 0..n-1')
    if X.shape[0] != genotype_ids.size or X.shape[0] != phenotype_ids.size:
        raise ValueError('Wheat2000 cache ID lengths disagree with rows')
    if len(trait_names) != Y.shape[1] or len(set(trait_names)) != len(trait_names):
        raise ValueError('Wheat2000 cache trait names are invalid')
    if len(set(genotype_ids.tolist())) != X.shape[0] \
            or len(set(phenotype_ids.tolist())) != X.shape[0]:
        raise ValueError('Wheat2000 cache IDs must be unique')
    if not np.isfinite(X).all() or not np.isin(X, (0, 1)).all():
        raise ValueError('Wheat2000 X must be finite binary 0/1')
    if not np.isfinite(Y).all():
        raise ValueError('Wheat2000 Y contains non-finite values')
    return Wheat2000Data(X, Y, trait_names, sample_ids, genotype_ids, phenotype_ids)


def make_splits(data, n_splits=10, seed=42, inner_splits=3):
    """Shared KFold10 outer and KFold3 inner splits over all complete materials."""
    Y = np.asarray(data.Y)
    sample_ids = np.asarray(data.sample_ids)
    n = len(sample_ids)
    if n < n_splits or n < inner_splits:
        raise ValueError('Not enough materials for requested splits')
    if sample_ids.tolist() != list(range(n)):
        raise ValueError('sample_ids must be 0..n-1')
    if Y.shape[0] != n or not np.isfinite(Y).all():
        raise ValueError('Shared Wheat2000 splits require complete phenotypes for all traits')
    folds = []
    for fold, (dev_index, test_index) in enumerate(
            KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(np.zeros(n))):
        dev_index = np.asarray(dev_index, dtype=np.int64)
        test_index = np.asarray(test_index, dtype=np.int64)
        inner = []
        for train_local, val_local in KFold(
                n_splits=inner_splits, shuffle=True, random_state=seed + fold).split(
                    np.zeros(len(dev_index))):
            train_index = dev_index[train_local]
            val_index = dev_index[val_local]
            if set(train_index) & set(val_index) \
                    or set(train_index) | set(val_index) != set(dev_index):
                raise ValueError('Inner split does not partition outer dev')
            inner.append({'train_idx': train_index.tolist(),
                          'val_idx': val_index.tolist()})
        if set(dev_index) & set(test_index) \
                or set(dev_index) | set(test_index) != set(range(n)):
            raise ValueError('Outer split does not partition materials')
        if set(test_index) & set(np.concatenate([item['val_idx'] for item in inner])):
            raise ValueError('Test materials entered an inner split')
        folds.append({'fold': fold, 'dev_idx': dev_index.tolist(),
                      'test_idx': test_index.tolist(), 'inner': inner})
    covered = [value for fold in folds for value in fold['test_idx']]
    if sorted(covered) != list(range(n)):
        raise ValueError('Outer test folds do not cover materials exactly once')
    return folds
