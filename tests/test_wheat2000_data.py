"""Checks for the Wheat2000 loader, R cache preparation and shared splits."""

import hashlib
import json
import shutil
import subprocess

import numpy as np
import pytest

from wheat599.wheat2000_data import (Wheat2000Data, file_sha256, load_wheat2000,
                                     make_splits, prepare)

RSCRIPT = shutil.which('Rscript')


def write_rdata(path, x, pheno, trait_names, genotype_ids, phenotype_ids):
    if RSCRIPT is None:
        pytest.skip('Rscript is unavailable')
    n, p = x.shape
    q = pheno.shape[1]
    assert pheno.shape[0] == n
    x_values = ', '.join(str(float(value)) for value in x.flatten(order='F'))
    y_values = ', '.join(str(float(value)) for value in pheno.flatten(order='F'))
    code = f"""
    X <- matrix(0, nrow={n}, ncol={p})
    X[] <- c({x_values})
    rownames(X) <- c({', '.join(repr(str(value)) for value in genotype_ids)})
    pheno <- matrix(0, nrow={n}, ncol={q})
    pheno[] <- c({y_values})
    rownames(pheno) <- c({', '.join(repr(str(value)) for value in phenotype_ids)})
    colnames(pheno) <- c({', '.join(repr(str(value)) for value in trait_names)})
    save(X, pheno, file={str(path)!r})
    """
    script = path.parent / 'write.R'
    script.write_text(code)
    subprocess.run([RSCRIPT, str(script)], check=True)


def tiny():
    rng = np.random.default_rng(31)
    x = (rng.random((12, 7)) < 0.45).astype(np.float64)
    y = rng.normal(size=(12, 3))
    return x, y, ('TKW', 'TW', 'GL'), [f'L{i}' for i in range(12)], [str(i + 1) for i in range(12)]


def test_prepare_and_load_roundtrip(tmp_path):
    x, y, traits, genotype_ids, phenotype_ids = tiny()
    raw = tmp_path / 'tiny.RData'
    write_rdata(raw, x, y, traits, genotype_ids, phenotype_ids)
    raw_sha = file_sha256(raw)
    cache = tmp_path / 'tiny.npz'
    metadata = prepare(raw, raw_sha, cache)
    assert cache.is_file()
    assert (tmp_path / 'tiny.npz.metadata.json').is_file()
    assert metadata['source']['sha256'] == raw_sha
    assert metadata['cache']['sha256'] == file_sha256(cache)
    assert metadata['r_python']['x_values_equal'] is True
    assert metadata['r_python']['y_values_equal'] is True
    assert metadata['arrays']['X']['dtype'] == 'float32'
    data = load_wheat2000(raw, raw_sha, cache, file_sha256(cache))
    assert data.X.dtype == np.float32
    assert data.Y.dtype == np.float64
    assert np.array_equal(data.X, x.astype(np.float32))
    assert np.array_equal(data.Y, y)
    assert data.trait_names == traits
    assert np.array_equal(data.sample_ids, np.arange(12))
    assert data.genotype_ids.tolist() == genotype_ids
    assert data.phenotype_ids.tolist() == phenotype_ids
    assert set(data.genotype_ids) != set(data.phenotype_ids)


def test_explicit_two_by_three_r_column_order(tmp_path):
    x = np.asarray([[0., 1., 0.], [1., 0., 1.]])
    y = np.asarray([[1., 2., 3.], [4., 5., 6.]])
    raw, cache = tmp_path / 'two_by_three.RData', tmp_path / 'two_by_three.npz'
    write_rdata(raw, x, y, ('A', 'B', 'C'), ('L1', 'L2'), ('7', '19'))
    prepare(raw, file_sha256(raw), cache)
    data = load_wheat2000(raw, file_sha256(raw), cache, file_sha256(cache))
    np.testing.assert_array_equal(data.X, x)
    np.testing.assert_array_equal(data.Y, y)


def test_prepare_raw_sha_is_checked(tmp_path):
    x, y, traits, genotype_ids, phenotype_ids = tiny()
    raw = tmp_path / 'tiny.RData'
    write_rdata(raw, x, y, traits, genotype_ids, phenotype_ids)
    with pytest.raises(ValueError, match='raw RData sha256'):
        prepare(raw, '0' * 64, tmp_path / 'bad.npz')


def test_load_rejects_bad_cache_and_values(tmp_path):
    x, y, traits, genotype_ids, phenotype_ids = tiny()
    raw = tmp_path / 'tiny.RData'
    write_rdata(raw, x, y, traits, genotype_ids, phenotype_ids)
    raw_sha = file_sha256(raw)
    cache = tmp_path / 'tiny.npz'
    prepare(raw, raw_sha, cache)
    good_sha = file_sha256(cache)
    load_wheat2000(raw, raw_sha, cache, good_sha)
    with pytest.raises(ValueError, match='cache sha256'):
        load_wheat2000(raw, raw_sha, cache, '0' * 64)
    bad = tmp_path / 'bad.npz'
    np.savez_compressed(bad, X=np.full_like(x, 2, dtype=np.float32), Y=y,
                        trait_names=np.asarray(traits),
                        sample_ids=np.arange(len(x), dtype=np.int64),
                        genotype_ids=np.asarray(genotype_ids),
                        phenotype_ids=np.asarray(phenotype_ids))
    with pytest.raises(ValueError, match='binary'):
        load_wheat2000(raw, raw_sha, bad, file_sha256(bad))
    bad_y = tmp_path / 'bad_y.npz'
    invalid_y = y.copy()
    invalid_y[0, 0] = np.nan
    np.savez_compressed(bad_y, X=x.astype(np.float32), Y=invalid_y,
                        trait_names=np.asarray(traits),
                        sample_ids=np.arange(len(x), dtype=np.int64),
                        genotype_ids=np.asarray(genotype_ids),
                        phenotype_ids=np.asarray(phenotype_ids))
    with pytest.raises(ValueError, match='non-finite'):
        load_wheat2000(raw, raw_sha, bad_y, file_sha256(bad_y))


def split_data(n=50, environments=8, seed=7):
    rng = np.random.default_rng(seed)
    return Wheat2000Data(
        X=(rng.random((n, 9)) < 0.4).astype(np.float32),
        Y=rng.normal(size=(n, environments)),
        trait_names=tuple(f'T{i}' for i in range(environments)),
        sample_ids=np.arange(n, dtype=np.int64),
        genotype_ids=np.asarray([f'L{i}' for i in range(n)]),
        phenotype_ids=np.asarray([str(i + 1) for i in range(n)]),
    )


def test_make_splits_cover_and_are_deterministic():
    data = split_data()
    first = make_splits(data, n_splits=10, seed=42)
    second = make_splits(data, n_splits=10, seed=42)
    assert first == second
    assert len(first) == 10
    covered = []
    for fold in first:
        assert len(fold['test_idx']) == 5
        assert len(fold['dev_idx']) == 45
        assert set(fold['dev_idx']) & set(fold['test_idx']) == set()
        assert sorted(fold['dev_idx'] + fold['test_idx']) == list(range(50))
        assert len(fold['inner']) == 3
        inner_cover = []
        for inner in fold['inner']:
            assert set(inner['train_idx']) & set(inner['val_idx']) == set()
            assert sorted(inner['train_idx'] + inner['val_idx']) == sorted(fold['dev_idx'])
            assert set(inner['val_idx']) & set(fold['test_idx']) == set()
            inner_cover.extend(inner['val_idx'])
        assert sorted(inner_cover) == sorted(fold['dev_idx'])
        covered.extend(fold['test_idx'])
    assert sorted(covered) == list(range(50))


def test_make_splits_require_complete_phenotypes_and_sample_ids():
    data = split_data()
    broken = data._replace(Y=data.Y.copy())
    broken.Y[0, 0] = np.nan
    with pytest.raises(ValueError, match='complete phenotypes'):
        make_splits(broken)
    wrong_ids = data._replace(sample_ids=np.arange(1, 51))
    with pytest.raises(ValueError, match='sample_ids'):
        make_splits(wrong_ids)
    other = split_data(seed=8)
    other = other._replace(Y=other.Y + 10.0)
    assert make_splits(data) == make_splits(other)
