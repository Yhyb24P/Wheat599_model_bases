"""Layout regression tests for the R -> numpy export (2026-10-29 order bug).

R exports matrices with as.vector() (COLUMN-MAJOR); a C-order reshape
silently scrambles X/Y (A is symmetric so it never showed). These checks pin
the load against R-truth values and against DATA_FACTS.md invariants.
Skipped cleanly when the .RData file is absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from wheat599.data import load

RDATA = Path(__file__).resolve().parent.parent / "data" / "raw" / "wheat.RData"
SHA = "8710523389007dd8b7c88a5deb74ee16aea0bae158496bc11f0626fb43b5ea21"

pytestmark = pytest.mark.skipif(not RDATA.exists(), reason="wheat.RData not present")


@pytest.fixture(scope="module")
def data():
    return load(RDATA, SHA)


def test_x_first_row_matches_r(data):
    # R: X[1, 1:5] = 0 1 1 1 1
    assert data.X.shape == (599, 1279)
    np.testing.assert_array_equal(data.X[0, :5], np.array([0, 1, 1, 1, 1]))


def test_y_first_row_matches_r(data):
    # R: Y[1, ] = 1.671629 -1.72747 -1.890285 0.0509159
    np.testing.assert_allclose(
        data.Y[0], np.array([1.671629, -1.72747, -1.890285, 0.0509159]),
        atol=1e-5)
    assert data.environments == ("1", "2", "4", "5")


def test_y_env_correlations_match_data_facts(data):
    r = np.corrcoef(data.Y.T)
    pairs = {(0, 1): -0.0198, (0, 2): -0.1934, (0, 3): -0.1230,
             (1, 2): 0.6610, (1, 3): 0.4110, (2, 3): 0.3879}
    for (i, j), expected in pairs.items():
        assert r[i, j] == pytest.approx(expected, abs=1e-3), (i, j, r[i, j])


def test_x_maf_count_matches_data_facts(data):
    maf = np.minimum(data.X.mean(0), 1 - data.X.mean(0))
    assert int((maf < 0.05).sum()) == 96
    assert data.X.min() == 0.0 and data.X.max() == 1.0


def test_cache_layout_stamp_written(tmp_path):
    d = load(RDATA, SHA, cache_dir=tmp_path / "cache")
    assert (tmp_path / "cache" / "layout.version").read_text().strip() == "f-order-v2"
    # reload reuses the cache and yields identical arrays
    d2 = load(RDATA, SHA, cache_dir=tmp_path / "cache")
    np.testing.assert_array_equal(d.X, d2.X)
    np.testing.assert_array_equal(d.Y, d2.Y)
