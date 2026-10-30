"""Non-square regression test for the R column-major -> NumPy reader."""

from __future__ import annotations

import numpy as np

from wheat599.data import _read_flat


def test_read_flat_fortran_order_non_square(tmp_path):
    m = np.array([[11.0, 12.0, 13.0], [21.0, 22.0, 23.0]])
    # R's as.vector() on a matrix is column-major: (11,21,12,22,13,23).
    flat = m.T.ravel()
    path = tmp_path / "m.f64"
    path.write_bytes(flat.astype(np.float64).tobytes())

    got = _read_flat(path, np.float64, (2, 3))

    assert np.array_equal(got, m)
    assert not np.array_equal(
        np.frombuffer(path.read_bytes(), dtype=np.float64).reshape(2, 3), m
    )
