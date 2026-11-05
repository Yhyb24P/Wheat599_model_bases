"""Checks for the fit-only orthogonal phenotype-target rotation."""

import json

import numpy as np
import pytest

from wheat599.phenotype_rotation import PhenotypeRotation


def data(seed=3, fit_rows=60, val_rows=40, environments=4):
    generator = np.random.default_rng(seed)
    latent = generator.normal(size=(fit_rows + val_rows, environments))
    loading = generator.normal(size=(environments, environments))
    y = latent @ loading + 0.1 * generator.normal(size=(fit_rows + val_rows, environments))
    return y[:fit_rows], y[fit_rows:]


def test_fit_only_metadata_and_val_does_not_change_rotation():
    y_fit, y_val = data()
    rotation = PhenotypeRotation.fit(y_fit)
    metadata = rotation.metadata()
    assert set(metadata) == {'mean', 'std', 'rotation', 'eigenvalues'}
    assert np.allclose(metadata['mean'], y_fit.mean(0), atol=1e-12, rtol=0)
    assert np.allclose(metadata['std'], y_fit.std(0, ddof=0), atol=1e-12, rtol=0)
    assert np.allclose(rotation.mean_, y_fit.mean(0), atol=1e-12, rtol=0)
    before = rotation.rotation_.copy()
    rotation.transform(y_val)
    assert np.array_equal(rotation.rotation_, before)
    assert rotation.metadata() == metadata
    json.dumps(metadata)


def test_orthogonal_full_rank_order_sign_and_roundtrip():
    y_fit, y_val = data(seed=5)
    rotation = PhenotypeRotation.fit(y_fit)
    q = rotation.rotation_
    eigenvalues = rotation.eigenvalues_
    assert q.shape == (4, 4)
    assert np.allclose(q.T @ q, np.eye(4), atol=1e-12, rtol=0)
    assert np.all(np.diff(eigenvalues) <= 0)
    assert np.isclose(eigenvalues.sum(), 4.0, atol=1e-10)
    for column in range(q.shape[1]):
        index = int(np.argmax(np.abs(q[:, column])))
        assert q[index, column] > 0
    for values in (y_fit, y_val):
        transformed = rotation.transform(values)
        assert transformed.dtype == np.float64
        assert np.allclose(rotation.inverse_transform(transformed), values,
                           atol=1e-12, rtol=0)
    standardised = (y_fit - rotation.mean_) / rotation.std_
    transformed = rotation.transform(y_fit)
    assert np.allclose(transformed.T @ transformed / len(y_fit),
                       np.diag(eigenvalues), atol=1e-12, rtol=0)


def test_single_inverse_and_fixed_orthogonal_operator():
    y_fit, _ = data(seed=7)
    rotation = PhenotypeRotation.fit(y_fit)
    generator = np.random.default_rng(11)
    z = generator.normal(size=(25, 4))
    recovered = rotation.inverse_transform(z)
    assert np.allclose(rotation.transform(recovered), z, atol=1e-12, rtol=0)
    standardised = (y_fit - rotation.mean_) / rotation.std_
    transformed = rotation.transform(y_fit)
    assert np.allclose(transformed @ transformed.T,
                       standardised @ standardised.T, atol=1e-12, rtol=0)


def test_degenerate_direction_fails_explicitly():
    y_fit, _ = data(seed=9)
    duplicate = y_fit.copy()
    duplicate[:, 3] = duplicate[:, 2]
    with pytest.raises(ValueError, match='Degenerate phenotype direction'):
        PhenotypeRotation.fit(duplicate)
    constant = y_fit.copy()
    constant[:, 1] = 2.0
    with pytest.raises(ValueError, match='Constant'):
        PhenotypeRotation.fit(constant)
    nonfinite = y_fit.copy()
    nonfinite[0, 0] = np.nan
    with pytest.raises(ValueError, match='non-finite'):
        PhenotypeRotation.fit(nonfinite)


def test_shape_and_fit_validation():
    y_fit, y_val = data(seed=13)
    rotation = PhenotypeRotation.fit(y_fit)
    with pytest.raises(ValueError, match='material-by-environment'):
        PhenotypeRotation.fit(np.zeros(5))
    with pytest.raises(ValueError, match='material-by-environment'):
        rotation.transform(y_val[:, :3])
    with pytest.raises(ValueError, match='material-by-direction'):
        rotation.inverse_transform(np.zeros((len(y_val), 3)))
    with pytest.raises(ValueError, match='non-empty'):
        PhenotypeRotation.fit(np.zeros((0, 4)))
