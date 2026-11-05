"""Fit-only orthogonal phenotype-target rotation for four independent directions.

``fit`` standardises the fit-row phenotype columns (population standard deviation),
builds their float64 correlation matrix, and keeps all eigenvectors of its
``eigh`` decomposition ordered by descending eigenvalue. Each column is sign-fixed
so its largest-absolute entry is positive. The rotation is orthonormal and no
direction is dropped.

    z = ((y - mean) / std) @ Q
    y = (z @ Q.T) * std + mean

Only fit rows enter ``fit``. Transformation of validation rows supplies rotated
targets but never changes ``mean``, ``std`` or ``Q``. The direction models own
their own rotated-target scaler and inverse; ``inverse_transform`` expects
unstandardised rotated predictions and applies the original phenotype scale once.
A constant phenotype or a rank-deficient correlation (a fully degenerate
direction) raises explicitly rather than silently removing a direction.
"""

import numpy as np


class PhenotypeRotation:
    def __init__(self, mean, std, rotation, eigenvalues, tolerance):
        self.mean_ = np.asarray(mean, dtype=np.float64)
        self.std_ = np.asarray(std, dtype=np.float64)
        self.rotation_ = np.asarray(rotation, dtype=np.float64)
        self.eigenvalues_ = np.asarray(eigenvalues, dtype=np.float64)
        self.tolerance_ = float(tolerance)

    @classmethod
    def fit(cls, y_fit, *, tolerance=1e-10):
        y = np.asarray(y_fit, dtype=np.float64)
        if y.ndim != 2 or y.shape[0] < 2 or y.shape[1] < 1:
            raise ValueError('y_fit must be a non-empty material-by-environment matrix')
        if not np.isfinite(y).all():
            raise ValueError('y_fit contains non-finite values')
        mean = y.mean(axis=0)
        std = y.std(axis=0, ddof=0)
        if np.any(std <= 0) or not np.isfinite(std).all():
            raise ValueError('Constant or non-finite phenotype column')
        standardised = (y - mean) / std
        correlation = standardised.T @ standardised / y.shape[0]
        eigenvalues, eigenvectors = np.linalg.eigh(correlation)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        rotation = eigenvectors[:, order]
        for column in range(rotation.shape[1]):
            index = int(np.argmax(np.abs(rotation[:, column])))
            if rotation[index, column] < 0:
                rotation[:, column] *= -1.0
        threshold = float(tolerance) * max(1.0, float(eigenvalues[0]))
        degenerate = np.flatnonzero(eigenvalues <= threshold)
        if degenerate.size:
            raise ValueError(
                'Degenerate phenotype direction(s): '
                + ', '.join(str(int(value)) for value in degenerate))
        return cls(mean, std, rotation, eigenvalues, tolerance)

    def transform(self, y):
        y = np.asarray(y, dtype=np.float64)
        if y.ndim != 2 or y.shape[1] != self.mean_.size:
            raise ValueError('y must be a material-by-environment matrix')
        return ((y - self.mean_) / self.std_) @ self.rotation_

    def inverse_transform(self, z):
        z = np.asarray(z, dtype=np.float64)
        if z.ndim != 2 or z.shape[1] != self.mean_.size:
            raise ValueError('z must be a material-by-direction matrix')
        return (z @ self.rotation_.T) * self.std_ + self.mean_

    def metadata(self):
        return {
            'mean': self.mean_.tolist(),
            'std': self.std_.tolist(),
            'rotation': self.rotation_.tolist(),
            'eigenvalues': self.eigenvalues_.tolist(),
        }
