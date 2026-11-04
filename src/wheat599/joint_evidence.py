"""Exact evidence operators for the pairwise interaction model.

For context rows ``Z`` (n x p, already centred with global RMS), ``Y`` (n x q) and
``A = Z / sqrt(p)``, the additive coefficients are eliminated with

    M = lambda_A (A A^T + lambda_A I)^{-1},

so the profiled interaction problem depends on ``Y`` only through

    score = Phi^T M e / n,      e = Y - Phi C,
    H     = Phi^T M Phi / n,

with ``Phi`` the context-centred, context-RMS-scaled pairwise products (constant
columns are zero).  The profiled objective uses ``M`` itself, not ``M^2``; ``M^2``
appears only in the conditional covariance of the score,
``(1/n^2) Phi^T M^2 Phi``, and ``M A A^T M`` in the additive part of that covariance.

``readout(indices, d)`` returns the exact per-candidate ridge solution

    C = D F^T (I + F D F^T)^{-1} R,    F = L Phi / sqrt(n),  R = L Y / sqrt(n),
    B = A^T (A A^T + lambda_A I)^{-1} (Y - Phi C),

i.e. the minimiser of ``||Y - A B - Phi C||^2 / (2n) + lambda_A ||B||^2 / (2n)
+ (1/2) sum_l ||C_l||^2 / d_l``.  The n x n Woodbury system is symmetric positive definite, so
``d = 0`` is exact (``C = 0``, ``B = B0``) and the gradient with respect to ``d`` is
finite at zero without any square root of ``d``.  This is exact linear algebra for a
neural weighting, not the original group-lasso objective; no KKT certificate is
provided.  No neural network or gate is implemented here.
"""

import math

import torch


class Evidence:
    """Exact score, statistics and readout operators for the pairwise model."""

    def __init__(self, z, y, lambda_a=3.0, block_size=4096):
        if not torch.is_tensor(z) or not torch.is_tensor(y):
            raise ValueError('z and y must be torch tensors')
        if z.ndim != 2 or y.ndim != 2 or z.shape[0] != y.shape[0]:
            raise ValueError('z and y must share the material axis')
        if z.dtype != y.dtype or z.device != y.device:
            raise ValueError('z and y must share dtype and device')
        if not lambda_a > 0:
            raise ValueError('lambda_a must be positive')
        self.z, self.y = z, y
        self.dtype, self.device = z.dtype, z.device
        self.materials, self.markers = z.shape
        self.environments = y.shape[1]
        self.lambda_a = float(lambda_a)
        self.block_size = int(block_size)
        left, right = torch.triu_indices(self.markers, self.markers, offset=1,
                                         device=self.device)
        self.pairs = torch.stack((left, right), dim=1).to(torch.long)
        self.pair_count = int(self.pairs.shape[0])
        self.a = z / math.sqrt(self.markers)
        identity = torch.eye(self.materials, dtype=self.dtype, device=self.device)
        self._factor = torch.linalg.cholesky(self.a @ self.a.T + self.lambda_a * identity)
        self.m = self.lambda_a * torch.cholesky_solve(identity, self._factor)
        # Any L with L^T L = M works; L = sqrt(lambda_a) * L_G^{-1} with
        # (A A^T + lambda_A I) = L_G L_G^T.
        self._root = math.sqrt(self.lambda_a) * torch.linalg.solve_triangular(
            self._factor, identity, upper=False)
        self._ridge_b = self.a.T @ torch.cholesky_solve(self.y, self._factor)
        self._mean = None
        self._scale = None
        self._constant = None

    # -- columns -----------------------------------------------------------
    def _raw(self, pairs):
        pairs = pairs.to(torch.long)
        return self.z[:, pairs[:, 0]] * self.z[:, pairs[:, 1]]

    def _statistics_built(self):
        if self._mean is not None:
            return
        means = torch.empty(self.pair_count, dtype=self.dtype, device=self.device)
        scales = torch.empty(self.pair_count, dtype=self.dtype, device=self.device)
        constants = torch.zeros(self.pair_count, dtype=torch.bool, device=self.device)
        for start in range(0, self.pair_count, self.block_size):
            stop = min(start + self.block_size, self.pair_count)
            raw = self._raw(self.pairs[start:stop])
            constant = raw.amax(dim=0) == raw.amin(dim=0)
            mean = raw.mean(dim=0)
            scale = (raw - mean).square().mean(dim=0).sqrt()
            safe = torch.where(constant | (scale <= 0), torch.ones_like(scale), scale)
            means[start:stop], scales[start:stop] = mean, safe
            constants[start:stop] = constant
        self._mean, self._scale, self._constant = means, scales, constants

    def _standardise(self, raw, mean, scale, constant):
        block = (raw - mean) / scale
        return torch.where(constant.unsqueeze(0), torch.zeros_like(block), block)

    def phi(self, indices, query=None):
        """Standardised columns for explicit pair indices (any order, repeats allowed)."""
        self._statistics_built()
        indices = torch.as_tensor(indices, dtype=torch.long, device=self.device).reshape(-1)
        if indices.numel() and (int(indices.min()) < 0 or int(indices.max()) >= self.pair_count):
            raise ValueError('Pair index outside the enumerated range')
        columns = self._standardise(self._raw(self.pairs[indices]), self._mean[indices],
                                    self._scale[indices], self._constant[indices])
        if query is None:
            return columns
        if not torch.is_tensor(query) or query.dtype != self.dtype or query.device != self.device:
            raise ValueError('query must be a tensor on the same dtype and device')
        if query.shape[1] != self.markers:
            raise ValueError('query markers disagree with the context markers')
        raw = query[:, self.pairs[indices, 0]] * query[:, self.pairs[indices, 1]]
        return columns, self._standardise(raw, self._mean[indices], self._scale[indices],
                                          self._constant[indices])

    # -- score and statistics ---------------------------------------------
    def restricted_score(self, indices, coefficients=None):
        """Rows of the conditional score for the given pair indices only."""
        columns = self.phi(indices)
        if coefficients is None:
            residual = self.y
        else:
            coefficients = torch.as_tensor(coefficients, dtype=self.dtype, device=self.device)
            if coefficients.shape != (columns.shape[1], self.environments):
                raise ValueError('coefficients must be pairs by environments')
            residual = self.y - columns @ coefficients
        return columns.T @ (self.m @ residual) / self.materials

    def score(self, indices=None, coefficients=None):
        """Full-space conditional score ``Phi^T M (Y - Phi_indices C) / n``.

        Always returns all ``L`` rows: the conditional scan is the KKT-relevant object
        for selecting new candidates.  ``indices=None`` means C = 0.
        """
        if indices is None:
            if coefficients is not None:
                raise ValueError('coefficients require indices')
            residual = self.y
        else:
            if coefficients is None:
                residual = self.y
            else:
                columns = self.phi(indices)
                coefficients = torch.as_tensor(coefficients, dtype=self.dtype,
                                               device=self.device)
                if coefficients.shape != (columns.shape[1], self.environments):
                    raise ValueError('coefficients must be pairs by environments')
                residual = self.y - columns @ coefficients
        target = self.m @ residual / self.materials
        out = torch.empty((self.pair_count, self.environments), dtype=self.dtype,
                          device=self.device)
        self._statistics_built()
        for start in range(0, self.pair_count, self.block_size):
            stop = min(start + self.block_size, self.pair_count)
            block = self._standardise(self._raw(self.pairs[start:stop]),
                                      self._mean[start:stop], self._scale[start:stop],
                                      self._constant[start:stop])
            out[start:stop] = block.T @ target
        return out

    def _block(self, indices):
        columns = self.phi(indices)
        target = self.m @ self.y / self.materials
        projected = self.m @ columns
        return {'columns': columns,
                'S': columns.T @ target,
                'H': (columns.T @ projected) / self.materials,
                'noise_diag': projected.square().sum(dim=0) / self.materials ** 2,
                'additive_diag': ((self.a.T @ self.m) @ columns).square().sum(dim=0)
                                 / self.materials ** 2}

    def statistics(self, indices):
        """``S``, ``H``, conditional and additive score variances, and the energy.

        ``noise_diag`` is ``diag(Phi^T M^2 Phi) / n^2`` and ``additive_diag`` is
        ``diag(Phi^T M A A^T M Phi) / n^2``; they are conditional covariance terms, not
        calibrated variances.  ``S`` is the C = 0 score.
        """
        self._statistics_built()
        indices = torch.as_tensor(indices, dtype=torch.long, device=self.device).reshape(-1)
        block = self._block(indices)
        return {'S': block['S'], 'H': block['H'], 'noise_diag': block['noise_diag'],
                'additive_diag': block['additive_diag'],
                'energy': self.energy(), 'indices': indices}

    def energy(self):
        """``Y^T M Y / n`` as a q-by-q tensor, preserving the environment structure."""
        return (self.y.T @ (self.m @ self.y)) / self.materials

    def constant_mask(self):
        """Boolean mask of pair columns that are exactly constant on the context."""
        self._statistics_built()
        return self._constant.clone()

    # -- readout -----------------------------------------------------------
    def readout(self, indices, d):
        """Exact ridge readout with per-candidate precision ``1 / d_l``."""
        indices = torch.as_tensor(indices, dtype=torch.long, device=self.device).reshape(-1)
        if indices.numel() == 0:
            zero = self.y.new_zeros((0, self.environments))
            return self._ridge_b.clone(), zero
        d = torch.as_tensor(d, dtype=self.dtype, device=self.device)
        if d.shape != (indices.numel(),):
            raise ValueError('d must be one non-negative value per candidate')
        if bool((d < 0).any()):
            raise ValueError('d must be non-negative')
        block = self._block(indices)
        # Observation-space Woodbury form: with F = L Phi / sqrt(n), R = L Y / sqrt(n)
        # (L^T L = M), C = D F^T (I + F D F^T)^{-1} R.  Identical to
        # (I + D H) C = D S, but the n x n system is SPD, d = 0 is exact and the
        # derivative at d = 0 is finite without any square root of d.
        root_n = math.sqrt(self.materials)
        features = (self._root @ block['columns']) / root_n
        target = (self._root @ self.y) / root_n
        system = torch.eye(self.materials, dtype=self.dtype, device=self.device) \
            + (features * d.unsqueeze(0)) @ features.T
        solved = torch.linalg.solve(system, target)
        coefficients = d.unsqueeze(1) * (features.T @ solved)
        additive = self.a.T @ torch.cholesky_solve(self.y - block['columns'] @ coefficients,
                                                   self._factor)
        return additive, coefficients

    def predict(self, query, indices, b, c):
        """Prediction on query rows using context statistics only."""
        if not torch.is_tensor(query) or query.dtype != self.dtype or query.device != self.device:
            raise ValueError('query must be a tensor on the same dtype and device')
        if query.shape[1] != self.markers:
            raise ValueError('query markers disagree with the context markers')
        scaled = query / math.sqrt(self.markers)
        prediction = scaled @ b
        indices = torch.as_tensor(indices, dtype=torch.long, device=self.device).reshape(-1)
        if indices.numel():
            _, columns = self.phi(indices, query=query)
            c = torch.as_tensor(c, dtype=self.dtype, device=self.device)
            prediction = prediction + columns @ c
        return prediction

    def ridge(self, query=None):
        """Original additive ridge prediction."""
        if query is None:
            return self.a @ self._ridge_b
        if not torch.is_tensor(query) or query.dtype != self.dtype or query.device != self.device:
            raise ValueError('query must be a tensor on the same dtype and device')
        return (query / math.sqrt(self.markers)) @ self._ridge_b
