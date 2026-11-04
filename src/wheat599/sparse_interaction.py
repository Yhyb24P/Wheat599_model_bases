"""Exact active-set group-lasso solver over all pairwise marker interactions.

Model, for context rows ``Z`` (n x p), phenotypes ``Y`` (n x q) and standardised
query rows ``Q`` (m x p):

    A    = Z / sqrt(p)                         (context RMS already normalised)
    Phi  = [ z_j (.) z_k for all j < k ]       centred and column-RMS-scaled on context
    Y   ~  A B + Phi C + noise
    f(B, C) = ||Y - A B - Phi C||_F^2 / (2n)
              + lambda_A ||B||_F^2 / (2n)
              + lambda_G * sum_l ||C_l||_2
              + lambda_E ||C||_F^2 / 2

Eliminating ``B`` gives the profiled objective

    f(C) = (1 / 2n) * (Y - Phi C)^T M (Y - Phi C) + lambda_G sum_l ||C_l||_2
           + (lambda_E / 2) ||C||_F^2,
    M    = lambda_A (A A^T + lambda_A I)^{-1},

where ``M`` already contains the eliminated B penalty: using ``M^2`` here would drop
it.  The residual after eliminating ``B`` is ``M (Y - Phi C)``, the additive
coefficients are ``B = A^T (A A^T + lambda_A I)^{-1} (Y - Phi C)``, and ``lambda_max``
is ``max_l ||Phi_l^T M Y / n||_2``.

Certification.  A candidate is ``optimal`` only when three exact conditions hold at the
returned point: the vector stationarity residual on the active groups
``||Phi_l^T M e / n - lambda_E C_l - lambda_G C_l / ||C_l||_2||`` (zero groups use
``max(||score_l|| - lambda_G, 0)``), the inactive condition
``max(||score_l|| - lambda_G, 0)``, and the duality gap

    gap = lambda_G sum_l ||C_l|| + lambda_E ||C||^2 / 2
          + sum_l max(||score_l|| - lambda_G, 0)^2 / (2 lambda_E) - <C, score>,

which bounds the primal suboptimality and needs no ``M^{-1}``.  Resource limits
(``max_active`` / ``max_outer`` / ``max_iter``) or a failed certificate return a status
with ``prediction=None``; nothing is accepted silently.  No time limit is applied.

The public ``phi(pairs, query=None)`` accessor exposes the exact same context
centring/scaling so an independent oracle can be built on identical columns.
"""

import math
import time
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import torch


@dataclass
class Candidate:
    lambda_a: float
    ratio: float
    lambda_g: float
    lambda_max: float
    status: str
    objective: float
    kkt_absolute: float
    kkt_relative: float
    dual_gap: float
    relative_gap: float
    full_scans: int
    iterations: int
    seconds: float
    active: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=np.int64))
    coefficients: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    prediction: np.ndarray | None = None
    linear_prediction: np.ndarray | None = None
    additive: np.ndarray | None = None

    def as_dict(self):
        return {'lambda_a': self.lambda_a, 'ratio': self.ratio, 'lambda_g': self.lambda_g,
                'lambda_max': self.lambda_max, 'status': self.status,
                'objective': self.objective, 'kkt_absolute': self.kkt_absolute,
                'kkt_relative': self.kkt_relative, 'dual_gap': self.dual_gap,
                'relative_gap': self.relative_gap, 'full_scans': self.full_scans,
                'iterations': self.iterations, 'seconds': self.seconds,
                'active_pairs': int(len(self.active))}


class SparseInteraction:
    """Pairwise interaction group lasso with exactly verified KKT and gap certificates."""

    def __init__(self, z, y, *, lambda_a, lambda_e=1e-3, device='cuda', dtype=torch.float64,
                 block_size=8192, cache_phi=True, max_active=4096, max_iter=10000,
                 max_outer=64, kkt_tolerance=1e-8, dual_gap_tolerance=1e-7,
                 inner_tolerance=1e-13):
        z = torch.as_tensor(np.asarray(z), dtype=dtype, device=device)
        y = torch.as_tensor(np.asarray(y), dtype=dtype, device=device)
        if z.ndim != 2 or y.ndim != 2 or z.shape[0] != y.shape[0]:
            raise ValueError('z and y must share the material axis')
        if not 0 < lambda_a or not 0 < lambda_e or kkt_tolerance <= 0:
            raise ValueError('Invalid regularisation or tolerance')
        if z.shape[0] < 3 or z.shape[1] < 2:
            raise ValueError('At least three materials and two markers are required')
        self.z, self.y = z, y
        self.materials, self.markers = z.shape
        self.environments = y.shape[1]
        self.lambda_a, self.lambda_e = float(lambda_a), float(lambda_e)
        self.device, self.dtype = torch.device(device), dtype
        self.block_size, self.cache_phi = int(block_size), bool(cache_phi)
        self.max_active, self.max_iter = int(max_active), int(max_iter)
        self.max_outer = int(max_outer)
        self.kkt_tolerance = float(kkt_tolerance)
        self.dual_gap_tolerance = float(dual_gap_tolerance)
        self.inner_tolerance = float(inner_tolerance)
        self.full_scans = 0
        self.pairs = None
        self._phi = None
        self._built = False
        self._mean = None
        self._scale = None
        self._constant = None
        self.lambda_max_value = 0.0
        self.a = z / math.sqrt(self.markers)
        identity = torch.eye(self.materials, dtype=self.dtype, device=self.device)
        self._factor = torch.linalg.cholesky(self.a @ self.a.T + self.lambda_a * identity)
        self.operator = self.lambda_a * torch.cholesky_solve(identity, self._factor)
        # Profiled loss including the eliminated B penalty is r^T M r with M = operator.
        self.r = self.operator
        # M = B B^T with B = sqrt(lambda_a) L^{-T}; the gradient Lipschitz constant is
        # the largest eigenvalue of B^T Phi Phi^T B / n.
        self._half = math.sqrt(self.lambda_a) * torch.linalg.solve_triangular(
            self._factor, identity, upper=False).T
        self._ridge_b = self.a.T @ torch.cholesky_solve(self.y, self._factor)

    # -- pairs and columns -------------------------------------------------
    def _ensure_pairs(self):
        if self.pairs is None:
            self.pairs = np.array(list(combinations(range(self.markers), 2)), dtype=np.int64)

    def _lex_index(self, pairs):
        """Position of (j, k) with j < k in combinations(range(markers), 2)."""
        pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
        if pairs.size and np.any(pairs[:, 0] >= pairs[:, 1]):
            raise ValueError('Pairs must satisfy j < k')
        j, k = pairs[:, 0], pairs[:, 1]
        return j * (2 * self.markers - j - 1) // 2 + k - j - 1

    def _raw_columns(self, pairs):
        pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
        if pairs.shape[0] == 0:
            return torch.empty((self.materials, 0), dtype=self.dtype, device=self.device)
        left = torch.as_tensor(pairs[:, 0], device=self.device)
        right = torch.as_tensor(pairs[:, 1], device=self.device)
        return self.z[:, left] * self.z[:, right]

    def _build_phi(self):
        self._ensure_pairs()
        total = self.pairs.shape[0]
        phi = torch.empty((self.materials, total), dtype=self.dtype, device=self.device) \
            if self.cache_phi else None
        means = torch.empty(total, dtype=self.dtype, device=self.device)
        scales = torch.empty(total, dtype=self.dtype, device=self.device)
        constants = torch.zeros(total, dtype=torch.bool, device=self.device)
        for start in range(0, total, self.block_size):
            stop = min(start + self.block_size, total)
            raw = self._raw_columns(self.pairs[start:stop])
            constant = raw.amax(dim=0) == raw.amin(dim=0)
            mean = raw.mean(dim=0)
            centred = raw - mean
            scale = centred.square().mean(dim=0).sqrt()
            safe = torch.where(constant | (scale <= 0), torch.ones_like(scale), scale)
            standardised = centred / safe
            standardised = torch.where(constant.unsqueeze(0), torch.zeros_like(standardised),
                                       standardised)
            means[start:stop], scales[start:stop] = mean, safe
            constants[start:stop] = constant
            if phi is not None:
                phi[:, start:stop] = standardised
        self._mean, self._scale, self._constant = means, scales, constants
        self._phi = phi
        self._built = True

    def _standardised_block(self, start, stop):
        raw = self._raw_columns(self.pairs[start:stop])
        centred = raw - self._mean[start:stop]
        block = centred / self._scale[start:stop]
        return torch.where(self._constant[start:stop].unsqueeze(0), torch.zeros_like(block), block)

    def _standardise_query(self, query):
        query = torch.as_tensor(np.asarray(query), dtype=self.dtype, device=self.device)
        if query.shape[1] != self.markers:
            raise ValueError('Query markers disagree with the context markers')
        return query

    def phi(self, pairs, query=None):
        """Standardised interaction columns for explicit (j, k) pairs.

        Context columns reuse the stored context centring and scale; query columns use
        the same statistics.  Empty pair lists return width-zero blocks, and an unknown
        pair raises instead of being silently skipped.
        """
        self._ensure_pairs()
        if not self._built:
            self._build_phi()
        pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
        if pairs.size and (pairs.min() < 0 or pairs.max() >= self.markers):
            raise ValueError('Pair index outside the marker range')
        index = self._lex_index(pairs)
        context = self._active_columns(index)
        if query is None:
            return context
        q = self._standardise_query(query)
        return context, self._query_columns(q, index)

    def _query_columns(self, q, active):
        active = np.asarray(active, dtype=np.int64)
        if active.shape[0] == 0:
            return torch.empty((q.shape[0], 0), dtype=self.dtype, device=self.device)
        index = torch.as_tensor(active, device=self.device)
        pairs = torch.as_tensor(self.pairs[active], device=self.device)
        raw = q[:, pairs[:, 0]] * q[:, pairs[:, 1]]
        block = (raw - self._mean[index]) / self._scale[index]
        return torch.where(self._constant[index].unsqueeze(0), torch.zeros_like(block), block)

    def _active_columns(self, active):
        active = np.asarray(active, dtype=np.int64)
        if self._phi is not None:
            return self._phi[:, torch.as_tensor(active, device=self.device)]
        if active.shape[0] == 0:
            return torch.empty((self.materials, 0), dtype=self.dtype, device=self.device)
        index = torch.as_tensor(active, device=self.device)
        raw = self._raw_columns(self.pairs[active])
        block = (raw - self._mean[index]) / self._scale[index]
        return torch.where(self._constant[index].unsqueeze(0), torch.zeros_like(block), block)

    # -- shared quantities -------------------------------------------------
    def lambda_max(self):
        self._ensure_pairs()
        if not self._built:
            self._build_phi()
        target = self.r @ self.y / self.materials
        if self._phi is not None:
            columns = self._phi.T @ target
        else:
            columns = torch.empty((self.pairs.shape[0], self.environments), dtype=self.dtype,
                                  device=self.device)
            for start in range(0, self.pairs.shape[0], self.block_size):
                stop = min(start + self.block_size, self.pairs.shape[0])
                columns[start:stop] = self._standardised_block(start, stop).T @ target
        norms = columns.norm(dim=1)
        return float(norms.max()) if norms.numel() else 0.0

    def _full_gradient(self, residual):
        """All-column score ``Phi^T M (Y - Phi C) / n`` with C extended by zeros."""
        self.full_scans += 1
        target = self.r @ residual / self.materials
        if self._phi is not None:
            return self._phi.T @ target
        out = torch.empty((self.pairs.shape[0], self.environments), dtype=self.dtype,
                          device=self.device)
        for start in range(0, self.pairs.shape[0], self.block_size):
            stop = min(start + self.block_size, self.pairs.shape[0])
            out[start:stop] = self._standardised_block(start, stop).T @ target
        return out

    def _lipschitz(self, columns):
        if columns.shape[1] == 0:
            return 0.0
        mapped = self._half.T @ columns
        eigenvalues = torch.linalg.eigvalsh(mapped @ mapped.T)
        return float(eigenvalues.max().clamp_min(0.0) / self.materials)

    # -- certificates ------------------------------------------------------
    def _active_stationarity(self, score, coefficients, active, lambda_g):
        """Exact per-group KKT residual; zero groups use the inactive threshold."""
        active = np.asarray(active, dtype=np.int64)
        if active.shape[0] == 0:
            return 0.0
        index = torch.as_tensor(active, device=self.device)
        group_score = score.index_select(0, index)
        norms = coefficients.norm(dim=1, keepdim=True)
        nonzero = norms[:, 0] > 0
        residual = group_score - self.lambda_e * coefficients
        if lambda_g > 0:
            residual = residual - lambda_g * coefficients / norms.clamp_min(1e-300)
        vector_error = residual.norm(dim=1)
        zero_error = (group_score.norm(dim=1) - lambda_g).clamp_min(0.0)
        return float(torch.where(nonzero, vector_error, zero_error).max())

    def _duality_gap(self, score, active, coefficients, lambda_g):
        """Primal-dual gap for lambda_E > 0; independent of the eliminated B."""
        if self.lambda_e <= 0:
            return float('nan')
        threshold = (score.norm(dim=1) - lambda_g).clamp_min(0.0)
        conjugate = threshold.square().sum() / (2.0 * self.lambda_e)
        if len(active):
            group = lambda_g * coefficients.norm(dim=1).sum()
            ridge = self.lambda_e / 2.0 * coefficients.square().sum()
            index = torch.as_tensor(np.asarray(active, dtype=np.int64), device=self.device)
            pairing = (coefficients * score.index_select(0, index)).sum()
        else:
            zero = torch.zeros((), dtype=self.dtype, device=self.device)
            group = ridge = pairing = zero
        return float(group + ridge + conjugate - pairing)

    def _objective(self, residual, coefficients, lambda_g):
        quadratic = (residual * (self.r @ residual)).sum() / (2.0 * self.materials)
        group = lambda_g * coefficients.norm(dim=1).sum() if len(coefficients) else 0.0
        ridge = self.lambda_e / 2.0 * coefficients.square().sum() if len(coefficients) else 0.0
        return float(quadratic + group + ridge)

    # -- inner solver ------------------------------------------------------
    def _fista(self, columns, residual, coefficients, lambda_g, kkt_target):
        if coefficients.shape[0] == 0:
            return coefficients, residual, 0, 'empty'
        lipschitz = self._lipschitz(columns)
        if lipschitz <= 0:
            return coefficients, residual, 0, 'degenerate'
        step = 1.0 / lipschitz
        c = coefficients.clone()
        previous = c.clone()
        t = 1.0
        objective = self._objective(residual, c, lambda_g)
        tolerance = self.inner_tolerance * max(1.0, abs(objective))
        for iterations in range(1, self.max_iter + 1):
            t_next = (1.0 + math.sqrt(1.0 + 4.0 * t * t)) / 2.0
            extrapolated = c + ((t - 1.0) / t_next) * (c - previous)
            gradient = columns.T @ (self.r @ (self.y - columns @ extrapolated)) / self.materials
            candidate = extrapolated + step * gradient
            norms = candidate.norm(dim=1, keepdim=True)
            updated = candidate * (1.0 - step * lambda_g / norms.clamp_min(1e-300)).clamp_min(0.0) \
                / (1.0 + step * self.lambda_e)
            new_residual = self.y - columns @ updated
            new_objective = self._objective(new_residual, updated, lambda_g)
            if new_objective > objective:
                t_next = 1.0
                gradient = columns.T @ (self.r @ (self.y - columns @ c)) / self.materials
                updated = (c + step * gradient)
                norms = updated.norm(dim=1, keepdim=True)
                updated = updated * (1.0 - step * lambda_g / norms.clamp_min(1e-300)).clamp_min(0.0) \
                    / (1.0 + step * self.lambda_e)
                new_residual = self.y - columns @ updated
                new_objective = self._objective(new_residual, updated, lambda_g)
            change = abs(objective - new_objective)
            previous, c, residual, objective = c, updated, new_residual, new_objective
            t = t_next
            if iterations % 10 == 0 or change <= tolerance:
                moved = self.y - columns @ c
                score = columns.T @ (self.r @ moved) / self.materials
                if self._active_stationarity(score, c, np.arange(c.shape[0]),
                                             lambda_g) <= kkt_target:
                    return c, moved, iterations, 'converged'
        return c, residual, iterations, 'max_iter'

    # -- active set --------------------------------------------------------
    def _solve(self, active, coefficients, lambda_g):
        active = np.asarray(active, dtype=np.int64)
        residual = self.y - self._active_columns(active) @ coefficients
        total_iterations, reason = 0, 'ok'
        kkt_threshold = self.kkt_tolerance * max(1.0, self.lambda_max_value)
        kkt_target = 0.1 * kkt_threshold
        for _ in range(self.max_outer):
            columns = self._active_columns(active)
            coefficients, residual, inner, inner_status = self._fista(
                columns, residual, coefficients, lambda_g, kkt_target)
            total_iterations += inner
            nonzero = coefficients.norm(dim=1) > 0
            if not bool(nonzero.all()):
                active, coefficients = active[nonzero.cpu().numpy()], coefficients[nonzero]
                residual = self.y - self._active_columns(active) @ coefficients
            score = self._full_gradient(residual)
            norms = score.norm(dim=1)
            inactive_mask = np.ones(self.pairs.shape[0], dtype=bool)
            inactive_mask[active] = False
            inactive_t = torch.as_tensor(inactive_mask, device=self.device)
            violation = float(((norms - lambda_g).masked_fill(~inactive_t, -math.inf)).max()) \
                if inactive_mask.any() else -math.inf
            active_error = self._active_stationarity(score, coefficients, active, lambda_g)
            objective = self._objective(residual, coefficients, lambda_g)
            gap = self._duality_gap(score, active, coefficients, lambda_g)
            gap_threshold = self.dual_gap_tolerance * max(1.0, abs(objective))
            if violation <= kkt_threshold and active_error <= kkt_threshold \
                    and gap <= gap_threshold:
                return active, coefficients, residual, total_iterations, 'ok'
            if violation > kkt_threshold:
                scores = (norms - lambda_g).masked_fill(~inactive_t, -math.inf)
                values, index = torch.topk(scores, min(128, scores.numel()))
                additions = index[values > kkt_threshold].cpu().numpy()
                if additions.size == 0:
                    additions = index[:1].cpu().numpy()
                room = max(0, self.max_active - len(active))
                additions = additions[:room]
                if additions.size == 0:
                    return active, coefficients, residual, total_iterations, 'active_limit'
                active = np.concatenate([active, additions])
                coefficients = torch.cat([coefficients, torch.zeros(
                    (len(additions), self.environments), dtype=self.dtype, device=self.device)])
                residual = self.y - self._active_columns(active) @ coefficients
                continue
            if inner_status == 'max_iter':
                return active, coefficients, residual, total_iterations, 'max_iter'
            # KKT is fine but the gap is not: refine on the same set in the next pass.
        return active, coefficients, residual, total_iterations, 'max_iter'

    def _certify(self, active, coefficients, residual, lambda_g):
        score = self._full_gradient(residual)
        norms = score.norm(dim=1)
        inactive_mask = np.ones(self.pairs.shape[0], dtype=bool)
        inactive_mask[active] = False
        objective = self._objective(residual, coefficients, lambda_g)
        active_error = self._active_stationarity(score, coefficients, active, lambda_g)
        inactive_error = float((norms[inactive_mask] - lambda_g).clamp_min(0.0).max()) \
            if inactive_mask.any() else 0.0
        absolute = max(active_error, inactive_error)
        scale = lambda_g if lambda_g > 1e-8 * self.lambda_max_value else self.lambda_max_value
        relative = absolute / max(scale, 1e-30)
        gap = self._duality_gap(score, active, coefficients, lambda_g)
        reference = max(1.0, abs(objective))
        relative_gap = gap / reference
        okay = (absolute <= self.kkt_tolerance * max(1.0, self.lambda_max_value)
                and gap <= self.dual_gap_tolerance * reference
                and gap >= -1e-12 * reference)
        status = 'optimal' if okay else 'kkt_failed'
        if self.pairs.shape[0] > 0 and len(active) > self.max_active:
            status = 'active_limit'
        return status, objective, absolute, relative, gap, relative_gap

    # -- public API --------------------------------------------------------
    def fit(self, ratios, query=None):
        self._ensure_pairs()
        if not self._built:
            self._build_phi()
        self.lambda_max_value = self.lambda_max()
        results = []
        active = np.empty(0, dtype=np.int64)
        coefficients = torch.empty((0, self.environments), dtype=self.dtype, device=self.device)
        for ratio in ratios:
            started = time.perf_counter()
            scans_before = self.full_scans
            lambda_g = float(ratio) * self.lambda_max_value
            if self.lambda_max_value <= 0:
                # Degenerate interaction space: C = 0 is exact for every ratio.
                status, objective = 'optimal', self._objective(self.y, coefficients, 0.0)
                kkt_abs = kkt_rel = gap = gap_rel = 0.0
                iterations, active = 0, np.empty(0, dtype=np.int64)
            else:
                active, coefficients, residual, iterations, reason = self._solve(
                    active, coefficients, lambda_g)
                status, objective, kkt_abs, kkt_rel, gap, gap_rel = self._certify(
                    active, coefficients, residual, lambda_g)
                if status != 'optimal' and reason in ('active_limit', 'max_iter'):
                    status = reason
            prediction, linear, additive = None, None, None
            if status == 'optimal':
                interaction = self._active_columns(active) @ coefficients
                additive = self.a.T @ torch.cholesky_solve(self.y - interaction, self._factor)
                if query is None:
                    linear = (self.a @ self._ridge_b).detach().cpu().numpy()
                    prediction = (self.a @ additive + interaction).detach().cpu().numpy()
                else:
                    q = self._standardise_query(query)
                    scaled = q / math.sqrt(self.markers)
                    linear = (scaled @ self._ridge_b).detach().cpu().numpy()
                    prediction = ((scaled @ additive)
                                  + self._query_columns(q, active) @ coefficients
                                  ).detach().cpu().numpy()
            results.append(Candidate(
                lambda_a=self.lambda_a, ratio=float(ratio), lambda_g=lambda_g,
                lambda_max=self.lambda_max_value, status=status, objective=objective,
                kkt_absolute=kkt_abs, kkt_relative=kkt_rel, dual_gap=gap,
                relative_gap=gap_rel, full_scans=self.full_scans - scans_before,
                seconds=time.perf_counter() - started, iterations=int(iterations),
                active=self.pairs[active] if len(active) else np.empty((0, 2), dtype=np.int64),
                coefficients=coefficients.detach().cpu().numpy(),
                prediction=prediction, linear_prediction=linear,
                additive=additive.detach().cpu().numpy() if additive is not None else None))
        return results

    def _linear_prediction(self, query):
        if query is None:
            return (self.a @ self._ridge_b).detach().cpu().numpy()
        q = self._standardise_query(query)
        return ((q / math.sqrt(self.markers)) @ self._ridge_b).detach().cpu().numpy()

    def ridge(self, query=None):
        return self._linear_prediction(query), self._ridge_b.detach().cpu().numpy()
