"""Analytic operators for a multi-environment marker-effect posterior."""

import torch


class ConditionalLikelihood:
    """Profile the unpenalized intercept and Gaussian residual background.

    Marker matrices must already use the training-only centering and scaling
    that define the additive kernel. All tensors share dtype and device.
    """

    def __init__(self, z, rbf, task_covariance, fraction, ridge):
        self.z = z
        self.rbf = rbf
        self.b = task_covariance
        self.fraction = fraction
        self.ridge = ridge
        self.n, self.p = z.shape
        self.q = task_covariance.shape[0]
        if not 0 <= fraction <= 1 or ridge <= 0:
            raise ValueError('Invalid kernel fraction or ridge')
        eye = torch.eye(self.n * self.q, dtype=z.dtype, device=z.device)
        self.chol = torch.linalg.cholesky(
            (1 - fraction) * torch.kron(rbf.contiguous(), task_covariance.contiguous())
            + ridge * eye
        )
        self.design = torch.eye(self.q, dtype=z.dtype, device=z.device).repeat(self.n, 1)
        self.vf = torch.cholesky_solve(self.design, self.chol)
        self.intercept_chol = torch.linalg.cholesky(self.design.T @ self.vf)
        self._gaussian_chol = None
        self._gaussian_design_solution = None

    def solve(self, value):
        return torch.cholesky_solve(value, self.chol)

    def project(self, value):
        solved = self.solve(value)
        intercept = torch.cholesky_solve(self.design.T @ solved, self.intercept_chol)
        return solved - self.vf @ intercept

    def marker_forward(self, weights):
        return self.fraction ** 0.5 * (self.z @ weights)

    def marker_transpose(self, values):
        return self.fraction ** 0.5 * (self.z.T @ values)

    def sufficient_score(self, y):
        return self.marker_transpose(self.project(y.reshape(-1, 1)).reshape(self.n, self.q))

    def hessian_apply(self, weights):
        values = self.marker_forward(weights)
        return self.marker_transpose(self.project(values.reshape(-1, 1)).reshape(self.n, self.q))

    def hessian_diagonal_blocks(self, batch_size=32):
        """Compute marker diagonal blocks through the separable covariance spectrum."""
        if self.fraction == 0:
            return self.z.new_zeros((self.p, self.q, self.q))
        values, vectors = torch.linalg.eigh(self.rbf)
        task_values, task_vectors = torch.linalg.eigh(self.b)
        inverse = 1 / ((1 - self.fraction) * values[:, None] * task_values[None, :] + self.ridge)
        basis = torch.stack([vectors @ (vectors.T * inverse[:, k, None]) for k in range(self.q)])
        blocks = torch.einsum('ek,fk,kab->efab', task_vectors, task_vectors, basis)
        design = blocks.sum(-1)
        gram = blocks.sum((-1, -2))
        solved = torch.linalg.solve(gram, design.permute(1, 0, 2).reshape(self.q, -1))
        solved = solved.reshape(self.q, self.q, self.n)
        projected = blocks - torch.einsum('xka,kyb->xyab', design, solved)
        out = self.z.new_empty((self.p, self.q, self.q))
        for e in range(self.q):
            for f in range(e, self.q):
                value = self.fraction * torch.einsum('aj,ab,bj->j', self.z, projected[e, f], self.z)
                out[:, e, f] = value
                out[:, f, e] = value
        return out

    def dense_hessian(self, batch_size=32):
        """Materialize H in marker-major order for sampling references."""
        task_eye = torch.eye(self.q, dtype=self.z.dtype, device=self.z.device)
        columns = []
        for start in range(0, self.p, batch_size):
            z_batch = self.z[:, start:start + batch_size]
            count = z_batch.shape[1]
            design = (self.fraction ** 0.5 * z_batch[:, None, :, None]
                      * task_eye[None, :, None, :]).reshape(self.n * self.q, count * self.q)
            projected = self.project(design).reshape(self.n, self.q, count * self.q)
            columns.append((self.fraction ** 0.5 * torch.einsum(
                'nj,nak->jak', self.z, projected)).reshape(self.p * self.q, count * self.q))
        return torch.cat(columns, dim=1).reshape(self.p, self.q, self.p, self.q)

    def predict_given_weights(self, y, weights, z_target, rbf_cross):
        """Conditional prediction with the intercept and background profiled."""
        remainder = (y - self.marker_forward(weights)).reshape(-1, 1)
        solved = self.solve(remainder)
        intercept = torch.cholesky_solve(self.design.T @ solved, self.intercept_chol)
        residual = (solved - self.vf @ intercept).reshape(self.n, self.q)
        background = (1 - self.fraction) * rbf_cross @ residual @ self.b
        markers = self.fraction ** 0.5 * z_target @ weights
        return intercept.reshape(1, self.q) + background + markers

    def prediction_correction(self, weights, z_target, rbf_cross):
        """Apply the conditional prediction operator R_* to effect changes."""
        return self.predict_given_weights(
            torch.zeros((self.n, self.q), dtype=self.z.dtype, device=self.z.device),
            weights, z_target, rbf_cross,
        )

    def gaussian_posterior_mean(self, y):
        """Use the observation-space system; no inverse of B is required."""
        return self.gaussian_posterior_mean_batch(y.unsqueeze(0))[0]

    def gaussian_posterior_mean_batch(self, y):
        """Reuse the same observation-space factor across tasks sharing a kernel."""
        if y.ndim != 3 or y.shape[1:] != (self.n, self.q):
            raise ValueError('Expected batch by material by environment')
        if self._gaussian_chol is None:
            additive = self.z @ self.z.T
            cov = self.fraction * torch.kron(additive.contiguous(), self.b.contiguous())
            cov = cov + self.chol @ self.chol.T
            self._gaussian_chol = torch.linalg.cholesky(cov)
            self._gaussian_design_solution = torch.cholesky_solve(self.design, self._gaussian_chol)
        design_solution = self._gaussian_design_solution
        rhs = y.reshape(y.shape[0], -1).T.contiguous()
        solved = torch.cholesky_solve(rhs, self._gaussian_chol)
        intercept = torch.linalg.solve(self.design.T @ design_solution, self.design.T @ solved)
        residual = (solved - design_solution @ intercept).T.reshape(-1, self.n, self.q)
        return self.fraction ** 0.5 * torch.matmul(self.z.T, residual) @ self.b
