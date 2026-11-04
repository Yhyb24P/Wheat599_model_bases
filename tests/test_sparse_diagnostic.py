"""Algebraic evidence and the independent support-only diagnostic."""
import itertools

import numpy as np
import torch

from wheat599.matrix_tasks import prepare_context
from wheat599.sparse_diagnostic import support_oracle
from wheat599.sparse_interaction import SparseInteraction


def test_legal_binary_same_gram_has_different_sparse_posterior():
    inputs = [['11011', '01000', '00010', '10111', '01110'],
              ['10111', '00100', '00010', '11011', '10000']]
    y = torch.tensor([[1.], [-1.], [1.], [-1.]], dtype=torch.float64)
    summaries, posterior, pair_cross = [], [], []
    for rows in inputs:
        x = torch.tensor([[int(c) for c in row] for row in rows], dtype=torch.float64)
        data = prepare_context(x[:4], y, x[4:])
        z, q = data.x, data.query
        summaries.append((z @ z.T, z @ q.T, q @ q.T))
        pairs = list(itertools.combinations(range(5), 2))
        w = torch.stack([z[:, j] * z[:, k] for j, k in pairs], dim=1)
        wq = torch.stack([q[:, j] * q[:, k] for j, k in pairs], dim=1)
        pair_cross.append(wq @ w.T)
        # Directly enumerate the supplied 20 equiprobable hypotheses.
        signed = 2 * x - 1
        f = torch.stack([signed[:, j] * signed[:, k] for j, k in pairs], dim=1)
        f = torch.cat([f, -f], dim=1)
        weights = torch.softmax(-.5 * (y - f[:4]).square().sum(0), dim=0)
        posterior.append(float(f[4] @ weights))
        raw_pair = .5 * ((q @ z.T).square() - q.square() @ z.square().T)
        torch.testing.assert_close(wq @ w.T, raw_pair)
        mu = x[:4].mean(0)
        scale = (x[:4] - mu).square().mean().sqrt()
        torch.testing.assert_close(z.square() - z.square().mean(0), (1 - 2 * mu) * z / scale)
    for a, b in zip(*summaries):
        torch.testing.assert_close(a, b, atol=1e-14, rtol=1e-14)
    np.testing.assert_allclose(posterior, [-.5697611490340134, -.18992038301133773], atol=1e-14)
    assert not torch.allclose(pair_cross[0], pair_cross[1])


def test_support_oracle_matches_full_solver_when_all_pairs_are_given():
    rng = np.random.default_rng(812)
    x = torch.from_numpy(rng.integers(0, 2, (24, 4)).astype(float))
    q = torch.from_numpy(rng.integers(0, 2, (3, 4)).astype(float))
    y = torch.from_numpy(rng.normal(size=(24, 2)))
    data = prepare_context(x, y, q)
    solver = SparseInteraction(data.x.numpy(), data.y.numpy(), lambda_a=3,
        lambda_e=.001, device='cpu', kkt_tolerance=1e-8, max_iter=10000)
    ratios = [1., .2]
    unknown = solver.fit(ratios, query=data.query.numpy())
    oracle = support_oracle(solver, solver.pairs, data.query.numpy(), ratios,
                             solver.lambda_max_value)
    for left, right in zip(unknown, oracle):
        assert left.status == 'optimal'
        np.testing.assert_allclose(left.prediction, right['prediction'], rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(left.objective, right['objective'], rtol=1e-7, atol=1e-8)
