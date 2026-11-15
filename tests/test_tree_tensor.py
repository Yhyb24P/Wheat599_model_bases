import numpy as np
import torch

from wheat599.tree_tensor import TreeTensor, geometry


def test_projection_uniform_degree_penalty_qr_and_local_graph_bound():
    torch.set_num_threads(1)
    states = (2 * ((np.arange(16)[:, None] >> np.arange(4)) & 1) - 1).astype(float)
    x = (states + 1) / 2
    children, edges, mixed = geometry(x)
    model = TreeTensor(children, rank=4, seed=42).to('cpu')
    s = torch.tensor(states)
    full = sum(model.components(s))[:, 0]
    products = np.prod(np.where(((np.arange(16)[:, None] >> np.arange(4)) & 1)[None],
                                states[:, None, :], 1), axis=-1)
    coefficients = torch.tensor(products.T) @ full / 16
    degrees = np.array([int(i).bit_count() for i in range(16)])
    high = torch.tensor(products[:, degrees >= 2]) @ coefficients[degrees >= 2]
    torch.testing.assert_close(model.interaction(s), high, atol=1e-6, rtol=1e-6)
    expected = (coefficients[degrees >= 2].square()
                * torch.tensor(1 + .3 * degrees[degrees >= 2] * (degrees[degrees >= 2] - 1))).sum()
    torch.testing.assert_close(model.uniform_penalty(.3), expected, atol=1e-6, rtol=1e-6)
    with torch.no_grad():
        model.cores[0].mul_(1.7)
    before = model(s).clone()
    penalty = model.uniform_penalty(.3).clone()
    model.orthogonalize()
    torch.testing.assert_close(model(s), before, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(model.uniform_penalty(.3), penalty, atol=1e-6, rtol=1e-6)
    all_s = torch.tensor(2 * np.vstack((x, mixed)) - 1)
    graph = torch.tensor(edges)
    h = model.interaction(all_s)
    lower = .25 * (h[graph[:, 0]] - h[graph[:, 1]]).square().mean()
    codes = (np.vstack((x, mixed)).astype(int) * (1 << np.arange(4))).sum(1)
    explicit = 1.7 * high[codes]
    expected_local = .5 * ((explicit[graph[:, 0]] - explicit[-len(edges):]).square()
                           + (explicit[graph[:, 1]] - explicit[-len(edges):]).square()).mean()
    observed = model.local_penalty(all_s, graph)
    torch.testing.assert_close(observed, expected_local, atol=1e-6, rtol=1e-6)
    assert observed >= lower - 1e-6
