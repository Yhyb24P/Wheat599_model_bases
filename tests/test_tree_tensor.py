import numpy as np
import torch

from genopred.models.structured.tree_tensor import TreeTensor, geometry


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


def test_als_coordinate_matches_explicit_walsh_system_and_sweep_decreases_loss():
    from genopred.models.structured.tree_tensor import ALSState, data_message, norm_message
    torch.set_num_threads(1)
    rng = np.random.default_rng(17)
    cube = (2 * ((np.arange(64)[:, None] >> np.arange(6)) & 1) - 1).astype(float)
    fit = rng.choice(64, size=20, replace=False)
    x = (cube[fit] + 1) / 2
    children, edges, mixed = geometry(x)
    mixed[:, :3] = x[edges[:, 1], :3]
    all_x = np.vstack((x, mixed))
    all_s = torch.tensor(2 * all_x - 1)
    graph = torch.tensor(edges)
    target = torch.tensor(.5 * cube[fit, 0] + cube[fit, 2] * cube[fit, 3] + rng.normal(size=20) * .1)
    model = TreeTensor(children, rank=4, seed=42)
    state = ALSState(model, all_s, graph, target, .3, .2, .1)
    node = max(range(len(model.cores) - 1), key=lambda i: model.cores[i].numel())

    def outside(node_id):
        if node_id == 2 * model.p - 2:
            return state.root_data, state.root_norm
        parent, axis = model.parents[node_id]
        data, norms = outside(model.p + parent)
        sibling = model.nodes[parent][1 if axis == 1 else 0]
        t = model.cores[parent].permute((1, 0, 2) if axis == 1 else (2, 0, 1))
        return data_message(t, data, state.data[sibling]), norm_message(t, norms, state.norms[sibling])

    q, b = state.system(node, *outside(model.p + node))
    old = model.cores[node].clone()
    products = np.prod(np.where(((np.arange(64)[:, None] >> np.arange(6)) & 1)[None],
                                cube[:, None, :], 1), axis=-1)
    walsh = torch.tensor(products)
    degrees = np.array([int(i).bit_count() for i in range(64)])
    high = degrees >= 2
    columns = []
    for j in range(old.numel()):
        model.cores[node].zero_()
        model.cores[node].flatten()[j] = 1
        full = sum(model.components(torch.tensor(cube)))[:, 0]
        columns.append((walsh.T @ full / 64)[high])
    model.cores[node].copy_(old)
    coefficients = torch.stack(columns, 1)
    codes = (all_x.astype(int) * (1 << np.arange(6))).sum(1)
    phi = (walsh[:, high] @ coefficients)[codes]
    difference = torch.cat((phi[graph[:, 0]] - phi[len(fit):], phi[graph[:, 1]] - phi[len(fit):]))
    weights = torch.tensor(1 + .1 * degrees[high] * (degrees[high] - 1))
    expected_q = phi[:len(fit)].T @ phi[:len(fit)] / len(fit) + .2 * (
        coefficients.T @ (weights[:, None] * coefficients)
        + difference.T @ difference / (2 * len(edges)))
    residual = target - model.intercept - all_s[:len(fit)] @ model.beta
    torch.testing.assert_close(q, expected_q, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(b, phi[:len(fit)].T @ residual / len(fit), atol=1e-6, rtol=1e-6)
    before = float(state.objective())
    state.sweep()
    after = float(state.objective())
    assert after <= before + 1e-6
    torch.testing.assert_close(state.data[-1][2][:, 0], model.interaction(all_s), atol=1e-6, rtol=1e-6)
    direct = ((target - model(all_s[:len(fit)])).square().mean() + .3 * model.beta.square().sum()
              + .2 * (model.uniform_penalty(.1) + model.local_penalty(all_s, graph)))
    torch.testing.assert_close(state.objective(), direct, atol=1e-6, rtol=1e-6)
