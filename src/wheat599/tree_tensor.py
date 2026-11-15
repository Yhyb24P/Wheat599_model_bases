"""Projected tree-tensor functions and fit-only local response regularization."""
import time
import numpy as np
import torch
from scipy.sparse.csgraph import minimum_spanning_tree
from torch import nn

from wheat599.metrics import pcc


def geometry(fit_x):
    """Balanced correlation tree, a connected material graph, and fixed block mixes."""
    x = np.asarray(fit_x, dtype=np.float64)
    n, p = x.shape
    centered = x - x.mean(0)
    norms = np.linalg.norm(centered, axis=0)
    z = np.divide(centered, norms, out=np.zeros_like(centered), where=norms > 0)
    correlation = np.abs(z.T @ z)
    children, groups, parents = [], {}, {}

    def split(markers):
        if len(markers) == 1:
            node = int(markers[0])
        else:
            a = markers[np.argmax(norms[markers])]
            b = markers[np.argmin(correlation[a, markers])]
            order = np.argsort(correlation[markers, a] - correlation[markers, b], kind='stable')
            left, right = np.array_split(markers[order], 2)
            l, r = split(left), split(right)
            node = p + len(children)
            children.append((l, r))
            parents[l] = parents[r] = node
        groups[node] = markers
        return node

    root = split(np.arange(p))
    blocks = [markers for node, markers in groups.items()
              if len(markers) <= 32 and (node == root or len(groups[parents[node]]) > 32)]
    counts = x.sum(1)
    distances = counts[:, None] + counts[None, :] - 2 * (x @ x.T)
    # Adding one preserves the MST and keeps identical genotypes connected.
    costs = distances + 1
    np.fill_diagonal(costs, 0)
    graph = minimum_spanning_tree(costs).tocoo()
    edges = np.column_stack((graph.row, graph.col)).astype(np.int64)
    if len(edges) != n - 1:
        raise ValueError('Material graph must connect every fitting material')
    mixed = x[edges[:, 0]].copy()
    generator = np.random.default_rng(42)
    for block in blocks:
        donor = generator.integers(0, 2, len(edges))
        mixed[:, block] = x[edges[np.arange(len(edges)), donor]][:, block]
    return children, edges, mixed


def bilinear(core, left, right):
    return torch.einsum('abc,nb,nc->na', core, left, right)


def gram(core, left, right):
    return torch.einsum('abc,nbd,nce,fde->naf', core, left, right, core)


def data_message(core, left, right):
    l0, l1, l2 = left
    r0, r1, r2 = right
    return (bilinear(core, l0, r0),
            bilinear(core, l1, r0) + bilinear(core, l0, r1),
            bilinear(core, l2, r0 + r1 + r2) + bilinear(core, l0, r2)
            + bilinear(core, l1, r1 + r2))


def norm_message(core, left, right):
    l0, l1, l2, ld1, ld2 = left
    r0, r1, r2, rd1, rd2 = right
    la, ra = l0 + l1 + l2, r0 + r1 + r2
    return (gram(core, l0, r0),
            gram(core, l1, r0) + gram(core, l0, r1),
            gram(core, l2, ra) + gram(core, l0, r2) + gram(core, l1, r1 + r2),
            gram(core, ld1, ra) + gram(core, la, rd1),
            gram(core, ld2, ra) + 2 * gram(core, ld1, rd1) + gram(core, la, rd2))


def high_terms(outside, left, right):
    e0, e1, e2 = outside[:3]
    l0, l1, l2 = left[:3]
    r0, r1, r2 = right[:3]
    return ((e2, l0 + l1 + l2, r0 + r1 + r2),
            (e0, l2, r0 + r1 + r2), (e0, l0, r2), (e0, l1, r1 + r2),
            (e1, l1 + l2, r0 + r1 + r2), (e1, l0, r1 + r2))


def covariance_product(a, b, c):
    value = torch.einsum('ad,be,cf->abcdef', a[0], b[0], c[0])
    size = a.shape[1] * b.shape[1] * c.shape[1]
    return value.reshape(size, size)


def node_norm(outside, left, right, eta):
    result = sum(covariance_product(*terms) for terms in high_terms(outside, left, right))
    if eta:
        ea, la, ra = sum(outside[:3]), sum(left[:3]), sum(right[:3])
        second = (covariance_product(outside[4], la, ra)
                  + covariance_product(ea, left[4], ra)
                  + covariance_product(ea, la, right[4])
                  + 2 * covariance_product(outside[3], left[3], ra)
                  + 2 * covariance_product(outside[3], la, right[3])
                  + 2 * covariance_product(ea, left[3], right[3]))
        result = result + eta * second
    return result


class TreeTensor(nn.Module):
    """Full additive path plus the degree-at-least-two part of a tree function."""
    def __init__(self, children, rank=4, seed=42):
        super().__init__()
        if not isinstance(rank, int) or rank < 1:
            raise ValueError('Tree rank must be a positive integer')
        self.nodes = children
        self.p = len(children) + 1
        self.intercept = nn.Parameter(torch.zeros((), dtype=torch.float64), requires_grad=False)
        self.beta = nn.Parameter(torch.zeros(self.p, dtype=torch.float64), requires_grad=False)
        self.cores = nn.ParameterList()
        sizes = [2] * self.p
        self.parents = {}
        generator = torch.Generator().manual_seed(seed)
        for i, (left, right) in enumerate(children):
            dl, dr = sizes[left], sizes[right]
            width = 1 if i == len(children) - 1 else min(rank, dl * dr)
            if width == 1 and i == len(children) - 1:
                flat = torch.randn(1, dl * dr, generator=generator, dtype=torch.float64)
                flat = .01 * flat / flat.norm()
            else:
                flat = torch.zeros(width, dl * dr, dtype=torch.float64)
                flat[0, 0] = 1
                q, _ = torch.linalg.qr(torch.randn(dl * dr - 1, width - 1,
                                                  generator=generator, dtype=torch.float64))
                flat[1:, 1:] = q.T
            self.cores.append(nn.Parameter(flat.reshape(width, dl, dr), requires_grad=False))
            sizes.append(width)
            self.parents[left] = (i, 1)
            self.parents[right] = (i, 2)

    def components(self, s):
        """Propagate disjoint 0, 1 and >=2 degree categories without subtraction."""
        values = {}

        def take(node):
            if node >= self.p:
                return values.pop(node)
            zero = torch.zeros_like(s[:, node])
            constant = torch.stack((torch.ones_like(zero), zero), 1)
            linear = torch.stack((zero, s[:, node]), 1)
            return constant, linear, torch.zeros_like(constant)

        for i, (left, right) in enumerate(self.nodes):
            values[self.p + i] = data_message(self.cores[i], take(left), take(right))
        return values[self.p + len(self.nodes) - 1]

    def interaction(self, s):
        return self.components(s)[2][:, 0]

    def forward(self, s):
        return self.intercept + s @ self.beta + self.interaction(s)

    def uniform_penalty(self, eta):
        """Original uniform-Walsh norm plus eta times its degree factorial moment."""
        values = {}
        leaf0 = torch.diag(self.beta.new_tensor([1., 0.]))[None]
        leaf1 = torch.diag(self.beta.new_tensor([0., 1.]))[None]
        zero = torch.zeros_like(leaf0)

        def take(node):
            return values.pop(node) if node >= self.p else (leaf0, leaf1, zero, leaf1, zero)

        for i, (left, right) in enumerate(self.nodes):
            values[self.p + i] = norm_message(self.cores[i], take(left), take(right))
        _, _, high, _, second = values[self.p + len(self.nodes) - 1]
        return high[0, 0, 0] + eta * second[0, 0, 0]

    def local_penalty(self, all_s, edges):
        """Fit rows followed by one fixed block recombination per graph edge."""
        h = self.interaction(all_s)
        child = h[-len(edges):]
        return .5 * ((h[edges[:, 0]] - child).square()
                     + (h[edges[:, 1]] - child).square()).mean()

    @torch.no_grad()
    def orthogonalize(self):
        for i in range(len(self.cores) - 1):
            self.orthogonalize_node(i)

    def orthogonalize_node(self, i):
        t = self.cores[i]
        q, r = torch.linalg.qr(t.flatten(1).T)
        t.copy_(q.T.reshape_as(t))
        parent, axis = self.parents[self.p + i]
        old = self.cores[parent]
        updated = (torch.einsum('abc,db->adc', old, r) if axis == 1 else
                   torch.einsum('abc,dc->abd', old, r))
        old.copy_(updated)


class ALSState:
    """Cached subtree messages; the current DFS path supplies outside messages."""
    def __init__(self, model, all_s, edges, target, ridge, interaction, eta):
        if ridge <= 0 or interaction < 0 or eta < 0:
            raise ValueError('Require positive additive ridge and nonnegative interaction penalties')
        self.model, self.s, self.edges, self.target = model, all_s, edges, target
        self.ridge, self.strength, self.eta = ridge, interaction, eta
        self.n = len(target)
        zero = torch.zeros_like(all_s[:, 0])
        constant = torch.stack((torch.ones_like(zero), zero), 1)
        empty = torch.zeros_like(constant)
        self.data = [(constant, torch.stack((zero, all_s[:, j]), 1), empty) for j in range(model.p)]
        leaf0 = torch.diag(model.beta.new_tensor([1., 0.]))[None]
        leaf1 = torch.diag(model.beta.new_tensor([0., 1.]))[None]
        leafz = torch.zeros_like(leaf0)
        self.norms = [(leaf0, leaf1, leafz, leaf1, leafz)] * model.p
        for i, (left, right) in enumerate(model.nodes):
            self.data.append(data_message(model.cores[i], self.data[left], self.data[right]))
            self.norms.append(norm_message(model.cores[i], self.norms[left], self.norms[right]))
        one = all_s.new_ones(len(all_s), 1)
        self.root_data = (one, torch.zeros_like(one), torch.zeros_like(one))
        one = all_s.new_ones(1, 1, 1)
        zero = torch.zeros_like(one)
        self.root_norm = (one, zero, zero, zero, zero)
        self.x_mean = all_s[:self.n].mean(0)
        self.centered = all_s[:self.n] - self.x_mean
        self.factor = torch.linalg.cholesky(
            self.centered @ self.centered.T
            + self.n * ridge * torch.eye(self.n, dtype=all_s.dtype, device=all_s.device))
        self.additive_update()

    def additive_update(self):
        y = self.target - self.data[-1][2][:self.n, 0]
        mean = y.mean()
        weights = torch.cholesky_solve((y - mean)[:, None], self.factor)[:, 0]
        self.model.beta.copy_(self.centered.T @ weights)
        self.model.intercept.copy_(mean - self.x_mean @ self.model.beta)

    def system(self, i, outside_data, outside_norm):
        left, right = self.model.nodes[i]
        phi = sum(torch.einsum('na,nb,nc->nabc', *parts).flatten(1)
                  for parts in high_terms(outside_data, self.data[left], self.data[right]))
        fit = phi[:self.n]
        difference = torch.cat((phi[self.edges[:, 0]] - phi[self.n:],
                                phi[self.edges[:, 1]] - phi[self.n:]))
        regularizer = node_norm(outside_norm, self.norms[left], self.norms[right], self.eta)
        q = fit.T @ fit / self.n + self.strength * (
            regularizer + difference.T @ difference / (2 * len(self.edges)))
        residual = self.target - self.model.intercept - self.s[:self.n] @ self.model.beta
        return (q + q.T) / 2, fit.T @ residual / self.n

    @torch.no_grad()
    def sweep(self):
        model = self.model

        def visit(i, outside_data, outside_norm):
            q, b = self.system(i, outside_data, outside_norm)
            model.cores[i].copy_((torch.linalg.pinv(q, hermitian=True) @ b).reshape_as(model.cores[i]))
            left, right = model.nodes[i]
            for child, sibling, permutation in ((left, right, (1, 0, 2)),
                                                 (right, left, (2, 0, 1))):
                if child >= model.p:
                    t = model.cores[i].permute(permutation)
                    visit(child - model.p,
                          data_message(t, outside_data, self.data[sibling]),
                          norm_message(t, outside_norm, self.norms[sibling]))
            if model.p + i in model.parents:
                model.orthogonalize_node(i)
            self.data[model.p + i] = data_message(model.cores[i], self.data[left], self.data[right])
            self.norms[model.p + i] = norm_message(model.cores[i], self.norms[left], self.norms[right])

        visit(len(model.cores) - 1, self.root_data, self.root_norm)
        self.additive_update()

    def objective(self):
        h = self.data[-1][2][:, 0]
        child = h[self.n:]
        local = .5 * ((h[self.edges[:, 0]] - child).square()
                      + (h[self.edges[:, 1]] - child).square()).mean()
        uniform = self.norms[-1][2][0, 0, 0] + self.eta * self.norms[-1][4][0, 0, 0]
        residual = self.target - self.model.intercept - self.s[:self.n] @ self.model.beta - h[:self.n]
        return residual.square().mean() + self.ridge * self.model.beta.square().sum() + self.strength * (uniform + local)


@torch.no_grad()
def train(x_fit, y_fit, x_val, y_val, config, seed, _index, settings, device):
    start = time.perf_counter()
    torch.manual_seed(int(seed))
    children, edges, mixed = geometry(x_fit)
    model = TreeTensor(children, rank=config['rank'], seed=seed).to(device)
    all_s = torch.as_tensor(2 * np.vstack((x_fit, mixed)) - 1, dtype=torch.float64, device=device)
    val_s = torch.as_tensor(2 * x_val - 1, dtype=torch.float64, device=device)
    graph = torch.as_tensor(edges, device=device)
    mean, std = float(np.mean(y_fit)), float(np.std(y_fit))
    if not np.isfinite(std) or std <= 0:
        raise ValueError('Invalid fit phenotype scale')
    target = torch.as_tensor((y_fit - mean) / std, dtype=torch.float64, device=device)
    state = ALSState(model, all_s, graph, target, config['lambda_A'], config['lambda_I'], config['eta'])
    training = settings['training']
    best_score, best_sweep, best_prediction = -np.inf, -1, None
    trace = []
    for sweep in range(training['max_sweeps'] + 1):
        if sweep:
            state.sweep()
        objective = float(state.objective())
        val_interaction = model.interaction(val_s)
        prediction = (model.intercept + val_s @ model.beta + val_interaction).cpu().numpy() * std + mean
        score = pcc(y_val, prediction)
        if not np.isfinite(objective) or not np.isfinite(score):
            raise FloatingPointError('Nonfinite ALS objective or undefined validation PCC')
        trace.append({'sweep': sweep, 'objective': objective, 'pcc': score,
                      'interaction_fit_rms': float(state.data[-1][2][:len(target), 0].square().mean().sqrt()),
                      'interaction_val_rms': float(val_interaction.square().mean().sqrt()),
                      'seconds': time.perf_counter() - start})
        if score > best_score + training.get('tie_tolerance', 1e-12):
            best_score, best_sweep, best_prediction = score, sweep, prediction.copy()
        elif sweep - best_sweep >= training['patience']:
            break
    return best_prediction, dict(best_epoch=best_sweep, epochs_run=sweep, iteration_unit='als_sweep',
        best_pcc=best_score, parameters=sum(p.numel() for p in model.parameters()),
        seconds=time.perf_counter() - start, trace=trace)
