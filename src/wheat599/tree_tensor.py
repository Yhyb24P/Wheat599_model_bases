"""Projected tree-tensor functions and fit-only local response regularization."""
import numpy as np
import torch
from scipy.sparse.csgraph import minimum_spanning_tree
from torch import nn


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
            l0, l1, l2 = take(left)
            r0, r1, r2 = take(right)
            t = self.cores[i]
            c0 = bilinear(t, l0, r0)
            c1 = bilinear(t, l1, r0) + bilinear(t, l0, r1)
            c2 = (bilinear(t, l2, r0 + r1 + r2) + bilinear(t, l0, r2)
                  + bilinear(t, l1, r1 + r2))
            values[self.p + i] = c0, c1, c2
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
            l0, l1, l2, ld1, ld2 = take(left)
            r0, r1, r2, rd1, rd2 = take(right)
            la, ra = l0 + l1 + l2, r0 + r1 + r2
            t = self.cores[i]
            c0 = gram(t, l0, r0)
            c1 = gram(t, l1, r0) + gram(t, l0, r1)
            c2 = gram(t, l2, ra) + gram(t, l0, r2) + gram(t, l1, r1 + r2)
            d1 = gram(t, ld1, ra) + gram(t, la, rd1)
            d2 = gram(t, ld2, ra) + 2 * gram(t, ld1, rd1) + gram(t, la, rd2)
            values[self.p + i] = c0, c1, c2, d1, d2
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
        for i, t in enumerate(self.cores[:-1]):
            q, r = torch.linalg.qr(t.flatten(1).T)
            t.copy_(q.T.reshape_as(t))
            parent, axis = self.parents[self.p + i]
            old = self.cores[parent]
            updated = (torch.einsum('abc,db->adc', old, r) if axis == 1 else
                       torch.einsum('abc,dc->abd', old, r))
            old.copy_(updated)
