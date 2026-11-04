"""Small shared network over joint-evidence candidates, with an exact readout.

The network never produces coefficients.  Each round it sees the current candidate set
through the conditional full-space score, the Hessian statistics and the previous
coefficients, emits a per-candidate non-negative precision scale ``d`` through a shared
node encoder plus one relation-message step, and the exact weighted readout of
``wheat599.joint_evidence`` turns ``d`` into ``(B, C)``.  Only ``d`` is learned: the
discrete indices come from a detached full-space score scan and carry no gradient.

Node inputs are sign-invariant: per-environment squared scores, log-scaled Hessian,
noise and additive diagonals, score norms, current coefficient norms and ``S . C``.
Relation features are the two sign-invariant combinations ``rho^2`` and
``rho * (S_i . S_j) / (||S_i|| ||S_j||)``.  Noise diagonals are geometry only and are
not claimed to be calibrated.  ``gate_features`` are context-only statistics and never
read the query.

``variant='independent'`` shares every parameter with ``'joint'`` and only zeros the
relation aggregation, so the two variants differ by the message step alone.
"""

import math

import torch
from torch import nn


class JointInference(nn.Module):
    """Joint-evidence candidate network with an exact linear-algebra readout."""

    def __init__(self, environments=4, width=32, rounds=3, add_per_round=16,
                 max_candidates=32, variant='joint', max_variance=100.0):
        super().__init__()
        if variant not in ('joint', 'independent'):
            raise ValueError("variant must be 'joint' or 'independent'")
        if environments < 1 or width < 1 or rounds < 1:
            raise ValueError('environments, width and rounds must be positive')
        if add_per_round < 1 or max_candidates < add_per_round:
            raise ValueError('candidate budget is inconsistent')
        self.environments = int(environments)
        self.width, self.rounds = int(width), int(rounds)
        self.add_per_round, self.max_candidates = int(add_per_round), int(max_candidates)
        self.variant, self.max_variance = variant, float(max_variance)
        self.node_feature_dim = self.environments + 8
        self.global_feature_dim = 12 + self.environments ** 2
        self.encoder = nn.Sequential(
            nn.Linear(self.node_feature_dim + self.global_feature_dim, width), nn.SiLU())
        self.edge = nn.Sequential(nn.Linear(2, width), nn.SiLU(), nn.Linear(width, 1))
        self.message = nn.Sequential(nn.Linear(2 * width, width), nn.SiLU())
        self.head = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 1))
        with torch.no_grad():
            # Initial variance scale is 0.1 in absolute units: solve
            # max_variance * sigmoid(bias) = 0.1 for the bias.
            target = min(0.1, 0.5 * self.max_variance)
            self.target_variance = target
            self.head[-1].bias.fill_(math.log(target / (self.max_variance - target)))

    @property
    def gate_feature_dim(self):
        return self.global_feature_dim

    # -- features ----------------------------------------------------------
    def _node_features(self, conditional_score, statistics, previous):
        norms = conditional_score.norm(dim=1)
        coefficient_norms = previous.norm(dim=1)
        denominator = (norms * coefficient_norms).clamp_min(1e-30)
        coupling = (conditional_score * previous).sum(dim=1) / denominator
        pooled = (norms - norms.mean()) / norms.std(unbiased=False).clamp_min(1e-30) \
            if norms.numel() > 1 else torch.zeros_like(norms)
        columns = [torch.log1p(conditional_score.square())[:, index]
                   for index in range(self.environments)]
        return torch.stack(columns + [
            torch.log1p(statistics['noise_diag']),
            torch.log1p(statistics['additive_diag']),
            torch.log1p(statistics['H'].diagonal()),
            norms,
            conditional_score.abs().max(dim=1).values,
            coefficient_norms,
            coupling,
            pooled,
        ], dim=1)

    def _context_features(self, evidence, energy, pool_norms, coefficients, additive_change):
        dtype, device = evidence.dtype, evidence.device
        scalar = lambda value: torch.as_tensor(float(value), dtype=dtype, device=device)
        selected = scalar(coefficients.shape[0])
        if coefficients.numel():
            coefficient_norms = coefficients.norm(dim=1)
            c_mean, c_max = coefficient_norms.mean(), coefficient_norms.max()
        else:
            c_mean = c_max = torch.zeros((), dtype=dtype, device=device)
        top = torch.topk(pool_norms, min(self.add_per_round, pool_norms.numel())).values
        count = (pool_norms > 0.5 * pool_norms.max().clamp_min(1e-30)).sum().to(dtype)
        signed = torch.sign(energy) * torch.log1p(energy.abs())
        return torch.cat([
            torch.stack([scalar(evidence.materials).log(), scalar(evidence.markers).log(),
                         scalar(evidence.pair_count).log(), scalar(evidence.lambda_a).log()]),
            signed.reshape(-1),
            torch.stack([pool_norms.max(), pool_norms.mean(), top.mean(),
                         count / scalar(evidence.pair_count)]),
            torch.stack([torch.log1p(selected), c_mean, c_max]),
            additive_change.reshape(1),
        ])

    def _aggregate(self, embedding, hessian, score):
        count = embedding.shape[0]
        if count < 2:
            return embedding + self.message(torch.cat([embedding,
                                                       torch.zeros_like(embedding)], dim=1))
        if self.variant == 'independent':
            return embedding + self.message(torch.cat([embedding,
                                                       torch.zeros_like(embedding)], dim=1))
        diagonal = hessian.diagonal().clamp_min(1e-30)
        rho = hessian / torch.sqrt(diagonal.unsqueeze(1) * diagonal.unsqueeze(0))
        norms = score.norm(dim=1)
        cosine = (score @ score.T) / (norms.unsqueeze(1) * norms.unsqueeze(0)).clamp_min(1e-30)
        features = torch.stack([rho.square(), rho * cosine], dim=-1)
        weights = self.edge(features).squeeze(-1)
        mask = torch.eye(count, dtype=torch.bool, device=embedding.device)
        weights = weights.masked_fill(mask, float('-inf'))
        weights = torch.softmax(weights, dim=1)
        return embedding + self.message(torch.cat([embedding, weights @ embedding], dim=1))

    def _additive_change(self, evidence, additive):
        fitted = evidence.a @ additive
        baseline = evidence.a @ evidence._ridge_b
        return (fitted - baseline).norm() / baseline.norm().clamp_min(1e-30)

    # -- forward -----------------------------------------------------------
    def forward(self, evidence, query):
        if not torch.is_tensor(query) or query.dtype != evidence.dtype \
                or query.device != evidence.device:
            raise ValueError('query must be a tensor on the evidence dtype and device')
        dtype, device = evidence.dtype, evidence.device
        if next(self.parameters()).dtype != dtype or next(self.parameters()).device != device:
            self.to(device=device, dtype=dtype)
        pool = evidence.score()
        pool_norms = pool.norm(dim=1)
        constants = evidence.constant_mask()
        indices = torch.empty(0, dtype=torch.long, device=device)
        coefficients = torch.zeros((0, self.environments), dtype=dtype, device=device)
        additive = evidence._ridge_b
        variance = torch.zeros(0, dtype=dtype, device=device)
        trace = []
        for round_index in range(self.rounds):
            with torch.no_grad():
                conditional = evidence.score(indices if indices.numel() else None,
                                             coefficients if indices.numel() else None)
                scores = conditional.norm(dim=1)
                available = ~constants.clone()
                if indices.numel():
                    available[indices] = False
                added = 0
                if bool(available.any()):
                    # Add a full batch of new candidates every round and trim to the
                    # budget afterwards, so a full set still refreshes add_per_round
                    # candidates instead of only one.
                    take = min(self.add_per_round, int(available.sum()))
                    values, chosen = torch.topk(scores.masked_fill(~available, float('-inf')),
                                                take)
                    indices = torch.cat([indices, chosen])
                    coefficients = torch.cat([coefficients, torch.zeros(
                        (chosen.numel(), self.environments), dtype=dtype, device=device)])
                    added = int(chosen.numel())
            if not indices.numel():
                trace.append({'round': round_index, 'selected': 0, 'added': 0})
                continue
            statistics = evidence.statistics(indices)
            conditional_score = evidence.restricted_score(indices, coefficients)
            context = self._context_features(
                evidence, statistics['energy'], pool_norms, coefficients,
                self._additive_change(evidence, additive).detach())
            nodes = self._node_features(conditional_score, statistics, coefficients)
            embedding = self.encoder(torch.cat([nodes, context.expand(nodes.shape[0], -1)],
                                               dim=1))
            embedding = self._aggregate(embedding, statistics['H'], conditional_score)
            variance = self.max_variance * torch.sigmoid(self.head(embedding).squeeze(-1))
            additive, coefficients = evidence.readout(indices, variance)
            if indices.numel() > self.max_candidates:
                with torch.no_grad():
                    keep = torch.sort(torch.topk(coefficients.norm(dim=1),
                                                 self.max_candidates).indices).values
                indices, variance = indices[keep], variance[keep]
                additive, coefficients = evidence.readout(indices, variance)
            trace.append({'round': round_index, 'selected': int(indices.numel()),
                          'added': added, 'variance_mean': float(variance.mean()),
                          'variance_zero': int((variance == 0).sum()),
                          'score_max': float(scores.max())})
        prediction = evidence.predict(query, indices, additive, coefficients)
        gate = self._context_features(evidence, evidence.energy(), pool_norms, coefficients,
                                      self._additive_change(evidence, additive))
        return {'prediction': prediction,
                'additive_prediction': evidence.ridge(query),
                'indices': indices,
                'd': variance if indices.numel() else torch.zeros(0, dtype=dtype,
                                                                  device=device),
                'coefficients': coefficients,
                'gate_features': gate,
                'trace': trace}
