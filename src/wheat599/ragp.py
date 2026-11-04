"""RAGP reference adaptation for Wheat599 at upstream commit 85675b1e.

The upstream model (https://github.com/l00907l/RAGP) is four independent single-output
regression networks, one per environment. Each network encodes fit materials, retrieves
the five cosine-nearest fit materials, fuses the query encoding with the retrieved
feature mean (0.4/0.6), appends 0.01 times the retrieved label mean, and passes the
result through residual blocks to a scalar output. The library is refreshed without
gradients at the end of every epoch. Training uses batch Pearson loss plus
``0.1 * loss_alpha`` times a cosine triplet loss.

This adaptation keeps those mechanisms and the official Wheat599 defaults
(``hidden_dims=(2500, 3000, 3000, 1500)``, three blocks, dropout 0.5, top-5 neighbours,
fusion 0.4, label scale 0.01, triplet anchors 32/5/5, margin 1, ``loss_alpha``
0.9/0.8/0.5/0.1). It returns one tensor ``(n, 1, q)``; the member axis is one because the
four environments are separate single-output models, not ensemble members.

Documented differences from upstream:

* PCA95 and phenotype standardisation are fitted on the current fit rows only. The
  released Wheat599 PC file is a scikit-learn PCA (centred, not scaled) fitted on all
  599 rows, and its Y files are globally z-scored.
* Self-retrieval is excluded by material ID equality, not by subtracting the identity
  from the cosine matrix (which sets self-cosine to zero rather than ``-inf`` and does
  not handle repeated IDs).
* The triplet indices are aligned with the embeddings used by the loss. Upstream
  samples a 32-row subset to choose positives/negatives but then embeds the first 32
  rows of the batch.
* The four environments share one module and one optimiser. Their parameters are
  disjoint and their losses are summed, so with equal learning-rate schedules this is
  equivalent to four separate Adam runs.
* The first epoch uses upstream's zero-label fallback because the library is empty
  until the first ``update_library``. Call ``update_library`` after every training epoch.
  ``load_state_dict`` rebuilds the library from the loaded encoders, so a selected
  checkpoint cannot be evaluated against a stale library.
* Pearson loss guards a zero-variance batch instead of returning NaN. The upstream
  unused ``alpha`` parameter is omitted. The upstream single 80/20 split is replaced by
  the project's outer/inner splits in the runner.
* Initialisation and dropout consume the caller's RNG across all four environments;
  upstream resets its torch seed per environment. This is a same-protocol adaptation,
  not a bitwise reproduction. Training batches need at least two materials because the
  official blocks use BatchNorm1d.

The upstream repository had no LICENSE file at the pinned commit. This module is an
independent reimplementation of the published mechanism, not copied upstream source.

The forward pass never accepts query labels. Reference labels are fixed fit-row buffers.
Library buffers are non-persistent: ``update_library`` is called after every training
epoch, and ``load_state_dict`` rebuilds them from the loaded encoder weights so a
selected checkpoint cannot be evaluated with a stale library.
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torch import nn

OFFICIAL_HIDDEN_DIMS = (2500, 3000, 3000, 1500)


class RAGPBlock(nn.Module):
    """The upstream residual block: LayerNorm, linear, BatchNorm, ReLU, dropout, linear."""

    def __init__(self, hidden_dim, dropout, dtype=None):
        super().__init__()
        self.layernorm = nn.LayerNorm(hidden_dim, dtype=dtype)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim, dtype=dtype)
        self.batchnorm = nn.BatchNorm1d(hidden_dim, dtype=dtype)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim, dtype=dtype)

    def forward(self, x):
        residual = x
        out = self.layernorm(x)
        out = self.linear1(out)
        out = self.batchnorm(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.linear2(out)
        return out + residual


class RAGPEnvironment(nn.Module):
    """One single-output RAGP model: encoder, retrieval library and prediction head."""

    def __init__(self, input_dim, hidden_dim, reference_rows, neighbors, n_blocks,
                 dropout, dtype=None):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.reference_rows = int(reference_rows)
        self.neighbors = int(neighbors)
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim, dtype=dtype),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim, dtype=dtype),
        )
        self.blocks = nn.ModuleList(
            [RAGPBlock(self.hidden_dim + 1, dropout, dtype=dtype) for _ in range(n_blocks)])
        self.fc = nn.Linear(self.hidden_dim + 1, 1, dtype=dtype)
        self.register_buffer(
            'library_features',
            torch.zeros(self.reference_rows, self.hidden_dim, dtype=dtype),
            persistent=False)
        self.register_buffer(
            'neighbor_indices',
            torch.zeros(self.reference_rows, self.neighbors, dtype=torch.long),
            persistent=False)
        self.register_buffer(
            'neighbor_similarities',
            torch.zeros(self.reference_rows, self.neighbors, dtype=dtype),
            persistent=False)
        self.library_ready = False

    def encode(self, features):
        return self.encoder(features)

    def forward(self, inputs):
        for block in self.blocks:
            inputs = block(inputs)
        return self.fc(inputs).squeeze(-1)

    @torch.no_grad()
    def update_library(self, reference_features, reference_ids):
        """Refresh features and strict-ID top-k neighbours from the current encoder."""
        features = self.encoder(reference_features)
        normalized = F.normalize(features, dim=-1)
        similarity = normalized @ normalized.transpose(0, 1)
        same_identity = reference_ids[:, None] == reference_ids[None, :]
        similarity = similarity.masked_fill(same_identity, float('-inf'))
        values, indices = similarity.topk(self.neighbors, dim=1, largest=True, sorted=True)
        self.library_features.copy_(features.detach())
        self.neighbor_similarities.copy_(values)
        self.neighbor_indices.copy_(indices)
        self.library_ready = True


def pearson_loss(prediction, target, eps=1e-8):
    """Upstream batch Pearson loss: 1 - cov / (std_pred * std_target)."""
    if prediction.numel() < 2:
        return prediction.sum() * 0.0 + 1.0
    covariance = ((prediction - prediction.mean()) * (target - target.mean())).mean()
    denominator = prediction.std() * target.std()
    return 1.0 - covariance / denominator.clamp_min(eps)


def triplet_loss(embeddings, labels, anchors=32, positives=5, negatives=5, margin=1.0):
    """Upstream cosine triplet objective with correctly aligned anchor indices.

    Anchors are sampled from the current batch; the closest and farthest labels within
    that sampled set select positives and negatives in the same set. The upstream
    implementation samples labels from a random subset but embeds the first rows of the
    batch, so this function intentionally fixes that mismatch.
    """
    if embeddings.ndim != 2 or labels.ndim != 1:
        raise ValueError('embeddings must be (n, d) and labels must be (n,)')
    if embeddings.shape[0] != labels.shape[0]:
        raise ValueError('embeddings and labels must share the batch axis')
    if embeddings.shape[0] < 2:
        return embeddings.sum() * 0.0
    chosen = torch.randperm(embeddings.shape[0], device=embeddings.device)
    chosen = chosen[:min(int(anchors), embeddings.shape[0])]
    encoded = F.normalize(embeddings[chosen], dim=-1)
    selected_labels = labels[chosen]
    difference = (selected_labels[:, None] - selected_labels[None, :]).abs()
    diagonal = torch.eye(len(chosen), dtype=torch.bool, device=embeddings.device)
    difference = difference.masked_fill(diagonal, float('inf'))
    count = len(chosen) - 1
    positive_count, negative_count = min(int(positives), count), min(int(negatives), count)
    if positive_count < 1 or negative_count < 1:
        return embeddings.sum() * 0.0
    positive = (-difference).topk(positive_count, dim=1).indices
    negative = difference.masked_fill(diagonal, float('-inf')).topk(
        negative_count, dim=1).indices
    positive_dot = (encoded[:, None, :] * encoded[positive]).sum(-1)
    negative_dot = (encoded[:, None, :] * encoded[negative]).sum(-1)
    losses = F.relu(margin - positive_dot[:, :, None] + negative_dot[:, None, :])
    return losses.sum(dim=(1, 2)).mean()


class RAGPGenomic(nn.Module):
    """Four independent RAGP environments with fit-only PCA95 and fit-local libraries."""

    def __init__(self, train_x, train_y, train_ids=None, *,
                 hidden_dims=OFFICIAL_HIDDEN_DIMS, n_blocks=3, dropout=0.5,
                 pca_variance=0.95, neighbors=5, fusion_alpha=0.4, label_scale=0.01,
                 loss_alpha=(0.9, 0.8, 0.5, 0.1), triplet_anchors=32,
                 triplet_positives=5, triplet_negatives=5, triplet_margin=1.0,
                 dtype=torch.float32):
        super().__init__()
        x = torch.as_tensor(train_x, dtype=torch.float64).detach().cpu()
        y = torch.as_tensor(train_y, dtype=dtype).detach().cpu()
        if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
            raise ValueError('train_x and train_y must share the material axis')
        if x.shape[0] < 2:
            raise ValueError('At least two fit materials are required')
        environments = int(y.shape[1])
        hidden_dims = tuple(int(value) for value in hidden_dims)
        if len(hidden_dims) != environments or any(value <= 0 for value in hidden_dims):
            raise ValueError('hidden_dims must provide one positive width per environment')
        if int(n_blocks) < 1:
            raise ValueError('n_blocks must be positive')
        if not 0.0 <= float(dropout) <= 1.0:
            raise ValueError('dropout must lie in [0, 1]')
        if not 0.0 < float(fusion_alpha) <= 1.0:
            raise ValueError('fusion_alpha must lie in (0, 1]')
        if float(label_scale) < 0.0:
            raise ValueError('label_scale must be non-negative')
        loss_alpha = tuple(float(value) for value in loss_alpha)
        if len(loss_alpha) != environments:
            raise ValueError('loss_alpha must provide one weight per environment')
        reference_rows = int(x.shape[0])
        neighbors = int(neighbors)
        if not 0 < neighbors < reference_rows:
            raise ValueError('neighbors must be positive and smaller than the fit set')

        pca = PCA(n_components=pca_variance, svd_solver='full').fit(x.numpy())
        mean = torch.as_tensor(pca.mean_, dtype=dtype)
        components = torch.as_tensor(pca.components_, dtype=dtype)
        reference_x = (x.to(dtype) - mean) @ components.T
        self.dtype = dtype
        self.markers = int(x.shape[1])
        self.environment_count = environments
        self.hidden_dims = hidden_dims
        self.n_blocks = int(n_blocks)
        self.dropout = float(dropout)
        self.pca_variance = pca_variance
        self.pca_explained_variance = float(pca.explained_variance_ratio_.sum())
        self.neighbors = neighbors
        self.fusion_alpha = float(fusion_alpha)
        self.label_scale = float(label_scale)
        self.loss_alpha = loss_alpha
        self.triplet_anchors = int(triplet_anchors)
        self.triplet_positives = int(triplet_positives)
        self.triplet_negatives = int(triplet_negatives)
        self.triplet_margin = float(triplet_margin)
        if min(self.triplet_anchors, self.triplet_positives, self.triplet_negatives) < 1:
            raise ValueError('triplet counts must be positive')

        self.register_buffer('pca_mean', mean.to(dtype))
        self.register_buffer('pca_components', components.to(dtype))
        self.register_buffer('reference_x', reference_x)
        self.register_buffer('y_mean', y.mean(dim=0))
        self.register_buffer('y_std', y.std(dim=0, unbiased=False).clamp_min(1e-12))
        self.register_buffer('reference_y', (y - self.y_mean) / self.y_std)

        if train_ids is None:
            ids = torch.arange(reference_rows, dtype=torch.long)
        else:
            ids = torch.as_tensor(train_ids, dtype=torch.long).detach().cpu().reshape(-1)
        if ids.numel() != reference_rows:
            raise ValueError('train_ids must provide one ID per fit material')
        if torch.unique(ids).numel() != reference_rows:
            raise ValueError('train_ids must be unique')
        self.register_buffer('reference_ids', ids)
        order = torch.argsort(ids)
        self.register_buffer('reference_order', order, persistent=False)
        self.register_buffer('reference_sorted_ids', ids[order], persistent=False)

        self.environments = nn.ModuleList(
            [RAGPEnvironment(reference_x.shape[1], hidden, reference_rows, neighbors,
                             self.n_blocks, self.dropout, dtype=dtype)
             for hidden in self.hidden_dims])

    # -- preprocessing -----------------------------------------------------
    def standardize_x(self, x):
        x = torch.as_tensor(x, dtype=self.dtype, device=self.pca_mean.device)
        if x.ndim != 2 or x.shape[1] != self.pca_components.shape[1]:
            raise ValueError('x must be a marker matrix with the fit feature count')
        return (x - self.pca_mean) @ self.pca_components.T

    def standardize_y(self, y):
        y = torch.as_tensor(y, dtype=self.dtype, device=self.y_mean.device)
        if y.ndim != 2 or y.shape[1] != self.environment_count:
            raise ValueError('y must be a material-by-environment matrix')
        return (y - self.y_mean) / self.y_std

    def inverse_y(self, prediction):
        return prediction * self.y_std + self.y_mean

    # -- identity lookup ---------------------------------------------------
    def _training_positions(self, query_ids, batch):
        ids = torch.as_tensor(query_ids, dtype=torch.long,
                              device=self.reference_ids.device).reshape(-1)
        if ids.numel() != batch:
            raise ValueError('query_ids must provide one ID per query material')
        positions = torch.searchsorted(self.reference_sorted_ids, ids)
        valid = positions < self.reference_sorted_ids.numel()
        positions = positions.clamp_max(self.reference_sorted_ids.numel() - 1)
        valid = valid & (self.reference_sorted_ids[positions] == ids)
        if not bool(valid.all()):
            raise ValueError('Training query_ids must all occur in train_ids')
        return self.reference_order[positions]

    def _evaluation_mask(self, query_ids, batch):
        if query_ids is None:
            return None
        ids = torch.as_tensor(query_ids, dtype=torch.long,
                              device=self.reference_ids.device).reshape(-1)
        if ids.numel() != batch:
            raise ValueError('query_ids must provide one ID per query material')
        return ids[:, None] == self.reference_ids[None, :]

    # -- retrieval ---------------------------------------------------------
    def _fuse(self, environment, encoded, indices, similarities, environment_index):
        weights = F.softmax(similarities, dim=-1)
        neighbors = environment.library_features[indices]
        fused = (self.fusion_alpha * encoded
                 + (1.0 - self.fusion_alpha) * (weights.unsqueeze(-1) * neighbors).sum(1))
        labels = self.reference_y[indices, environment_index]
        label_mean = (weights * labels).sum(1, keepdim=True) * self.label_scale
        return torch.cat([fused, label_mean], dim=-1)

    @staticmethod
    def _fallback(encoded):
        return torch.cat([encoded, encoded.new_zeros(encoded.shape[0], 1)], dim=-1)

    def forward(self, x, query_ids=None, return_embeddings=False):
        features = self.standardize_x(x)
        batch = features.shape[0]
        embeddings = []
        if self.training:
            if query_ids is None:
                raise ValueError('Training requires query_ids for strict self exclusion')
            positions = self._training_positions(query_ids, batch)
            predictions = []
            for index, environment in enumerate(self.environments):
                encoded = environment.encode(features)
                if environment.library_ready:
                    neighbors = environment.neighbor_indices[positions]
                    similarities = environment.neighbor_similarities[positions]
                    inputs = self._fuse(environment, encoded, neighbors, similarities, index)
                else:
                    inputs = self._fallback(encoded)
                predictions.append(environment(inputs))
                embeddings.append(encoded)
            prediction = torch.stack(predictions, dim=-1).unsqueeze(1)
        else:
            with torch.no_grad():
                mask = self._evaluation_mask(query_ids, batch)
                predictions = []
                for index, environment in enumerate(self.environments):
                    encoded = environment.encode(features)
                    if environment.library_ready:
                        normalized_query = F.normalize(encoded, dim=-1)
                        normalized_library = F.normalize(environment.library_features, dim=-1)
                        similarities = normalized_query @ normalized_library.T
                        if mask is not None:
                            similarities = similarities.masked_fill(mask, float('-inf'))
                        available = torch.isfinite(similarities).sum(dim=1)
                        if int(available.min()) < self.neighbors:
                            raise ValueError('Not enough reference materials after ID exclusion')
                        values, neighbors = similarities.topk(
                            self.neighbors, dim=1, largest=True, sorted=True)
                        inputs = self._fuse(environment, encoded, neighbors, values, index)
                    else:
                        inputs = self._fallback(encoded)
                    predictions.append(environment(inputs))
                    embeddings.append(encoded)
                prediction = torch.stack(predictions, dim=-1).unsqueeze(1)
        if return_embeddings:
            return prediction, embeddings
        return prediction

    # -- losses ------------------------------------------------------------
    def loss(self, prediction, target, embeddings=None, *, include_triplet=True,
             return_terms=False):
        if prediction.ndim != 3 or prediction.shape[1] != 1 \
                or prediction.shape[2] != self.environment_count:
            raise ValueError('prediction must have shape (n, 1, environments)')
        if target.ndim == 3 and target.shape[1] == 1:
            target = target[:, 0, :]
        if target.ndim != 2 or target.shape != (prediction.shape[0], self.environment_count):
            raise ValueError('target must have shape (n, environments)')
        target = target.to(device=prediction.device, dtype=prediction.dtype)
        flat = prediction[:, 0, :]
        pearson = [pearson_loss(flat[:, index], target[:, index])
                   for index in range(self.environment_count)]
        terms = list(pearson)
        triplet, weighted = [], []
        if include_triplet:
            if embeddings is None or len(embeddings) != self.environment_count:
                raise ValueError('embeddings must provide one tensor per environment')
            for index, encoded in enumerate(embeddings):
                if encoded.ndim != 2 or encoded.shape[0] != prediction.shape[0]:
                    raise ValueError('each embedding must have shape (n, hidden width)')
                if encoded.device != prediction.device:
                    raise ValueError('embeddings must share the prediction device')
                value = triplet_loss(encoded, target[:, index],
                                     anchors=self.triplet_anchors,
                                     positives=self.triplet_positives,
                                     negatives=self.triplet_negatives,
                                     margin=self.triplet_margin)
                weight = 0.1 * self.loss_alpha[index]
                triplet.append(value)
                weighted.append(weight * value)
                terms.append(weight * value)
        total = torch.stack(terms).sum()
        if return_terms:
            return total, {
                'pearson': [float(value.detach().cpu()) for value in pearson],
                'triplet': [float(value.detach().cpu()) for value in triplet],
                'weighted_triplet': [float(value.detach().cpu()) for value in weighted],
            }
        return total

    # -- reference library -------------------------------------------------
    @torch.no_grad()
    def update_library(self):
        for environment in self.environments:
            environment.update_library(self.reference_x, self.reference_ids)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Load weights and rebuild every library from the loaded encoders."""
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        order = torch.argsort(self.reference_ids)
        self.reference_order.copy_(order)
        self.reference_sorted_ids.copy_(self.reference_ids[order])
        self.update_library()
        return result

    # -- bookkeeping -------------------------------------------------------
    def parameter_count(self):
        return int(sum(parameter.numel() for parameter in self.parameters()))

    def ridge_penalty(self):
        return self.reference_x.new_zeros(())

    def metadata(self):
        return {
            'variant': 'ragp',
            'markers': self.markers,
            'environments': self.environment_count,
            'pca_components': int(self.pca_components.shape[0]),
            'pca_variance_requested': self.pca_variance,
            'pca_variance_explained': self.pca_explained_variance,
            'hidden_dims': list(self.hidden_dims),
            'n_blocks': self.n_blocks,
            'dropout': self.dropout,
            'neighbors': self.neighbors,
            'fusion_alpha': self.fusion_alpha,
            'label_scale': self.label_scale,
            'loss_alpha': list(self.loss_alpha),
            'triplet_anchors': self.triplet_anchors,
            'triplet_positives': self.triplet_positives,
            'triplet_negatives': self.triplet_negatives,
            'triplet_margin': self.triplet_margin,
            'reference_rows': int(self.reference_x.shape[0]),
            'library_ready': [bool(environment.library_ready)
                              for environment in self.environments],
            'library_update': 'manual after each training epoch; rebuilt by load_state_dict',
            'parameters': self.parameter_count(),
            'upstream': {
                'url': 'https://github.com/l00907l/RAGP',
                'commit': '85675b1e8054fe3e91fdb3e974150f1dde81eee3',
                'license': 'no LICENSE file at the pinned commit; independent implementation',
            },
        }
