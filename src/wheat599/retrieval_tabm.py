"""Full-marker TabM with member-specific TabR-style reference aggregation.

References are encoded using current weights on every forward. Encoder dropout is
disabled for BOTH query and reference retrieval representations, keeping distances
in the same deterministic space; predictor dropout retains the training mode.
The first official TabM block produces the member representation, with retrieval
inserted before the remaining blocks. All reference keys retain gradients.
"""
import torch
from torch import nn

from wheat599.genomic_tabm import GenomicTabM


class RetrievalTabM(GenomicTabM):
    def __init__(self, train_x, train_y, *, variant='retrieval', n_blocks=3,
                 d_block=256, dropout=0.1, k=32, context_size=8,
                 dtype=torch.float32):
        if variant not in ('raw', 'retrieval', 'single_retrieval'):
            raise ValueError('Unknown retrieval variant')
        if n_blocks < 2:
            raise ValueError('At least two blocks are required for internal retrieval')
        if variant == 'single_retrieval':
            k = 1
        super().__init__(train_x, train_y, variant='raw', n_blocks=n_blocks,
                         d_block=d_block, dropout=dropout, k=k, dtype=dtype)
        self.variant = variant
        self.k = k
        self.context_size = int(context_size)
        if self.context_size < 1:
            raise ValueError('context_size must be positive')
        self.register_buffer('reference_x', torch.as_tensor(train_x, dtype=dtype).detach().cpu().clone())
        self.register_buffer('reference_y', self.standardize_y(
            torch.as_tensor(train_y, dtype=dtype).detach().cpu()).clone())
        if len(self.reference_x) < 2:
            raise ValueError('At least two fit materials required')
        if variant != 'raw':
            self.key = nn.Linear(d_block, d_block, dtype=dtype)
            self.label_encoder = nn.Linear(self.environments, d_block, dtype=dtype)
            self.difference = nn.Sequential(nn.Linear(d_block, d_block, dtype=dtype),
                                            nn.ReLU(),
                                            nn.Linear(d_block, d_block, bias=False, dtype=dtype))

    def _encode(self, x):
        block = self.backbone.backbone.blocks[0]
        # Restore every submodule's mode even if encoding raises. No running
        # statistics exist in official TabM blocks, so this changes dropout only.
        modes = [(module, module.training) for module in block.modules()]
        block.eval()
        try:
            return block(self.backbone.ensemble_view(self.standardize_x(x)))
        finally:
            for module, mode in modes:
                module.training = mode

    def encode_reference(self):
        if self.variant == 'raw':
            return None
        return self.key(self._encode(self.reference_x))

    def forward(self, x, query_ids=None):
        if self.variant == 'raw':
            return self.backbone(self.standardize_x(x))
        if self.training and query_ids is None:
            raise ValueError('Training requires fit-local query_ids for self exclusion')
        h = self._encode(x)
        query = self.key(h)
        references = self.encode_reference()
        # (members, queries, references), in a shared learned key space.
        q, r = query.transpose(0, 1), references.transpose(0, 1)
        distance = (q.square().sum(-1, keepdim=True)
                    + r.square().sum(-1).unsqueeze(1) - 2 * q @ r.transpose(1, 2))
        count = len(self.reference_x)
        if query_ids is not None:
            ids = torch.as_tensor(query_ids, device=distance.device, dtype=torch.long)
            if ids.shape != (len(x),) or bool(((ids < 0) | (ids >= count)).any()):
                raise ValueError('query_ids must be fit-local row IDs')
            mask = torch.arange(count, device=distance.device)[None, :] == ids[:, None]
            distance = distance.masked_fill(mask.unsqueeze(0), float('inf'))
            count -= 1
        neighbors = min(self.context_size, count)
        indices = distance.detach().topk(neighbors, dim=-1, largest=False).indices
        scores = -distance.gather(-1, indices)
        attention = scores.softmax(-1)
        member = torch.arange(self.k, device=r.device)[:, None, None]
        selected_keys = r[member, indices]
        values = (self.label_encoder(self.reference_y[indices])
                  + self.difference(q.unsqueeze(2) - selected_keys))
        context = (attention.unsqueeze(-1) * values).sum(2).transpose(0, 1)
        self.last_neighbor_indices = indices.detach().transpose(0, 1)
        self.last_neighbor_weights = attention.detach().transpose(0, 1)
        h = h + context
        for block in self.backbone.backbone.blocks[1:]:
            h = block(h)
        return self.backbone.output(h)

    def ridge_penalty(self):
        return self.reference_x.new_zeros(())

    def group_metadata(self):
        return {'variant': self.variant, 'markers': self.markers, 'members': self.k,
                'reference_rows': len(self.reference_x), 'context_size': self.context_size,
                'retrieval_after_block': 1, 'encoder_dropout': False,
                'reference_encoding': 'current weights, every forward, with gradients'}
