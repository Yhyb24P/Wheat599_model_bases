"""Direct multi-output in-context predictor for matched synthetic tasks."""

import torch
from torch import nn


class DirectPredictor(nn.Module):
    """Predict each query from genotypes and labelled context materials."""

    def __init__(self, markers, environments=4, width=64, heads=4, hidden=128, layers=4):
        super().__init__()
        self.markers = markers
        self.environments = environments
        self.input = nn.Linear(markers + environments + 1 + environments ** 2 + 3, width)
        layer = nn.TransformerEncoderLayer(width, heads, hidden, dropout=0.,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.output = nn.Linear(width, environments)

    def forward(self, context_x, context_y, query_x, task_covariance, fraction, ridge, bandwidth):
        n, q = context_y.shape
        m = query_x.shape[0]
        if (context_x.shape != (n, self.markers) or query_x.shape[1] != self.markers
                or q != self.environments or task_covariance.shape != (q, q)):
            raise ValueError('Incompatible context or query dimensions')
        x = torch.cat((context_x, query_x), dim=0)
        mean = context_y.mean(dim=0, keepdim=True)
        y = torch.cat((context_y - mean, context_y.new_zeros((m, q))), dim=0)
        observed = torch.cat((context_y.new_ones((n, 1)), context_y.new_zeros((m, 1))))
        scalars = torch.as_tensor([fraction, ridge, bandwidth], dtype=x.dtype, device=x.device)
        parameters = torch.cat((task_covariance.reshape(-1), scalars)).expand(n + m, -1)
        tokens = self.input(torch.cat((x, y, observed, parameters), dim=1)).unsqueeze(0)
        # Context tokens cannot attend queries; each query may attend context
        # and itself, so predictions do not depend on other query genotypes.
        mask = torch.ones((n + m, n + m), dtype=torch.bool, device=x.device)
        mask[:, :n] = False
        indices = torch.arange(m, device=x.device) + n
        mask[indices, indices] = False
        return mean + self.output(self.encoder(tokens, mask=mask).squeeze(0)[n:])
