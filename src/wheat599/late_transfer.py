"""Independent TabM specialists with cross-fitted, target-specific prediction transfer.

Each specialist trains on one phenotype. After freezing it, linear readouts of its
hidden member representations predict the other phenotypes. Outputs are in phenotype
units, so banks trained on different material subsets have aligned stacking features
even though their hidden coordinates need not align.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.model_selection import KFold
from torch import nn

from wheat599.genomic_tabm import GenomicTabM


class Source(nn.Module):
    def __init__(self, network, hidden_size, targets):
        super().__init__()
        self.network = network
        self.register_buffer("readout", network.y_mean.new_zeros(hidden_size, targets))
        self.register_buffer("intercept", network.y_mean.new_zeros(targets))

    def encode(self, x):
        net = self.network
        tabm = net.backbone
        hidden = tabm.backbone(tabm.ensemble_view(net.standardize_x(x)))
        own = net.inverse_y(tabm.output(hidden).mean(dim=1))
        return own, hidden.flatten(start_dim=1)

    def forward(self, x):
        own, hidden = self.encode(x)
        return own, hidden @ self.readout + self.intercept


def fit_transfer_head(hidden, y, ridge):
    """Primal coefficients of a linear readout. Normalise the Gram mean diagonal to one."""
    if ridge <= 0:
        raise ValueError("The transfer ridge must be positive")
    h, y = hidden.double(), y.double()
    h0, y0 = h.mean(0), y.mean(0)
    centered = h - h0
    scale = centered.square().sum() / len(h)
    if scale <= torch.finfo(h.dtype).eps:
        return h.new_zeros(h.shape[1], y.shape[1]), y0
    gram = (centered @ centered.T) / scale
    gram.diagonal().add_(ridge)
    dual = torch.cholesky_solve(y - y0, torch.linalg.cholesky(gram))
    weights = centered.T @ dual / scale
    return weights, y0 - h0 @ weights


def new_source(x, y, profile, seed, targets):
    torch.manual_seed(seed)
    architecture = {key: profile[key] for key in ("n_blocks", "d_block", "dropout", "k")}
    network = GenomicTabM(x, y, variant="raw", **architecture).to(x.device)
    return Source(network, profile["k"] * profile["d_block"], targets)


def train_source(x, y, target, profile, seed, recipe):
    source = new_source(x, y[:, target:target + 1], profile, seed, y.shape[1])
    net = source.network
    opt = torch.optim.AdamW(net.parameters(), lr=profile["learning_rate"],
                            weight_decay=profile["weight_decay"])
    truth = net.standardize_y(y[:, target:target + 1]).unsqueeze(1)
    rng = torch.Generator(device="cpu").manual_seed(seed * 1000 + recipe["profile_indices"][target])
    net.train()
    for _ in range(recipe["epochs"]):
        order = torch.randperm(len(x), generator=rng).to(x.device)
        for batch in order.split(recipe["batch_size"]):
            opt.zero_grad(set_to_none=True)
            loss = (net(x[batch]) - truth[batch]).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite specialist training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), recipe["gradient_clip"])
            opt.step()
    source.eval().requires_grad_(False)
    with torch.no_grad():
        _, hidden = source.encode(x)
        weights, bias = fit_transfer_head(hidden, y, recipe["transfer_ridge"])
        source.readout.copy_(weights)
        source.intercept.copy_(bias)
    return source


class Bank(nn.Module):
    def __init__(self, sources):
        super().__init__()
        self.sources = nn.ModuleList(nn.ModuleList(group) for group in sources)

    def forward(self, x):
        own, transferred = [], []
        for group in self.sources:
            outputs = [source(x) for source in group]
            own.append(torch.stack([p for p, _ in outputs]).mean(0))
            transferred.append(torch.stack([v for _, v in outputs]).mean(0))
        # Axes: materials x target (for the transfer head) x source.
        return torch.cat(own, dim=1), torch.stack(transferred, dim=-1)


def fit_bank(x, y, recipe):
    return Bank([
        [train_source(x, y, t, profile, seed, recipe) for seed in recipe["seeds"]]
        for t, profile in enumerate(recipe["profiles"])
    ]).eval()


def readout_features(own, transferred, mode):
    q = own.shape[1]
    if mode == "self_calibration":
        return own.unsqueeze(-1)
    scalar = own[:, None, :].expand(-1, q, -1)
    if mode == "score_transfer":
        return scalar
    if mode != "hidden_transfer":
        raise ValueError(f"Unknown readout: {mode}")
    off_diagonal = ~torch.eye(q, dtype=torch.bool, device=own.device)
    other = transferred[:, off_diagonal].reshape(len(own), q, q - 1)
    return torch.cat([scalar, other], dim=-1)


class ResidualReadout(nn.Module):
    def __init__(self, targets, features):
        super().__init__()
        self.register_buffer("weights", torch.zeros(targets, features))
        self.register_buffer("offset", torch.zeros(targets))

    @torch.no_grad()
    def fit(self, features, residual, ridge):
        if ridge <= 0:
            raise ValueError("The stacking ridge must be positive")
        # Per-target standardisation and ridge on mean squared residual error.
        x = features.double().transpose(0, 1)
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, unbiased=False, keepdim=True)
        std = torch.where(std > 1e-8, std, 1.0)
        z = (x - mean) / std
        y = residual.double().T.unsqueeze(-1)
        gram = z.transpose(1, 2) @ z
        gram.diagonal(dim1=1, dim2=2).add_(ridge * features.shape[0])
        beta = torch.cholesky_solve(z.transpose(1, 2) @ (y - y.mean(1, keepdim=True)),
                                    torch.linalg.cholesky(gram))
        weights = beta.squeeze(-1) / std.squeeze(1)
        self.weights.copy_(weights)
        self.offset.copy_(y.mean(1).squeeze(-1) - (mean.squeeze(1) * weights).sum(1))

    def forward(self, x):
        return (x * self.weights).sum(-1) + self.offset


class LateTransferTabM(nn.Module):
    def __init__(self, bank):
        super().__init__()
        self.bank = bank
        q = len(bank.sources)
        self.heads = nn.ModuleDict({
            "self_calibration": ResidualReadout(q, 1),
            "score_transfer": ResidualReadout(q, q),
            "hidden_transfer": ResidualReadout(q, 2 * q - 1),
        })

    def systems(self, x):
        own, transferred = self.bank(x)
        results = {"independent": own}
        for mode, head in self.heads.items():
            results[mode] = own + head(readout_features(own, transferred, mode))
        return results

    def forward(self, x):
        return self.systems(x)["hidden_transfer"]


def fit_late_transfer(x, y, recipe):
    """Only fit materials and fit phenotypes enter this API, including all readouts."""
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or y.shape[1] < 2:
        raise ValueError("Expected matching material rows and multiple target columns")
    if len(recipe["profiles"]) != y.shape[1] or not recipe["seeds"]:
        raise ValueError("Need one profile per target and at least one seed")
    if (recipe["epochs"] < 1 or recipe["batch_size"] < 1 or
            recipe["transfer_ridge"] <= 0 or recipe["stack_ridge"] <= 0):
        raise ValueError("Invalid training or readout settings")
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError("Nonfinite fit inputs")
    n, q = y.shape
    own = x.new_full((n, q), float("nan"))
    transferred = x.new_full((n, q, q), float("nan"))
    kfold = KFold(n_splits=recipe["crossfit_folds"], shuffle=True,
                  random_state=recipe["crossfit_seed"])
    for train, held in kfold.split(np.arange(n)):
        a, b = [torch.as_tensor(i, device=x.device) for i in (train, held)]
        bank = fit_bank(x[a], y[a], recipe)
        with torch.no_grad():
            own[b], transferred[b] = bank(x[b])
        del bank
    if not torch.isfinite(own).all() or not torch.isfinite(transferred).all():
        raise FloatingPointError("Incomplete or nonfinite cross-fitted features")
    model = LateTransferTabM(fit_bank(x, y, recipe)).to(x.device)
    for mode, head in model.heads.items():
        head.fit(readout_features(own, transferred, mode), y - own, recipe["stack_ridge"])
    return model.eval().requires_grad_(False)


def save_predictor(path, model, recipe, markers):
    torch.save({"format": 1, "recipe": recipe, "markers": markers,
                "state": model.state_dict()}, path)


def load_predictor(path, device="cpu"):
    """Restore a genotype-only predictor, without the training matrix or phenotypes."""
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if obj["format"] != 1:
        raise ValueError("Unsupported late-transfer checkpoint")
    recipe, p = obj["recipe"], obj["markers"]
    x, y = torch.zeros(2, p), torch.zeros(2, 1)
    q = len(recipe["profiles"])
    sources = [[new_source(x, y, prof, seed, q) for seed in recipe["seeds"]]
               for prof in recipe["profiles"]]
    model = LateTransferTabM(Bank(sources))
    model.load_state_dict(obj["state"])
    return model.to(device).eval().requires_grad_(False)
