"""Wheat2000 baselines used for the multi-dataset comparison.

DeepGS.  Independent PyTorch adaptation of the official R/mxnet network in
``cma2015/DeepGS`` (commit 33ae2dc1ed7bda4e38b76460b18756649d4478a5, package
version 1.2, GPL-2|GPL-3).  The official source builds, for a marker matrix with
``p`` columns, a (1, p, 1, N) image followed by Conv(kernel 1x18, 8 filters,
stride 1), ReLU, MaxPool(1x4, stride 1x4), Dropout(0.2), Flatten,
FullyConnected(32), Sigmoid, Dropout(0.1), FullyConnected(1), Dropout(0.05) and
a linear regression output.  The official training defaults are num_round=6000,
batch=30, lr=0.01, momentum=0.5, wd=1e-5, initializer=0.01 and early stopping
after 600 bad steps; the trainer owns those settings.  This module only mirrors
the network structure: it is not a bitwise reproduction of mxnet training,
initialisation or RNG behaviour.

BiMSGP.  The official MIT implementation is loaded in place from a pinned,
clean checkout (``liuxumail-create/BiMSGP``) and is never copied into this
repository.  The caller supplies the checkout path and commit; the wide decoder
and all official defaults are left untouched.
"""

from __future__ import annotations

import hashlib
import importlib.util
import subprocess
from pathlib import Path

import torch
from torch import nn


DEEPGS_DEFAULTS = {
    'conv_channels': 8,
    'kernel_size': 18,
    'pool_size': 4,
    'hidden': 32,
    'dropouts': (0.2, 0.1, 0.05),
}
BIMSGP_DEFAULTS = {
    'd_model': 64,
    'd_state': 64,
    'n_layers': 1,
    'residual': True,
    'direction': 'bidirectional',
    'decoder': 'wide',
    'dropout': 0.1,
    'n_outputs': 1,
}


def parameter_count(model):
    return int(sum(parameter.numel() for parameter in model.parameters()))


class DeepGS(nn.Module):
    """PyTorch reimplementation of the official DeepGS convolutional network."""

    def __init__(self, n_markers, conv_channels=8, kernel_size=18, pool_size=4, hidden=32,
                 dropouts=(0.2, 0.1, 0.05)):
        super().__init__()
        n_markers, conv_channels, kernel_size, pool_size, hidden = (
            int(n_markers), int(conv_channels), int(kernel_size), int(pool_size), int(hidden))
        if n_markers <= kernel_size:
            raise ValueError('n_markers must exceed the convolution kernel size')
        if pool_size < 1 or hidden < 1 or conv_channels < 1:
            raise ValueError('DeepGS sizes must be positive')
        if len(dropouts) != 3 or any(not 0 <= float(value) < 1 for value in dropouts):
            raise ValueError('DeepGS needs three dropout probabilities in [0, 1)')
        self.n_markers = n_markers
        self.pooled = (n_markers - kernel_size + 1 - pool_size) // pool_size + 1
        if self.pooled < 1:
            raise ValueError('Marker length is shorter than one pooled window')
        self.conv = nn.Conv1d(1, conv_channels, kernel_size, stride=1)
        self.pool = nn.MaxPool1d(pool_size, stride=pool_size)
        self.drop_input = nn.Dropout(float(dropouts[0]))
        self.fc1 = nn.Linear(conv_channels * self.pooled, hidden)
        self.drop_hidden = nn.Dropout(float(dropouts[1]))
        self.fc2 = nn.Linear(hidden, 1)
        self.drop_output = nn.Dropout(float(dropouts[2]))

    def forward(self, x):
        if x.ndim != 2 or x.shape[1] != self.n_markers:
            raise ValueError(f'Expected a (n, {self.n_markers}) marker matrix')
        z = self.pool(torch.relu(self.conv(x.unsqueeze(1))))
        z = self.drop_input(z).flatten(1)
        z = torch.sigmoid(self.fc1(z))
        z = self.fc2(self.drop_hidden(z))
        return self.drop_output(z).squeeze(1)

    def parameter_count(self):
        return parameter_count(self)


def _run_git(path, *arguments):
    return subprocess.check_output(['git', *arguments], cwd=path, text=True).strip()


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_bimsgp_source(source_path, source_commit):
    """Check the pinned official checkout is complete, clean and MIT licensed."""
    path = Path(source_path)
    if not path.is_dir():
        raise ValueError(f'BiMSGP source directory does not exist: {path}')
    model_path = path / 'bimsgp' / 'model.py'
    license_path = path / 'LICENSE'
    if not model_path.is_file() or not license_path.is_file():
        raise ValueError('BiMSGP source lacks bimsgp/model.py or LICENSE')
    license_text = license_path.read_text()
    if 'MIT License' not in license_text:
        raise ValueError('BiMSGP source is not the MIT-licensed official tree')
    try:
        head = _run_git(path, 'rev-parse', 'HEAD')
        dirty = _run_git(path, 'status', '--porcelain')
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(f'BiMSGP source is not a readable git checkout: {error}') from error
    source_commit = str(source_commit)
    if not (head == source_commit or (len(source_commit) >= 7 and head.startswith(source_commit))):
        raise ValueError(f'BiMSGP checkout is at {head}, expected {source_commit}')
    if dirty:
        raise ValueError(f'BiMSGP checkout is not clean: {dirty}')
    return {
        'path': str(path), 'commit': head, 'clean': True,
        'model_sha256': _sha256(model_path), 'license_sha256': _sha256(license_path),
        'license': 'MIT',
    }


def build_bimsgp(source_path, source_commit, n_markers, d_model=64, d_state=64, n_layers=1,
                 residual=True, direction='bidirectional', decoder='wide', dropout=0.1,
                 n_outputs=1):
    """Load the official BiMSGP class from the pinned checkout and build one trait model."""
    info = verify_bimsgp_source(source_path, source_commit)
    model_path = Path(source_path) / 'bimsgp' / 'model.py'
    spec = importlib.util.spec_from_file_location('wheat2000_official_bimsgp_model', model_path)
    if spec is None or spec.loader is None:
        raise ValueError(f'Cannot import {model_path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    builder = getattr(module, 'BiMSGP', None)
    if builder is None:
        raise ValueError('BiMSGP source does not define BiMSGP')
    model = builder(n_markers=int(n_markers), d_model=int(d_model), d_state=int(d_state),
                    n_layers=int(n_layers), residual=bool(residual), direction=direction,
                    decoder=decoder, dropout=float(dropout), n_outputs=int(n_outputs))
    info['parameter_count'] = parameter_count(model)
    model.source_info = info
    return model
