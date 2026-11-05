import subprocess
from pathlib import Path

import pytest
import torch

from wheat599 import wheat2000_baselines as baselines


STUB_MODEL = '''
import torch
from torch import nn


class BiMSGP(nn.Module):
    def __init__(self, n_markers, d_model=64, d_state=64, n_layers=1, residual=True,
                 direction="bidirectional", decoder="wide", dropout=0.1, n_outputs=1):
        super().__init__()
        self.linear = nn.Linear(n_markers, n_outputs)

    def forward(self, x):
        return self.linear(x.flatten(1)).squeeze(1)
'''


def make_repo(tmp_path, license_text='MIT License\n\nPermission is hereby granted.\n',
              model_code=STUB_MODEL, name='BiMSGP'):
    repo = tmp_path / name
    (repo / 'bimsgp').mkdir(parents=True)
    (repo / 'LICENSE').write_text(license_text)
    (repo / 'bimsgp' / '__init__.py').write_text('')
    if model_code is not None:
        (repo / 'bimsgp' / 'model.py').write_text(model_code)
    subprocess.check_output(['git', 'init', '-q'], cwd=repo)
    subprocess.check_output(['git', 'add', '.'], cwd=repo)
    subprocess.check_output(['git', '-c', 'user.name=t', '-c', 'user.email=t@e',
                             'commit', '-q', '-m', 'stub'], cwd=repo)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    return repo, commit


def test_deepgs_matches_the_official_structure_and_parameter_count():
    assert baselines.DEEPGS_DEFAULTS == {'conv_channels': 8, 'kernel_size': 18, 'pool_size': 4,
                                         'hidden': 32, 'dropouts': (0.2, 0.1, 0.05)}
    small = baselines.DeepGS(100)
    assert small.pooled == 20
    assert small.parameter_count() == 152 + (160 * 32 + 32) + 33 == 5337
    prediction = small(torch.randn(3, 100))
    assert prediction.shape == (3,) and torch.isfinite(prediction).all()
    large = baselines.DeepGS(33709)
    assert large.pooled == 8423
    assert large.parameter_count() == 2_156_505
    with pytest.raises(ValueError):
        baselines.DeepGS(18)
    with pytest.raises(ValueError):
        baselines.DeepGS(100, dropouts=(0.2, 0.1))


def test_bimsgp_loader_verifies_the_clean_pinned_checkout(tmp_path):
    repo, commit = make_repo(tmp_path)
    model_file = repo / 'bimsgp' / 'model.py'
    before = baselines._sha256(model_file)
    model = baselines.build_bimsgp(repo, commit, 10)
    assert model(torch.randn(4, 10)).shape == (4,)
    assert model.source_info['commit'] == commit and model.source_info['clean'] is True
    assert model.source_info['license'] == 'MIT'
    assert model.source_info['parameter_count'] == 11
    assert baselines._sha256(model_file) == before
    with pytest.raises(ValueError):
        baselines.build_bimsgp(repo, '0' * 40, 10)
    (repo / 'extra.txt').write_text('dirty')
    with pytest.raises(ValueError):
        baselines.build_bimsgp(repo, commit, 10)


def test_bimsgp_loader_rejects_wrong_license_and_missing_model(tmp_path):
    apache, apache_commit = make_repo(
        tmp_path, license_text='Apache License\nVersion 2.0\n', name='apache')
    with pytest.raises(ValueError):
        baselines.build_bimsgp(apache, apache_commit, 10)
    missing, missing_commit = make_repo(tmp_path, model_code=None, name='missing')
    with pytest.raises(ValueError):
        baselines.build_bimsgp(missing, missing_commit, 10)
    with pytest.raises(ValueError):
        baselines.build_bimsgp(tmp_path / 'absent', '0' * 40, 10)
