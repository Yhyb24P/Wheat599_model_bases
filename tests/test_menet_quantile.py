from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from torch import nn
from torch.nn import functional as F

from wheat599.menet_quantile import RelationMap, Trainer, triplet_loss, official_modules, DualQuantile
from wheat599.quantile_tabm import build_model, quantile_loss


def test_triplet_per_example_not_cross_batch():
    torch.manual_seed(7)
    a, s, t = [torch.randn(4, 6, requires_grad=True) for _ in range(3)]
    ay, sy, ty = [torch.tensor(v) for v in ([0., 1., 2., 3.], [2., 1.1, 5., 4.], [.1, 4., 2.2, 1.])]
    batch_loss = triplet_loss(a, ay[:, None], s, sy[:, None], t, ty[:, None], .1)
    scalar_loss = torch.stack([triplet_loss(a[i:i+1], ay[i:i+1], s[i:i+1], sy[i:i+1], t[i:i+1], ty[i:i+1], .1) for i in range(4)]).mean()
    torch.testing.assert_close(batch_loss, scalar_loss)
    batch_loss.backward()
    assert all(torch.isfinite(v.grad).all() for v in (a, s, t))


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 4)

    def forward_once(self, x):
        return self.linear(x[:, 0])


def test_relation_fit_only_and_query_batch_invariance():
    torch.manual_seed(1)
    fit, query = torch.randn(7, 3), torch.randn(4, 3)
    relation = RelationMap(Encoder(), fit)
    original_state = {k: v.clone() for k, v in relation.state_dict().items()}
    expected = relation(query)
    actual = torch.cat([relation(row[None]) for row in query])
    torch.testing.assert_close(expected, actual, atol=1e-6, rtol=1e-6)
    relation(torch.full((100, 3), 1e6))
    torch.testing.assert_close(expected, relation(query))
    assert expected.shape == (4, 7)
    assert not expected.requires_grad
    assert all(torch.equal(v, original_state[k]) for k, v in relation.state_dict().items())


@pytest.fixture
def settings():
    cfg = yaml.safe_load(Path('configs/menet_quantile.yaml').read_text())
    if not Path(cfg['upstream_path']).exists():
        pytest.skip('Pinned academic-use MeNet checkout is required')
    cfg['relation_training'].update(epochs=2, batch_size=8)
    cfg['training'].update(max_epochs=2, patience=2, batch_size=8)
    return cfg


def test_dual_forward_both_residual_paths_train(settings):
    torch.set_num_threads(1)
    rng = np.random.default_rng(2)
    x, y = rng.normal(size=(8, 80)), rng.normal(size=8)
    config = dict(n_blocks=2, d_block=16, dropout=0., k=4, learning_rate=.001, weight_decay=.001)
    _, module = official_modules(settings)
    marker = build_model(x, y, config, 42, settings, 'cpu')
    model = DualQuantile(marker, 8, 16, module)
    inputs = torch.cat((torch.tensor(x, dtype=torch.float32), torch.rand(8, 8)), dim=1)
    result = model(inputs)
    assert result.shape == (8, 4, 99)
    quantile_loss(result, torch.tensor(y[:, None], dtype=torch.float32), torch.linspace(.001, .999, 99)).backward()
    for branch in (model.marker, model.relation, model.marker_update, model.relation_update, model.head):
        assert sum(p.grad.abs().sum().item() for p in branch.parameters() if p.grad is not None) > 0
    model.eval()
    with torch.no_grad():
        original = model(inputs)
        changed = inputs.clone(); changed[:, 80:] = torch.rand(8, 8)
        assert not torch.equal(original, model(changed))


def test_two_stage_training_reuses_frozen_encoder(tmp_path, settings):
    torch.set_num_threads(1)
    rng = np.random.default_rng(8)
    x = rng.integers(0, 2, size=(24, 1279)).astype('float32')
    y = rng.normal(size=24).astype('float32')
    cfg = dict(n_blocks=2, d_block=16, dropout=0., k=4, learning_rate=.001, weight_decay=.001)
    trainer = Trainer(settings, tmp_path)
    p1, info = trainer(x[:16], y[:16], x[16:], y[16:], cfg, 42, 0, settings, 'cpu')
    p2, _ = trainer(x[:16], y[:16], x[16:], y[16:], cfg, 42, 0, settings, 'cpu')
    assert len(trainer.pretraining) == 1
    np.testing.assert_array_equal(p1, p2)
    assert len(p1) == 8 and np.isfinite(p1).all()
    assert 0 <= info['best_epoch'] <= 2
    assert len(list((tmp_path / 'relations').glob('*.pt'))) == 1
