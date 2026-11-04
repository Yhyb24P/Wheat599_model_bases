import copy

import pytest
import torch

from wheat599.genomic_tabm import GenomicTabM
from wheat599.retrieval_tabm import RetrievalTabM


def build(variant='retrieval'):
    torch.manual_seed(31)
    x, y = torch.randn(12, 9), torch.randn(12, 4)
    torch.manual_seed(42)
    model = RetrievalTabM(x, y, variant=variant, n_blocks=2, d_block=8,
                         k=2, context_size=3, dropout=0.1)
    return model, x, y


def test_raw_matches_official_path():
    model, x, y = build('raw')
    torch.manual_seed(42)
    raw = GenomicTabM(x, y, variant='raw', n_blocks=2, d_block=8,
                     k=2, dropout=0.1)
    model.eval()
    raw.eval()
    assert torch.equal(model(x), raw(x))


def test_exclusion_gradients_and_shape():
    model, x, y = build()
    ids = torch.tensor([0, 4, 7])
    prediction = model(x[ids], query_ids=ids)
    assert prediction.shape == (3, 2, 4)
    assert not (model.last_neighbor_indices == ids[:, None, None]).any()
    prediction.square().mean().backward()
    for module in (model.key, model.label_encoder, model.difference):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
    with pytest.raises(ValueError, match='query_ids'):
        model(x[:3])


def test_own_label_excluded_and_reference_labels_used():
    model, x, y = build()
    model.eval()
    before = model(x[:1], query_ids=torch.tensor([0]))
    with torch.no_grad():
        model.reference_y[0] += 100
    assert torch.equal(before, model(x[:1], query_ids=torch.tensor([0])))
    with torch.no_grad():
        model.reference_y[1:] += 10
    assert not torch.allclose(before, model(x[:1], query_ids=torch.tensor([0])))


def test_checkpoint_and_current_reference_encoding():
    model, x, y = build('single_retrieval')
    model.eval()
    prediction = model(x)
    assert prediction.shape == (12, 1, 4)
    other, _, _ = build('single_retrieval')
    other.load_state_dict(copy.deepcopy(model.state_dict()))
    other.eval()
    assert torch.equal(prediction, other(x))
    keys = model.encode_reference().detach().clone()
    with torch.no_grad():
        model.key.weight.add_(0.1)
    assert not torch.equal(keys, model.encode_reference())
