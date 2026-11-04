"""Synthetic CPU checks for the RAGP Wheat599 adaptation."""

import copy
import inspect

import pytest
import torch
from sklearn.decomposition import PCA

from wheat599.ragp import (OFFICIAL_HIDDEN_DIMS, RAGPGenomic, pearson_loss,
                           triplet_loss)

torch.set_num_threads(1)

TINY = {
    'hidden_dims': (6, 5, 4, 3),
    'n_blocks': 1,
    'dropout': 0.0,
    'pca_variance': 0.95,
    'neighbors': 3,
    'fusion_alpha': 0.4,
    'label_scale': 0.01,
    'loss_alpha': (0.9, 0.8, 0.5, 0.1),
    'triplet_anchors': 8,
    'triplet_positives': 2,
    'triplet_negatives': 2,
    'triplet_margin': 1.0,
    'dtype': torch.float64,
}


def data(seed=5, materials=20, markers=12, environments=4):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(materials, markers, generator=generator, dtype=torch.float64)
    y = torch.randn(materials, environments, generator=generator, dtype=torch.float64)
    return x, y


def build(x, y, ids=None, **overrides):
    settings = dict(TINY)
    settings.update(overrides)
    torch.manual_seed(17)
    return RAGPGenomic(x, y, train_ids=ids, **settings)


def test_fit_only_pca_and_standardisation():
    x, y = data()
    model = build(x, y)
    reference = PCA(n_components=0.95, svd_solver='full').fit(x.numpy())
    assert model.pca_components.shape == (reference.n_components_, x.shape[1])
    assert model.pca_components.shape[0] <= min(x.shape)
    assert model.pca_explained_variance >= 0.95 - 1e-12
    assert torch.allclose(model.pca_mean, torch.tensor(reference.mean_),
                          atol=1e-12, rtol=0)
    assert torch.allclose(model.pca_components, torch.tensor(reference.components_),
                          atol=1e-12, rtol=0)
    expected = (x - model.pca_mean) @ model.pca_components.T
    assert torch.allclose(model.standardize_x(x), expected, atol=1e-12, rtol=0)
    standardised = model.standardize_y(y)
    assert torch.allclose(standardised.mean(0), torch.zeros(4, dtype=torch.float64),
                          atol=1e-12, rtol=0)
    assert torch.allclose(standardised.std(0, unbiased=False),
                          torch.ones(4, dtype=torch.float64), atol=1e-12, rtol=0)
    assert torch.allclose(model.inverse_y(standardised), y, atol=1e-12, rtol=0)
    reference_x = model.reference_x.clone()
    x.add_(1.0)
    assert torch.equal(model.reference_x, reference_x)


def test_forward_shape_metadata_and_official_defaults():
    x, y = data()
    model = build(x, y)
    assert OFFICIAL_HIDDEN_DIMS == (2500, 3000, 3000, 1500)
    model.eval()
    with torch.no_grad():
        prediction, embeddings = model(x, return_embeddings=True)
    assert prediction.shape == (x.shape[0], 1, 4)
    assert len(embeddings) == 4
    for index, hidden in enumerate(model.hidden_dims):
        assert embeddings[index].shape == (x.shape[0], hidden)
    metadata = model.metadata()
    assert metadata['environments'] == 4
    assert metadata['pca_components'] == model.pca_components.shape[0]
    assert metadata['hidden_dims'] == list(model.hidden_dims)
    assert metadata['parameters'] == model.parameter_count()
    assert metadata['library_ready'] == [False, False, False, False]
    assert metadata['upstream']['commit'].startswith('85675b1e')
    assert torch.equal(model.ridge_penalty(), torch.zeros((), dtype=torch.float64))


def test_training_requires_ids_and_uses_material_namespace():
    x, y = data()
    ids = torch.arange(100, 120)
    model = build(x, y, ids=ids)
    model.train()
    with pytest.raises(ValueError, match='query_ids'):
        model(x)
    with pytest.raises(ValueError, match='occur in train_ids'):
        model(x, query_ids=torch.arange(x.shape[0]))
    prediction = model(x, query_ids=ids)
    assert prediction.shape == (x.shape[0], 1, 4)
    with pytest.raises(ValueError, match='one ID per query'):
        model(x, query_ids=ids[:-1])


def test_library_update_excludes_self_by_id_not_feature_equality():
    x, y = data()
    x[1] = x[0]
    model = build(x, y, ids=torch.arange(10, 30))
    assert not any(environment.library_ready for environment in model.environments)
    model.update_library()
    for environment in model.environments:
        assert environment.library_ready
        assert not environment.library_features.requires_grad
        indices = environment.neighbor_indices
        for row in range(x.shape[0]):
            assert row not in indices[row].tolist()
        assert 1 in indices[0].tolist()
    before = [environment.library_features.clone() for environment in model.environments]
    with torch.no_grad():
        for environment in model.environments:
            environment.encoder[0].weight.add_(0.1)
    model.update_library()
    assert any(not torch.allclose(old, environment.library_features)
               for old, environment in zip(before, model.environments))


def test_training_fallback_loss_and_gradient_paths():
    x, y = data()
    ids = torch.arange(x.shape[0])
    model = build(x, y, ids=ids)
    assert not any(environment.library_ready for environment in model.environments)
    model.train()
    prediction, embeddings = model(x, query_ids=ids, return_embeddings=True)
    target = model.standardize_y(y)
    loss, terms = model.loss(prediction, target, embeddings, return_terms=True)
    assert torch.isfinite(loss)
    loss.backward()
    for environment in model.environments:
        assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0
                   for parameter in environment.parameters())
        assert environment.library_features.grad is None
        assert environment.neighbor_indices.grad is None
    assert model.reference_x.grad is None
    assert len(terms['pearson']) == 4
    assert len(terms['triplet']) == 4
    assert len(terms['weighted_triplet']) == 4
    expected = sum(terms['pearson']) + sum(terms['weighted_triplet'])
    assert abs(float(loss.detach()) - expected) < 1e-10


def test_pearson_and_triplet_objectives():
    target = torch.linspace(-1.0, 1.0, 9, dtype=torch.float64)
    prediction = 2.0 * target + 3.0
    # Upstream mixes population covariance with unbiased std, so the loss is
    # 1 - ((n - 1) / n) * Pearson rather than 1 - Pearson.
    factor = (len(target) - 1) / len(target)
    assert abs(float(pearson_loss(prediction, target)) - (1.0 - factor)) < 1e-12
    assert abs(float(pearson_loss(prediction, -target)) - (1.0 + factor)) < 1e-12
    assert torch.isfinite(pearson_loss(torch.ones_like(target), target))

    labels = torch.tensor([0.0, 0.1, 10.0, 10.1, 20.0, 20.1], dtype=torch.float64)
    embeddings = torch.tensor(
        [[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0], [-0.9, -0.1], [0.0, 1.0], [0.1, 0.9]],
        dtype=torch.float64)
    good = triplet_loss(embeddings, labels, anchors=6, positives=1, negatives=1,
                        margin=0.5)
    assert float(good) == 0.0
    broken = embeddings.clone()
    broken[1], broken[4] = embeddings[4].clone(), embeddings[1].clone()
    bad = triplet_loss(broken, labels, anchors=6, positives=1, negatives=1, margin=0.5)
    assert float(bad) > 0.0


def test_evaluation_excludes_self_by_id_and_uses_reference_labels():
    x, y = data()
    x[1] = x[0]
    ids = torch.arange(100, 120)
    model = build(x, y, ids=ids)
    model.update_library()
    model.eval()
    with torch.no_grad():
        before = model(x, query_ids=ids)
    with torch.no_grad():
        model.reference_y[0] += 100.0
    with torch.no_grad():
        after = model(x, query_ids=ids)
    assert torch.allclose(before[0], after[0], atol=1e-12, rtol=0)
    with torch.no_grad():
        model.reference_y[1] += 50.0
    with torch.no_grad():
        after_duplicate = model(x, query_ids=ids)
    assert not torch.allclose(before[0], after_duplicate[0])
    with torch.no_grad():
        unknown = model(x, query_ids=torch.arange(1000, 1020))
    assert unknown.shape == (x.shape[0], 1, 4)


def test_checkpoint_load_rebuilds_library_reproducibly():
    x, y = data()
    ids = torch.arange(x.shape[0])
    model = build(x, y, ids=ids)
    model.update_library()
    best = copy.deepcopy(model.state_dict())
    expected = [environment.library_features.clone() for environment in model.environments]
    assert not any('library_features' in key or 'neighbor_indices' in key for key in best)
    with torch.no_grad():
        for environment in model.environments:
            environment.encoder[0].weight.add_(0.3)
    model.update_library()
    assert any(not torch.allclose(old, environment.library_features)
               for old, environment in zip(expected, model.environments))
    model.load_state_dict(best)
    for old, environment in zip(expected, model.environments):
        assert environment.library_ready
        assert torch.allclose(environment.library_features, old, atol=1e-12, rtol=0)
    other = build(x, y, ids=ids)
    other.load_state_dict(best)
    model.eval()
    other.eval()
    with torch.no_grad():
        assert torch.allclose(model(x), other(x), atol=1e-12, rtol=0)


def test_parameter_count_matches_official_block_formula():
    x, y = data()
    model = build(x, y)
    input_dim = model.pca_components.shape[0]

    def expected(hidden):
        wide = hidden + 1
        encoder = input_dim * hidden + hidden * hidden + 2 * hidden
        block = 2 * wide * wide + 6 * wide
        return encoder + model.n_blocks * block + wide + 1

    assert model.parameter_count() == sum(expected(hidden) for hidden in model.hidden_dims)


def test_constructor_validation_and_loss_shapes():
    x, y = data()
    with pytest.raises(ValueError, match='hidden_dims'):
        build(x, y, hidden_dims=(6, 5, 4))
    with pytest.raises(ValueError, match='neighbors'):
        build(x, y, neighbors=x.shape[0])
    with pytest.raises(ValueError, match='unique'):
        build(x, y, ids=torch.zeros(x.shape[0], dtype=torch.long))
    model = build(x, y)
    prediction = torch.randn(x.shape[0], 1, 4, dtype=torch.float64)
    target = torch.randn(x.shape[0], 4, dtype=torch.float64)
    total = model.loss(prediction, target, include_triplet=False)
    expected = sum(pearson_loss(prediction[:, 0, index], target[:, index])
                   for index in range(4))
    assert torch.allclose(total, expected, atol=1e-12, rtol=0)
    with pytest.raises(ValueError, match='embeddings'):
        model.loss(prediction, target)
    with pytest.raises(ValueError, match='shape'):
        model.loss(prediction[:, :, :3], target)


def test_forward_has_no_query_label_argument():
    parameters = inspect.signature(RAGPGenomic.forward).parameters
    assert 'query_ids' in parameters
    assert not ({'y', 'target', 'labels'} & set(parameters))
