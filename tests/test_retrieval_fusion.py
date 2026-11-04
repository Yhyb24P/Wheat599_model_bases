import numpy as np

from wheat599.retrieval_fusion import choose_weight


def test_convex_fusion_uses_prediction_vectors():
    rng = np.random.default_rng(42)
    y = rng.normal(size=(40, 4))
    error = rng.normal(size=y.shape)
    units = [(y + error, y - error, y)]
    weight, scores = choose_weight(units, [0., .25, .5, .75, 1.])
    assert weight == .5
    assert abs(scores[2] - 1.) < 1e-12


def test_fusion_tie_uses_first_weight():
    y = np.arange(40).reshape(10, 4).astype(float)
    weight, _ = choose_weight([(y, y, y)], [0., .5, 1.])
    assert weight == 0.
