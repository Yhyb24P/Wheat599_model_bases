"""MtCro MMOE wiring on a tiny CPU problem."""
import numpy as np
import torch

from wheat599.mtcro_unified import build, fit_predict


def test_build_and_fit_tiny():
    rng = np.random.default_rng(0)
    x = rng.integers(0, 2, size=(20, 12)).astype(float)
    y = np.column_stack([x[:, :3].mean(1) + rng.normal(0, 0.05, 20) for _ in range(4)])
    spec = {'experts': 2, 'expert_dims': [8], 'tower_dims': [8], 'dropout': 0.0}
    cfg = {'device': 'cpu', 'training': {'learning_rate': 1e-3, 'weight_decay': 0.0,
                                         'batch_size': 8, 'patience': 5}}
    model, names = build(x.shape[1], spec)
    assert len(names) == x.shape[1]
    assert model({name: torch.as_tensor(x[:4, i]).float().unsqueeze(1)
                  for i, name in enumerate(names)}).shape == (4, 4)
    pred, best_epoch, n_params = fit_predict(x, y, np.arange(14), np.arange(14, 20),
                                             spec, cfg, 42, 3, True)
    assert pred.shape == (6, 4)
    assert np.all(np.isfinite(pred)) and best_epoch >= 1 and n_params > 0
