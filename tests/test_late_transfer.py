import numpy as np
import torch

from wheat599 import late_transfer as lt


def test_transfer_readout_matches_primal_ridge_and_handles_constant_hidden():
    torch.manual_seed(17)
    h = torch.randn(12, 4, dtype=torch.float64) + 2
    y = torch.randn(12, 3, dtype=h.dtype)
    weights, bias = lt.fit_transfer_head(h, y, ridge=1.5)
    z = h - h.mean(0)
    expected = torch.linalg.solve(z.T @ z + 1.5 * z.square().sum() / len(h) *
                                  torch.eye(4, dtype=h.dtype), z.T @ (y - y.mean(0)))
    torch.testing.assert_close(weights, expected)
    torch.testing.assert_close(h @ weights + bias, z @ expected + y.mean(0))
    w, c = lt.fit_transfer_head(torch.ones_like(h), y, ridge=1)
    torch.testing.assert_close(w, torch.zeros_like(w))
    torch.testing.assert_close(c, y.mean(0))


def test_source_target_axes_and_matched_score_control():
    own = torch.tensor([[2., 3., 5.]])
    aux = torch.arange(9.).reshape(1, 3, 3)
    full = lt.readout_features(own, aux, "hidden_transfer")
    torch.testing.assert_close(full[0, 1], torch.tensor([2., 3., 5., 3., 5.]))
    changed = aux + 100 * torch.eye(3)
    torch.testing.assert_close(lt.readout_features(own, changed, "hidden_transfer"), full)
    torch.testing.assert_close(lt.readout_features(own, aux, "score_transfer"), full[..., :3])


def test_oof_sources_exclude_the_scored_rows_and_checkpoint_predicts_without_labels(monkeypatch, tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(19)
    x, y = torch.randn(24, 9), torch.randn(24, 3)
    x[:, 0] = torch.arange(len(x))
    profile = dict(n_blocks=2, d_block=8, k=2, dropout=0.1, learning_rate=0.001, weight_decay=0.01)
    recipe = dict(profiles=[profile] * 3, profile_indices=[0] * 3, seeds=[42, 2026],
                  epochs=1, batch_size=8, gradient_clip=5, crossfit_folds=3, crossfit_seed=17,
                  transfer_ridge=1., stack_ridge=0.1)
    original, held_ids = lt.fit_bank, []

    def checked_bank(train, truth, recipe):
        train_ids = train[:, 0].long()
        torch.testing.assert_close(truth, y[train_ids])
        bank = original(train, truth, recipe)
        if len(train) < len(x):
            def check_queries(module, inputs):
                ids = inputs[0][:, 0].long().tolist()
                assert not set(ids).intersection(train_ids.tolist())
                held_ids.extend(ids)
            bank.register_forward_pre_hook(check_queries)
        return bank

    monkeypatch.setattr(lt, "fit_bank", checked_bank)
    model = lt.fit_late_transfer(x, y, recipe)
    assert sorted(held_ids) == list(range(len(x)))
    query = torch.randn(5, 9)
    predictions = model.systems(query)
    raw, _ = model.bank(query)
    torch.testing.assert_close(predictions["independent"], raw)
    path = tmp_path / "predictor.pt"
    lt.save_predictor(path, model, recipe, x.shape[1])
    restored = lt.load_predictor(path)
    for key, value in predictions.items():
        torch.testing.assert_close(restored.systems(query)[key], value)
        assert np.isfinite(value.numpy()).all()
