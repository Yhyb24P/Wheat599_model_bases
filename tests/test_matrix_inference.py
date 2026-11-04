"""Structural contracts of material--marker inference, not accuracy claims."""

import torch
from torch import nn

from wheat599.matrix_inference import MatrixInference


def example():
    rng = torch.Generator().manual_seed(271)
    z = torch.randn(11, 17, generator=rng, dtype=torch.float64)
    z = z - z.mean(0)
    y = torch.randn(11, 3, generator=rng, dtype=torch.float64)
    y = y - y.mean(0)
    query = torch.randn(5, 17, generator=rng, dtype=torch.float64)
    torch.manual_seed(17)
    model = MatrixInference(3, width=8, layers=3, psi_hidden=12,
                            gate_hidden=12, dtype=torch.float64)
    return model, z, y, query


def test_material_and_marker_symmetries_and_query_independence():
    model, z, y, query = example()
    expected = model(z, y, query)
    row_order = torch.tensor([7, 0, 9, 3, 6, 1, 4, 10, 5, 8, 2])
    marker_order = torch.arange(16, -1, -1)
    sign = torch.where(torch.arange(17) % 3 == 0, -1., 1.).double()
    for actual in (
        model(z[row_order], y[row_order], query),
        model(z[:, marker_order], y, query[:, marker_order]),
        model(z * sign, y, query * sign),
        torch.cat([model(z, y, query[i:i+1]) for i in range(len(query))]),
        torch.cat([model(z, y, query[:2]), model(z, y, query[2:])]),
    ):
        torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
    changed = query.clone()
    changed[1:] = 100 * changed[1:]
    torch.testing.assert_close(model(z, y, changed)[:1], expected[:1],
                               atol=2e-12, rtol=2e-12)


def test_context_labels_drive_predictions_with_valid_gradients():
    model, z, y, query = example()
    y.requires_grad_()
    assert torch.autograd.gradcheck(lambda response: model(z, response, query),
                                   (y,), fast_mode=True, atol=1e-5, rtol=1e-4)
    prediction = model(z, y, query)
    target = torch.arange(prediction.numel(), dtype=prediction.dtype).reshape_as(prediction)
    (prediction * (1 + target)).sum().backward()
    assert y.grad is not None and torch.isfinite(y.grad).all()
    assert (y.grad.norm(dim=0) > 0).all()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.norm() > 0, name
    assert not torch.allclose(model(z, y.detach().roll(1, 0), query), prediction)


class ProductOnly(nn.Module):
    def forward(self, values):
        # width=q=1: the assembled channels are [A, Y, A*Y].
        return values[..., 2:3]


class ReplaceState(nn.Module):
    def forward(self, values):
        return torch.full_like(values[..., :1], 1000.)


def test_actual_message_path_recovers_explicit_quadratic_statistic():
    torch.manual_seed(91)
    model = MatrixInference(1, width=1, layers=1, psi_hidden=2,
                            gate_hidden=2, dtype=torch.float64)
    with torch.no_grad():
        model.initial.weight.copy_(torch.tensor([[1., 0.]]))
        model.update.weight.fill_(1)
        model.readout.weight.fill_(1)
        model.readout.bias.zero_()
    model.psi, model.gate = ProductOnly(), ReplaceState()
    z = torch.randn(13, 7, dtype=torch.float64)
    y = torch.randn(13, 1, dtype=torch.float64)
    query = torch.randn(4, 7, dtype=torch.float64)
    # Independent explicit p-by-p reference is used only in this small test.
    cross_moment = z.T @ torch.diag(y[:, 0]) @ z
    explicit = torch.stack([x @ cross_moment @ x for x in query])[:, None]
    explicit = explicit / (len(z) * z.shape[1])
    torch.testing.assert_close(model(z, y, query), explicit, atol=1e-12, rtol=1e-12)


def test_one_model_accepts_different_marker_and_material_counts():
    model, z, y, query = example()
    count = model.parameter_count()
    assert model(z, y, query).shape == (5, 3)
    assert model(z[:6, :9], y[:6], query[:2, :9]).shape == (2, 3)
    assert model.parameter_count() == count
