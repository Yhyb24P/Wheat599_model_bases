import torch
from wheat599.conditional_quantile import ConditionalQuantile, entmax15


def test_connections_and_continuous_training():
    torch.manual_seed(4)
    weights = entmax15(torch.randn(8, 100))
    assert torch.allclose(weights.sum(-1), torch.ones(8), atol=1e-5)
    assert (weights == 0).any()
    x = torch.randint(0, 2, (16, 19)).float()
    model = ConditionalQuantile(x.numpy(), torch.randn(16).numpy())
    result = model(x)
    assert result.shape == (16, 4, 99)
    result.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    model.eval()
    assert torch.equal(model(x), model(x))


def test_constant_fit_marker_does_not_amplify_query():
    x = torch.zeros(8, 3)
    model = ConditionalQuantile(x.numpy(), torch.randn(8).numpy())
    with torch.no_grad():
        model.linear.fill_(1)
        model.readout.zero_()
    assert torch.equal(model(torch.ones(2, 3)), model(torch.zeros(2, 3)))
