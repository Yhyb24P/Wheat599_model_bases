import numpy as np
import torch

from wheat599.sparse_residual import (
    SparseResidualQuantile, l12_prox, make_optimizer, proximal_step)


def _objective(z, u, tau):
    return 0.5 * (z - u) ** 2 + tau * np.sqrt(np.abs(z))


def _model(seed=0, n=24, markers=6, width=8, members=3, quantiles=5, n_blocks=3):
    torch.manual_seed(seed)
    x = torch.randn(n, markers)
    y = torch.randn(n)
    return SparseResidualQuantile(x, y, width=width, members=members,
                                  quantiles=quantiles, n_blocks=n_blocks), x, y


# -- proximal operator math ----------------------------------------------------

def test_prox_zero_tau_is_identity():
    u = torch.linspace(-5, 5, 101)
    torch.testing.assert_close(l12_prox(u, torch.zeros_like(u)), u, atol=1e-6, rtol=1e-6)


def test_prox_sign_symmetry_and_zero_input():
    u = torch.rand(50) * 4
    tau = torch.rand(50) * 2
    torch.testing.assert_close(l12_prox(-u, tau), -l12_prox(u, tau), atol=1e-6, rtol=1e-6)
    assert torch.all(l12_prox(torch.zeros(10), torch.ones(10)) == 0)


def test_prox_threshold_branches():
    tau = 0.7
    threshold = 1.5 * tau ** (2.0 / 3.0)
    # c multiplies the FULL threshold: c<1 -> zero (c in (0.794,1) still has a nonzero
    # local minimum but zero is global), c>1 -> nonzero.
    for c in (0.0, 0.5, 0.8, 0.9, 0.99):
        u = c * threshold
        assert l12_prox(torch.tensor([u]), torch.tensor([tau])).item() == 0.0
    for c in (1.01, 1.19, 1.3, 1.49, 1.5, 2.0, 10.0):
        u = c * threshold
        assert l12_prox(torch.tensor([u]), torch.tensor([tau])).item() > 0.0


def test_prox_jump_at_boundary():
    tau = 0.7
    threshold = 1.5 * tau ** (2.0 / 3.0)
    just_above = threshold * (1 + 1e-7)
    val = l12_prox(torch.tensor([just_above]), torch.tensor([tau])).item()
    # The half-threshold is discontinuous: it jumps from 0 to ~tau^(2/3).
    assert abs(val - tau ** (2.0 / 3.0)) < 1e-3 * tau ** (2.0 / 3.0)


def test_prox_matches_independent_cubic_roots():
    rng = np.random.default_rng(0)
    for _ in range(60):
        tau = float(10 ** rng.uniform(-3, 1))
        u = float(10 ** rng.uniform(-1, 2))
        threshold = 1.5 * tau ** (2.0 / 3.0)
        val = l12_prox(torch.tensor([u]), torch.tensor([tau])).item()
        if u <= threshold:
            assert val == 0.0
            continue
        roots = np.roots([1.0, 0.0, -u, tau / 2.0])
        real = roots[np.abs(roots.imag) < 1e-8].real
        t = real[real > 0].max()
        assert abs(val - t * t) <= 1e-6 * max(1.0, t * t)


def test_prox_global_optimal_on_grid():
    # The band (27/16)^(1/3)..1.5 * tau^(2/3) is c in (0.794,1) of the full threshold:
    # a nonzero local minimum exists there but zero is the global minimum.
    tau = 0.4
    threshold = 1.5 * tau ** (2.0 / 3.0)
    cases = [c * threshold for c in (0.3, 0.5, 0.8, 0.9, 0.95, 0.99, 1.01, 1.5, 2.0, 4.0)]
    cases += [0.05, 0.5, 2.0, 7.0]
    for u in cases:
        z = l12_prox(torch.tensor([u]), torch.tensor([tau])).item()
        fz = _objective(z, u, tau)
        grid = np.linspace(0.0, max(u, 1e-12), 4001)
        fgrid = _objective(grid, u, tau).min()
        assert fz <= fgrid + 1e-7 * max(1.0, fgrid)
        assert fz <= _objective(0.0, u, tau) + 1e-9


# -- model structure -----------------------------------------------------------

def test_forward_shape_and_member_diversity():
    model, x, _ = _model()
    out = model(x)
    assert out.shape == (len(x), model.members, model.quantiles)
    assert torch.isfinite(out).all()
    assert not torch.allclose(out[:, 0], out[:, 1])


def test_y_roundtrip_and_train_only_buffers():
    model, x, y = _model()
    yt = torch.as_tensor(y[:, None])
    torch.testing.assert_close(model.inverse_y(model.standardize_y(yt)), yt, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(model.x_mean, x.mean(0), atol=1e-6, rtol=1e-6)
    before = model.x_mean.clone()
    model.standardize_x(torch.randn(40, x.shape[1]) * 100)
    torch.testing.assert_close(model.x_mean, before, atol=0, rtol=0)


def test_constant_marker_is_zeroed():
    torch.manual_seed(2)
    x = torch.randn(20, 4)
    x[:, 1] = 3.0
    model = SparseResidualQuantile(x, torch.randn(20), width=6, members=2, quantiles=3)
    assert bool(model.x_constant[1])
    z = model.standardize_x(x)
    assert torch.all(z[:, 1] == 0)


def test_parameter_grouping():
    model, _, _ = _model(n_blocks=3)
    prox = model.proximal_parameters()
    decay = model.decay_parameters()
    assert len(prox) == 2 * 3
    assert all(p.shape == (model.width, model.width) for p in prox)
    expected = {id(w) for b in model.blocks for w in (b.fc1.weight, b.fc2.weight)}
    assert {id(p) for p in prox} == expected
    prox_ids, decay_ids = {id(p) for p in prox}, {id(p) for p in decay}
    assert prox_ids.isdisjoint(decay_ids)
    assert prox_ids | decay_ids == {id(p) for p in model.parameters()}


def test_make_optimizer_partition():
    model, _, _ = _model()
    opt = make_optimizer(model, learning_rate=0.001, weight_decay=0.0003)
    prox_ids = {id(p) for p in model.proximal_parameters()}
    assert opt.param_groups[0]['weight_decay'] == 0.0
    assert opt.param_groups[1]['weight_decay'] == 0.0003
    assert {id(p) for p in opt.param_groups[0]['params']} == prox_ids
    assert {id(p) for p in opt.param_groups[1]['params']} == {id(p) for p in model.decay_parameters()}
    covered = [id(p) for g in opt.param_groups for p in g['params']]
    assert len(covered) == len(set(covered)) == len(list(model.parameters()))


# -- proximal step integration -------------------------------------------------

def test_proximal_step_matches_adam_preconditioner():
    model, x, y = _model(seed=3)
    opt = make_optimizer(model, learning_rate=0.05, weight_decay=0.0003)
    lam = 0.02
    model(x).square().mean().backward()
    opt.step()
    group = opt.param_groups[0]
    beta1, beta2 = group['betas']
    lr, eps = group['lr'], group['eps']
    snapshot = []
    for p in model.proximal_parameters():
        state = opt.state[p]
        step = float(state['step'])
        m_hat = state['exp_avg'] / (1 - beta1 ** step)
        v_hat = state['exp_avg_sq'] / (1 - beta2 ** step)
        d = lr / (v_hat.sqrt() + eps)
        snapshot.append((p, p.detach().clone(), lam * d))
    proximal_step(model, opt, lam)
    for p, u, tau in snapshot:
        torch.testing.assert_close(p, l12_prox(u, tau), atol=1e-6, rtol=1e-6)


def test_proximal_step_leaves_decay_params_unchanged():
    model, x, y = _model(seed=4)
    opt = make_optimizer(model, 0.05, 0.0003)
    model(x).square().mean().backward()
    opt.step()
    before = [p.detach().clone() for p in model.decay_parameters()]
    proximal_step(model, opt, 0.05)
    for p, b in zip(model.decay_parameters(), before):
        torch.testing.assert_close(p, b, atol=0, rtol=0)


def test_proximal_step_sparsifies_with_large_lambda():
    model, x, y = _model(seed=5)
    opt = make_optimizer(model, 0.05, 0.0)
    model(x).square().mean().backward()
    opt.step()
    proximal_step(model, opt, lam=5.0)
    zeros = sum(int((p == 0).sum()) for p in model.proximal_parameters())
    total = sum(p.numel() for p in model.proximal_parameters())
    assert zeros > 0.5 * total


def test_zero_can_reactivate():
    model, x, _ = _model(seed=1, n=16, markers=5, width=6, members=3, quantiles=4, n_blocks=2)
    opt = make_optimizer(model, 0.5, 0.0)
    p = model.proximal_parameters()[0]
    model(x).square().mean().backward()
    opt.step()
    proximal_step(model, opt, 1e-4)
    with torch.no_grad():
        p.zero_()
    for q in model.parameters():
        q.grad = None
    p.grad = torch.zeros_like(p)
    p.grad[0, 0] = -1e4
    opt.step()
    proximal_step(model, opt, 1e-4)
    assert p[0, 0].item() != 0.0


def test_initialization_is_seed_reproducible():
    a, _, _ = _model(seed=11)
    b, _, _ = _model(seed=11)
    c, _, _ = _model(seed=12)
    for pa, pb in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(pa, pb, atol=0, rtol=0)
    assert any(not torch.allclose(pa, pc) for pa, pc in zip(a.parameters(), c.parameters()))