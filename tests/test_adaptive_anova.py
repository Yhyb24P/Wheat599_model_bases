"""Independent verification of the adaptive ANOVA-kernel GP.

Checks exact kernel expansions against brute-force combination enumeration, PSD,
gradients, the zero-delta / RBF fallback identities, duplicate-template handling,
and the Cholesky readout against a direct numpy solve. No Wheat599 training.
"""

import itertools
import math

import numpy as np
import pytest
import torch

from wheat599 import adaptive_anova as aa


def binary(rows, p, seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=(rows, p)).astype(np.float64)


def pm1(rows, p, seed):
    return 2 * binary(rows, p, seed) - 1


def exhaustive_kd(s, t, d):
    p = len(s)
    total = 0.0
    for comb in itertools.combinations(range(p), d):
        idx = list(comb)
        total += float(np.prod(s[idx]) * np.prod(t[idx]))
    return total / math.comb(p, d)


def test_base_kernels_match_exhaustive_combinations():
    p = 6
    S = torch.as_tensor(pm1(4, p, 1))
    T = torch.as_tensor(pm1(5, p, 2))
    K1, K2, K3 = aa.base_kernels(S, T, p)
    for d, K in [(1, K1), (2, K2), (3, K3)]:
        ref = np.array([[exhaustive_kd(S[i].numpy(), T[j].numpy(), d)
                         for j in range(5)] for i in range(4)])
        np.testing.assert_allclose(K.numpy(), ref, atol=1e-12, rtol=1e-12)
    # unit diagonal for binary inputs
    np.testing.assert_allclose(torch.diagonal(aa.base_kernel(S, S, p, 2)).numpy(), 1.0, atol=1e-12)


def test_template_kernel_and_gram_match_exhaustive():
    p = 6
    S = torch.as_tensor(pm1(4, p, 3))
    V = torch.as_tensor(np.random.default_rng(4).normal(size=(3, p)))
    for d in (2, 3):
        tk = aa.template_kernel(S, V, p, d).numpy()
        ref = np.array([[exhaustive_kd(V[r].numpy(), S[i].numpy(), d)
                         for r in range(3)] for i in range(4)])
        np.testing.assert_allclose(tk, ref, atol=1e-11, rtol=1e-11)
        G = aa.gram_matrix(V, p, d).numpy()
        refG = np.array([[exhaustive_kd(V[r].numpy(), V[t].numpy(), d)
                          for t in range(3)] for r in range(3)])
        np.testing.assert_allclose(G, refG, atol=1e-11, rtol=1e-11)
        np.testing.assert_allclose(G, G.T, atol=1e-12)


def test_zero_delta_recovers_base_kernel_exactly():
    p = 8
    S = torch.as_tensor(pm1(6, p, 5))
    V = torch.as_tensor(np.random.default_rng(6).normal(size=(4, p)))
    V = V / V.norm(dim=1, keepdim=True)
    zero = torch.zeros(4, dtype=torch.float64)
    for d in (2, 3):
        base = aa.base_kernel(S, S, p, d)
        adj = aa.adjusted_kernel(S, S, V, zero, p, d)
        np.testing.assert_allclose(adj.numpy(), base.numpy(), atol=1e-11, rtol=1e-11)


def test_adjusted_and_tail_kernels_are_psd():
    p = 10
    S = torch.as_tensor(pm1(12, p, 7))
    V = torch.as_tensor(np.random.default_rng(8).normal(size=(5, p)))
    V = V / V.norm(dim=1, keepdim=True)
    delta = torch.as_tensor(np.random.default_rng(9).normal(size=5))
    for d in (2, 3):
        K = aa.adjusted_kernel(S, S, V, delta, p, d).numpy()
        np.testing.assert_allclose(K, K.T, atol=1e-10)
        assert np.linalg.eigvalsh(K).min() >= -1e-8
    Kt = aa.tail_kernel(S, S, 0.03, p).numpy()
    np.testing.assert_allclose(Kt, Kt.T, atol=1e-10)
    assert np.linalg.eigvalsh(Kt).min() >= -1e-8
    np.testing.assert_allclose(np.diag(Kt), 1.0, atol=1e-10)


def test_tail_plus_low_orders_reconstructs_rbf():
    p = 9
    rho = 0.02
    S = torch.as_tensor(pm1(7, p, 10))
    P1 = (S @ S.T).numpy()
    hamming = (p - P1) / 2
    rbf = ((1 - rho) / (1 + rho)) ** hamming
    a = [math.comb(p, d) * rho ** d / (1 + rho) ** p for d in range(4)]
    K1, K2, K3 = [k.numpy() for k in aa.base_kernels(S, S, p)]
    tail = aa.tail_kernel(S, S, rho, p).numpy()
    norm = 1 - sum(a)
    reconstructed = tail * norm + a[0] + a[1] * K1 + a[2] * K2 + a[3] * K3
    np.testing.assert_allclose(reconstructed, rbf, atol=1e-10, rtol=1e-10)


def test_full_kernel_with_rbf_weights_equals_rbf_minus_constant():
    # delta = 0 and sigma_d^2 = a_d (d=1,2,3), sigma_>3^2 = norm  =>  k_theta = RBF - a_0.
    p = 9
    rho = 0.02
    x = binary(10, p, 11)
    y = np.random.default_rng(12).normal(size=10)
    model = aa.AdaptiveAnovaGP(x, y, rho=rho, rank=4, seed=1)
    with torch.no_grad():
        model.delta2.zero_()
        model.delta3.zero_()
        a = [math.comb(p, d) * rho ** d / (1 + rho) ** p for d in range(4)]
        norm = 1 - sum(a)
        model.log_sigma.copy_(torch.log(torch.tensor([a[1], a[2], a[3], norm], dtype=torch.float64)))
    S = model.s_fit
    K = model.kernel(S, S).detach().numpy()
    P1 = (S @ S.T).numpy()
    rbf = ((1 - rho) / (1 + rho)) ** ((p - P1) / 2)
    np.testing.assert_allclose(K, rbf - a[0], atol=1e-9, rtol=1e-9)


def _small_model(seed=0, p=12, n=24, rank=4, rho=0.03):
    x = binary(n, p, seed)
    y = np.random.default_rng(seed + 1).normal(size=n)
    return aa.AdaptiveAnovaGP(x, y, rho=rho, rank=rank, seed=seed), x, y


def test_loss_gradients_finite_and_match_finite_difference():
    model, _, _ = _small_model()
    loss = model.loss(0.1)
    assert torch.isfinite(loss)
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name
    # central finite differences on a few representative parameters
    def loss_val():
        with torch.no_grad():
            return float(model.loss(0.1))
    for param, idx in [(model.mu, 0), (model.log_noise, None), (model.delta2, 0), (model.log_sigma, 1)]:
        h = 1e-6
        flat = param.detach().clone()
        orig = flat.clone()
        if idx is None:
            param.data.add_(h); up = loss_val()
            param.data.copy_(orig); param.data.sub_(h); dn = loss_val()
            param.data.copy_(orig)
            numeric = (up - dn) / (2 * h)
            analytic = float(param.grad)
        else:
            param.data[idx] += h; up = loss_val()
            param.data.copy_(orig); param.data[idx] -= h; dn = loss_val()
            param.data.copy_(orig)
            numeric = (up - dn) / (2 * h)
            analytic = float(param.grad[idx])
        assert abs(numeric - analytic) <= 1e-4 * (1 + abs(analytic)), (numeric, analytic)


def test_forward_matches_direct_numpy_solve_and_original_scale():
    model, x, y = _small_model(seed=2)
    with torch.no_grad():
        model.delta2.normal_(0, 0.3)
        model.log_sigma.normal_(0, 0.2)
    with torch.no_grad():
        pred = model(x).numpy()
    # independent numpy reference using the same kernel
    S = model.s_fit.numpy()
    V2, V3 = [v.detach().numpy() for v in model._templates()]
    d2 = model.delta2.detach().numpy()
    d3 = model.delta3.detach().numpy()
    sig = np.exp(model.log_sigma.detach().numpy())
    noise = max(float(np.exp(model.log_noise.detach().numpy())), 1e-8)
    p = model.p
    rho = float(model.rho)

    def kern(A, B):
        P1 = A @ B.T
        K1 = P1 / p
        K2 = (P1 ** 2 - p) / (p * (p - 1))
        K3 = (P1 ** 3 - 3 * p * P1 + 2 * P1) / (p * (p - 1) * (p - 2))

        def adj(V, delta, d):
            P1t = A @ V.T
            P2 = (V ** 2).sum(1)
            P3 = A @ (V ** 3).T
            den = p * (p - 1) if d == 2 else p * (p - 1) * (p - 2)
            tkA = (P1t ** 2 - P2) / den if d == 2 else (P1t ** 3 - 3 * P1t * P2 + 2 * P3) / den
            P1tB = B @ V.T
            P3B = B @ (V ** 3).T
            tkB = (P1tB ** 2 - P2) / den if d == 2 else (P1tB ** 3 - 3 * P1tB * P2 + 2 * P3B) / den
            G_P1 = V @ V.T
            G_P2 = (V ** 2) @ (V ** 2).T
            G_P3 = (V ** 3) @ (V ** 3).T
            G = (G_P1 ** 2 - G_P2) / den if d == 2 else (G_P1 ** 3 - 3 * G_P1 * G_P2 + 2 * G_P3) / den
            w, U = np.linalg.eigh(G)
            Gis = (U / np.sqrt(w)) @ U.T
            Hs, Ht = tkA @ Gis, tkB @ Gis
            base = K2 if d == 2 else K3
            return base - Hs @ Ht.T + (Hs * np.exp(delta)) @ Ht.T

        ham = (p - P1) / 2
        rbf = ((1 - rho) / (1 + rho)) ** ham
        a = [math.comb(p, d) * rho ** d / (1 + rho) ** p for d in range(4)]
        tail = (rbf - a[0] - a[1] * K1 - a[2] * K2 - a[3] * K3) / (1 - sum(a))
        return sig[0] * K1 + sig[1] * adj(V2, d2, 2) + sig[2] * adj(V3, d3, 3) + sig[3] * tail

    K = kern(S, S)
    C = K + noise * np.eye(len(S))
    ym, ys = float(y.mean()), float(y.std())
    ystd = (y - ym) / ys
    mu = float(model.mu.detach().numpy()[0])
    alpha = np.linalg.solve(C, ystd - mu)
    cross = kern(S, S)
    ref = ym + ys * (mu + cross @ alpha)
    np.testing.assert_allclose(pred, ref, atol=1e-8, rtol=1e-8)


def test_templates_initialized_nonduplicate_and_full_rank():
    model, _, _ = _small_model(rank=6)
    for V, d in [(model.V2, 2), (model.V3, 3)]:
        Vn = V / V.norm(dim=1, keepdim=True)
        G = aa.gram_matrix(Vn, model.p, d)
        eig = torch.linalg.eigvalsh(G)
        assert eig.min() > 1e-10 * eig.max()
    # duplicate fit materials must not create duplicate template directions
    x = binary(30, 12, 20)
    x = np.vstack([x, x[:5]])  # exact duplicates
    y = np.random.default_rng(21).normal(size=35)
    m = aa.AdaptiveAnovaGP(x, y, rho=0.03, rank=6, seed=3)
    for V, d in [(m.V2, 2), (m.V3, 3)]:
        Vn = V / V.norm(dim=1, keepdim=True)
        eig = torch.linalg.eigvalsh(aa.gram_matrix(Vn, m.p, d))
        assert eig.min() > 1e-10 * eig.max()


def test_duplicate_templates_stay_finite():
    model, x, _ = _small_model(rank=4)
    with torch.no_grad():
        model.V3[1].copy_(model.V3[0])  # force a rank-deficient direction set
    pred = model(x)
    assert torch.isfinite(pred).all()
    assert torch.isfinite(model.loss(0.1))


def test_constant_target_and_rho_validation():
    x = binary(10, 8, 30)
    with pytest.raises(ValueError):
        aa.AdaptiveAnovaGP(x, np.ones(10), rho=0.03, rank=4, seed=0)
    with pytest.raises(ValueError):
        aa.AdaptiveAnovaGP(x, np.zeros(10) + np.arange(10), rho=0.0, rank=4, seed=0)
    with pytest.raises(ValueError):
        aa.AdaptiveAnovaGP(x, np.arange(10), rho=1.0, rank=4, seed=0)


def test_degenerate_fit_set_raises():
    # all-identical materials span only one direction: cannot place rank=4
    x = np.tile(binary(1, 10, 50), (12, 1))
    y = np.arange(12, dtype=np.float64)
    with pytest.raises(ValueError):
        aa.AdaptiveAnovaGP(x, y, rho=0.03, rank=4, seed=0)
    # fewer materials than requested rank
    x = binary(3, 10, 51)
    with pytest.raises(ValueError):
        aa.AdaptiveAnovaGP(x, np.arange(3, dtype=np.float64), rho=0.03, rank=4, seed=0)


def test_deterministic_initialization():
    x = binary(20, 10, 40)
    y = np.random.default_rng(41).normal(size=20)
    a = aa.AdaptiveAnovaGP(x, y, rho=0.03, rank=5, seed=7)
    b = aa.AdaptiveAnovaGP(x, y, rho=0.03, rank=5, seed=7)
    np.testing.assert_allclose(a.V2.detach().numpy(), b.V2.detach().numpy(), atol=0, rtol=0)
    np.testing.assert_allclose(a.V3.detach().numpy(), b.V3.detach().numpy(), atol=0, rtol=0)