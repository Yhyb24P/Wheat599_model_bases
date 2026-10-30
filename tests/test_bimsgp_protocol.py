"""BiMSGP-protocol runner tests (CPU only; no GPU, no real training runs).

Locks the frozen protocol pieces:
* standard 10-fold construction (599 = 9*60 + 59; every material tested once;
  val=54; train 485/486; deterministic under seed 42);
* train-only scalers and the inverse-transform roundtrip;
* failure policy (explicit rows, never filtered/nanmean'ed);
* a tiny CPU training wiring check (checkpoint + scheduler + BF16-off path);
* the manifest-only CLI path (must not require torch/GPU/data).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import run_bimsgp as rb  # noqa: E402

N = 599
SIZES = [60] * 9 + [59]


def test_fold_construction_coverage_and_sizes():
    folds = rb.make_folds(n=N, seed=42, test_sizes=SIZES, val_size=54)
    assert [f["test"].size for f in folds] == SIZES
    all_test = np.concatenate([f["test"] for f in folds])
    assert np.array_equal(np.sort(all_test), np.arange(N))  # each material once
    for f in folds:
        assert f["val"].size == 54
        assert f["train"].size in (485, 486)
        sets = [set(f["train"]), set(f["val"]), set(f["test"])]
        assert sets[0].isdisjoint(sets[1]) and sets[0].isdisjoint(sets[2])
        assert sets[1].isdisjoint(sets[2])
        assert len(sets[0] | sets[1] | sets[2]) == N


def test_fold_determinism_and_seed_sensitivity():
    a = rb.make_folds(n=N, seed=42, test_sizes=SIZES, val_size=54)
    b = rb.make_folds(n=N, seed=42, test_sizes=SIZES, val_size=54)
    for fa, fb in zip(a, b):
        assert np.array_equal(fa["test"], fb["test"])
        assert np.array_equal(fa["val"], fb["val"])
        assert np.array_equal(fa["train"], fb["train"])
    c = rb.make_folds(n=N, seed=43, test_sizes=SIZES, val_size=54)
    assert not np.array_equal(a[0]["test"], c[0]["test"])


def test_scaler_train_only_and_roundtrip():
    rng = np.random.default_rng(0)
    X_train = rng.normal(5, 2, size=(100, 4))
    X_val = rng.normal(50, 10, size=(20, 4))  # wildly different on purpose
    sc = rb.Scaler().fit(X_train)
    # fitted statistics come from the train slice only
    assert np.allclose(sc.mean, X_train.mean(axis=0))
    transformed = sc.transform(X_val)
    refit_with_val = rb.Scaler().fit(np.vstack([X_train, X_val]))
    assert not np.allclose(sc.mean, refit_with_val.mean)
    back = sc.inverse_transform(sc.transform(X_train))
    assert np.allclose(back, X_train)


def test_evaluate_fold_failure_policy():
    y = np.linspace(-1, 1, 60)
    ok = rb.evaluate_fold("1", 0, y, y + 0.01)
    assert not ok["failed"] and ok["pcc"] > 0.99
    bad = rb.evaluate_fold("1", 0, y, np.full(60, np.nan))
    assert bad["failed"] and bad["reason"] == "non_finite_prediction"
    const_y = rb.evaluate_fold("1", 0, np.full(60, 0.3), y)
    assert const_y["failed"] and const_y["reason"] == "zero_variance_y_true"
    summary = rb.summarize([ok, bad, const_y])
    entry = summary["1"]
    assert entry["n_folds"] == 3 and entry["n_valid"] == 1 and entry["n_failed"] == 2
    assert entry["pcc"]["n"] == 1  # failures excluded, not nanmean-masked


def _tiny_cfg():
    return {
        "n_blocks": 1, "d_block": 8, "attention_n_heads": 2,
        "attention_dropout": 0.0, "ffn_d_hidden": 16,
        "ffn_d_hidden_multiplier": None, "ffn_dropout": 0.0,
        "residual_dropout": 0.0,
    }


def _tiny_train_cfg():
    return {
        "lr": 0.005, "weight_decay": 0.01, "batch_size": 8, "epochs": 3,
        "lr_scheduler": {"mode": "min", "factor": 0.5, "patience": 10},
    }


def test_tiny_cpu_training_wiring():
    torch = pytest.importorskip("torch")
    rtdl = pytest.importorskip("rtdl_revisiting_models")
    rng = np.random.default_rng(0)
    n, p = 40, 8
    X = rng.normal(size=(n, p))
    y = X @ rng.normal(size=p) + rng.normal(0, 0.1, n)
    Xtr, ytr = X[:30], y[:30]
    Xva, yva = X[30:], y[30:]
    model = rb.build_model(p, _tiny_cfg(), seed=42)
    model, hist, fit = rb.train_one_fold(
        model, Xtr, ytr, Xva, yva, _tiny_train_cfg(), device="cpu", seed=42)
    assert len(hist) == 3
    assert 0 <= fit["best_pcc_epoch"] < 3
    assert np.isfinite(fit["best_pcc"])
    assert 0 <= fit["best_rmse_epoch"] < 3
    assert np.isfinite(fit["best_rmse"])
    assert fit["best_rmse_state"] is not None
    assert fit["best_pcc"] == max(h["val_pcc"] for h in hist)
    assert fit["best_rmse"] == min(h["val_rmse"] for h in hist)
    pred = rb.predict(model, Xva, "cpu")
    assert pred.shape == (10,) and np.isfinite(pred).all()
    assert sum(p_.numel() for p_ in model.parameters()) > 0


def test_build_model_seeded_initialization():
    torch = pytest.importorskip("torch")
    pytest.importorskip("rtdl_revisiting_models")
    p = 8
    a = rb.build_model(p, _tiny_cfg(), seed=7)
    b = rb.build_model(p, _tiny_cfg(), seed=7)
    c = rb.build_model(p, _tiny_cfg(), seed=8)
    sa, sb, sc = a.state_dict(), b.state_dict(), c.state_dict()
    assert all(torch.equal(sa[k], sb[k]) for k in sa)
    assert any(not torch.equal(sa[k], sc[k]) for k in sa)


def test_manifest_only_cli_writes_coverage(tmp_path):
    out = tmp_path / "manifest_run"
    rb.main(["--manifest-only", "--out", str(out)])
    manifest = json.loads((out / "splits_manifest.json").read_text())
    assert manifest["seed"] == 42
    assert manifest["val_size"] == 54
    folds = manifest["folds"]
    assert len(folds) == 10
    seen = []
    for f in folds:
        assert f["n_val"] == 54
        assert f["n_train"] in (485, 486)
        seen.extend(f["test_idx"])
    assert sorted(seen) == list(range(599))
    assert not (out / "provenance.json").exists()  # no data/torch touched
