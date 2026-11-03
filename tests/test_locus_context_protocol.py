"""The neural development runner never trains on scoring rows or tests before freezing."""
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from wheat599.data import sha256_of
from wheat599 import locus_context_run as protocol


def test_development_and_frozen_refit(tmp_path, monkeypatch):
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/locus_context.yaml").read_text())
    rng = np.random.default_rng(4)
    X = rng.normal(size=(60, 9))
    X[:, -1] = np.arange(60)
    Y = X[:, :4].copy()
    data = SimpleNamespace(X=X, Y=Y, n=60, environments=["1", "2", "4", "5"])
    folds, bases, final_bases = [], [], []
    raw_base = {"bandwidth": 1., "fraction": .5, "ridge": [.1] * 4}
    for k in range(10):
        test = list(range(k*6, (k+1)*6))
        dev = [i for i in range(60) if i not in test]
        inner = []
        for j in range(3):
            val = dev[j*18:(j+1)*18]
            train = [i for i in dev if i not in val]
            inner.append({"train_idx": train, "val_idx": val})
            bases.append({"fold": k, "inner_fold": j, "homogeneous": raw_base})
        folds.append({"fold": k, "dev_idx": dev, "test_idx": test, "inner": inner})
        final_bases.append({"fold": k, "squared": {"eta": .8}, "base": raw_base})
    split_path, base_path, refit_path = [tmp_path / name for name in ("splits.json", "base.json", "refit.json")]
    split_path.write_text(json.dumps({"folds": folds}))
    base_path.write_text(json.dumps(bases))
    refit_path.write_text(json.dumps({"selected": final_bases, "splits_sha256": sha256_of(split_path),
                                     "base_source_sha256": sha256_of(base_path)}))
    cfg.update(splits=str(split_path), base_choices=str(base_path), refit_base_choices=str(refit_path),
               device="cpu", training_steps=1, checkpoints=[0, 1], gammas=[.1], weight_decays=[.001])
    monkeypatch.setattr(protocol, "load", lambda *a: data)
    calls = {"phase": "development", "predicted_test": []}

    def train(X_fit, Y_fit, base, variant, gamma, decay, seed, config):
        k, j = divmod(seed - cfg["seed"], 100)
        expected = folds[k]["dev_idx"] if j == 99 else folds[k]["inner"][j]["train_idx"]
        assert X_fit[:, -1].astype(int).tolist() == expected
        np.testing.assert_array_equal(Y_fit, Y[expected])
        return {step: {"step": step} for step in config["checkpoints"]}, {"seconds": 0.}

    def predict(X_fit, Y_fit, X_target, base, variant, gamma, state, seed, config):
        k, j = divmod(seed - cfg["seed"], 100)
        expected = folds[k]["test_idx"] if j == 99 else folds[k]["inner"][j]["val_idx"]
        assert X_target[:, -1].astype(int).tolist() == expected
        if j == 99:
            assert calls["phase"] == "final"
            calls["predicted_test"].append(k)
        optimal = gamma > 0 and variant == "full" and state["step"] == 1
        return X_target[:, :4] + (0 if optimal else X_target[:, 4:8])

    fake_core = SimpleNamespace(train_context=train, predict_context=predict)
    # All model training is stubbed; no external checkout, GPU or real phenotype is used.
    monkeypatch.setitem(sys.modules, "wheat599.locus_context", fake_core)
    selected = tmp_path / "selected"
    protocol.run(cfg, SimpleNamespace(out=selected, selection=None), Path(protocol.__file__))
    assert calls["predicted_test"] == []
    selection_path = selected / "selection.json"
    frozen = json.loads(selection_path.read_text())
    assert frozen["development_gate"]["pass"]
    assert all(v["chosen"]["full"]["steps"] == 1 for v in frozen["selected"])
    calls["phase"] = "final"
    final = tmp_path / "final"
    protocol.run(cfg, SimpleNamespace(out=final, selection=selection_path), Path(protocol.__file__))
    assert len(calls["predicted_test"]) == 60
    summary = json.loads((final / "summary.json").read_text())
    assert summary["models"]["full"]["macro"]["pcc"] == pytest.approx(1.)
    failed = copy.deepcopy(frozen)
    failed["development_gate"]["pass"] = False
    selection_path.write_text(json.dumps(failed))
    with pytest.raises(ValueError, match="development condition"):
        protocol.run(cfg, SimpleNamespace(out=tmp_path / "rejected", selection=selection_path),
                     Path(protocol.__file__))
