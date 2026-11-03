"""Selection must round-trip into refit without touching real held-out data."""
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

runner = importlib.import_module("wheat599.dnawhisper.run")


def test_selection_refit_roundtrip(tmp_path, monkeypatch):
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] /
                          "configs/dnawhisper_nested.yaml").read_text())
    rng = np.random.default_rng(7)
    data = SimpleNamespace(n=24, X=rng.integers(0, 2, (24, 12)),
                           Y=rng.normal(size=(24, 4)), environments=["1", "2", "4", "5"])
    folds = []
    for k in range(2):
        test = list(range(k * 12, (k + 1) * 12))
        dev = [i for i in range(24) if i not in test]
        inner = []
        for j in range(3):
            val = dev[j * 4:(j + 1) * 4]
            inner.append({"train_idx": [i for i in dev if i not in val], "val_idx": val})
        folds.append({"fold": k, "dev_idx": dev, "test_idx": test, "inner": inner})
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({"folds": folds}))
    cfg.update(splits=str(splits), device="cpu", precision="32-true",
               pretrain={"epochs": 2}, finetune={"epochs": 3})
    monkeypatch.setattr(runner, "load", lambda *a: data)
    monkeypatch.setattr(runner, "verify_upstream", lambda *a: {})
    monkeypatch.setattr(runner, "official_files", lambda *a: {})
    cost = {"seconds": 0.0, "optimizer_steps": 1, "peak_megabytes": 0.0}
    calls = {"refit": [], "predict": []}
    monkeypatch.setattr(runner, "pretrain", lambda *a: ({}, 123, dict(cost)))

    def finetune(*args, **kwargs):
        return object(), [0.1, 0.3, 0.2], dict(cost)

    def refit(checkout, config, pretrained, rows, seed, epochs, **kwargs):
        calls["refit"].append((epochs, len(rows[0])))
        return object(), dict(cost)

    def predict(module, markers, environments, *args, **kwargs):
        calls["predict"].append(len(markers))
        return rng.normal(size=(len(markers), environments))

    monkeypatch.setattr(runner, "finetune_curve", finetune)
    monkeypatch.setattr(runner, "refit", refit)
    monkeypatch.setattr(runner, "predict", predict)
    selected = tmp_path / "selection"
    runner.run(cfg, SimpleNamespace(out=selected, selection=None), Path(__file__))
    assert not calls["predict"]
    frozen_path = selected / "selection.json"
    frozen = json.loads(frozen_path.read_text())
    assert frozen["config"] == cfg
    assert [s["epoch"] for s in frozen["selected"]] == [2, 2]
    final = tmp_path / "final"
    runner.run(cfg, SimpleNamespace(out=final, selection=frozen_path), Path(__file__))
    assert calls["refit"] == [(2, 12), (2, 12)]
    assert calls["predict"] == [12, 12]
    summary = json.loads((final / "summary.json").read_text())
    assert summary["n_runs"] == 8
    assert np.isfinite(list(summary["macro"].values())).all()
    frozen["config"]["batch_size"] += 1
    frozen_path.write_text(json.dumps(frozen))
    with pytest.raises(ValueError, match="Frozen selection"):
        runner.run(cfg, SimpleNamespace(out=tmp_path / "rejected", selection=frozen_path),
                   Path(__file__))
