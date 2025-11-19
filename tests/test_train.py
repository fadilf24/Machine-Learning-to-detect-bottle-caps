# tests/test_train.py
"""Unit test for training runner that ensures train_model gets called."""

from types import SimpleNamespace
import yaml
from pathlib import Path
import pytest
from bsort import train as train_module


def make_cfg(tmp_path: Path) -> str:
    cfg = {
        "dataset": {"train_path": "/tmp/train", "val_path": "/tmp/val"},
        "training": {"epochs": 1, "batch_size": 2, "img_size": 640, "learning_rate": 0.001},
        "model": {"name": "yolov8n.pt", "save_path": "runs/train"},
        "inference": {"conf_threshold": 0.2, "imgsz": 640},
    }
    p = tmp_path / "cfg.yaml"
    with open(p, "w", encoding="utf8") as f:
        yaml.dump(cfg, f)
    return str(p)


def test_run_train_calls_train_model(tmp_path, monkeypatch):
    cfg_path = make_cfg(tmp_path)

    called = {"ok": False}

    def fake_train_model(cfg):
        # ensure cfg has the expected structure
        assert hasattr(cfg, "training")
        called["ok"] = True
        return SimpleNamespace(status="trained")

    monkeypatch.setattr("bsort.model.train_model", fake_train_model)

    result = train_module.run_train(cfg_path)
    assert called["ok"] is True
