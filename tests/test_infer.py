# tests/test_infer.py
"""Unit tests for inference runner.

We monkeypatch the underlying model.infer_image to avoid running actual model.
"""

from types import SimpleNamespace
import builtins
import pytest
from bsort import infer as infer_module
from bsort.config import FullConfig
import yaml
from pathlib import Path


def make_dummy_cfg(tmp_path: Path) -> str:
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


def test_run_inference_monkeypatched(tmp_path, monkeypatch):
    """When model.infer_image is monkeypatched, run_inference should return its value."""
    cfg_file = make_dummy_cfg(tmp_path)

    dummy_result = SimpleNamespace()
    dummy_result.dummy = True

    # patch bsort.model.infer_image used by infer.run_inference
    def fake_infer(cfg: FullConfig, image_path: str):
        assert image_path == "some.jpg"
        return dummy_result

    monkeypatch.setattr("bsort.model.infer_image", fake_infer)

    result = infer_module.run_inference(cfg_file, "some.jpg")
    assert result is dummy_result
