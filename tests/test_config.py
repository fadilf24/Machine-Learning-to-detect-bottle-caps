# tests/test_config.py
"""Unit tests for config loader."""

import yaml
from pathlib import Path
import tempfile
from bsort.config import load_config


def test_load_config(tmp_path: Path):
    """Test that load_config reads YAML and returns expected attributes."""
    data = {
        "dataset": {"train_path": "/tmp/train", "val_path": "/tmp/val"},
        "training": {"epochs": 1, "batch_size": 2, "img_size": 640, "learning_rate": 0.001},
        "model": {"name": "yolov8n.pt", "save_path": "runs/train"},
        "inference": {"conf_threshold": 0.2, "imgsz": 640},
    }
    config_file = tmp_path / "cfg.yaml"
    with open(config_file, "w", encoding="utf8") as f:
        yaml.dump(data, f)

    cfg = load_config(str(config_file))
    assert cfg.dataset.train_path == "/tmp/train"
    assert cfg.training.epochs == 1
    assert cfg.model.name == "yolov8n.pt"
    assert cfg.inference.conf_threshold == 0.2
