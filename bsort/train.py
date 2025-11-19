from .config import load_config
from .model import train_model


def run_train(config_path: str):
    """Run training pipeline."""
    cfg = load_config(config_path)
    train_model(cfg)
