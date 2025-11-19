import click
from bsort.train import train_model
from bsort.infer import run_inference
from bsort.config import load_config

@click.group()
def main():
    """bsort CLI for training and inference."""
    pass

@main.command()
@click.option("--config", required=True, help="Path to YAML config.")
def train(config: str):
    """Train YOLO model."""
    cfg = load_config(config)
    train_model(cfg)

@main.command()
@click.option("--config", required=True)
@click.option("--image", required=True)
def infer(config: str, image: str):
    """Run inference on an image."""
    cfg = load_config(config)
    run_inference(cfg, image)
