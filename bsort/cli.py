import typer
from .train import run_train
from .infer import run_inference

app = typer.Typer(help="Bottle cap detector CLI")


@app.command()
def train(config: str):
    """Train YOLO model using settings.yaml."""
    typer.echo(f"Training model with config: {config}")
    run_train(config)


@app.command()
def infer(config: str, image: str):
    """Run inference on a single image."""
    typer.echo("Running inference...")
    results = run_inference(config, image)
    results[0].show()


if __name__ == "__main__":
    app()
