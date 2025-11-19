import yaml
from pydantic import BaseModel


class DatasetConfig(BaseModel):
    train_path: str
    val_path: str


class TrainingConfig(BaseModel):
    epochs: int
    batch_size: int
    img_size: int
    learning_rate: float


class ModelConfig(BaseModel):
    name: str
    save_path: str


class InferenceConfig(BaseModel):
    conf_threshold: float
    imgsz: int


class FullConfig(BaseModel):
    dataset: DatasetConfig
    training: TrainingConfig
    model: ModelConfig
    inference: InferenceConfig


def load_config(path: str) -> FullConfig:
    """Load YAML config into structured Python object."""
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return FullConfig(**data)
