from ultralytics import YOLO
from .config import FullConfig


def load_yolo(model_path: str) -> YOLO:
    return YOLO(model_path)


def train_model(cfg: FullConfig):
    """Train YOLO model using wandb logging."""
    import wandb

    wandb.init(project="bottle-cap-detector", config=cfg.dict())

    model = YOLO(cfg.model.name)

    results = model.train(
        data={
            "train": f"{cfg.dataset.train_path}/images",
            "val": f"{cfg.dataset.val_path}/images",
        },
        epochs=cfg.training.epochs,
        imgsz=cfg.training.img_size,
        lr0=cfg.training.learning_rate,
        batch=cfg.training.batch_size,
        project=cfg.model.save_path,
        name="exp",
    )

    wandb.finish()
    return results


def infer_image(cfg: FullConfig, image_path: str):
    model = YOLO(f"{cfg.model.save_path}/exp/weights/best.pt")
    results = model(image_path, conf=cfg.inference.conf_threshold, imgsz=cfg.inference.imgsz)
    return results
