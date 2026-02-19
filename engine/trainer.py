from __future__ import annotations

from ultralytics import YOLO

from models import register_ultralytics_modules


def train(model_cfg: str, data_cfg: str, **kwargs):
    """Train a custom YOLO-like model using Ultralytics."""
    register_ultralytics_modules()
    model = YOLO(model_cfg)

    return model.train(data=data_cfg, **kwargs)
