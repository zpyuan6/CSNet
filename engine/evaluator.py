from __future__ import annotations

from ultralytics import YOLO

from models import register_ultralytics_modules


def evaluate(weights: str, data_cfg: str, **kwargs):
    """Evaluate a trained model."""
    register_ultralytics_modules()
    model = YOLO(weights)
    return model.val(data=data_cfg, **kwargs)
