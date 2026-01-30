from __future__ import annotations

from ultralytics import YOLO

from models import register_ultralytics_modules


def export(weights: str, **kwargs):
    """Export a trained model for deployment."""
    register_ultralytics_modules()
    model = YOLO(weights)
    return model.export(**kwargs)
