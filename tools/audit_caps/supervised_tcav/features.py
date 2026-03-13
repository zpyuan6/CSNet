from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from ultralytics import YOLO

from models import register_ultralytics_modules
from tools.audit_caps.hooks import CapsuleHookManager
from tools.audit_caps.relevance import load_image_tensor, select_target_detection, unpack_model_outputs

from .data import ConceptAnnotation


def load_yolo_model(model_path: str, device: str = "") -> tuple[YOLO, torch.nn.Module]:
    register_ultralytics_modules()
    yolo = YOLO(model_path)
    model = yolo.model.eval()
    if device:
        model.to(device)
    return yolo, model


def _roi_pool(feature: torch.Tensor, box_xyxy: tuple[float, float, float, float] | None, imgsz: int) -> torch.Tensor:
    # feature: [1, C, H, W]
    fmap = feature[0]
    _, h, w = fmap.shape
    if box_xyxy is None:
        return fmap.mean(dim=(1, 2))

    x1, y1, x2, y2 = box_xyxy
    sx = w / float(imgsz)
    sy = h / float(imgsz)
    ix1 = max(0, min(w - 1, int(x1 * sx)))
    iy1 = max(0, min(h - 1, int(y1 * sy)))
    ix2 = max(ix1 + 1, min(w, int(max(x2 * sx, ix1 + 1))))
    iy2 = max(iy1 + 1, min(h, int(max(y2 * sy, iy1 + 1))))
    roi = fmap[:, iy1:iy2, ix1:ix2]
    return roi.mean(dim=(1, 2))


def _resize_box_to_imgsz(box_xyxy: tuple[float, float, float, float], orig_w: int, orig_h: int, imgsz: int) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box_xyxy
    return (
        x1 * imgsz / float(orig_w),
        y1 * imgsz / float(orig_h),
        x2 * imgsz / float(orig_w),
        y2 * imgsz / float(orig_h),
    )


def extract_feature_dataset(
    model_path: str,
    annotations: list[ConceptAnnotation],
    layer: str,
    imgsz: int = 640,
    device: str = "",
) -> tuple[torch.Tensor, torch.Tensor]:
    _, model = load_yolo_model(model_path, device=device)
    hook_manager = CapsuleHookManager(model)
    hook_manager.register()

    by_image: dict[str, list[ConceptAnnotation]] = defaultdict(list)
    for ann in annotations:
        by_image[ann.image_path].append(ann)

    features: list[torch.Tensor] = []
    labels: list[float] = []
    model_device = next(model.parameters()).device

    with torch.no_grad():
        for image_path, rows in by_image.items():
            hook_manager.clear()
            image_tensor, raw_image = load_image_tensor(image_path, imgsz)
            image_tensor = image_tensor.to(model_device)
            _ = model(image_tensor)
            activation = hook_manager.activations.get(layer)
            if activation is None:
                raise ValueError(f"Layer '{layer}' not found in captured capsule activations.")

            orig_w, orig_h = raw_image.size
            for row in rows:
                box = None
                if row.has_box:
                    box = _resize_box_to_imgsz((row.x1, row.y1, row.x2, row.y2), orig_w, orig_h, imgsz)
                pooled = _roi_pool(activation.output.detach(), box, imgsz)
                features.append(pooled.cpu())
                labels.append(float(row.label))

    hook_manager.close()
    x = torch.stack(features, dim=0).float()
    y = torch.tensor(labels, dtype=torch.float32)
    return x, y


def extract_detection_feature(
    model_path: str,
    image_path: str,
    layer: str,
    imgsz: int = 640,
    device: str = "",
    class_id: int | None = None,
    det_index: int | None = None,
) -> tuple[torch.Tensor, dict[str, Any], torch.Tensor, tuple[float, float, float, float]]:
    _, model = load_yolo_model(model_path, device=device)
    hook_manager = CapsuleHookManager(model)
    hook_manager.register()

    image_tensor, _ = load_image_tensor(image_path, imgsz)
    image_tensor = image_tensor.to(next(model.parameters()).device)
    image_tensor.requires_grad_(True)

    outputs = model(image_tensor)
    decoded, _ = unpack_model_outputs(outputs)
    target = select_target_detection(decoded, class_id=class_id, det_index=det_index)

    activation = hook_manager.activations.get(layer)
    if activation is None:
        hook_manager.close()
        raise ValueError(f"Layer '{layer}' not found in captured capsule activations.")

    cx, cy, bw, bh = target["bbox_xywh"]
    x1 = max(cx - bw / 2.0, 0.0)
    y1 = max(cy - bh / 2.0, 0.0)
    x2 = min(cx + bw / 2.0, float(imgsz))
    y2 = min(cy + bh / 2.0, float(imgsz))
    box_xyxy = (x1, y1, x2, y2)

    feature_vec = _roi_pool(activation.output, box_xyxy, imgsz)
    return feature_vec, target, activation.output, box_xyxy


def roi_pool_gradient(grad: torch.Tensor, box_xyxy: tuple[float, float, float, float], imgsz: int) -> torch.Tensor:
    return _roi_pool(grad, box_xyxy, imgsz)
