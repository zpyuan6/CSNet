from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

from models import register_ultralytics_modules

from .hooks import CapsuleHookManager


def load_image_tensor(image_path: str | Path, imgsz: int) -> tuple[torch.Tensor, Image.Image]:
    image = Image.open(image_path).convert("RGB")
    resized = image.resize((imgsz, imgsz), Image.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).contiguous()
    return tensor, image


def unpack_model_outputs(outputs: Any) -> tuple[torch.Tensor, dict[str, Any]]:
    if isinstance(outputs, tuple) and len(outputs) == 2 and isinstance(outputs[1], dict):
        preds = outputs[1]
        decoded = outputs[0]
        if isinstance(decoded, tuple):
            decoded = decoded[0]
        if isinstance(decoded, tuple):
            decoded = decoded[0]
        return decoded, preds
    raise TypeError(f"Unsupported model output structure: {type(outputs)}")


def select_target_detection(decoded: torch.Tensor, class_id: int | None = None, det_index: int | None = None) -> dict[str, Any]:
    if decoded.ndim != 3 or decoded.shape[-1] < 6:
        raise ValueError(f"Expected decoded detections shaped [B, N, >=6], got {tuple(decoded.shape)}")

    sample = decoded[0]
    if det_index is None:
        if class_id is None:
            det_index = int(sample[:, 4].argmax().item())
        else:
            class_mask = sample[:, 5].long() == int(class_id)
            if not bool(class_mask.any()):
                raise ValueError(f"No decoded detection for class_id={class_id}")
            masked_scores = sample[:, 4].clone()
            masked_scores[~class_mask] = -1.0
            det_index = int(masked_scores.argmax().item())

    target = sample[det_index]
    return {
        "det_index": int(det_index),
        "score": target[4],
        "class_id": int(target[5].item()),
        "bbox_xywh": [float(v) for v in target[:4].detach().cpu().tolist()],
    }


def compute_concept_relevance(activation: torch.Tensor, gradient: torch.Tensor, k_out: int, p_out: int) -> dict[str, Any]:
    b, c, h, w = activation.shape
    expected_c = k_out * (p_out + 1)
    if c != expected_c:
        raise ValueError(f"Packed capsule channels mismatch: got {c}, expected {expected_c}")

    rel = (activation * gradient).detach()
    caps = rel.view(b, k_out, p_out + 1, h, w)[0]
    pose_rel = caps[:, :p_out]
    act_rel = caps[:, p_out]
    heatmaps = pose_rel.abs().sum(dim=1) + act_rel.abs()
    scores = heatmaps.sum(dim=(1, 2))

    concepts = []
    for type_idx in range(k_out):
        heat = heatmaps[type_idx]
        flat_idx = int(heat.argmax().item())
        y, x = divmod(flat_idx, w)
        concepts.append(
            {
                "type_idx": int(type_idx),
                "score": float(scores[type_idx].item()),
                "peak_xy": [int(x), int(y)],
                "heatmap": heat.detach().cpu().float(),
            }
        )

    concepts.sort(key=lambda item: item["score"], reverse=True)
    return {"spatial_size": [int(w), int(h)], "concepts": concepts}


def audit_single_image(
    model_path: str,
    image_path: str,
    imgsz: int = 640,
    device: str = "",
    class_id: int | None = None,
    det_index: int | None = None,
    topk: int = 5,
) -> dict[str, Any]:
    register_ultralytics_modules()
    yolo = YOLO(model_path)
    model = yolo.model
    model.eval()
    if device:
        model.to(device)

    hook_manager = CapsuleHookManager(model)
    hook_manager.register()

    image_tensor, raw_image = load_image_tensor(image_path, imgsz)
    image_tensor = image_tensor.to(next(model.parameters()).device)
    image_tensor.requires_grad_(True)

    outputs = model(image_tensor)
    decoded, _preds = unpack_model_outputs(outputs)
    target = select_target_detection(decoded, class_id=class_id, det_index=det_index)

    model.zero_grad(set_to_none=True)
    target["score"].backward(retain_graph=False)

    concepts_by_layer = []
    for name, activation in hook_manager.activations.items():
        grad = activation.output.grad
        if grad is None:
            continue
        layer_result = compute_concept_relevance(activation.output, grad, activation.k_out, activation.p_out)
        top_concepts = layer_result["concepts"][:topk]
        layer_entries = []
        for concept in top_concepts:
            heat = concept.pop("heatmap")
            layer_entries.append({**concept, "heatmap": heat.numpy().tolist()})
        concepts_by_layer.append(
            {
                "layer": name,
                "k_out": activation.k_out,
                "p_out": activation.p_out,
                "spatial_size": layer_result["spatial_size"],
                "concepts": layer_entries,
            }
        )

    hook_manager.close()
    return {
        "image_path": str(image_path),
        "image_size": list(raw_image.size),
        "imgsz": int(imgsz),
        "target": {
            "det_index": target["det_index"],
            "class_id": target["class_id"],
            "score": float(target["score"].detach().cpu().item()),
            "bbox_xywh": target["bbox_xywh"],
        },
        "concepts_by_layer": concepts_by_layer,
    }


def save_audit_json(result: dict[str, Any], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
