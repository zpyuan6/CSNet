from __future__ import annotations

import json
from pathlib import Path

import torch

from .features import extract_detection_feature, roi_pool_gradient
from .probe import load_probe


def _apply_probe(feature_vec: torch.Tensor, probe: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = torch.tensor(probe["mean"], dtype=feature_vec.dtype, device=feature_vec.device)
    std = torch.tensor(probe["std"], dtype=feature_vec.dtype, device=feature_vec.device)
    weight = torch.tensor(probe["state_dict"]["linear.weight"], dtype=feature_vec.dtype, device=feature_vec.device).view(-1)
    bias = torch.tensor(probe["state_dict"]["linear.bias"], dtype=feature_vec.dtype, device=feature_vec.device).view(())
    x_std = (feature_vec - mean) / std
    logit = torch.dot(x_std, weight) + bias
    return logit, weight, x_std


def audit_single_detection_with_probes(
    model_path: str,
    image_path: str,
    probe_paths: list[str],
    imgsz: int = 640,
    device: str = "",
    class_id: int | None = None,
    det_index: int | None = None,
) -> dict:
    if not probe_paths:
        raise ValueError("At least one probe path is required.")

    probes = [load_probe(p) for p in probe_paths]
    layer_names = {probe["layer"] for probe in probes}
    if len(layer_names) != 1:
        raise ValueError("All probes passed to a single audit call must use the same layer.")
    layer = next(iter(layer_names))

    feature_vec, target, activation, box_xyxy = extract_detection_feature(
        model_path=model_path,
        image_path=image_path,
        layer=layer,
        imgsz=imgsz,
        device=device,
        class_id=class_id,
        det_index=det_index,
    )

    model_device = activation.device
    score = target["score"]
    if not isinstance(score, torch.Tensor):
        raise TypeError("Target score must be a Tensor for TCAV gradient computation.")

    score.backward(retain_graph=False)
    grad = activation.grad
    if grad is None:
        raise RuntimeError("No gradient retained for target activation.")
    grad_vec = roi_pool_gradient(grad, box_xyxy, imgsz).to(model_device)

    results = []
    for probe in probes:
        feat = feature_vec.to(model_device)
        logit, weight, _ = _apply_probe(feat, probe)
        prob = torch.sigmoid(logit)
        direction = weight / weight.norm().clamp_min(1e-6)
        tcav = torch.dot(grad_vec, direction)
        results.append(
            {
                "concept": probe["concept"],
                "layer": probe["layer"],
                "probe_logit": float(logit.detach().cpu().item()),
                "probe_prob": float(prob.detach().cpu().item()),
                "tcav_score": float(tcav.detach().cpu().item()),
            }
        )

    results.sort(key=lambda item: item["probe_prob"], reverse=True)
    return {
        "image_path": str(image_path),
        "layer": layer,
        "target": {
            "det_index": target["det_index"],
            "class_id": target["class_id"],
            "bbox_xywh": target["bbox_xywh"],
            "score": float(target["score"].detach().cpu().item()),
        },
        "concept_results": results,
    }


def save_tcav_json(result: dict, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
