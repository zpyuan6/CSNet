from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps


@dataclass
class TargetMatch:
    class_id: int
    raw_index: int
    selected_score: torch.Tensor
    selected_box: torch.Tensor
    score_scalar: torch.Tensor


@dataclass
class HeadSeedInfo:
    level_idx: int
    local_index: int
    grid_xy: list[int]
    feature_hw: list[int]
    cls_strength: float
    box_strength: float


@dataclass
class ImagePrep:
    image_size: tuple[int, int]
    resized_size: tuple[int, int]
    scale: float
    pad: tuple[float, float]


def load_image_tensor(image_path: str | Path, imgsz: int) -> tuple[torch.Tensor, Image.Image, ImagePrep]:
    image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    image_w, image_h = image.size
    scale = min(float(imgsz) / max(image_w, 1), float(imgsz) / max(image_h, 1))
    new_w = max(1, int(round(image_w * scale)))
    new_h = max(1, int(round(image_h * scale)))
    resized = image.resize((new_w, new_h), Image.BILINEAR)
    pad_w = imgsz - new_w
    pad_h = imgsz - new_h
    left = pad_w // 2
    top = pad_h // 2
    canvas = Image.new("RGB", (imgsz, imgsz), (114, 114, 114))
    canvas.paste(resized, (left, top))
    array = np.asarray(canvas, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).contiguous()
    meta = ImagePrep(
        image_size=(int(image_w), int(image_h)),
        resized_size=(int(new_w), int(new_h)),
        scale=float(scale),
        pad=(float(left), float(top)),
    )
    return tensor, image, meta


def unpack_model_outputs(outputs: Any) -> tuple[torch.Tensor, dict[str, Any]]:
    if isinstance(outputs, tuple) and len(outputs) == 2 and isinstance(outputs[1], dict):
        decoded = outputs[0]
        preds = outputs[1]
        if isinstance(decoded, tuple):
            decoded = decoded[0]
        if isinstance(decoded, tuple):
            decoded = decoded[0]
        return decoded, preds
    raise TypeError(f"Unsupported model output structure: {type(outputs)}")


def get_detect_head(model: torch.nn.Module) -> torch.nn.Module:
    head = model.model[-1] if hasattr(model, "model") else None
    if head is None:
        raise TypeError("Unable to locate detection head on model.")
    return head


def get_bbox_format(head: torch.nn.Module) -> str:
    return "xyxy" if bool(getattr(head, "end2end", False)) else "xywh"


def _xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    x, y, w, h = boxes.unbind(dim=-1)
    return torch.stack((x - w / 2.0, y - h / 2.0, x + w / 2.0, y + h / 2.0), dim=-1)


def remap_bbox_to_image(bbox: list[float], bbox_format: str, prep: ImagePrep) -> list[float]:
    image_w, image_h = prep.image_size
    scale = max(float(prep.scale), 1e-8)
    pad_x, pad_y = prep.pad
    box = torch.tensor(bbox, dtype=torch.float32)
    if bbox_format == "xyxy":
        box = box - torch.tensor([pad_x, pad_y, pad_x, pad_y], dtype=torch.float32)
        box = box / scale
        box[0::2] = box[0::2].clamp(0, image_w)
        box[1::2] = box[1::2].clamp(0, image_h)
        return [float(v) for v in box.tolist()]
    box[:2] = box[:2] - torch.tensor([pad_x, pad_y], dtype=torch.float32)
    box = box / scale
    xyxy = _xywh_to_xyxy(box.unsqueeze(0))[0]
    xyxy[0::2] = xyxy[0::2].clamp(0, image_w)
    xyxy[1::2] = xyxy[1::2].clamp(0, image_h)
    return [float(v) for v in xyxy.tolist()]


def select_target_detection(
    decoded: torch.Tensor,
    class_id: int | None = None,
    det_index: int | None = None,
    bbox_format: str = "xyxy",
    image_scale: float = 640.0,
) -> dict[str, Any]:
    sample = decoded[0]
    if det_index is None:
        if class_id is None:
            det_index = int(sample[:, 4].argmax().item())
        else:
            mask = sample[:, 5].long() == int(class_id)
            if not bool(mask.any()):
                raise ValueError(f"No detection for class_id={class_id}")
            masked = sample[:, 4].clone()
            masked[~mask] = -1.0
            det_index = int(masked.argmax().item())
    target = sample[det_index]
    return {
        "det_index": int(det_index),
        "score": float(target[4].detach().cpu().item()),
        "class_id": int(target[5].item()),
        "bbox_model": [float(v) for v in target[:4].detach().cpu().tolist()],
        "bbox_format": bbox_format,
        "image_scale": float(image_scale),
    }


def decode_raw_boxes(raw: dict[str, Any], head: torch.nn.Module) -> torch.Tensor:
    if "boxes" not in raw:
        raise TypeError("Expected raw prediction dict with 'boxes'.")
    return head._get_decode_boxes(raw)


def _match_final_detection_to_train_branch(
    target: dict[str, Any], train_scores: torch.Tensor, train_boxes: torch.Tensor
) -> tuple[int, int]:
    target_class = int(target["class_id"])
    target_box = torch.tensor(target["bbox_model"], device=train_boxes.device, dtype=train_boxes.dtype)
    if target["bbox_format"] == "xywh":
        target_box = _xywh_to_xyxy(target_box.unsqueeze(0))[0]
        candidate_boxes = _xywh_to_xyxy(train_boxes[0].transpose(0, 1))
    else:
        candidate_boxes = train_boxes[0].transpose(0, 1)
    class_scores = train_scores[0, target_class].sigmoid()
    l1 = (candidate_boxes - target_box.unsqueeze(0)).abs().sum(dim=1)
    norm = max(float(target["image_scale"]), 1.0)
    match_score = class_scores - (l1 / norm)
    return target_class, int(match_score.argmax().item())


def resolve_target_match(
    preds: dict[str, Any], head: torch.nn.Module, target: dict[str, Any], bbox_weight: float = 0.05
) -> TargetMatch:
    train_raw = preds.get("one2many", preds)
    if not isinstance(train_raw, dict) or "scores" not in train_raw:
        raise TypeError("Expected differentiable prediction dict with 'scores'.")
    train_scores = train_raw["scores"]
    train_boxes = decode_raw_boxes(train_raw, head)
    class_id, raw_index = _match_final_detection_to_train_branch(target, train_scores, train_boxes)
    selected_score = train_scores[0, class_id, raw_index]
    selected_box = train_boxes[0, :, raw_index]
    if (not selected_score.requires_grad) and isinstance(train_raw.get("feats"), list) and hasattr(head, "forward_head"):
        try:
            grad_raw = head.forward_head(train_raw["feats"], **getattr(head, "one2many", {}))
        except Exception:
            grad_raw = None
        if isinstance(grad_raw, dict) and "scores" in grad_raw and "boxes" in grad_raw:
            grad_boxes = decode_raw_boxes(grad_raw, head)
            grad_score = grad_raw["scores"][0, class_id, raw_index]
            grad_box = grad_boxes[0, :, raw_index]
            if grad_score.requires_grad:
                selected_score = grad_score
                selected_box = grad_box
    box_term = selected_box.mean() / max(float(target["image_scale"]), 1.0)
    score_scalar = selected_score + float(bbox_weight) * box_term
    return TargetMatch(
        class_id=class_id,
        raw_index=raw_index,
        selected_score=selected_score,
        selected_box=selected_box,
        score_scalar=score_scalar,
    )


def resolve_head_seed_info(preds: dict[str, Any], head: torch.nn.Module, match: TargetMatch) -> HeadSeedInfo | None:
    train_raw = preds.get("one2many", preds)
    if not isinstance(train_raw, dict):
        return None
    feats = train_raw.get("feats")
    offset = 0
    raw_index = int(match.raw_index)
    if isinstance(feats, list):
        for level_idx, feat in enumerate(feats):
            if not isinstance(feat, torch.Tensor) or feat.ndim != 4:
                continue
            h = int(feat.shape[-2])
            w = int(feat.shape[-1])
            cells = h * w
            if raw_index < offset + cells:
                local_index = raw_index - offset
                gy, gx = divmod(local_index, w)
                local_feat = feat[0, :, gy, gx].detach().abs()
                feat_strength = float(local_feat.mean().item())
                box_strength = feat_strength
                cls_strength = feat_strength

                scores = train_raw.get("scores")
                if isinstance(scores, torch.Tensor) and scores.ndim == 3:
                    cls_slice = scores[0, :, raw_index].detach().abs()
                    cls_strength = float(cls_slice.mean().item())

                boxes = train_raw.get("boxes")
                if isinstance(boxes, torch.Tensor) and boxes.ndim == 3:
                    box_slice = boxes[0, :, raw_index].detach().abs()
                    box_strength = float(box_slice.mean().item())

                return HeadSeedInfo(
                    level_idx=level_idx,
                    local_index=int(local_index),
                    grid_xy=[int(gx), int(gy)],
                    feature_hw=[int(w), int(h)],
                    cls_strength=cls_strength,
                    box_strength=box_strength,
                )
            offset += cells

    if feats is None or not hasattr(head, "_build_feats"):
        return None
    try:
        box_feats, cls_feats, _ = head._build_feats(feats)
    except Exception:
        # Some heads (e.g. capsule segmentation variants) may store already-processed
        # features in preds["feats"], which no longer match _build_feats() input layout.
        # In that case head seeding is optional, so fall back to no head seed.
        return None
    offset = 0
    for level_idx, (box_feat, cls_feat) in enumerate(zip(box_feats, cls_feats)):
        h = int(box_feat.shape[-2])
        w = int(box_feat.shape[-1])
        cells = h * w
        if raw_index < offset + cells:
            local_index = raw_index - offset
            gy, gx = divmod(local_index, w)
            cls_strength = float(cls_feat[0, :, gy, gx].detach().abs().mean().item())
            box_strength = float(box_feat[0, :, gy, gx].detach().abs().mean().item())
            return HeadSeedInfo(
                level_idx=level_idx,
                local_index=int(local_index),
                grid_xy=[int(gx), int(gy)],
                feature_hw=[int(w), int(h)],
                cls_strength=cls_strength,
                box_strength=box_strength,
            )
        offset += cells
    return None
