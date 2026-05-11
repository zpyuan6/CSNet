from __future__ import annotations

import torch


def split_capsule_channels(activation: torch.Tensor, k_out: int, p_out: int) -> torch.Tensor:
    b, c, h, w = activation.shape
    expected_c = k_out * (p_out + 1)
    if c != expected_c:
        raise ValueError(f"Packed capsule channels mismatch: got {c}, expected {expected_c}")
    return activation.view(b, k_out, p_out + 1, h, w)


def grouped_channel_support_map(activation: torch.Tensor, type_idx: int, k_out: int, p_out: int) -> torch.Tensor:
    caps = split_capsule_channels(activation, k_out, p_out)[0]
    group = caps[type_idx]
    return torch.relu(group).sum(dim=0)


def project_bbox_to_feature(box_xyxy: list[float], image_scale: float, width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = box_xyxy
    sx = float(width) / max(float(image_scale), 1.0)
    sy = float(height) / max(float(image_scale), 1.0)
    ix1 = max(0, min(width - 1, int(x1 * sx)))
    iy1 = max(0, min(height - 1, int(y1 * sy)))
    ix2 = max(ix1 + 1, min(width, int(max(x2 * sx, ix1 + 1))))
    iy2 = max(iy1 + 1, min(height, int(max(y2 * sy, iy1 + 1))))
    return [ix1, iy1, ix2, iy2]


def roi_sum(feature_map: torch.Tensor, roi_xyxy: list[int]) -> torch.Tensor:
    x1, y1, x2, y2 = roi_xyxy
    return feature_map[y1:y2, x1:x2].sum()
