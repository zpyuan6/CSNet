from __future__ import annotations

import numpy as np
import torch


def epsilon_stabilizer(z: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    sign = torch.where(z >= 0, torch.ones_like(z), -torch.ones_like(z))
    return z + sign * eps


def epsilon_lrp_redistribute(
    support_map: torch.Tensor,
    relevance: float,
    eps: float = 1e-6,
    clamp_positive: bool = True,
) -> torch.Tensor:
    z = support_map
    if clamp_positive:
        z = torch.clamp(z, min=0)
    z = z.to(dtype=torch.float32)
    denom = epsilon_stabilizer(z.sum(), eps=eps)
    if float(torch.abs(denom).detach().cpu().item()) <= eps:
        return torch.zeros_like(z)
    return (z / denom) * float(relevance)


def bbox_mask_for_feature_map(
    box_xyxy_model: list[float],
    image_scale: float,
    width: int,
    height: int,
    device: torch.device,
) -> torch.Tensor:
    x1, y1, x2, y2 = box_xyxy_model
    sx = float(width) / max(float(image_scale), 1.0)
    sy = float(height) / max(float(image_scale), 1.0)
    ix1 = max(0, min(width - 1, int(x1 * sx)))
    iy1 = max(0, min(height - 1, int(y1 * sy)))
    ix2 = max(ix1 + 1, min(width, int(max(x2 * sx, ix1 + 1))))
    iy2 = max(iy1 + 1, min(height, int(max(y2 * sy, iy1 + 1))))
    mask = torch.zeros((height, width), dtype=torch.float32, device=device)
    mask[iy1:iy2, ix1:ix2] = 1.0
    return mask


def spatial_lcrp_relevance(
    support_map: torch.Tensor,
    relevance: float,
    box_xyxy_model: list[float] | None,
    image_scale: float,
    bbox_focus: float = 0.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    support = torch.clamp(support_map, min=0).to(dtype=torch.float32)
    if box_xyxy_model is not None and support.ndim == 2:
        mask = bbox_mask_for_feature_map(
            box_xyxy_model=box_xyxy_model,
            image_scale=image_scale,
            width=int(support.shape[1]),
            height=int(support.shape[0]),
            device=support.device,
        )
        support = support * (1.0 + float(bbox_focus) * mask)
    return epsilon_lrp_redistribute(support, relevance=relevance, eps=eps, clamp_positive=True)


def heatmap_overlap_score(low_heatmap: list[list[float]], high_heatmap: list[list[float]]) -> float:
    low = np.asarray(low_heatmap, dtype=np.float32)
    high = np.asarray(high_heatmap, dtype=np.float32)
    if low.size == 0 or high.size == 0:
        return 0.0
    low = low - low.min()
    high = high - high.min()
    if float(low.max()) > 0:
        low = low / float(low.max())
    if float(high.max()) > 0:
        high = high / float(high.max())
    if low.shape != high.shape:
        high_img = torch.from_numpy(high).unsqueeze(0).unsqueeze(0)
        high_resized = torch.nn.functional.interpolate(
            high_img,
            size=low.shape,
            mode="bilinear",
            align_corners=False,
        )[0, 0].numpy()
    else:
        high_resized = high
    overlap = float((low * high_resized).sum())
    norm = max(float(low.sum()), float(high_resized.sum()), 1e-6)
    return overlap / norm
