from __future__ import annotations

import torch

from .concepts import split_capsule_channels
from .crp_core import CapsuleLayerRelevance
from .hooks import CapturedActivation


def _peak_centered_local_sum(heatmap: torch.Tensor, radius: int = 1) -> tuple[float, list[int]]:
    if heatmap.numel() == 0:
        return 0.0, [0, 0]
    peak_flat = int(heatmap.reshape(-1).argmax().item())
    width = int(heatmap.shape[-1])
    y, x = divmod(peak_flat, width)
    y1 = max(0, int(y) - radius)
    y2 = min(int(heatmap.shape[-2]), int(y) + radius + 1)
    x1 = max(0, int(x) - radius)
    x2 = min(int(heatmap.shape[-1]), int(x) + radius + 1)
    local_sum = float(heatmap[y1:y2, x1:x2].sum().detach().cpu().item())
    return local_sum, [int(x), int(y)]


def _normalize_heatmap(heatmap: torch.Tensor) -> list[list[float]]:
    heatmap = heatmap.detach().float()
    if heatmap.numel() == 0:
        return []
    heatmap = heatmap - heatmap.min()
    max_value = float(heatmap.max().item())
    if max_value > 0:
        heatmap = heatmap / max_value
    return heatmap.cpu().tolist()


def _peak_xy(heatmap: torch.Tensor) -> list[int]:
    if heatmap.numel() == 0:
        return [0, 0]
    flat_idx = int(heatmap.reshape(-1).argmax().item())
    width = int(heatmap.shape[-1])
    y, x = divmod(flat_idx, width)
    return [int(x), int(y)]


def run_capsule_actgrad(
    captures: list[CapturedActivation],
    score_scalar: torch.Tensor,
    topk: int,
    model: torch.nn.Module | None = None,
) -> list[CapsuleLayerRelevance]:
    if not captures:
        return []
    if model is not None:
        model.zero_grad(set_to_none=True)
    score_scalar.backward(retain_graph=False)

    concepts: list[CapsuleLayerRelevance] = []
    for capture in captures:
        grad = capture.output.grad
        if grad is None:
            continue
        acts = split_capsule_channels(capture.output, capture.k_out, capture.p_out)[0]
        grads = split_capsule_channels(grad, capture.k_out, capture.p_out)[0]
        layer_concepts: list[CapsuleLayerRelevance] = []
        for type_idx in range(capture.k_out):
            attr = torch.relu(acts[type_idx] * grads[type_idx])
            heat = attr.sum(dim=0)
            score, peak_xy = _peak_centered_local_sum(heat, radius=1)
            if score <= 0:
                # Fallback keeps the layer usable when positive support vanishes.
                heat = (acts[type_idx] * grads[type_idx]).abs().sum(dim=0)
                score, peak_xy = _peak_centered_local_sum(heat, radius=1)
            if score <= 0:
                continue
            layer_concepts.append(
                CapsuleLayerRelevance(
                    layer=capture.name,
                    type_idx=int(type_idx),
                    score=score,
                    support=score,
                    heatmap=_normalize_heatmap(heat),
                    peak_xy=peak_xy,
                )
            )
        layer_concepts.sort(key=lambda item: float(item.score), reverse=True)
        concepts.extend(layer_concepts[:topk])
    return concepts
