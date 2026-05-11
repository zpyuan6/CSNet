from __future__ import annotations

import torch

from .concepts import grouped_channel_support_map, project_bbox_to_feature, roi_sum
from .crp_core import CapsuleLayerRelevance, OutputRelevanceSeed
from .hooks import CapturedActivation
from .lrp import heatmap_overlap_score, spatial_lcrp_relevance
from .rules import route_rule_weight


def _conv_weight_tensor(layer) -> torch.Tensor | None:
    if layer is None:
        return None
    if hasattr(layer, "weight") and isinstance(layer.weight, torch.Tensor):
        return layer.weight.detach()
    conv = getattr(layer, "conv", None)
    if conv is not None and hasattr(conv, "weight") and isinstance(conv.weight, torch.Tensor):
        return conv.weight.detach()
    return None


def _grouped_conv_gain(layer, type_idx: int, num_types: int) -> float | None:
    weight = _conv_weight_tensor(layer)
    if weight is None or weight.ndim != 4 or num_types <= 0:
        return None
    out_channels = int(weight.shape[0])
    if out_channels % int(num_types) != 0:
        return None
    group_size = out_channels // int(num_types)
    start = int(type_idx) * group_size
    end = start + group_size
    if start < 0 or end > out_channels:
        return None
    return float(weight[start:end].abs().mean().item())


def make_output_seed(class_relevance: float, box_relevance: float) -> OutputRelevanceSeed:
    class_rel = max(float(class_relevance), 0.0)
    box_rel = max(float(box_relevance), 0.0)
    return OutputRelevanceSeed(
        class_relevance=class_rel,
        box_relevance=box_rel,
        total_relevance=class_rel + box_rel,
    )


def _project_head_point_to_feature(head_xy: list[int], head_hw: list[int], out_hw: tuple[int, int]) -> tuple[int, int]:
    head_w = max(int(head_hw[0]), 1)
    head_h = max(int(head_hw[1]), 1)
    out_h, out_w = int(out_hw[0]), int(out_hw[1])
    x = min(max(int(round(float(head_xy[0]) * out_w / head_w)), 0), max(out_w - 1, 0))
    y = min(max(int(round(float(head_xy[1]) * out_h / head_h)), 0), max(out_h - 1, 0))
    return x, y


def _point_support(fmap: torch.Tensor, point_xy: tuple[int, int]) -> float:
    if fmap.ndim != 2:
        return 0.0
    x, y = int(point_xy[0]), int(point_xy[1])
    h, w = int(fmap.shape[0]), int(fmap.shape[1])
    x0, x1 = max(0, x - 1), min(w, x + 2)
    y0, y1 = max(0, y - 1), min(h, y + 2)
    patch = fmap[y0:y1, x0:x1]
    if patch.numel() == 0:
        return 0.0
    return float(patch.detach().mean().item())


def _route_weight(
    parent_capture: CapturedActivation,
    parent_type_idx: int,
    child_capture: CapturedActivation,
    child_type_idx: int,
    parent_map: torch.Tensor,
    child_map: torch.Tensor,
) -> float:
    overlap = heatmap_overlap_score(child_map.detach().cpu().tolist(), parent_map.detach().cpu().tolist())
    support_term = float(child_map.sum().detach().cpu().item())
    source_support = _source_input_support(parent_capture, child_capture, child_type_idx)
    pre_route_gain = _grouped_conv_gain(getattr(parent_capture.module, "conv_route", None), child_type_idx, int(getattr(parent_capture.module, "K_cat", child_capture.k_out)))
    post_route_gain = _grouped_conv_gain(getattr(parent_capture.module, "spagg", None), parent_type_idx, parent_capture.k_out)
    rule_weight = route_rule_weight(
        module=parent_capture.module,
        parent_type_idx=parent_type_idx,
        child_k=child_capture.k_out,
        child_p=child_capture.p_out,
        child_type_idx=child_type_idx,
    )
    conv_gain = 1.0
    if pre_route_gain is not None:
        conv_gain *= max(pre_route_gain, 1e-6)
    if post_route_gain is not None:
        conv_gain *= max(post_route_gain, 1e-6)
    if rule_weight is None:
        return float(overlap) * (max(support_term, 0.0) + 1e-6) * (source_support + 1e-6) * conv_gain
    return max(float(rule_weight), 0.0) * (float(overlap) + 1e-6) * (max(support_term, 0.0) + 1e-6) * (source_support + 1e-6) * conv_gain


def _match_source_tensor(parent_capture: CapturedActivation, child_capture: CapturedActivation) -> torch.Tensor | None:
    expected_channels = int(child_capture.k_out) * (int(child_capture.p_out) + 1)
    candidates = [tensor for tensor in parent_capture.inputs if tensor.ndim == 4 and int(tensor.shape[1]) == expected_channels]
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    spatial_matches = [
        tensor
        for tensor in candidates
        if int(tensor.shape[-2]) == int(child_capture.output.shape[-2]) and int(tensor.shape[-1]) == int(child_capture.output.shape[-1])
    ]
    if spatial_matches:
        return spatial_matches[0]
    return candidates[0]


def _source_input_support(parent_capture: CapturedActivation, child_capture: CapturedActivation, child_type_idx: int) -> float:
    source_tensor = _match_source_tensor(parent_capture, child_capture)
    if source_tensor is None:
        return 1.0
    source_map = grouped_channel_support_map(source_tensor.detach(), child_type_idx, child_capture.k_out, child_capture.p_out)
    return max(float(source_map.sum().detach().cpu().item()), 0.0)


def rank_top_capsules(
    captured: CapturedActivation,
    bbox_xyxy_model: list[float],
    image_scale: float,
    topk: int,
    head_seed: dict | None = None,
) -> list[tuple[int, float, float, float, torch.Tensor]]:
    _, _, h, w = captured.output.shape
    roi = project_bbox_to_feature(bbox_xyxy_model, image_scale=image_scale, width=w, height=h)
    rows: list[tuple[int, float, torch.Tensor]] = []
    point_xy: tuple[int, int] | None = None
    cls_gain = 0.0
    box_gain = 0.0
    if head_seed is not None:
        point_xy = _project_head_point_to_feature(head_seed["grid_xy"], head_seed["feature_hw"], (h, w))
        cls_gain = float(head_seed.get("cls_strength", 0.0))
        box_gain = float(head_seed.get("box_strength", 0.0))
    for type_idx in range(captured.k_out):
        fmap = grouped_channel_support_map(captured.output.detach(), type_idx, captured.k_out, captured.p_out)
        roi_score = float(roi_sum(fmap, roi).detach().cpu().item())
        point_score = 0.0 if point_xy is None else _point_support(fmap, point_xy)
        score = roi_score + cls_gain * point_score + box_gain * point_score
        rows.append((type_idx, score, roi_score, point_score, fmap.detach()))
    rows.sort(key=lambda item: item[1], reverse=True)
    return rows[:topk]


def seed_top_capsule_layer(
    captured: CapturedActivation,
    bbox_xyxy_model: list[float],
    image_scale: float,
    seed: OutputRelevanceSeed,
    topk: int,
    head_seed: dict | None = None,
) -> list[CapsuleLayerRelevance]:
    candidates = rank_top_capsules(
        captured,
        bbox_xyxy_model=bbox_xyxy_model,
        image_scale=image_scale,
        topk=topk,
        head_seed=head_seed,
    )
    total_cls_support = max(sum(max(roi_score + point_score * float(head_seed.get("cls_strength", 0.0) if head_seed else 0.0), 0.0) for _, _, roi_score, point_score, _ in candidates), 1e-6)
    total_box_support = max(sum(max(roi_score + point_score * float(head_seed.get("box_strength", 0.0) if head_seed else 0.0), 0.0) for _, _, roi_score, point_score, _ in candidates), 1e-6)
    concepts: list[CapsuleLayerRelevance] = []
    cls_gain = float(head_seed.get("cls_strength", 0.0)) if head_seed else 0.0
    box_gain = float(head_seed.get("box_strength", 0.0)) if head_seed else 0.0
    for type_idx, local_score, roi_score, point_score, fmap in candidates:
        cls_support = max(roi_score + cls_gain * point_score, 0.0)
        box_support = max(roi_score + box_gain * point_score, 0.0)
        concept_rel = float(seed.class_relevance) * cls_support / total_cls_support
        concept_rel += float(seed.box_relevance) * box_support / total_box_support
        heat = spatial_lcrp_relevance(
            fmap,
            relevance=concept_rel,
            box_xyxy_model=bbox_xyxy_model,
            image_scale=image_scale,
            bbox_focus=0.0,
            eps=1e-6,
        ).detach().cpu().float()
        flat_idx = int(heat.argmax().item()) if heat.numel() > 0 else 0
        py, px = divmod(flat_idx, heat.shape[1] if heat.ndim == 2 and heat.shape[1] > 0 else 1)
        concepts.append(
            CapsuleLayerRelevance(
                layer=captured.name,
                type_idx=type_idx,
                score=float(concept_rel),
                support=float(local_score),
                heatmap=heat.tolist(),
                peak_xy=[int(px), int(py)],
            )
        )
    return concepts


def propagate_capsule_layer(
    child_capture: CapturedActivation,
    parent_capture: CapturedActivation,
    parent_concepts: list[CapsuleLayerRelevance],
    topk: int,
) -> list[CapsuleLayerRelevance]:
    child_maps = {
        type_idx: grouped_channel_support_map(child_capture.output.detach(), type_idx, child_capture.k_out, child_capture.p_out)
        for type_idx in range(child_capture.k_out)
    }
    parent_maps = {
        concept.type_idx: grouped_channel_support_map(parent_capture.output.detach(), concept.type_idx, parent_capture.k_out, parent_capture.p_out)
        for concept in parent_concepts
    }
    child_scores: dict[int, float] = {type_idx: 0.0 for type_idx in range(child_capture.k_out)}
    for parent in parent_concepts:
        parent_map = parent_maps[parent.type_idx]
        weighted: list[tuple[int, float]] = []
        for child_type_idx, child_map in child_maps.items():
            weight = _route_weight(
                parent_capture=parent_capture,
                parent_type_idx=parent.type_idx,
                child_capture=child_capture,
                child_type_idx=child_type_idx,
                parent_map=parent_map,
                child_map=child_map,
            )
            weighted.append((child_type_idx, max(weight, 0.0)))
        total = max(sum(weight for _, weight in weighted), 1e-6)
        for child_type_idx, weight in weighted:
            child_scores[child_type_idx] += float(parent.score) * weight / total

    ranked = sorted(child_scores.items(), key=lambda item: item[1], reverse=True)[:topk]
    concepts: list[CapsuleLayerRelevance] = []
    for type_idx, relevance in ranked:
        fmap = child_maps[type_idx]
        heat = spatial_lcrp_relevance(
            fmap,
            relevance=float(relevance),
            box_xyxy_model=None,
            image_scale=1.0,
            bbox_focus=0.0,
            eps=1e-6,
        ).detach().cpu().float()
        flat_idx = int(heat.argmax().item()) if heat.numel() > 0 else 0
        py, px = divmod(flat_idx, heat.shape[1] if heat.ndim == 2 and heat.shape[1] > 0 else 1)
        concepts.append(
            CapsuleLayerRelevance(
                layer=child_capture.name,
                type_idx=type_idx,
                score=float(relevance),
                support=float(fmap.sum().detach().cpu().item()),
                heatmap=heat.tolist(),
                peak_xy=[int(px), int(py)],
            )
        )
    return concepts
