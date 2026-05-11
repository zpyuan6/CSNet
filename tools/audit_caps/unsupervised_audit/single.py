from __future__ import annotations

from pathlib import Path

import numpy as np
from ultralytics import YOLO

from models import register_ultralytics_modules

from .attribution import get_attribution_backend, run_attribution
from .hooks import CapsuleHookManager
from .io import save_json
from .lrp import heatmap_overlap_score
from .crp_core import CapsuleLayerRelevance
from .schema import ConceptEdge, ConceptHeatmap, ConceptNode, DetectionTarget, SampleAttribution
from .target import (
    get_bbox_format,
    get_detect_head,
    load_image_tensor,
    remap_bbox_to_image,
    resolve_head_seed_info,
    resolve_target_match,
    select_target_detection,
    unpack_model_outputs,
)
from .visualize import render_detection, render_heatmap_overlay


def _heatmap_crop_box(heatmap: list[list[float]], image_size: tuple[int, int]) -> list[int]:
    arr = np.asarray(heatmap, dtype=np.float32)
    if arr.size == 0:
        return [0, 0, image_size[0], image_size[1]]
    arr = arr - arr.min()
    mx = float(arr.max())
    if mx <= 0:
        return [0, 0, image_size[0], image_size[1]]
    mask = arr >= max(np.percentile(arr, 90.0), mx * 0.5)
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return [0, 0, image_size[0], image_size[1]]
    x1 = int(xs.min() * image_size[0] / arr.shape[1])
    y1 = int(ys.min() * image_size[1] / arr.shape[0])
    x2 = int((xs.max() + 1) * image_size[0] / arr.shape[1])
    y2 = int((ys.max() + 1) * image_size[1] / arr.shape[0])
    return [max(0, x1), max(0, y1), min(image_size[0], x2), min(image_size[1], y2)]


def _edge_weight_from_heatmaps(low_node: ConceptNode, high_node: ConceptNode) -> float:
    low = (low_node.heatmap.heatmap if low_node.heatmap else []) or []
    high = (high_node.heatmap.heatmap if high_node.heatmap else []) or []
    if len(low) == 0 or len(high) == 0:
        return min(float(low_node.score), float(high_node.score))
    overlap_ratio = heatmap_overlap_score(low, high)
    return min(float(low_node.score), float(high_node.score)) * overlap_ratio


def build_sample_edges(nodes: list[ConceptNode], class_id: int) -> list[ConceptEdge]:
    edges: list[ConceptEdge] = []
    by_layer: dict[str, list[ConceptNode]] = {}
    for node in nodes:
        by_layer.setdefault(str(node.layer), []).append(node)
    layers = list(by_layer.keys())
    for low_layer, high_layer in zip(layers[:-1], layers[1:]):
        for high_node in by_layer[high_layer]:
            weighted: list[tuple[float, ConceptNode]] = []
            for low_node in by_layer[low_layer]:
                weight = _edge_weight_from_heatmaps(low_node, high_node)
                weighted.append((weight, low_node))
            weighted.sort(key=lambda item: item[0], reverse=True)
            for rank, (weight, low_node) in enumerate(weighted[:3]):
                edges.append(
                    ConceptEdge(
                        source=low_node.id,
                        target=high_node.id,
                        kind="sample_concept_to_concept",
                        weight=float(weight),
                        metadata={"rank": rank},
                    )
                )
    if layers:
        top_layer = layers[-1]
        for rank, node in enumerate(sorted(by_layer[top_layer], key=lambda item: float(item.score), reverse=True)):
            edges.append(
                ConceptEdge(
                    source=node.id,
                    target=f"class:{class_id}",
                    kind="sample_concept_to_class",
                    weight=float(node.score),
                    metadata={"rank": rank},
                )
            )
    return edges


def _concept_to_node(
    concept: CapsuleLayerRelevance,
    image_path: str,
    raw_image_size: tuple[int, int],
    overlay_path: str | Path,
) -> ConceptNode:
    overlay = render_heatmap_overlay(image_path, concept.heatmap, overlay_path)
    crop_box = _heatmap_crop_box(concept.heatmap, raw_image_size)
    return ConceptNode(
        id=f"{concept.layer}:type{concept.type_idx}",
        kind="concept",
        layer=concept.layer,
        type_idx=concept.type_idx,
        score=float(concept.score),
        support=float(concept.support),
        heatmap=ConceptHeatmap(
            layer=concept.layer,
            concept_id=f"{concept.layer}:type{concept.type_idx}",
            type_idx=concept.type_idx,
            relevance_score=float(concept.score),
            peak_xy=concept.peak_xy,
            heatmap=concept.heatmap,
            sample_overlay_path=overlay,
            crop_box_image=crop_box,
        ),
    )


def run_single_audit(
    model_path: str,
    image_path: str,
    outdir: str | Path,
    imgsz: int = 640,
    device: str = "",
    class_id: int | None = None,
    det_index: int | None = None,
    topk: int = 5,
    bbox_weight: float = 0.05,
    attribution_method: str = "capsule_lcrp",
    layers: list[str] | None = None,
) -> SampleAttribution:
    backend = get_attribution_backend(attribution_method)
    register_ultralytics_modules()
    yolo = YOLO(model_path)
    model = yolo.model.eval()
    if bool(backend.get("requires_model_grad", False)):
        model.requires_grad_(True)
    if device:
        model.to(device)

    hook_manager = CapsuleHookManager(model, target_layers=layers)
    hook_manager.register()

    image_tensor, raw_image, image_prep = load_image_tensor(image_path, imgsz)
    image_tensor = image_tensor.to(next(model.parameters()).device)
    image_tensor.requires_grad_(False)

    outputs = model(image_tensor)
    decoded, preds = unpack_model_outputs(outputs)
    head = get_detect_head(model)
    bbox_format = get_bbox_format(head)
    target_raw = select_target_detection(
        decoded, class_id=class_id, det_index=det_index, bbox_format=bbox_format, image_scale=float(imgsz)
    )
    base_match = resolve_target_match(preds, head, target_raw, bbox_weight=bbox_weight)
    head_seed = resolve_head_seed_info(preds, head, base_match)
    base_score = float(base_match.score_scalar.detach().cpu().item())
    class_relevance = float(base_match.selected_score.detach().cpu().item())
    box_relevance = float((base_match.score_scalar - base_match.selected_score).detach().cpu().item())

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    mapped_bbox_xyxy = remap_bbox_to_image(target_raw["bbox_model"], bbox_format, image_prep)
    label = f"class={target_raw['class_id']} score={target_raw['score']:.4f}"
    render_detection(image_path, mapped_bbox_xyxy, label, outdir / "target_detection.png")

    target = DetectionTarget(
        det_index=target_raw["det_index"],
        class_id=target_raw["class_id"],
        score=target_raw["score"],
        bbox_xywh_image=mapped_bbox_xyxy,
        bbox_xywh_model=target_raw["bbox_model"],
        selector_branch="one2one" if "one2one" in preds else "single",
        gradient_branch="one2many" if "one2many" in preds else "single",
        raw_index=base_match.raw_index,
    )

    captures = list(hook_manager.activations.values())
    propagated = run_attribution(
        method=attribution_method,
        captures=captures,
        bbox_xyxy_model=target_raw["bbox_model"],
        image_scale=float(target_raw["image_scale"]),
        class_relevance=class_relevance,
        box_relevance=box_relevance,
        topk=topk,
        score_scalar=base_match.score_scalar,
        model=model,
        head_seed=None if head_seed is None else {
            "level_idx": head_seed.level_idx,
            "grid_xy": head_seed.grid_xy,
            "feature_hw": head_seed.feature_hw,
            "cls_strength": head_seed.cls_strength,
            "box_strength": head_seed.box_strength,
        },
    )
    overlays_dir = outdir / "concept_overlays"
    nodes = [
        _concept_to_node(
            concept,
            image_path=str(image_path),
            raw_image_size=raw_image.size,
            overlay_path=overlays_dir / f"{concept.layer.replace('.', '_')}_type{concept.type_idx}.png",
        )
        for concept in propagated
    ]

    hook_manager.close()
    nodes.sort(key=lambda item: (str(item.layer), -float(item.score)))
    edges = build_sample_edges(nodes, class_id=target.class_id)
    record = SampleAttribution(
        method=str(backend["sample_method"]),
        image_path=str(image_path),
        image_size=[int(raw_image.size[0]), int(raw_image.size[1])],
        model_path=str(model_path),
        target=target,
        nodes=nodes,
        edges=edges,
        metadata={
            "imgsz": int(imgsz),
            "bbox_format": bbox_format,
            "base_target_score": base_score,
            "backward_target": {
                "class_id": base_match.class_id,
                "raw_index": base_match.raw_index,
                "bbox_weight": float(bbox_weight),
                "selected_box": [float(v) for v in base_match.selected_box.detach().cpu().tolist()],
                "selected_score_logit": float(base_match.selected_score.detach().cpu().item()),
                "class_relevance_seed": class_relevance,
                "box_relevance_seed": box_relevance,
            },
            "head_seed": None if head_seed is None else {
                "level_idx": head_seed.level_idx,
                "local_index": head_seed.local_index,
                "grid_xy": head_seed.grid_xy,
                "feature_hw": head_seed.feature_hw,
                "cls_strength": head_seed.cls_strength,
                "box_strength": head_seed.box_strength,
            },
            "attribution_method": attribution_method,
            "attribution_style": str(backend["attribution_style"]),
            "concept_mapping": str(backend["concept_mapping"]),
            "implementation_note": str(backend["implementation_note"]),
            "spatial_relevance_rule": str(backend["spatial_relevance_rule"]),
            "edge_semantics": str(backend["edge_semantics"]),
            "target_layers": [] if not layers else [str(layer) for layer in layers],
        },
    )
    save_json(record.to_dict(), outdir / "sample_attribution.json")
    (outdir / "summary.txt").write_text(
        "\n".join(
            [
                f"image={image_path}",
                f"class_id={target.class_id}",
                f"score={target.score:.6f}",
                f"bbox_xyxy_image={target.bbox_xywh_image}",
                f"raw_index={target.raw_index}",
                f"base_target_score={base_score:.6f}",
                f"num_concepts={len(nodes)}",
                f"attribution_method={attribution_method}",
                f"attribution_style={backend['attribution_style']}",
                f"concept_mapping={backend['concept_mapping']}",
                f"implementation_note={backend['implementation_note']}",
                f"spatial_relevance_rule={backend['spatial_relevance_rule']}",
                f"edge_semantics={backend['edge_semantics']}",
                f"target_layers={[] if not layers else [str(layer) for layer in layers]}",
            ]
        ),
        encoding="utf-8",
    )
    return record
