from __future__ import annotations

from .crp_core import CapsuleLayerRelevance
from .crp_rules import make_output_seed, propagate_capsule_layer, seed_top_capsule_layer
from .hooks import CapturedActivation


def run_capsule_lcrp(
    captures: list[CapturedActivation],
    bbox_xyxy_model: list[float],
    image_scale: float,
    class_relevance: float,
    box_relevance: float,
    topk: int,
    head_seed: dict | None = None,
) -> list[CapsuleLayerRelevance]:
    if not captures:
        return []
    ordered = captures
    seed = make_output_seed(class_relevance=class_relevance, box_relevance=box_relevance)
    propagated_by_layer: dict[str, list[CapsuleLayerRelevance]] = {}
    top_capture = ordered[-1]
    top_concepts = seed_top_capsule_layer(
        top_capture,
        bbox_xyxy_model=bbox_xyxy_model,
        image_scale=image_scale,
        seed=seed,
        topk=topk,
        head_seed=head_seed,
    )
    propagated_by_layer[top_capture.name] = top_concepts
    parent_capture = top_capture
    parent_concepts = top_concepts
    for child_capture in reversed(ordered[:-1]):
        child_concepts = propagate_capsule_layer(
            child_capture=child_capture,
            parent_capture=parent_capture,
            parent_concepts=parent_concepts,
            topk=topk,
        )
        propagated_by_layer[child_capture.name] = child_concepts
        parent_capture = child_capture
        parent_concepts = child_concepts

    concepts: list[CapsuleLayerRelevance] = []
    for captured in ordered:
        concepts.extend(propagated_by_layer.get(captured.name, []))
    return concepts
