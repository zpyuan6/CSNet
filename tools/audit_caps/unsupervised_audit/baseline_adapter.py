from __future__ import annotations

from pathlib import Path

from .io import load_json
from .schema import ConceptEdge, ConceptHeatmap, ConceptNode, DetectionTarget, SampleAttribution


def _build_detection_target(record: dict) -> DetectionTarget:
    target = record["target"]
    backward = record.get("backward_target", {})
    return DetectionTarget(
        det_index=int(target["det_index"]),
        class_id=int(target["class_id"]),
        score=float(target["score"]),
        bbox_xywh_image=[float(v) for v in target["bbox"]],
        bbox_xywh_model=[float(v) for v in target.get("bbox_model", [])] or None,
        selector_branch=backward.get("selector_branch"),
        gradient_branch=backward.get("gradient_branch"),
        raw_index=backward.get("raw_index"),
    )


def _build_nodes(record: dict) -> list[ConceptNode]:
    nodes: list[ConceptNode] = []
    for layer_entry in record.get("concepts_by_layer", []):
        layer = str(layer_entry["layer"])
        for concept in layer_entry.get("concepts", []):
            type_idx = int(concept["type_idx"])
            concept_id = f"{layer}:type{type_idx}"
            heatmap = ConceptHeatmap(
                layer=layer,
                concept_id=concept_id,
                type_idx=type_idx,
                relevance_score=float(concept["score"]),
                peak_xy=[int(v) for v in concept.get("peak_xy", [0, 0])],
                heatmap=concept.get("heatmap"),
            )
            nodes.append(
                ConceptNode(
                    id=concept_id,
                    kind="concept",
                    layer=layer,
                    type_idx=type_idx,
                    score=float(concept["score"]),
                    heatmap=heatmap,
                )
            )
    return nodes


def _build_edges(nodes: list[ConceptNode], target: DetectionTarget) -> list[ConceptEdge]:
    edges: list[ConceptEdge] = []
    nodes_by_layer: dict[str, list[ConceptNode]] = {}
    for node in nodes:
        nodes_by_layer.setdefault(str(node.layer), []).append(node)

    ordered_layers = list(nodes_by_layer.keys())
    for low_layer, high_layer in zip(ordered_layers[:-1], ordered_layers[1:]):
        low_nodes = nodes_by_layer[low_layer]
        high_nodes = nodes_by_layer[high_layer]
        for low_node in low_nodes:
            for high_node in high_nodes:
                weight = min(float(low_node.score), float(high_node.score))
                edges.append(
                    ConceptEdge(
                        source=low_node.id,
                        target=high_node.id,
                        kind="baseline_coactivation",
                        weight=weight,
                    )
                )

    class_node_id = f"class:{target.class_id}"
    top_layer = ordered_layers[-1] if ordered_layers else None
    if top_layer is not None:
        for node in nodes_by_layer[top_layer]:
            edges.append(
                ConceptEdge(
                    source=node.id,
                    target=class_node_id,
                    kind="baseline_concept_to_class",
                    weight=float(node.score),
                )
            )
    return edges


def load_baseline_sample_attribution(audit_json: str | Path, model_path: str = "") -> SampleAttribution:
    record = load_json(audit_json)
    target = _build_detection_target(record)
    nodes = _build_nodes(record)
    edges = _build_edges(nodes, target)
    return SampleAttribution(
        method="baseline_unsupervised_adapter",
        image_path=str(record["image_path"]),
        image_size=[int(v) for v in record.get("image_size", [])],
        model_path=model_path,
        target=target,
        nodes=nodes,
        edges=edges,
        metadata={
            "source_audit_json": str(audit_json),
            "imgsz": record.get("imgsz"),
            "relevance_mode": record.get("relevance_mode"),
            "backward_target": record.get("backward_target", {}),
        },
    )
