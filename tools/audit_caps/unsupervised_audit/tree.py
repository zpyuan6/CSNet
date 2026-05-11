from __future__ import annotations

import re
from pathlib import Path

from .io import load_json, save_json
from .labels import apply_label_metadata, load_concept_labels
from .visualize import render_inference_tree


def _layer_sort_key(layer: str) -> tuple[int, str]:
    match = re.search(r"(\d+)", str(layer))
    return (int(match.group(1)) if match else -1, str(layer))


def build_inference_tree(audit_json: str | Path, atlas_json: str | Path, out_path: str | Path) -> dict:
    sample = load_json(audit_json)
    atlas = load_json(atlas_json)
    atlas_path = Path(atlas_json)
    sample_path = Path(audit_json)
    labels = load_concept_labels(atlas_path.with_name("concept_labels.csv"))
    atlas_by_concept = {item["concept_id"]: item for item in atlas.get("concepts", [])}

    concept_nodes: dict[str, dict] = {}
    for node in sample.get("nodes", []):
        merged = dict(node)
        merged = apply_label_metadata(merged, labels)
        merged["atlas_refs"] = atlas_by_concept.get(node["id"], {}).get("references", [])[:4]
        concept_nodes[str(node["id"])] = merged

    concept_layers = sorted(
        {str(node.get("layer")) for node in concept_nodes.values() if node.get("layer") is not None},
        key=_layer_sort_key,
    )
    kept_layers = set(concept_layers[-3:])
    concept_nodes = {
        node_id: node
        for node_id, node in concept_nodes.items()
        if str(node.get("layer")) in kept_layers
    }

    target = sample["target"]
    class_node_id = f"class:{target['class_id']}"
    class_node = {
        "id": class_node_id,
        "kind": "class",
        "layer": "class",
        "class_id": target["class_id"],
        "class_name": f"class_{target['class_id']}",
        "score": float(target["score"]),
        "target_detection_path": str(sample_path.with_name("target_detection.png")),
    }

    incoming: dict[str, list[dict]] = {}
    for edge in sample.get("edges", []):
        source_id = str(edge["source"])
        target_id = str(edge["target"])
        if not source_id.startswith("class:") and source_id not in concept_nodes:
            continue
        if not target_id.startswith("class:") and target_id not in concept_nodes:
            continue
        incoming.setdefault(target_id, []).append(
            {
                "source": source_id,
                "target": target_id,
                "kind": edge.get("kind", "sample_concept_to_concept"),
                "weight": float(edge.get("weight", 0.0)),
                "metadata": dict(edge.get("metadata", {})),
            }
        )

    selected_edges: list[dict] = []
    selected_node_ids: set[str] = {class_node_id}
    frontier = [class_node_id]
    expanded_targets: set[str] = set()
    while frontier:
        target_id = frontier.pop(0)
        if target_id in expanded_targets:
            continue
        expanded_targets.add(target_id)
        ranked = sorted(incoming.get(target_id, []), key=lambda item: float(item["weight"]), reverse=True)[:2]
        for idx, edge in enumerate(ranked):
            picked = dict(edge)
            picked["highlight"] = idx == 0
            selected_edges.append(picked)
            source_id = str(picked["source"])
            selected_node_ids.add(source_id)
            if source_id in concept_nodes:
                frontier.append(source_id)

    nodes: list[dict] = [concept_nodes[node_id] for node_id in concept_nodes if node_id in selected_node_ids]
    nodes.sort(key=lambda item: (str(item.get("layer")), int(item.get("type_idx", 10**9))))
    nodes.append(class_node)

    tree = {
        "method": "capsule_lcrp_tree_v3",
        "target": target,
        "nodes": nodes,
        "edges": selected_edges,
        "metadata": {
            "sample_json": str(audit_json),
            "atlas_json": str(atlas_json),
            "sample_method": sample.get("method"),
            "sample_attribution_style": sample.get("metadata", {}).get("attribution_style"),
            "atlas_method": atlas.get("method"),
            "atlas_attribution_styles": atlas.get("metadata", {}).get("attribution_styles", []),
            "reference_strategy": atlas.get("metadata", {}).get("reference_strategy"),
            "edge_policy": "sample_backtrace_top2_per_target_highlight_top1",
        },
    }
    save_json(tree, out_path)
    render_inference_tree(target, nodes, selected_edges, Path(out_path).with_suffix(".png"), title="Inference Tree")
    return tree
