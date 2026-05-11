from __future__ import annotations

import re
from pathlib import Path

from .io import load_json, save_json
from .labels import apply_label_metadata, load_concept_labels
from .visualize import render_layered_graph


def _layer_sort_key(layer: str) -> tuple[int, str]:
    if layer == "class":
        return 10**9, layer
    match = re.search(r"(\d+)", str(layer))
    return (int(match.group(1)) if match else 10**8, str(layer))


def build_layered_concept_graph(atlas_json: str | Path, out_path: str | Path) -> dict:
    atlas = load_json(atlas_json)
    atlas_path = Path(atlas_json)
    labels = load_concept_labels(atlas_path.with_name("concept_labels.csv"))
    concept_nodes: list[dict] = []
    edges: list[dict] = []
    class_nodes: dict[int, dict] = {}
    concept_layers: dict[str, str] = {}
    concept_supports: dict[str, float] = {}
    raw_class_edges: list[dict] = []

    for concept in atlas.get("concepts", []):
        node = {
            "id": concept["concept_id"],
            "kind": "concept",
            "layer": concept.get("layer"),
            "type_idx": concept.get("type_idx"),
            "score": float(concept.get("support", 0.0)),
            "support": float(concept.get("support", 0.0)),
            "concept_name": concept.get("concept_name"),
            "semantic_group": concept.get("semantic_group"),
            "confidence": concept.get("confidence"),
        }
        node = apply_label_metadata(node, labels)
        concept_nodes.append(node)
        concept_layers[concept["concept_id"]] = str(concept.get("layer"))
        concept_supports[concept["concept_id"]] = float(concept.get("support", 0.0))
        for item in concept.get("top_classes", []):
            cid = int(item["class_id"])
            if cid not in class_nodes:
                class_nodes[cid] = {"id": f"class:{cid}", "kind": "class", "layer": "class", "class_id": cid, "class_name": f"class_{cid}", "score": 0.0}
            class_nodes[cid]["score"] += float(item["weight"])
            raw_class_edges.append(
                {
                    "source": concept["concept_id"],
                    "target": f"class:{cid}",
                    "kind": "concept_to_class",
                    "weight": float(item["weight"]),
                }
            )

    ordered_layers = sorted({layer for layer in concept_layers.values()}, key=_layer_sort_key)
    next_layer_of = {layer: ordered_layers[idx + 1] for idx, layer in enumerate(ordered_layers[:-1])}
    incoming: dict[str, list[dict]] = {}

    for edge in atlas.get("edges", []):
        source = str(edge["source"])
        target = str(edge["target"])
        source_layer = concept_layers.get(source)
        target_layer = concept_layers.get(target)
        if source_layer is None or target_layer is None:
            continue
        if next_layer_of.get(source_layer) != target_layer:
            continue
        incoming.setdefault(target, []).append(
            {
                "source": source,
                "target": target,
                "kind": edge.get("kind", "concept_to_concept"),
                "weight": float(edge.get("weight", 0.0)),
                "count": int(edge.get("count", 0)),
            }
        )

    top_layer = ordered_layers[-1] if ordered_layers else None
    if top_layer is not None:
        for edge in raw_class_edges:
            if concept_layers.get(str(edge["source"])) != top_layer:
                continue
            incoming.setdefault(str(edge["target"]), []).append(edge)

        top_layer_concepts = sorted(
            (
                node for node in concept_nodes
                if str(node.get("layer")) == top_layer
            ),
            key=lambda item: float(item.get("support", item.get("score", 0.0))),
            reverse=True,
        )
        for class_node in class_nodes.values():
            target = str(class_node["id"])
            existing = incoming.setdefault(target, [])
            used_sources = {str(edge["source"]) for edge in existing}
            if len(existing) >= 2:
                continue
            for concept_node in top_layer_concepts:
                source = str(concept_node["id"])
                if source in used_sources:
                    continue
                existing.append(
                    {
                        "source": source,
                        "target": target,
                        "kind": "concept_to_class_fallback",
                        "weight": float(concept_supports.get(source, 0.0)),
                        "fallback": True,
                    }
                )
                used_sources.add(source)
                if len(existing) >= 2:
                    break

    selected_nodes: set[str] = set()
    expanded_targets: set[str] = set()
    frontier = [str(item["id"]) for item in sorted(class_nodes.values(), key=lambda item: item["class_id"])]

    while frontier:
        target = frontier.pop(0)
        if target in expanded_targets:
            continue
        expanded_targets.add(target)
        candidate_edges = incoming.get(target, [])
        ranked = sorted(candidate_edges, key=lambda item: float(item.get("weight", 0.0)), reverse=True)[:2]
        for idx, edge in enumerate(ranked):
            edge = dict(edge)
            edge["highlight"] = idx == 0
            edges.append(edge)
            selected_nodes.add(str(edge["source"]))
            selected_nodes.add(str(edge["target"]))
            if str(edge["source"]).startswith("class:"):
                continue
            frontier.append(str(edge["source"]))

    nodes = [node for node in concept_nodes if str(node["id"]) in selected_nodes]
    nodes.extend(sorted(class_nodes.values(), key=lambda item: item["class_id"]))
    graph = {
        "method": "capsule_lcrp_graph_v3",
        "graph_type": "layered_concept_graph",
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "atlas_json": str(atlas_json),
            "num_concepts": len([node for node in nodes if node["kind"] == "concept"]),
            "num_classes": len(class_nodes),
            "atlas_method": atlas.get("method"),
            "sample_methods": atlas.get("metadata", {}).get("sample_methods", []),
            "attribution_styles": atlas.get("metadata", {}).get("attribution_styles", []),
            "reference_strategy": atlas.get("metadata", {}).get("reference_strategy"),
            "edge_policy": "from_each_class_backtrace_adjacent_layer_top2_per_target_highlight_top1",
        },
    }
    save_json(graph, out_path)
    render_layered_graph(nodes, edges, Path(out_path).with_suffix(".png"), title="Layered Concept Graph")
    return graph
