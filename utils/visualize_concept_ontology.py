from __future__ import annotations

import argparse
import html
from pathlib import Path

import yaml


LEVEL_COLORS = {
    "L1": "#DCEBFA",
    "L2": "#DFF3E4",
    "L3": "#FFF0CC",
    "L4": "#F8D7DA",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize concept ontology YAML as an SVG graph.")
    parser.add_argument("--concept-yaml", required=True, help="Path to concept ontology yaml.")
    parser.add_argument(
        "--morphology-mapping",
        default=None,
        help="Optional YAML mapping file that provides morphology-oriented display names.",
    )
    parser.add_argument("--out", required=True, help="Output SVG path.")
    parser.add_argument("--title", default=None, help="Optional title shown at the top of the figure.")
    parser.add_argument(
        "--class-outdir",
        default=None,
        help="Optional directory for per-class subtree SVG files. If omitted, no per-class files are written.",
    )
    return parser.parse_args()


def load_ontology(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("Concept ontology must be a YAML mapping.")
    return data


def load_morphology_mapping(path: str | Path | None) -> dict[str, str]:
    if not path:
        return {}
    data = load_ontology(path)
    mapping = data.get("mapping", {})
    if not isinstance(mapping, dict):
        return {}
    out: dict[str, str] = {}
    for concept_id, item in mapping.items():
        if not isinstance(item, dict):
            continue
        term = item.get("morphology_term")
        if term:
            out[str(concept_id)] = str(term)
    return out


def build_layout(data: dict, display_names: dict[str, str] | None = None) -> tuple[list[dict], list[tuple[str, str]], str]:
    levels = data.get("levels", [])
    level_concepts = data.get("level_concepts", {})
    concepts = data.get("concepts", [])
    meta = data.get("meta", {})
    display_names = display_names or {}

    concept_defs = {str(item["id"]): item for item in concepts}
    level_ids = [str(level["id"]) for level in levels]

    x_step = 340
    y_step = 120
    box_w = 240
    box_h = 54
    left = 60
    top = 100

    nodes: list[dict] = []
    positions: dict[str, tuple[int, int]] = {}

    for col, level_id in enumerate(level_ids):
        concept_map = level_concepts.get(level_id, {})
        concept_ids = list(concept_map.keys()) if isinstance(concept_map, dict) else list(concept_map)
        for row, concept_id in enumerate(concept_ids):
            item = concept_defs.get(concept_id)
            if item is None:
                continue
            x = left + col * x_step
            y = top + row * y_step
            positions[concept_id] = (x, y)
            nodes.append(
                {
                    "id": concept_id,
                    "name": display_names.get(concept_id, str(item.get("name", concept_id))),
                    "level": str(item.get("level", level_id)),
                    "scope": concept_map.get(concept_id, "instance") if isinstance(concept_map, dict) else "instance",
                    "description": str(item.get("description", "")),
                    "x": x,
                    "y": y,
                    "w": box_w,
                    "h": box_h,
                }
            )

    edges: list[tuple[str, str]] = []
    for item in concepts:
        parent_id = str(item["id"])
        for child_id in item.get("children", []):
            child_id = str(child_id)
            if parent_id in positions and child_id in positions:
                edges.append((parent_id, child_id))

    title = str(meta.get("dataset", "concept_ontology")).replace("_", " ").title()
    return nodes, edges, title


def build_subtree_layout(
    data: dict,
    root_id: str,
    display_names: dict[str, str] | None = None,
) -> tuple[list[dict], list[tuple[str, str]], str]:
    levels = data.get("levels", [])
    level_concepts = data.get("level_concepts", {})
    concepts = data.get("concepts", [])
    meta = data.get("meta", {})
    display_names = display_names or {}

    concept_defs = {str(item["id"]): item for item in concepts}
    if root_id not in concept_defs:
        raise KeyError(f"Unknown concept root: {root_id}")

    children_map = {str(item["id"]): [str(child) for child in item.get("children", [])] for item in concepts}
    keep: set[str] = set()

    def visit(node_id: str) -> None:
        if node_id in keep:
            return
        keep.add(node_id)
        for child_id in children_map.get(node_id, []):
            if child_id in concept_defs:
                visit(child_id)

    visit(root_id)

    x_step = 340
    y_step = 120
    box_w = 240
    box_h = 54
    left = 60
    top = 100

    nodes: list[dict] = []
    positions: dict[str, tuple[int, int]] = {}
    level_ids = [str(level["id"]) for level in levels]

    for col, level_id in enumerate(level_ids):
        concept_map = level_concepts.get(level_id, {})
        concept_ids = list(concept_map.keys()) if isinstance(concept_map, dict) else list(concept_map)
        filtered_ids = [concept_id for concept_id in concept_ids if concept_id in keep]
        for row, concept_id in enumerate(filtered_ids):
            item = concept_defs[concept_id]
            x = left + col * x_step
            y = top + row * y_step
            positions[concept_id] = (x, y)
            nodes.append(
                {
                    "id": concept_id,
                    "name": display_names.get(concept_id, str(item.get("name", concept_id))),
                    "level": str(item.get("level", level_id)),
                    "scope": concept_map.get(concept_id, "instance") if isinstance(concept_map, dict) else "instance",
                    "description": str(item.get("description", "")),
                    "x": x,
                    "y": y,
                    "w": box_w,
                    "h": box_h,
                }
            )

    edges: list[tuple[str, str]] = []
    for parent_id in keep:
        for child_id in children_map.get(parent_id, []):
            if child_id in keep and parent_id in positions and child_id in positions:
                edges.append((parent_id, child_id))

    dataset_title = str(meta.get("dataset", "concept_ontology")).replace("_", " ").title()
    root_name = display_names.get(root_id, str(concept_defs[root_id].get("name", root_id)))
    return nodes, edges, f"{dataset_title} - {root_name}"


def render_svg(nodes: list[dict], edges: list[tuple[str, str]], title: str) -> str:
    if not nodes:
        raise ValueError("No ontology nodes found to render.")

    width = max(node["x"] + node["w"] for node in nodes) + 80
    height = max(node["y"] + node["h"] for node in nodes) + 80

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>',
        'text { font-family: "Segoe UI", Arial, sans-serif; fill: #1f2937; }',
        '.title { font-size: 24px; font-weight: 700; }',
        '.node-title { font-size: 15px; font-weight: 700; }',
        '.node-meta { font-size: 11px; fill: #475569; }',
        '.edge { stroke: #94a3b8; stroke-width: 2; fill: none; }',
        '.box { stroke: #475569; stroke-width: 1.4; rx: 12; ry: 12; }',
        "</style>",
        f'<text x="60" y="48" class="title">{html.escape(title)} Concept Ontology</text>',
    ]

    pos = {node["id"]: node for node in nodes}
    for parent_id, child_id in edges:
        parent = pos[parent_id]
        child = pos[child_id]
        x1 = parent["x"] + 12
        y1 = parent["y"] + parent["h"] / 2
        x2 = child["x"] + child["w"] - 12
        y2 = child["y"] + child["h"] / 2
        cx1 = x1 - 70
        cx2 = x2 + 70
        lines.append(
            f'<path class="edge" d="M {x1:.1f} {y1:.1f} C {cx1:.1f} {y1:.1f}, {cx2:.1f} {y2:.1f}, {x2:.1f} {y2:.1f}"/>'
        )

    for node in nodes:
        fill = LEVEL_COLORS.get(node["level"], "#E5E7EB")
        x = node["x"]
        y = node["y"]
        w = node["w"]
        h = node["h"]
        lines.append(f'<rect class="box" x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}"/>')
        lines.append(f'<text x="{x + 14}" y="{y + 22}" class="node-title">{html.escape(node["name"])}</text>')
        meta = f'{node["level"]} | {node["scope"]}'
        lines.append(f'<text x="{x + 14}" y="{y + 40}" class="node-meta">{html.escape(meta)}</text>')

    lines.append("</svg>")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    data = load_ontology(args.concept_yaml)
    display_names = load_morphology_mapping(args.morphology_mapping)
    nodes, edges, default_title = build_layout(data, display_names=display_names)
    title = args.title or default_title
    svg = render_svg(nodes, edges, title=title)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(svg, encoding="utf-8")
    print(f"Wrote ontology visualization to {out_path}")

    if args.class_outdir:
        class_outdir = Path(args.class_outdir)
        class_outdir.mkdir(parents=True, exist_ok=True)
        for item in data.get("concepts", []):
            concept_id = str(item["id"])
            if str(item.get("level")) != "L4":
                continue
            class_nodes, class_edges, class_title = build_subtree_layout(data, concept_id, display_names=display_names)
            class_svg = render_svg(class_nodes, class_edges, title=class_title)
            class_path = class_outdir / f"{concept_id}.svg"
            class_path.write_text(class_svg, encoding="utf-8")
            print(f"Wrote class subtree to {class_path}")


if __name__ == "__main__":
    main()
