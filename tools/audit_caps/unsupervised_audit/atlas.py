from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from .dataset import resolve_split_paths
from .io import load_json
from .io import save_json
from .labels import apply_label_metadata, concept_sort_key, load_concept_labels
from .single import run_single_audit
from .visualize import render_atlas_overview, render_concept_reference_samples


def _save_crop(image_path: str, crop_box: list[int], out_path: str | Path) -> str | None:
    try:
        image = Image.open(image_path).convert("RGB")
    except OSError:
        return None
    x1, y1, x2, y2 = crop_box
    if x2 <= x1 or y2 <= y1:
        return None
    patch = image.crop((x1, y1, x2, y2))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    patch.save(out_path)
    return str(out_path)


def _reference_feature(crop_path: str | None) -> np.ndarray | None:
    if not crop_path or not Path(crop_path).exists():
        return None
    try:
        patch = Image.open(crop_path).convert("RGB").resize((32, 32), Image.BILINEAR)
    except OSError:
        return None
    arr = np.asarray(patch, dtype=np.float32) / 255.0
    mean_rgb = arr.mean(axis=(0, 1))
    std_rgb = arr.std(axis=(0, 1))
    gray = arr.mean(axis=2)
    grad_x = np.abs(np.diff(gray, axis=1)).mean()
    grad_y = np.abs(np.diff(gray, axis=0)).mean()
    feat = np.concatenate([mean_rgb, std_rgb, np.asarray([grad_x, grad_y], dtype=np.float32)], axis=0)
    norm = float(np.linalg.norm(feat))
    if norm > 0:
        feat = feat / norm
    return feat.astype(np.float32)


def _select_references_crp(rows: list[dict], max_refs: int = 12, candidate_pool: int = 32, similarity_threshold: float = 0.96) -> list[dict]:
    if not rows:
        return []
    ranked = sorted(rows, key=lambda item: float(item["score"]), reverse=True)[:candidate_pool]
    selected: list[dict] = []
    selected_feats: list[np.ndarray] = []
    for row in ranked:
        feat = _reference_feature(row.get("crop_path"))
        if feat is None:
            if len(selected) < max_refs:
                selected.append(row)
            continue
        if selected_feats:
            similarities = [float(np.dot(feat, prev)) for prev in selected_feats]
            if max(similarities) >= float(similarity_threshold):
                continue
        row = dict(row)
        row["reference_score"] = float(row["score"])
        selected.append(row)
        selected_feats.append(feat)
        if len(selected) >= max_refs:
            break
    if len(selected) < min(max_refs, len(ranked)):
        chosen = {str(item.get("crop_path")) + "::" + item["image_path"] for item in selected}
        for row in ranked:
            key = str(row.get("crop_path")) + "::" + row["image_path"]
            if key in chosen:
                continue
            row = dict(row)
            row["reference_score"] = float(row["score"])
            selected.append(row)
            if len(selected) >= max_refs:
                break
    return selected[:max_refs]


def build_concept_atlas(
    model_path: str,
    data_yaml: str,
    outdir: str | Path,
    split: str = "val",
    imgsz: int = 640,
    limit: int = 100,
    topk_per_image: int = 3,
    device: str = "",
    attribution_method: str = "capsule_lcrp",
    layers: list[str] | None = None,
) -> dict:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    image_paths = resolve_split_paths(data_yaml, split=split)[:limit]
    samples_dir = outdir / "samples"
    refs_dir = outdir / "references"
    concept_rows: dict[str, list[dict]] = defaultdict(list)
    concept_to_classes: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    adjacency: dict[tuple[str, str, str], dict[str, float]] = defaultdict(lambda: {"weight": 0.0, "count": 0.0})
    sample_methods: set[str] = set()
    attribution_styles: set[str] = set()

    for idx, image_path in enumerate(image_paths):
        sample_outdir = samples_dir / f"{idx:06d}"
        sample = run_single_audit(
            model_path=model_path,
            image_path=image_path,
            outdir=sample_outdir,
            imgsz=imgsz,
            device=device,
            topk=topk_per_image,
            attribution_method=attribution_method,
            layers=layers,
        )
        sample_methods.add(str(sample.method))
        attribution_style = sample.metadata.get("attribution_style") if isinstance(sample.metadata, dict) else None
        if attribution_style:
            attribution_styles.add(str(attribution_style))
        ordered_nodes = sorted(sample.nodes, key=lambda item: (str(item.layer), -float(item.score)))
        by_layer: dict[str, list] = defaultdict(list)
        for node in ordered_nodes:
            by_layer[str(node.layer)].append(node)
            if node.heatmap and node.heatmap.crop_box_image:
                crop_path = _save_crop(
                    sample.image_path,
                    node.heatmap.crop_box_image,
                    refs_dir / node.id.replace(":", "_") / f"{idx:06d}.png",
                )
            else:
                crop_path = None
            concept_rows[node.id].append(
                {
                    "concept_id": node.id,
                    "layer": node.layer,
                    "type_idx": node.type_idx,
                    "image_path": sample.image_path,
                    "score": float(node.score),
                    "crop_path": crop_path,
                    "sample_overlay_path": None if not node.heatmap else node.heatmap.sample_overlay_path,
                    "bbox_xyxy_image": sample.target.bbox_xywh_image,
                    "class_id": sample.target.class_id,
                }
            )
            concept_to_classes[node.id][str(sample.target.class_id)] += float(node.score)
        if getattr(sample, "edges", None):
            for edge in sample.edges:
                key = (str(edge.source), str(edge.target), str(edge.kind))
                adjacency[key]["weight"] += float(edge.weight)
                adjacency[key]["count"] += 1.0
        else:
            layers = list(by_layer.keys())
            for low_layer, high_layer in zip(layers[:-1], layers[1:]):
                for low_node in by_layer[low_layer]:
                    for high_node in by_layer[high_layer]:
                        key = (low_node.id, high_node.id, "atlas_concept_to_concept")
                        adjacency[key]["weight"] += min(float(low_node.score), float(high_node.score))
                        adjacency[key]["count"] += 1.0

    labels = load_concept_labels(outdir / "concept_labels.csv")
    concepts: list[dict] = []
    reference_samples_dir = outdir / "reference_samples"
    for concept_id, rows in concept_rows.items():
        rows.sort(key=lambda item: float(item["score"]), reverse=True)
        references = _select_references_crp(rows, max_refs=12, candidate_pool=32, similarity_threshold=0.96)
        top_classes = sorted(concept_to_classes[concept_id].items(), key=lambda item: item[1], reverse=True)
        concept_ref_figure = render_concept_reference_samples(
            {"concept_id": concept_id, "references": references},
            reference_samples_dir / f"{concept_id.replace(':', '_')}.png",
        )
        concept = {
            "concept_id": concept_id,
            "layer": rows[0]["layer"],
            "type_idx": rows[0]["type_idx"],
            "support": float(sum(float(row["score"]) for row in rows)),
            "num_refs": len(rows),
            "references": references,
            "reference_samples_figure": str(concept_ref_figure),
            "top_classes": [{"class_id": int(cid), "weight": float(weight)} for cid, weight in top_classes[:5]],
            "reference_selection": {
                "strategy": "relevance_maximization_reference_selection",
                "candidate_pool": min(32, len(rows)),
                "max_refs": 12,
                "similarity_threshold": 0.96,
            },
        }
        concept = apply_label_metadata(concept, labels)
        concepts.append(concept)
    concepts.sort(key=concept_sort_key)

    edge_rows = [
        {
            "source": source,
            "target": target,
            "kind": kind,
            "weight": float(meta["weight"]),
            "count": int(meta["count"]),
        }
        for (source, target, kind), meta in adjacency.items()
    ]
    atlas = {
        "method": "capsule_lcrp_atlas_v3",
        "model_path": model_path,
        "data_yaml": data_yaml,
        "split": split,
        "imgsz": int(imgsz),
        "num_images": len(image_paths),
        "topk_per_image": int(topk_per_image),
        "concepts": concepts,
        "edges": edge_rows,
        "metadata": {
            "attribution_method": attribution_method,
            "target_layers": [] if not layers else [str(layer) for layer in layers],
            "sample_methods": sorted(sample_methods),
            "attribution_styles": sorted(attribution_styles),
            "reference_strategy": "relevance_maximization_reference_selection",
        },
    }
    save_json(atlas, outdir / "concept_atlas.json")
    render_atlas_overview(concepts, outdir / "concept_atlas_overview.png")
    return atlas


def relabel_concept_atlas(
    atlas_json: str | Path,
    labels_csv: str | Path,
    outdir: str | Path | None = None,
    refresh_visuals: bool = True,
) -> dict:
    atlas_path = Path(atlas_json)
    atlas = load_json(atlas_path)
    labels = load_concept_labels(labels_csv)
    concepts = [apply_label_metadata(dict(concept), labels) for concept in atlas.get("concepts", [])]
    concepts.sort(key=concept_sort_key)
    atlas["concepts"] = concepts

    target_dir = Path(outdir) if outdir is not None else atlas_path.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    metadata = atlas.setdefault("metadata", {})
    metadata["labels_csv"] = str(labels_csv)
    save_json(atlas, target_dir / "concept_atlas.json")

    if refresh_visuals:
        render_atlas_overview(concepts, target_dir / "concept_atlas_overview.png")
        reference_samples_dir = target_dir / "reference_samples"
        for concept in concepts:
            concept_ref_figure = render_concept_reference_samples(
                concept,
                reference_samples_dir / f"{str(concept.get('concept_id', '')).replace(':', '_')}.png",
            )
            concept["reference_samples_figure"] = str(concept_ref_figure)
        save_json(atlas, target_dir / "concept_atlas.json")
    return atlas
