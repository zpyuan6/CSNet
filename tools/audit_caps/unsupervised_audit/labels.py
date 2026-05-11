from __future__ import annotations

import csv
import re
from pathlib import Path


def concept_sort_key(concept: dict) -> tuple[int, int, str]:
    layer = str(concept.get("layer") or "")
    type_idx = concept.get("type_idx")
    concept_id = str(concept.get("concept_id") or concept.get("id") or "")
    if type_idx is None:
        match = re.search(r"type(\d+)", concept_id)
        type_idx = int(match.group(1)) if match else 10**9
    layer_match = re.search(r"(\d+)", layer)
    layer_idx = int(layer_match.group(1)) if layer_match else 10**9
    return layer_idx, int(type_idx), concept_id


def load_concept_labels(path: str | Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    csv_path = Path(path)
    if not csv_path.exists():
        return {}
    rows: dict[str, dict[str, str]] = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            concept_id = (row.get("concept_id") or "").strip()
            if concept_id:
                rows[concept_id] = row
    return rows


def apply_label_metadata(node_dict: dict, labels: dict[str, dict[str, str]]) -> dict:
    concept_id = str(node_dict.get("concept_id") or node_dict.get("id") or "")
    row = labels.get(concept_id)
    if not row:
        return node_dict
    node_dict = dict(node_dict)
    node_dict["concept_name"] = row.get("concept_name") or node_dict.get("concept_name")
    node_dict["semantic_group"] = row.get("semantic_group") or node_dict.get("semantic_group")
    node_dict["confidence"] = row.get("confidence") or node_dict.get("confidence")
    return node_dict


def export_concept_labels_csv(atlas_json: str | Path, out_csv: str | Path) -> Path:
    import json

    atlas_path = Path(atlas_json)
    out_path = Path(out_csv)
    data = json.loads(atlas_path.read_text(encoding="utf-8"))
    concepts = sorted(data.get("concepts", []), key=concept_sort_key)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["concept_id", "concept_name", "semantic_group", "confidence", "notes"]
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for concept in concepts:
            writer.writerow(
                {
                    "concept_id": concept.get("concept_id", ""),
                    "concept_name": concept.get("concept_name", ""),
                    "semantic_group": concept.get("semantic_group", ""),
                    "confidence": concept.get("confidence", ""),
                    "notes": concept.get("notes", ""),
                }
            )
    return out_path
