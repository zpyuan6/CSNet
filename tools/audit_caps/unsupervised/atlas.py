from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import yaml
from PIL import Image

from .relevance import audit_single_image


def load_dataset_yaml(data_yaml: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8"))


def resolve_split_paths(data_yaml: str | Path, split: str = "val") -> list[Path]:
    cfg = load_dataset_yaml(data_yaml)
    root = Path(cfg["path"])
    entry = cfg[split]

    if isinstance(entry, list):
        items = []
        for value in entry:
            items.extend(resolve_entry(root, value))
        return items
    return resolve_entry(root, entry)


def resolve_entry(root: Path, entry: str) -> list[Path]:
    path = Path(entry)
    path = path if path.is_absolute() else root / path
    if path.suffix.lower() == ".txt":
        lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return [Path(line) if Path(line).is_absolute() else root / line for line in lines]
    if path.is_dir():
        return sorted([p for p in path.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])
    if path.is_file():
        return [path]
    raise FileNotFoundError(f"Unsupported split entry: {entry}")


def iter_top_concepts(record: dict[str, Any]) -> Iterable[tuple[str, dict[str, Any]]]:
    for layer_item in record.get("concepts_by_layer", []):
        layer_name = layer_item["layer"]
        spatial_w, spatial_h = layer_item["spatial_size"]
        image_w, image_h = record["image_size"]
        sx = image_w / max(spatial_w, 1)
        sy = image_h / max(spatial_h, 1)
        for concept in layer_item["concepts"]:
            peak_x, peak_y = concept["peak_xy"]
            yield (
                f"{layer_name}:type{concept['type_idx']}",
                {
                    **concept,
                    "layer": layer_name,
                    "crop_box": [
                        max(int((peak_x - 2) * sx), 0),
                        max(int((peak_y - 2) * sy), 0),
                        min(int((peak_x + 3) * sx), image_w),
                        min(int((peak_y + 3) * sy), image_h),
                    ],
                },
            )


def build_concept_atlas(
    model_path: str,
    data_yaml: str,
    outdir: str,
    split: str = "val",
    imgsz: int = 640,
    limit: int = 100,
    topk_per_image: int = 3,
    device: str = "",
) -> dict[str, Any]:
    out_root = Path(outdir)
    atlas_dir = out_root / "atlas"
    atlas_dir.mkdir(parents=True, exist_ok=True)

    concept_bank: dict[str, list[dict[str, Any]]] = defaultdict(list)
    image_paths = resolve_split_paths(data_yaml, split=split)[:limit]

    for image_path in image_paths:
        record = audit_single_image(model_path, str(image_path), imgsz=imgsz, device=device, topk=topk_per_image)
        source = Image.open(image_path).convert("RGB")
        for concept_key, concept in iter_top_concepts(record):
            crop = source.crop(tuple(concept["crop_box"]))
            concept_dir = atlas_dir / concept_key.replace(":", "_")
            concept_dir.mkdir(parents=True, exist_ok=True)
            rank = len(concept_bank[concept_key])
            crop_name = f"{rank:04d}.png"
            crop.save(concept_dir / crop_name)
            concept_bank[concept_key].append(
                {
                    "image_path": str(image_path),
                    "score": float(concept["score"]),
                    "class_id": int(record["target"]["class_id"]),
                    "crop_path": str((concept_dir / crop_name).resolve()),
                    "peak_xy": concept["peak_xy"],
                    "crop_box": concept["crop_box"],
                }
            )

    summary = {
        "model_path": model_path,
        "data_yaml": data_yaml,
        "split": split,
        "imgsz": imgsz,
        "num_images": len(image_paths),
        "concepts": {key: sorted(items, key=lambda item: item["score"], reverse=True) for key, items in concept_bank.items()},
    }
    (out_root / "concept_atlas.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
