from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

from .data import load_annotations
from .probe import load_probe
from .tcav import audit_single_detection_with_probes


def _load_probe_paths(manifest: str | None = None, probes: list[str] | None = None) -> list[str]:
    probe_paths: list[str] = []
    if manifest:
        data = json.loads(Path(manifest).read_text(encoding="utf-8"))
        probe_paths.extend([item["path"] for item in data.get("probes", [])])
    if probes:
        probe_paths.extend(probes)
    probe_paths = list(dict.fromkeys(probe_paths))
    if not probe_paths:
        raise ValueError("No probes provided. Use --manifest or --probes.")
    return probe_paths


def _resolve_split_images(data_yaml: str | Path, split: str) -> list[str]:
    data_yaml = Path(data_yaml)
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(str(data.get("path", "")))
    split_value = data.get(split)
    if split_value is None:
        raise ValueError(f"Split '{split}' not found in {data_yaml}")

    split_path = Path(str(split_value))
    if not split_path.is_absolute():
        split_path = root / split_path

    if split_path.suffix.lower() == ".txt":
        base_dir = split_path.parent
        lines = [line.strip() for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        images = []
        for line in lines:
            p = Path(line)
            if not p.is_absolute():
                p = (base_dir / p).resolve()
            images.append(str(p))
        return images

    if split_path.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        return [str(p.resolve()) for p in sorted(split_path.iterdir()) if p.suffix.lower() in exts]

    raise ValueError(f"Unsupported split path: {split_path}")


def analyze_dataset_with_probes(
    model_path: str,
    outdir: str,
    annotations_csv: str | None = None,
    data_yaml: str | None = None,
    split: str = "val",
    manifest: str | None = None,
    probes: list[str] | None = None,
    imgsz: int = 640,
    device: str = "",
    class_id: int | None = None,
    limit: int | None = None,
) -> dict:
    probe_paths = _load_probe_paths(manifest=manifest, probes=probes)
    probe_defs = [load_probe(p) for p in probe_paths]

    if annotations_csv is not None:
        rows = load_annotations(annotations_csv)
        image_paths = []
        seen = set()
        for row in rows:
            if row.image_path not in seen:
                seen.add(row.image_path)
                image_paths.append(row.image_path)
    elif data_yaml is not None:
        image_paths = _resolve_split_images(data_yaml, split)
    else:
        raise ValueError("Provide either annotations_csv or data_yaml.")

    if limit is not None:
        image_paths = image_paths[:limit]

    by_layer: dict[str, list[str]] = defaultdict(list)
    for path, probe in zip(probe_paths, probe_defs):
        by_layer[probe["layer"]].append(path)

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    image_results = []
    concept_scores: dict[str, list[dict]] = defaultdict(list)
    class_counter: dict[str, Counter] = defaultdict(Counter)
    by_class_concept_scores: dict[int, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))

    for image_path in image_paths:
        for layer, layer_probe_paths in by_layer.items():
            result = audit_single_detection_with_probes(
                model_path=model_path,
                image_path=image_path,
                probe_paths=layer_probe_paths,
                imgsz=imgsz,
                device=device,
                class_id=class_id,
                det_index=None,
            )
            image_results.append(result)
            det_class = int(result["target"]["class_id"])
            for item in result["concept_results"]:
                concept_scores[item["concept"]].append(item)
                class_counter[item["concept"]][det_class] += 1
                by_class_concept_scores[det_class][item["concept"]].append(item)

    summary = {"images": len(image_paths), "probes": len(probe_paths), "split": split, "concepts": [], "by_class": []}
    for concept, items in sorted(concept_scores.items()):
        mean_prob = sum(x["probe_prob"] for x in items) / len(items)
        mean_tcav = sum(x["tcav_score"] for x in items) / len(items)
        pos_tcav_ratio = sum(1 for x in items if x["tcav_score"] > 0) / len(items)
        summary["concepts"].append(
            {
                "concept": concept,
                "count": len(items),
                "mean_probe_prob": mean_prob,
                "mean_tcav_score": mean_tcav,
                "positive_tcav_ratio": pos_tcav_ratio,
                "top_classes": class_counter[concept].most_common(5),
            }
        )

    for class_id_key in sorted(by_class_concept_scores):
        class_items = []
        for concept, items in sorted(by_class_concept_scores[class_id_key].items()):
            mean_prob = sum(x["probe_prob"] for x in items) / len(items)
            mean_tcav = sum(x["tcav_score"] for x in items) / len(items)
            pos_tcav_ratio = sum(1 for x in items if x["tcav_score"] > 0) / len(items)
            class_items.append(
                {
                    "concept": concept,
                    "count": len(items),
                    "mean_probe_prob": mean_prob,
                    "mean_tcav_score": mean_tcav,
                    "positive_tcav_ratio": pos_tcav_ratio,
                }
            )
        summary["by_class"].append({"class_id": class_id_key, "concepts": class_items})

    (outdir / "dataset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (outdir / "dataset_results.json").write_text(json.dumps(image_results, ensure_ascii=False, indent=2), encoding="utf-8")

    with (outdir / "concept_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["concept", "count", "mean_probe_prob", "mean_tcav_score", "positive_tcav_ratio", "top_classes"])
        for item in summary["concepts"]:
            writer.writerow(
                [
                    item["concept"],
                    item["count"],
                    f"{item['mean_probe_prob']:.6f}",
                    f"{item['mean_tcav_score']:.6f}",
                    f"{item['positive_tcav_ratio']:.6f}",
                    json.dumps(item["top_classes"], ensure_ascii=False),
                ]
            )

    with (outdir / "concept_summary_by_class.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "concept", "count", "mean_probe_prob", "mean_tcav_score", "positive_tcav_ratio"])
        for class_entry in summary["by_class"]:
            class_id_key = class_entry["class_id"]
            for item in class_entry["concepts"]:
                writer.writerow(
                    [
                        class_id_key,
                        item["concept"],
                        item["count"],
                        f"{item['mean_probe_prob']:.6f}",
                        f"{item['mean_tcav_score']:.6f}",
                        f"{item['positive_tcav_ratio']:.6f}",
                    ]
                )

    lines = ["# Dataset-Level Supervised TCAV Summary", ""]
    lines.append(f"- images: `{summary['images']}`")
    lines.append(f"- probes: `{summary['probes']}`")
    lines.append(f"- split: `{summary['split']}`")
    lines.append("")
    lines.append("| concept | count | mean_probe_prob | mean_tcav_score | positive_tcav_ratio |")
    lines.append("|---|---:|---:|---:|---:|")
    for item in summary["concepts"]:
        lines.append(
            f"| {item['concept']} | {item['count']} | {item['mean_probe_prob']:.4f} | {item['mean_tcav_score']:.4f} | {item['positive_tcav_ratio']:.4f} |"
        )
    lines.append("")
    lines.append("## By Class")
    lines.append("")
    for class_entry in summary["by_class"]:
        lines.append(f"### class_id {class_entry['class_id']}")
        lines.append("")
        lines.append("| concept | count | mean_probe_prob | mean_tcav_score | positive_tcav_ratio |")
        lines.append("|---|---:|---:|---:|---:|")
        for item in class_entry["concepts"]:
            lines.append(
                f"| {item['concept']} | {item['count']} | {item['mean_probe_prob']:.4f} | {item['mean_tcav_score']:.4f} | {item['positive_tcav_ratio']:.4f} |"
            )
        lines.append("")
    (outdir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return summary
