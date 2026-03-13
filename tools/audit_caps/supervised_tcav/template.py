from __future__ import annotations

import csv
from pathlib import Path


def export_concept_annotation_template(out_csv: str | Path) -> None:
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "concept", "label", "layer", "x1", "y1", "x2", "y2", "notes"])
        writer.writerow(
            [
                r"C:\Datasets\coco-yolo\images\val2017\000000000139.jpg",
                "wheel",
                1,
                "model.15",
                250,
                280,
                360,
                410,
                "box-level concept annotation example",
            ]
        )
        writer.writerow(
            [
                r"C:\Datasets\coco-yolo\images\val2017\000000000139.jpg",
                "striped",
                0,
                "model.15",
                "",
                "",
                "",
                "",
                "image-level concept annotation example",
            ]
        )
