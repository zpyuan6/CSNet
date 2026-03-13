from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ConceptAnnotation:
    image_path: str
    concept: str
    label: float
    x1: float | None = None
    y1: float | None = None
    x2: float | None = None
    y2: float | None = None
    layer: str | None = None

    @property
    def has_box(self) -> bool:
        return None not in (self.x1, self.y1, self.x2, self.y2)


def load_annotations(csv_path: str | Path, concept: str | None = None, layer: str | None = None) -> list[ConceptAnnotation]:
    csv_path = Path(csv_path)
    rows: list[ConceptAnnotation] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"image", "concept", "label"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required annotation columns: {sorted(missing)}")

        for raw in reader:
            row_concept = str(raw["concept"]).strip()
            row_layer = str(raw.get("layer", "")).strip() or None
            if concept is not None and row_concept != concept:
                continue
            if layer is not None and row_layer not in (None, layer):
                continue

            image_value = str(raw["image"]).strip()
            image_path = Path(image_value)
            if not image_path.is_absolute():
                image_path = (csv_path.parent / image_path).resolve()

            def parse_opt(name: str) -> float | None:
                value = str(raw.get(name, "")).strip()
                return None if value == "" else float(value)

            rows.append(
                ConceptAnnotation(
                    image_path=str(image_path),
                    concept=row_concept,
                    label=float(raw["label"]),
                    x1=parse_opt("x1"),
                    y1=parse_opt("y1"),
                    x2=parse_opt("x2"),
                    y2=parse_opt("y2"),
                    layer=row_layer,
                )
            )

    if not rows:
        raise ValueError("No annotations matched the requested concept/layer.")
    return rows
