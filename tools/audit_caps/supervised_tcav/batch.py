from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .data import load_annotations
from .probe import train_concept_probe


def train_all_concepts(
    model_path: str,
    annotations_csv: str,
    outdir: str,
    default_layer: str | None = None,
    imgsz: int = 640,
    device: str = "",
    epochs: int = 200,
    lr: float = 1e-2,
    weight_decay: float = 1e-4,
) -> dict:
    rows = load_annotations(annotations_csv)
    grouped: dict[tuple[str, str], list] = defaultdict(list)
    for row in rows:
        layer = row.layer or default_layer
        if layer is None:
            raise ValueError("Each annotation row needs a layer, or provide --default-layer.")
        grouped[(row.concept, layer)].append(row)

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    manifest = {"model": model_path, "imgsz": imgsz, "probes": []}
    for (concept, layer), _items in sorted(grouped.items()):
        safe_name = f"{concept}__{layer.replace('.', '_')}.json"
        out_path = outdir / safe_name
        result = train_concept_probe(
            model_path=model_path,
            annotations_csv=annotations_csv,
            concept=concept,
            layer=layer,
            out_path=str(out_path),
            imgsz=imgsz,
            device=device,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
        )
        manifest["probes"].append(
            {
                "concept": concept,
                "layer": layer,
                "path": str(out_path),
                "samples": result["samples"],
                "train_loss": result["train_loss"],
                "train_acc": result["train_acc"],
            }
        )

    (outdir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest
