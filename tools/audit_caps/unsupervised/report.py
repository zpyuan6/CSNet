from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image


def _overlay_heatmap(image: Image.Image, heatmap: list[list[float]]) -> Image.Image:
    base = image.convert("RGBA")
    heat = np.asarray(heatmap, dtype=np.float32)
    heat = heat - heat.min()
    if heat.max() > 0:
        heat = heat / heat.max()
    heat = Image.fromarray(np.uint8(heat * 255), mode="L").resize(base.size, Image.BILINEAR)
    rgba = Image.new("RGBA", base.size, (255, 0, 0, 0))
    rgba.putalpha(heat)
    return Image.blend(base, rgba, 0.35)


def save_audit_report(record: dict, outdir: str | Path, topk: int = 3) -> None:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    image = Image.open(record["image_path"]).convert("RGB")
    (outdir / "audit.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        f"image: {record['image_path']}",
        f"class_id: {record['target']['class_id']}",
        f"score: {record['target']['score']:.4f}",
        f"bbox_xywh: {record['target']['bbox_xywh']}",
        "",
    ]

    for layer_item in record.get("concepts_by_layer", []):
        lines.append(f"[{layer_item['layer']}]")
        for idx, concept in enumerate(layer_item.get("concepts", [])[:topk]):
            lines.append(f"type={concept['type_idx']} score={concept['score']:.4f} peak={tuple(concept['peak_xy'])}")
            overlay = _overlay_heatmap(image, concept["heatmap"])
            overlay.save(outdir / f"{layer_item['layer'].replace('.', '_')}_type{concept['type_idx']}_{idx}.png")
        lines.append("")

    (outdir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
