from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw


def save_supervised_report(result: dict, outdir: str | Path) -> None:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    (outdir / "tcav_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    lines: list[str] = []
    target = result["target"]
    lines.append("# Supervised TCAV Audit")
    lines.append("")
    lines.append(f"- image: `{result['image_path']}`")
    lines.append(f"- layer: `{result['layer']}`")
    lines.append(f"- det_index: `{target['det_index']}`")
    lines.append(f"- class_id: `{target['class_id']}`")
    lines.append(f"- score: `{target['score']:.6f}`")
    lines.append(f"- bbox_xywh: `{target['bbox_xywh']}`")
    lines.append("")
    lines.append("## Concepts")
    lines.append("")
    lines.append("| concept | probe_prob | tcav_score | probe_logit |")
    lines.append("|---|---:|---:|---:|")
    for item in result["concept_results"]:
        lines.append(
            f"| {item['concept']} | {item['probe_prob']:.4f} | {item['tcav_score']:.4f} | {item['probe_logit']:.4f} |"
        )
    lines.append("")
    lines.append("Interpretation:")
    lines.append("- `probe_prob`: the supervised probe's estimate that the concept is present in the selected detection feature.")
    lines.append("- `tcav_score`: directional sensitivity of the selected detection score to the concept direction.")
    lines.append("- larger positive `tcav_score` means the concept supports the detection more strongly.")

    (outdir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def save_supervised_overlay(result: dict, outdir: str | Path, topk: int = 5) -> None:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    image = Image.open(result["image_path"]).convert("RGB")
    draw = ImageDraw.Draw(image)
    cx, cy, bw, bh = result["target"]["bbox_xywh"]
    x1 = cx - bw / 2.0
    y1 = cy - bh / 2.0
    x2 = cx + bw / 2.0
    y2 = cy + bh / 2.0
    draw.rectangle((x1, y1, x2, y2), outline=(255, 0, 0), width=3)

    text_lines = [f"cls={result['target']['class_id']} score={result['target']['score']:.3f}"]
    for item in result["concept_results"][:topk]:
        text_lines.append(f"{item['concept']}: p={item['probe_prob']:.2f}, tcav={item['tcav_score']:.2f}")

    tx = max(4, int(x1))
    ty = max(4, int(y1) - 16 * len(text_lines) - 6)
    for idx, line in enumerate(text_lines):
        y = ty + idx * 16
        draw.rectangle((tx - 2, y - 1, tx + max(120, len(line) * 7), y + 13), fill=(255, 255, 255))
        draw.text((tx, y), line, fill=(0, 0, 0))

    image.save(outdir / "overlay.png")
