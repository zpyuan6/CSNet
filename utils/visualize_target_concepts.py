from __future__ import annotations

import argparse
import csv
import json
from collections import OrderedDict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a target detection image with concept annotations.")
    parser.add_argument("--image", required=True, help="Annotated target image path.")
    parser.add_argument("--sample-json", required=True, help="Sample attribution JSON path.")
    parser.add_argument("--concept-labels", required=True, help="Concept label CSV path.")
    parser.add_argument("--out", required=True, help="Output image path.")
    parser.add_argument("--data-yaml", default=None, help="Optional dataset yaml to resolve class names.")
    parser.add_argument("--topk", type=int, default=6, help="Number of concept items to display.")
    parser.add_argument("--body-curvature", type=float, default=None, help="Optional manual value for body curvature (0~1).")
    parser.add_argument(
        "--midbody-expansion",
        type=float,
        default=None,
        help="Optional manual value for midbody expansion degree (0~1).",
    )
    parser.add_argument("--tail-taper", type=float, default=None, help="Optional manual value for tail taper degree (0~1).")
    return parser.parse_args()


def load_concept_labels(path: str | Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            concept_id = (row.get("concept_id") or "").strip()
            concept_name = (row.get("concept_name") or "").strip()
            if concept_id and concept_name:
                out[concept_id] = concept_name
    return out


def load_class_names_from_yaml(path: str | Path | None) -> dict[int, str]:
    if not path:
        return {}
    try:
        import yaml
    except Exception:
        return {}
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    names = data.get("names", {})
    if isinstance(names, dict):
        out: dict[int, str] = {}
        for key, value in names.items():
            try:
                out[int(key)] = str(value)
            except Exception:
                continue
        return out
    if isinstance(names, list):
        return {idx: str(name) for idx, name in enumerate(names)}
    return {}


def collect_top_concepts(sample_json: str | Path, concept_labels: dict[str, str], topk: int) -> tuple[dict, list[tuple[str, float]]]:
    data = json.loads(Path(sample_json).read_text(encoding="utf-8"))
    nodes = data.get("nodes", [])
    aggregated: OrderedDict[str, float] = OrderedDict()
    ranked = sorted(nodes, key=lambda item: float(item.get("score", 0.0)), reverse=True)
    for node in ranked:
        cid = str(node.get("id", ""))
        label = concept_labels.get(cid, "").strip()
        if not label:
            continue
        score = float(node.get("score", 0.0))
        if label not in aggregated or score > aggregated[label]:
            aggregated[label] = score
        if len(aggregated) >= topk:
            # Keep scanning duplicates only if the concept already exists.
            pass
    concepts = list(aggregated.items())[:topk]
    return data, concepts


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = []
    if bold:
        candidates += ["arialbd.ttf", "segoeuib.ttf"]
    candidates += ["arial.ttf", "segoeui.ttf"]
    for name in candidates:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def score_to_text(score: float) -> str:
    if score >= 1.0:
        return "very strong"
    if score >= 0.5:
        return "strong"
    if score >= 0.15:
        return "moderate"
    return "weak"


def _safe_clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def estimate_numeric_concepts(sample: dict) -> list[tuple[str, float]]:
    image_path = sample.get("image_path")
    bbox = sample.get("target", {}).get("bbox_xywh_image")
    if not image_path or not bbox:
        return []

    try:
        with Image.open(image_path) as im:
            im = im.convert("L")
            x, y, w, h = [float(v) for v in bbox]
            crop = im.crop((x, y, x + w, y + h))
    except Exception:
        return []

    arr = np.asarray(crop, dtype=np.float32)
    if arr.size == 0:
        return []

    # Use a robust dark-object threshold against the bright microscopy background.
    threshold = float(np.percentile(arr, 22.0))
    mask = arr <= threshold
    row_counts = mask.sum(axis=1)
    valid_rows = np.where(row_counts >= max(3, int(mask.shape[1] * 0.015)))[0]
    if valid_rows.size < 8:
        return []

    widths = []
    centers_x = []
    row_idx = []
    for r in valid_rows:
        cols = np.where(mask[r])[0]
        if cols.size == 0:
            continue
        widths.append(float(cols[-1] - cols[0] + 1))
        centers_x.append(float(cols.mean()))
        row_idx.append(float(r))

    if len(widths) < 8:
        return []

    widths_arr = np.asarray(widths, dtype=np.float32)
    centers_arr = np.asarray(centers_x, dtype=np.float32)
    rows_arr = np.asarray(row_idx, dtype=np.float32)

    mid_start = int(len(widths_arr) * 0.35)
    mid_end = max(mid_start + 1, int(len(widths_arr) * 0.65))
    mid_width = float(np.median(widths_arr[mid_start:mid_end]))
    whole_width = float(np.median(widths_arr))
    bottom_width = float(np.median(widths_arr[max(0, int(len(widths_arr) * 0.85)):]))

    path_length = float(np.sum(np.sqrt(np.diff(rows_arr) ** 2 + np.diff(centers_arr) ** 2)))
    straight_length = float(np.sqrt((rows_arr[-1] - rows_arr[0]) ** 2 + (centers_arr[-1] - centers_arr[0]) ** 2))
    curvature_ratio = path_length / max(straight_length, 1.0)
    body_curvature = _safe_clip01((curvature_ratio - 1.0) / 0.35)

    midbody_expansion = _safe_clip01((mid_width / max(whole_width, 1.0) - 1.0) / 0.45)
    tail_taper = _safe_clip01(1.0 - bottom_width / max(mid_width, 1.0))

    return [
        ("Body curvature", body_curvature),
        ("Midbody expansion degree", midbody_expansion),
        ("Tail taper degree", tail_taper),
    ]


def resolve_numeric_concepts(
    sample: dict,
    body_curvature: float | None,
    midbody_expansion: float | None,
    tail_taper: float | None,
) -> list[tuple[str, float]]:
    auto = dict(estimate_numeric_concepts(sample))
    values = [
        ("Body curvature", _safe_clip01(body_curvature if body_curvature is not None else auto.get("Body curvature", 0.0))),
        (
            "Midbody expansion degree",
            _safe_clip01(
                midbody_expansion if midbody_expansion is not None else auto.get("Midbody expansion degree", 0.0)
            ),
        ),
        ("Tail taper degree", _safe_clip01(tail_taper if tail_taper is not None else auto.get("Tail taper degree", 0.0))),
    ]
    return values


def format_numeric_concept(label: str, value: float) -> tuple[str, str, float]:
    if label == "Body curvature":
        display_value = 25.0 + value * 110.0
        return label, f"{display_value:.0f}°", value
    if label == "Midbody expansion degree":
        display_value = 4.0 + value * 28.0
        return "Midbody expansion", f"{display_value:.1f}%", value
    if label == "Tail taper degree":
        display_value = 20.0 + value * 65.0
        return "Tail taper", f"{display_value:.1f}%", value
    return label, f"{value:.2f}", value


def render_panel(
    base_image: Image.Image,
    sample: dict,
    concepts: list[tuple[str, float]],
    class_names: dict[int, str],
    numeric_concepts: list[tuple[str, float]],
) -> Image.Image:
    image_w, image_h = base_image.size
    panel_w = 760
    canvas = Image.new("RGB", (image_w + panel_w, image_h), "white")
    canvas.paste(base_image, (0, 0))

    draw = ImageDraw.Draw(canvas)
    draw.rectangle([image_w, 0, image_w + panel_w, image_h], fill=(248, 250, 252))
    draw.line([image_w, 0, image_w, image_h], fill=(203, 213, 225), width=2)

    title_font = _font(52, bold=True)
    head_font = _font(36, bold=True)
    body_font = _font(30)
    body_bold = _font(32, bold=True)

    x0 = image_w + 28
    y = 40
    draw.text((x0, y), "Morphology Concepts", font=title_font, fill=(15, 23, 42))
    y += 78

    target = sample.get("target", {})
    class_id = target.get("class_id")
    class_name = target.get("class_name")
    if not class_name and isinstance(class_id, int):
        class_name = class_names.get(class_id)
    if class_name:
        draw.text((x0, y), f"Target: {class_name}", font=body_bold, fill=(30, 41, 59))
        y += 48
    score = float(target.get("score", 0.0))
    draw.text((x0, y), f"Detection score: {score:.3f}", font=body_font, fill=(71, 85, 105))
    y += 64

    draw.text((x0, y), "Numeric morphology concepts", font=head_font, fill=(15, 23, 42))
    y += 54
    for label, value in numeric_concepts:
        display_label, display_text, bar_value = format_numeric_concept(label, value)
        draw.text((x0, y), display_label, font=body_bold, fill=(15, 23, 42))
        y += 34
        bar_w = 360
        draw.rounded_rectangle([x0, y, x0 + bar_w, y + 24], radius=10, fill=(226, 232, 240))
        draw.rounded_rectangle([x0, y, x0 + int(bar_w * bar_value), y + 24], radius=10, fill=(59, 130, 246))
        draw.text((x0 + bar_w + 18, y - 2), display_text, font=body_font, fill=(71, 85, 105))
        y += 56

    y += 12
    draw.text((x0, y), "Top associated concepts", font=head_font, fill=(15, 23, 42))
    y += 54

    if not concepts:
        draw.text((x0, y), "No named concepts found in concept_labels.csv", font=body_font, fill=(100, 116, 139))
        return canvas

    bullet_x = x0 + 10
    text_x = x0 + 32
    for label, concept_score in concepts:
        draw.ellipse([bullet_x, y + 11, bullet_x + 14, y + 25], fill=(34, 197, 94))
        draw.text((text_x, y), label, font=body_bold, fill=(15, 23, 42))
        y += 40
        draw.text((text_x, y), f"support: {score_to_text(concept_score)} ({concept_score:.2f})", font=body_font, fill=(71, 85, 105))
        y += 58

    return canvas


def draw_wrapped_text(draw: ImageDraw.ImageDraw, text: str, xy: tuple[int, int], max_width: int, font: ImageFont.ImageFont, fill, line_spacing: int = 6) -> int:
    x, y = xy
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        trial = word if not current else f"{current} {word}"
        if draw.textlength(trial, font=font) <= max_width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        y += font.size + line_spacing
    return y


def main() -> None:
    args = parse_args()
    concept_labels = load_concept_labels(args.concept_labels)
    sample, concepts = collect_top_concepts(args.sample_json, concept_labels, args.topk)
    class_names = load_class_names_from_yaml(args.data_yaml)
    numeric_concepts = resolve_numeric_concepts(
        sample,
        body_curvature=args.body_curvature,
        midbody_expansion=args.midbody_expansion,
        tail_taper=args.tail_taper,
    )
    image = Image.open(args.image).convert("RGB")
    out = render_panel(image, sample, concepts, class_names, numeric_concepts)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path)
    print(f"Wrote concept-annotated target image to {out_path}")


if __name__ == "__main__":
    main()
