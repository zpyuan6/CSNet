from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _normalize_heatmap(heatmap: list[list[float]] | np.ndarray) -> np.ndarray:
    arr = np.asarray(heatmap, dtype=np.float32)
    if arr.size == 0:
        return arr
    arr = arr - arr.min()
    mx = float(arr.max())
    if mx > 0:
        arr = arr / mx
    return arr


def _load_font(size: int) -> ImageFont.ImageFont:
    for name in ("arial.ttf", "Arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_heatmap_overlay(
    image_path: str | Path,
    heatmap: list[list[float]] | np.ndarray,
    out_path: str | Path,
    alpha: float = 0.85,
) -> str:
    image = Image.open(image_path).convert("RGBA")
    arr = _normalize_heatmap(heatmap)
    if arr.size == 0:
        image.convert("RGB").save(out_path)
        return str(out_path)
    heat = Image.fromarray(np.uint8(arr * 255), mode="L").resize(image.size, Image.BILINEAR)
    overlay = Image.new("RGBA", image.size, (255, 0, 0, 0))
    overlay.putalpha(Image.fromarray(np.uint8(np.asarray(heat, dtype=np.float32) * float(alpha))))
    out = Image.alpha_composite(image, overlay).convert("RGB")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path)
    return str(out_path)


def render_detection(image_path: str | Path, bbox_xyxy: list[float], label: str, out_path: str | Path) -> str:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle(bbox_xyxy, outline=(0, 255, 0), width=3)
    tx = max(2, int(bbox_xyxy[0]))
    ty = max(2, int(bbox_xyxy[1]) - 18)
    draw.rectangle((tx, ty, tx + max(120, len(label) * 7), ty + 16), fill=(0, 255, 0))
    draw.text((tx + 2, ty + 1), label, fill=(0, 0, 0))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    return str(out_path)


def render_atlas_overview(concepts: list[dict], out_path: str | Path, title: str = "Concept Atlas") -> str:
    card_w = 188
    card_h = 180
    cols = 3
    rows = max(1, math.ceil(len(concepts) / cols))
    canvas = Image.new("RGB", (cols * card_w + 20, rows * card_h + 50), (248, 248, 248))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), title, fill=(0, 0, 0))
    for idx, concept in enumerate(concepts):
        col = idx % cols
        row = idx // cols
        x0 = 10 + col * card_w
        y0 = 30 + row * card_h
        draw.rectangle((x0, y0, x0 + card_w - 10, y0 + card_h - 10), outline=(180, 180, 180), width=1)
        concept_id = str(concept.get("concept_id", ""))
        concept_name = str(concept.get("concept_name") or "").strip()
        if concept_name:
            label = f"{concept_id} | {concept_name}"
        else:
            label = concept_id
        draw.text((x0 + 6, y0 + 6), label[:40], fill=(0, 0, 0))
        refs = concept.get("references", [])[:6]
        for ridx, ref in enumerate(refs):
            crop_path = ref.get("crop_path")
            if not crop_path or not Path(crop_path).exists():
                continue
            try:
                patch = Image.open(crop_path).convert("RGB").resize((52, 52), Image.BILINEAR)
            except OSError:
                continue
            px = x0 + 6 + (ridx % 3) * 58
            py = y0 + 28 + (ridx // 3) * 58
            canvas.paste(patch, (px, py))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return str(out_path)


def render_concept_reference_samples(concept: dict, out_path: str | Path, title: str | None = None) -> str:
    refs = concept.get("references", [])[:6]
    cols = 3
    card_w = 220
    card_h = 170
    rows = max(1, math.ceil(len(refs) / cols))
    canvas = Image.new("RGB", (cols * card_w + 20, rows * card_h + 60), (250, 250, 250))
    draw = ImageDraw.Draw(canvas)
    header = title or (concept.get("concept_name") or concept.get("concept_id") or "Concept References")
    draw.text((10, 10), header[:60], fill=(0, 0, 0))
    for idx, ref in enumerate(refs):
        col = idx % cols
        row = idx // cols
        x0 = 10 + col * card_w
        y0 = 30 + row * card_h
        draw.rectangle((x0, y0, x0 + card_w - 10, y0 + card_h - 10), outline=(180, 180, 180), width=1)
        overlay_path = ref.get("sample_overlay_path")
        crop_path = ref.get("crop_path")
        image_box = None
        if overlay_path and Path(overlay_path).exists():
            try:
                image_box = Image.open(overlay_path).convert("RGB").resize((140, 96), Image.BILINEAR)
            except OSError:
                image_box = None
        if image_box is not None:
            canvas.paste(image_box, (x0 + 6, y0 + 6))
        patch = None
        if crop_path and Path(crop_path).exists():
            try:
                patch = Image.open(crop_path).convert("RGB").resize((56, 56), Image.BILINEAR)
            except OSError:
                patch = None
        if patch is not None:
            canvas.paste(patch, (x0 + 152, y0 + 6))
        score = float(ref.get("reference_score", ref.get("score", 0.0)))
        class_id = ref.get("class_id")
        draw.text((x0 + 6, y0 + 110), f"score={score:.3f}", fill=(40, 40, 40))
        draw.text((x0 + 6, y0 + 126), f"class={class_id}", fill=(60, 60, 60))
        img_name = Path(str(ref.get("image_path", ""))).name
        draw.text((x0 + 6, y0 + 142), img_name[:28], fill=(80, 80, 80))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return str(out_path)


def render_layered_graph(nodes: list[dict], edges: list[dict], out_path: str | Path, title: str) -> str:
    layers = []
    by_layer: dict[str, list[dict]] = {}
    for node in nodes:
        layer = node.get("layer") or ("class" if node.get("kind") == "class" else "misc")
        if layer not in by_layer:
            layers.append(layer)
            by_layer[layer] = []
        by_layer[layer].append(node)
    width = max(900, 220 * max(1, len(layers)))
    height = 140 + max((len(v) for v in by_layer.values()), default=1) * 80
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.text((10, 10), title, fill=(0, 0, 0))
    positions: dict[str, tuple[int, int]] = {}
    for lidx, layer in enumerate(layers):
        x = 120 + lidx * max(180, (width - 180) // max(1, len(layers)))
        draw.text((x - 20, 40), str(layer), fill=(20, 20, 20))
        if layer == "class":
            layer_nodes = sorted(by_layer[layer], key=lambda item: int(item.get("class_id", 10**9)))
        else:
            layer_nodes = sorted(
                by_layer[layer],
                key=lambda item: (int(item.get("type_idx", 10**9)), str(item.get("id", ""))),
            )
        for nidx, node in enumerate(layer_nodes):
            y = 90 + nidx * 70
            positions[str(node["id"])] = (x, y)
            fill = (255, 243, 230) if node.get("kind") == "concept" else (230, 240, 255)
            draw.rounded_rectangle((x - 72, y - 24, x + 72, y + 24), radius=8, outline=(120, 120, 120), fill=fill)
            if node.get("kind") == "concept":
                concept_id = str(node.get("id", ""))
                concept_name = str(node.get("concept_name") or "").strip()
                draw.text((x - 66, y - 16), concept_id[:22], fill=(0, 0, 0))
                if concept_name:
                    draw.text((x - 66, y), concept_name[:22], fill=(60, 60, 60))
            else:
                label = node.get("class_name") or str(node["id"])
                draw.text((x - 66, y - 7), str(label)[:20], fill=(0, 0, 0))
    for edge in edges:
        src = positions.get(str(edge["source"]))
        dst = positions.get(str(edge["target"]))
        if src is None or dst is None:
            continue
        strong = bool(edge.get("highlight", False))
        color = (180, 30, 30) if strong else (170, 170, 170)
        width_px = _edge_line_width(float(edge.get("weight", 0.0)), strong)
        draw.line((src[0] + 72, src[1], dst[0] - 72, dst[1]), fill=color, width=width_px)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    return str(out_path)


def _open_thumb(path: str | Path | None, size: tuple[int, int]) -> Image.Image | None:
    if not path:
        return None
    image_path = Path(path)
    if not image_path.exists():
        return None
    try:
        return Image.open(image_path).convert("RGB").resize(size, Image.BILINEAR)
    except OSError:
        return None


def _draw_soft_curve(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int],
    width: int = 2,
) -> None:
    sx, sy = start
    ex, ey = end
    points: list[tuple[int, int]] = []
    mid_x = (sx + ex) / 2.0
    dx = ex - sx
    dy = ey - sy
    bend = max(30.0, abs(dx) * 0.10 + abs(dy) * 0.18)
    direction = -1.0 if dy >= 0 else 1.0
    left_ctrl1 = (sx + dx * 0.18, sy)
    left_ctrl2 = (mid_x - dx * 0.12, sy + direction * bend)
    center = (mid_x, (sy + ey) / 2.0)
    right_ctrl1 = (mid_x + dx * 0.12, ey - direction * bend)
    right_ctrl2 = (ex - dx * 0.18, ey)

    for idx in range(13):
        t = idx / 12.0
        omt = 1.0 - t
        x = (
            omt * omt * omt * sx
            + 3.0 * omt * omt * t * left_ctrl1[0]
            + 3.0 * omt * t * t * left_ctrl2[0]
            + t * t * t * center[0]
        )
        y = (
            omt * omt * omt * sy
            + 3.0 * omt * omt * t * left_ctrl1[1]
            + 3.0 * omt * t * t * left_ctrl2[1]
            + t * t * t * center[1]
        )
        points.append((int(round(x)), int(round(y))))
    for idx in range(1, 13):
        t = idx / 12.0
        omt = 1.0 - t
        x = (
            omt * omt * omt * center[0]
            + 3.0 * omt * omt * t * right_ctrl1[0]
            + 3.0 * omt * t * t * right_ctrl2[0]
            + t * t * t * ex
        )
        y = (
            omt * omt * omt * center[1]
            + 3.0 * omt * omt * t * right_ctrl1[1]
            + 3.0 * omt * t * t * right_ctrl2[1]
            + t * t * t * ey
        )
        points.append((int(round(x)), int(round(y))))
    draw.line(points, fill=color, width=width)


def _edge_line_width(weight: float, strong: bool) -> int:
    w = max(0.0, float(weight))
    base = 1.0 + (w ** 0.7) * 4.0
    if strong:
        base += 1.0
    return max(1, min(8, int(round(base))))


def _horizontal_cubic_spline_link(
    x_from_to: tuple[float, float],
    y_from_to: tuple[float, float],
    n_segments: int = 64,
) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for idx in range(n_segments + 1):
        t = idx / float(n_segments)
        x = x_from_to[0] + (x_from_to[1] - x_from_to[0]) * t
        y = y_from_to[0] + (y_from_to[1] - y_from_to[0]) * (t * t * (3.0 - 2.0 * t))
        points.append((x, y))
    return points


def _draw_flow_band(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int],
    width: int,
) -> list[tuple[int, int]]:
    spline = _horizontal_cubic_spline_link((float(start[0]), float(end[0])), (float(start[1]), float(end[1])))
    half_w = max(1.0, float(width) * 0.55)
    left_side: list[tuple[int, int]] = []
    right_side: list[tuple[int, int]] = []
    for idx, (x, y) in enumerate(spline):
        prev_pt = spline[max(0, idx - 1)]
        next_pt = spline[min(len(spline) - 1, idx + 1)]
        tx = next_pt[0] - prev_pt[0]
        ty = next_pt[1] - prev_pt[1]
        norm = max(1.0, (tx * tx + ty * ty) ** 0.5)
        nx = -ty / norm
        ny = tx / norm
        taper = 0.80 + 0.20 * (1.0 - abs(idx - len(spline) / 2.0) / max(1.0, len(spline) / 2.0))
        offset = half_w * taper
        left_side.append((int(round(x + nx * offset)), int(round(y + ny * offset))))
        right_side.append((int(round(x - nx * offset)), int(round(y - ny * offset))))
    polygon = left_side + list(reversed(right_side))
    draw.polygon(polygon, fill=color)
    centerline = [(int(round(x)), int(round(y))) for x, y in spline]
    return centerline


def render_inference_tree(
    target: dict,
    nodes: list[dict],
    edges: list[dict],
    out_path: str | Path,
    title: str = "Inference Tree",
) -> str:
    scale = 2
    title_font = _load_font(18 * scale)
    layer_font = _load_font(15 * scale)
    node_font = _load_font(13 * scale)
    sub_font = _load_font(12 * scale)
    weight_font = _load_font(12 * scale)
    node_by_id = {str(node["id"]): node for node in nodes}
    layers = []
    by_layer: dict[str, list[dict]] = {}
    for node in nodes:
        layer = str(node.get("layer") or "misc")
        if layer not in by_layer:
            layers.append(layer)
            by_layer[layer] = []
        by_layer[layer].append(node)

    card_w = 320 * scale
    card_h = 190 * scale
    gap_x = 50 * scale
    gap_y = 26 * scale
    left_pad = 20 * scale
    top_pad = 50 * scale
    width = left_pad * 2 + max(1, len(layers)) * card_w + max(0, len(layers) - 1) * gap_x
    max_rows = max((len(items) for items in by_layer.values()), default=1)
    height = top_pad + max_rows * card_h + max(0, max_rows - 1) * gap_y + 30 * scale
    canvas = Image.new("RGB", (width, height), (250, 250, 250))
    draw = ImageDraw.Draw(canvas)
    draw.text((16 * scale, 12 * scale), title, fill=(0, 0, 0), font=title_font)

    positions: dict[str, tuple[int, int]] = {}
    for lidx, layer in enumerate(layers):
        x0 = left_pad + lidx * (card_w + gap_x)
        draw.text((x0 + 6 * scale, 30 * scale), layer, fill=(30, 30, 30), font=layer_font)
        layer_nodes = by_layer[layer]
        if layer == "class":
            ordered = layer_nodes
        else:
            ordered = sorted(
                layer_nodes,
                key=lambda item: (int(item.get("type_idx", 10**9)), str(item.get("id", ""))),
            )
        for nidx, node in enumerate(ordered):
            y0 = top_pad + nidx * (card_h + gap_y)
            positions[str(node["id"])] = (x0, y0)
            draw.rounded_rectangle((x0, y0, x0 + card_w, y0 + card_h), radius=10 * scale, outline=(140, 140, 140), fill=(255, 255, 255))
            if node.get("kind") == "class":
                title_text = f"class:{node.get('class_id')} score={float(node.get('score', 0.0)):.3f}"
                draw.text((x0 + 8 * scale, y0 + 8 * scale), title_text[:34], fill=(0, 0, 0), font=node_font)
                det_thumb = _open_thumb(node.get("target_detection_path"), (card_w - 16 * scale, card_h - 40 * scale))
                if det_thumb is not None:
                    canvas.paste(det_thumb, (x0 + 8 * scale, y0 + 30 * scale))
                continue

            concept_id = str(node.get("id", ""))
            concept_name = str(node.get("concept_name") or "").strip()
            draw.text((x0 + 8 * scale, y0 + 8 * scale), concept_id[:34], fill=(0, 0, 0), font=node_font)
            if concept_name:
                draw.text((x0 + 8 * scale, y0 + 24 * scale), concept_name[:34], fill=(70, 70, 70), font=sub_font)
            overlay_path = (((node.get("heatmap") or {}) if isinstance(node.get("heatmap"), dict) else {}).get("sample_overlay_path"))
            atlas_refs = node.get("atlas_refs") or []
            overlay_thumb = _open_thumb(overlay_path, (150 * scale, 112 * scale))
            if overlay_thumb is not None:
                canvas.paste(overlay_thumb, (x0 + 8 * scale, y0 + 48 * scale))
            atlas_origin_x = x0 + 170 * scale
            atlas_origin_y = y0 + 48 * scale
            for ridx, ref in enumerate(atlas_refs[:4]):
                atlas_thumb = _open_thumb(ref.get("crop_path"), (64 * scale, 64 * scale))
                if atlas_thumb is None:
                    continue
                rx = atlas_origin_x + (ridx % 2) * 70 * scale
                ry = atlas_origin_y + (ridx // 2) * 70 * scale
                canvas.paste(atlas_thumb, (rx, ry))

    for edge in edges:
        src = positions.get(str(edge["source"]))
        dst = positions.get(str(edge["target"]))
        if src is None or dst is None:
            continue
        sx = src[0] + card_w
        sy = src[1] + card_h // 2
        dx = dst[0]
        dy = dst[1] + card_h // 2
        strong = bool(edge.get("highlight", False))
        color = (180, 30, 30) if strong else (150, 150, 150)
        width_px = _edge_line_width(float(edge.get("weight", 0.0)), strong) * scale
        centerline = _draw_flow_band(draw, (sx, sy), (dx, dy), color=color, width=width_px + 3 * scale)
        mid_idx = len(centerline) // 2
        mid_x, mid_y = centerline[mid_idx]
        label_x = mid_x + 6 * scale
        label_y = mid_y - 18 * scale
        draw.text((label_x, label_y), f"{float(edge.get('weight', 0.0)):.2f}", fill=color, font=weight_font)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas = canvas.resize((width // scale, height // scale), Image.Resampling.LANCZOS)
    canvas.save(out_path)
    return str(out_path)
