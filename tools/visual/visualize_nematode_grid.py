from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from ultralytics import YOLO
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import register_ultralytics_modules


DEFAULT_MODEL = r"runs\detect\CapsNeckHeadv8_nematode_pt\weights\best.pt"
DEFAULT_IMAGE_DIR = r"C:\Datasets\Yolo_11Dec\images\val"
DEFAULT_OUTPUT = r"runs\detect\CapsNeckHeadv8_nematode_pt\viz_grid_2x2.png"
DEFAULT_DATA_YAML = r"configs\data\nematode.yaml"
DEFAULT_IMAGES = [
    "pcn_rln_x5_Image002_ch00.jpg",
    # "Globodera pallida (180).JPG",
    "PCN (205).jpg",
    # "PCN (188).jpg",
    # "nemotode031.jpg",
    "Image046_ch00.jpg",
    # "MC170020_1.JPG",
    # "cMC170001 (72).JPG",
    "pcn_rln_x5_Image031_ch00.jpg",
]
PALETTE = [
    (230, 25, 75),
    (60, 180, 75),
    (255, 225, 25),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
]


def _load_font(size: int) -> ImageFont.ImageFont:
    for name in ("arial.ttf", "Arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a 2x2 nematode detection visualization grid with color-coded boxes and clear legends."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--image-dir", default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--data-yaml", default=DEFAULT_DATA_YAML)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--cell-width", type=int, default=400)
    parser.add_argument("--cell-height", type=int, default=320)
    parser.add_argument("--images", nargs="*", default=DEFAULT_IMAGES)
    return parser.parse_args()


def draw_result_tile(
    image_path: Path,
    result,
    cell_width: int,
    cell_height: int,
    palette: list[tuple[int, int, int]],
) -> tuple[Image.Image, OrderedDict[int, dict]]:
    image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    has_masks = result.masks is not None and result.boxes is not None
    if has_masks:
        mask_overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        mask_draw = ImageDraw.Draw(mask_overlay, "RGBA")
        polygons = getattr(result.masks, "xy", None) or []
        mask_cls = result.boxes.cls.cpu().tolist()
        for polygon, cls_id_f in zip(polygons, mask_cls):
            if polygon is None or len(polygon) < 3:
                continue
            cls_id = int(cls_id_f)
            color = palette[cls_id % len(palette)]
            points = [(float(x), float(y)) for x, y in polygon.tolist()]
            mask_draw.polygon(points, fill=(color[0], color[1], color[2], 110))
        image = Image.alpha_composite(image.convert("RGBA"), mask_overlay).convert("RGB")
    draw = ImageDraw.Draw(image)
    present: OrderedDict[int, dict] = OrderedDict()

    boxes = result.boxes
    if boxes is not None:
        xyxy = boxes.xyxy.cpu().tolist()
        cls = boxes.cls.cpu().tolist()
        conf = boxes.conf.cpu().tolist()
        box_width = 2 if has_masks else 6
        for box, cls_id_f, conf_f in zip(xyxy, cls, conf):
            _ = conf_f
            cls_id = int(cls_id_f)
            color = palette[cls_id % len(palette)]
            draw.rectangle(box, outline=color, width=box_width)
            if cls_id not in present:
                present[cls_id] = {"name": result.names[cls_id], "color": color, "count": 0}
            present[cls_id]["count"] += 1

    tile = Image.new("RGB", (cell_width, cell_height), (255, 255, 255))
    title_h = 4
    render_h = cell_height - title_h - 4
    scaled = image.copy()
    scaled.thumbnail((cell_width - 12, render_h), Image.Resampling.BILINEAR)
    paste_x = (cell_width - scaled.width) // 2
    paste_y = title_h + (render_h - scaled.height) // 2
    tile.paste(scaled, (paste_x, paste_y))

    td = ImageDraw.Draw(tile)
    td.rectangle((0, 0, cell_width - 1, cell_height - 1), outline=(180, 180, 180), width=1)
    return tile, present


def build_legend(
    class_items: OrderedDict[int, dict],
    width: int,
) -> Image.Image:
    cols = 2
    rows = max(1, (len(class_items) + cols - 1) // cols)
    font = _load_font(22)
    legend = Image.new("RGB", (width, 18 + rows * 32), (250, 250, 250))
    draw = ImageDraw.Draw(legend)
    draw.rectangle((0, 0, width - 1, legend.height - 1), outline=(180, 180, 180), width=1)
    col_w = max(width // cols, 1)
    for idx, (_, item) in enumerate(class_items.items()):
        col = idx % cols
        row = idx // cols
        x = 10 + col * col_w
        y = 10 + row * 32
        color = item["color"]
        draw.rectangle((x, y, x + 18, y + 18), fill=color, outline=color)
        draw.text((x + 28, y - 2), item["name"], fill=(20, 20, 20), font=font)
    return legend


def load_class_items(data_yaml: str | Path, palette: list[tuple[int, int, int]]) -> OrderedDict[int, dict]:
    yaml_path = Path(data_yaml)
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    names = data.get("names", {})
    items: OrderedDict[int, dict] = OrderedDict()
    if isinstance(names, dict):
        ordered = sorted(((int(k), str(v)) for k, v in names.items()), key=lambda item: item[0])
    else:
        ordered = list(enumerate([str(v) for v in names]))
    for cls_id, name in ordered:
        items[int(cls_id)] = {"name": name, "color": palette[int(cls_id) % len(palette)]}
    return items


def main() -> None:
    args = parse_args()
    register_ultralytics_modules()
    image_dir = Path(args.image_dir)
    image_paths = [image_dir / name for name in args.images]
    missing = [str(path) for path in image_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing images: {missing}")

    model = YOLO(args.model)
    results = model([str(path) for path in image_paths], conf=args.conf, imgsz=args.imgsz, verbose=False)

    tiles: list[Image.Image] = []
    class_items = load_class_items(args.data_yaml, PALETTE)
    for image_path, result in zip(image_paths, results):
        tile, present = draw_result_tile(image_path, result, args.cell_width, args.cell_height, PALETTE)
        tiles.append(tile)
        for cls_id, item in present.items():
            if cls_id in class_items:
                class_items[cls_id]["color"] = item["color"]

    cols = 2
    rows = 2
    grid_w = cols * args.cell_width
    grid_h = rows * args.cell_height
    legend = build_legend(class_items, grid_w - 20)
    canvas = Image.new("RGB", (grid_w, grid_h + legend.height + 12), (245, 245, 245))

    for idx, tile in enumerate(tiles[: cols * rows]):
        x = (idx % cols) * args.cell_width
        y = (idx // cols) * args.cell_height
        canvas.paste(tile, (x, y))

    canvas.paste(legend, (10, grid_h + 6))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    print(out_path)


if __name__ == "__main__":
    main()
