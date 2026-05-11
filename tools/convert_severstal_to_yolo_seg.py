import argparse
import base64
import json
import random
import shutil
import zlib
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Severstal-DatasetNinja annotations to YOLO segmentation format.")
    parser.add_argument("--src", type=Path, required=True, help="Path to severstal-DatasetNinja root.")
    parser.add_argument("--dst", type=Path, required=True, help="Output YOLO-seg dataset root.")
    parser.add_argument(
        "--task",
        choices=["multiclass", "binary"],
        default="multiclass",
        help="multiclass keeps defect_1..defect_4. binary merges all defects into class 0.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Fraction of original train split used as val.")
    parser.add_argument("--approx-eps", type=float, default=1.5, help="Polygon simplification epsilon in pixels.")
    parser.add_argument("--min-points", type=int, default=3, help="Minimum polygon points retained.")
    parser.add_argument("--copy", action="store_true", help="Copy files instead of hard-linking them.")
    return parser.parse_args()


def ensure_dirs(root: Path) -> None:
    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path, copy_files: bool) -> None:
    if dst.exists():
        return
    if copy_files:
        shutil.copy2(src, dst)
        return
    try:
        dst.hardlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def load_class_map(src: Path, task: str) -> tuple[dict[str, int], list[str]]:
    meta = json.loads((src / "meta.json").read_text(encoding="utf-8"))
    raw_names = [c["title"] for c in meta["classes"]]
    if task == "binary":
        return {name: 0 for name in raw_names}, ["defect"]
    raw_names = sorted(raw_names)
    return {name: i for i, name in enumerate(raw_names)}, raw_names


def decode_bitmap_mask(bitmap: dict, image_h: int, image_w: int) -> np.ndarray:
    raw = zlib.decompress(base64.b64decode(bitmap["data"]))
    arr = np.frombuffer(raw, dtype=np.uint8)
    decoded = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if decoded is None:
        raise RuntimeError("Failed to decode bitmap annotation.")
    if decoded.ndim == 3:
        local_mask = decoded[:, :, 3] if decoded.shape[2] == 4 else decoded[:, :, 0]
    else:
        local_mask = decoded
    local_mask = (local_mask > 0).astype(np.uint8) * 255

    x0, y0 = bitmap["origin"]
    h, w = local_mask.shape[:2]
    canvas = np.zeros((image_h, image_w), dtype=np.uint8)
    x1 = min(image_w, x0 + w)
    y1 = min(image_h, y0 + h)
    canvas[y0:y1, x0:x1] = local_mask[: y1 - y0, : x1 - x0]
    return canvas


def mask_to_polygons(mask: np.ndarray, approx_eps: float, min_points: int) -> list[list[float]]:
    h, w = mask.shape[:2]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if cv2.contourArea(contour) <= 1.0:
            continue
        approx = cv2.approxPolyDP(contour, approx_eps, True).reshape(-1, 2)
        if approx.shape[0] < min_points:
            continue
        poly = []
        for x, y in approx:
            poly.extend([float(x) / w, float(y) / h])
        polygons.append(poly)
    return polygons


def parse_annotation(ann_path: Path, class_map: dict[str, int], task: str, approx_eps: float, min_points: int) -> tuple[list[str], int, int]:
    data = json.loads(ann_path.read_text(encoding="utf-8"))
    image_h = int(data["size"]["height"])
    image_w = int(data["size"]["width"])
    lines: list[str] = []
    for obj in data.get("objects", []):
        if obj.get("geometryType") != "bitmap":
            continue
        class_title = obj["classTitle"]
        class_id = 0 if task == "binary" else class_map[class_title]
        mask = decode_bitmap_mask(obj["bitmap"], image_h=image_h, image_w=image_w)
        polygons = mask_to_polygons(mask, approx_eps=approx_eps, min_points=min_points)
        for poly in polygons:
            lines.append(f"{class_id} " + " ".join(f"{v:.6f}" for v in poly))
    return lines, image_w, image_h


def collect_pairs(split_dir: Path) -> list[tuple[Path, Path]]:
    img_dir = split_dir / "img"
    ann_dir = split_dir / "ann"
    pairs = []
    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue
        ann_path = ann_dir / f"{img_path.name}.json"
        if not ann_path.exists():
            raise FileNotFoundError(f"Missing annotation for image: {img_path}")
        pairs.append((img_path, ann_path))
    return pairs


def write_label(label_path: Path, lines: list[str]) -> None:
    label_path.write_text("\n".join(lines), encoding="utf-8")


def save_pair(
    img_path: Path,
    ann_path: Path,
    split: str,
    dst: Path,
    class_map: dict[str, int],
    task: str,
    approx_eps: float,
    min_points: int,
    copy_files: bool,
) -> None:
    out_img = dst / "images" / split / img_path.name
    out_label = dst / "labels" / split / f"{img_path.stem}.txt"
    link_or_copy(img_path, out_img, copy_files)
    lines, _, _ = parse_annotation(ann_path, class_map, task, approx_eps, min_points)
    write_label(out_label, lines)


def write_yaml(dst: Path, names: list[str]) -> Path:
    yaml_path = dst / "severstal_yolo_seg.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {dst.as_posix()}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "",
                f"nc: {len(names)}",
                f"names: {names}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return yaml_path


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    ensure_dirs(args.dst)
    class_map, names = load_class_map(args.src, args.task)

    train_pairs = collect_pairs(args.src / "train")
    test_pairs = collect_pairs(args.src / "test")

    rng.shuffle(train_pairs)
    n_val = int(round(len(train_pairs) * args.val_ratio))
    val_pairs = train_pairs[:n_val]
    train_pairs = train_pairs[n_val:]

    for img_path, ann_path in train_pairs:
        save_pair(img_path, ann_path, "train", args.dst, class_map, args.task, args.approx_eps, args.min_points, args.copy)
    for img_path, ann_path in val_pairs:
        save_pair(img_path, ann_path, "val", args.dst, class_map, args.task, args.approx_eps, args.min_points, args.copy)
    for img_path, ann_path in test_pairs:
        save_pair(img_path, ann_path, "test", args.dst, class_map, args.task, args.approx_eps, args.min_points, args.copy)

    yaml_path = write_yaml(args.dst, names)
    print(f"Converted Severstal dataset to YOLO-seg at: {args.dst}")
    print(f"train={len(train_pairs)} val={len(val_pairs)} test={len(test_pairs)}")
    print(f"Classes: {names}")
    print(f"Dataset YAML: {yaml_path}")


if __name__ == "__main__":
    main()
