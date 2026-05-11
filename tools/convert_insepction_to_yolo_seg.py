import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Insepction defect dataset to YOLO segmentation format.")
    parser.add_argument("--src", type=Path, required=True, help="Path to Insepction dataset root.")
    parser.add_argument("--dst", type=Path, required=True, help="Output YOLO-seg dataset root.")
    parser.add_argument(
        "--task",
        choices=["multiclass", "binary"],
        default="multiclass",
        help="multiclass uses each top-level folder as one class; binary merges all defects into class 0.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--threshold", type=int, default=127, help="Mask binarization threshold.")
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


def split_items(items: list[dict], train_ratio: float, val_ratio: float, test_ratio: float, rng: random.Random) -> dict[str, list[dict]]:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError("train/val/test ratios must sum to 1.")
    rng.shuffle(items)
    n = len(items)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    if n_train + n_val > n:
        n_val = max(0, n - n_train)
    return {
        "train": items[:n_train],
        "val": items[n_train:n_train + n_val],
        "test": items[n_train + n_val:],
    }


def mask_to_polygons(mask_path: Path, threshold: int, approx_eps: float, min_points: int) -> list[list[float]]:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Failed to read mask: {mask_path}")
    _, binary = cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = binary.shape[:2]
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


def collect_items(src: Path, task: str) -> tuple[list[dict], list[str]]:
    class_dirs = sorted([p for p in src.iterdir() if p.is_dir()])
    names = ["defect"] if task == "binary" else [p.name for p in class_dirs]
    name_to_id = {name: idx for idx, name in enumerate(names)}
    items = []

    for class_dir in class_dirs:
        imgs_dir = class_dir / "Imgs"
        if not imgs_dir.exists():
            continue
        class_name = "defect" if task == "binary" else class_dir.name
        class_id = name_to_id[class_name]

        jpgs = sorted([p for p in imgs_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg"}])
        for img_path in jpgs:
            mask_path = imgs_dir / f"{img_path.stem}.png"
            if not mask_path.exists():
                continue
            items.append(
                {
                    "class_name": class_dir.name,
                    "class_id": class_id,
                    "img_path": img_path,
                    "mask_path": mask_path,
                }
            )
    return items, names


def unique_name(item: dict) -> str:
    return f"{item['class_name']}__{item['img_path'].name}"


def save_item(item: dict, split: str, dst: Path, threshold: int, approx_eps: float, min_points: int, copy_files: bool) -> None:
    out_image = dst / "images" / split / unique_name(item)
    out_label = dst / "labels" / split / f"{Path(unique_name(item)).stem}.txt"
    link_or_copy(item["img_path"], out_image, copy_files)
    polygons = mask_to_polygons(item["mask_path"], threshold, approx_eps, min_points)
    lines = [f"{item['class_id']} " + " ".join(f"{v:.6f}" for v in poly) for poly in polygons]
    out_label.write_text("\n".join(lines), encoding="utf-8")


def write_yaml(dst: Path, names: list[str]) -> Path:
    yaml_path = dst / "insepction_yolo_seg.yaml"
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
    items, names = collect_items(args.src, args.task)
    if not items:
        raise RuntimeError("No image-mask pairs found.")

    grouped: dict[str, list[dict]] = {}
    for item in items:
        grouped.setdefault(item["class_name"], []).append(item)

    split_buckets = {"train": [], "val": [], "test": []}
    for _, class_items in grouped.items():
        split = split_items(class_items, args.train_ratio, args.val_ratio, args.test_ratio, rng)
        for key in split_buckets:
            split_buckets[key].extend(split[key])

    for split, split_items_list in split_buckets.items():
        for item in split_items_list:
            save_item(item, split, args.dst, args.threshold, args.approx_eps, args.min_points, args.copy)

    yaml_path = write_yaml(args.dst, names)
    print(f"Converted Insepction dataset to YOLO-seg at: {args.dst}")
    print(f"train={len(split_buckets['train'])} val={len(split_buckets['val'])} test={len(split_buckets['test'])}")
    print(f"Classes: {names}")
    print(f"Dataset YAML: {yaml_path}")


if __name__ == "__main__":
    main()
