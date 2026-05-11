from __future__ import annotations

import argparse
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from datetime import datetime


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Objects365 COCO annotations to YOLO labels without per-class rescans."
    )
    parser.add_argument(
        "--src-root",
        type=Path,
        default=Path(r"C:\Datasets\Objects365"),
        help="Objects365 root containing images/ and zhiyuan_objv2_*.json",
    )
    parser.add_argument(
        "--dst-root",
        type=Path,
        default=Path(r"C:\Datasets\Objects365_fast"),
        help="Independent output root for generated labels and helper yaml/txt files.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        choices=["train", "val"],
        help="Dataset splits to convert.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="File-writing workers. Start low on Windows SSDs and increase only if useful.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Limit conversion to the first N images in the selected split. 0 means all images.",
    )
    parser.add_argument(
        "--label-subdir",
        type=str,
        default="",
        help="Optional subdirectory under labels/, e.g. test_train.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Optional progress log file path.",
    )
    return parser.parse_args()


def xywh_to_yolo(box: list[float], width: int, height: int) -> tuple[float, float, float, float]:
    x, y, w, h = box
    x1 = min(max(x, 0.0), float(width))
    y1 = min(max(y, 0.0), float(height))
    x2 = min(max(x + w, 0.0), float(width))
    y2 = min(max(y + h, 0.0), float(height))
    clipped_w = max(0.0, x2 - x1)
    clipped_h = max(0.0, y2 - y1)
    xc = (x1 + x2) / 2.0 / width
    yc = (y1 + y2) / 2.0 / height
    wn = clipped_w / width
    hn = clipped_h / height
    return (
        min(max(xc, 0.0), 1.0),
        min(max(yc, 0.0), 1.0),
        min(max(wn, 0.0), 1.0),
        min(max(hn, 0.0), 1.0),
    )


def build_image_index(images_root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    duplicates: set[str] = set()

    for path in images_root.rglob("*"):
        if path.suffix.lower() not in IMG_EXTS or not path.is_file():
            continue
        rel = path.relative_to(images_root)
        key = path.name
        if key in index and index[key] != rel:
            duplicates.add(key)
            continue
        index[key] = rel

    if duplicates:
        sample = ", ".join(sorted(list(duplicates))[:5])
        raise RuntimeError(f"Found duplicate image basenames under {images_root}: {sample}")
    return index


def load_coco(annotation_file: Path) -> dict:
    with annotation_file.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_category_map(coco: dict) -> tuple[dict[int, int], list[str]]:
    categories = sorted(coco["categories"], key=lambda x: x["id"])
    cat_id_to_idx = {cat["id"]: idx for idx, cat in enumerate(categories)}
    names = [cat["name"] for cat in categories]
    return cat_id_to_idx, names


def log_message(log_file: Path | None, message: str) -> None:
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def write_dataset_yaml(dst_root: Path, names: list[str]) -> None:
    yaml_path = dst_root / "Objects365_fast.yaml"
    lines = [
        f"path: {dst_root.as_posix()}",
        "train: train.txt",
        "val: val.txt",
        "names:",
    ]
    lines.extend(f"  {idx}: {name}" for idx, name in enumerate(names))
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def convert_split(
    src_root: Path,
    dst_root: Path,
    split: str,
    workers: int,
    max_images: int,
    label_subdir: str,
    log_file: Path | None,
) -> None:
    images_root = src_root / "images" / split
    annotation_file = src_root / f"zhiyuan_objv2_{split}.json"
    if not images_root.exists():
        raise FileNotFoundError(f"Missing images directory: {images_root}")
    if not annotation_file.exists():
        raise FileNotFoundError(f"Missing annotation file: {annotation_file}")

    log_message(log_file, f"[{split}] indexing images under {images_root}")
    image_index = build_image_index(images_root)
    log_message(log_file, f"[{split}] indexed {len(image_index):,} image files")

    log_message(log_file, f"[{split}] loading {annotation_file.name}")
    coco = load_coco(annotation_file)
    cat_id_to_idx, names = build_category_map(coco)

    images = coco["images"]
    if max_images > 0:
        images = images[:max_images]
    image_meta = {im["id"]: im for im in images}
    selected_image_ids = set(image_meta)
    labels_by_image: dict[int, list[str]] = defaultdict(list)

    total_annotations = len(coco["annotations"])
    log_message(log_file, f"[{split}] grouping {total_annotations:,} annotations")
    for i, ann in enumerate(coco["annotations"], start=1):
        image_id = ann["image_id"]
        if image_id not in selected_image_ids:
            continue
        im = image_meta.get(image_id)
        if im is None:
            continue
        width, height = im["width"], im["height"]
        if width <= 0 or height <= 0:
            continue
        cls = cat_id_to_idx[ann["category_id"]]
        x, y, w, h = xywh_to_yolo(ann["bbox"], width, height)
        labels_by_image[image_id].append(f"{cls} {x:.5f} {y:.5f} {w:.5f} {h:.5f}\n")
        if i % 200000 == 0:
            log_message(log_file, f"[{split}] grouped {i:,}/{total_annotations:,} annotations")

    labels_root = dst_root / "labels" / (label_subdir or split)
    labels_root.mkdir(parents=True, exist_ok=True)

    log_message(log_file, f"[{split}] writing {len(labels_by_image):,} label files to {labels_root}")

    def write_one(item: tuple[int, list[str]]) -> None:
        image_id, rows = item
        im = image_meta[image_id]
        basename = Path(im["file_name"]).name
        rel_image = image_index.get(basename)
        if rel_image is None:
            return
        label_path = (labels_root / rel_image).with_suffix(".txt")
        label_path.parent.mkdir(parents=True, exist_ok=True)
        with label_path.open("w", encoding="utf-8") as f:
            f.writelines(rows)

    items = list(labels_by_image.items())
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        for j, _ in enumerate(executor.map(write_one, items), start=1):
            if j % 50000 == 0 or j == len(items):
                log_message(log_file, f"[{split}] wrote {j:,}/{len(items):,} label files")

    image_list_path = dst_root / f"{split}.txt"
    with image_list_path.open("w", encoding="utf-8") as f:
        selected_basenames = {Path(im["file_name"]).name for im in image_meta.values()}
        for rel_image in sorted(image_index.values()):
            if max_images > 0 and rel_image.name not in selected_basenames:
                continue
            f.write(str((images_root / rel_image).resolve()) + "\n")

    names_path = dst_root / "names.json"
    if not names_path.exists():
        with names_path.open("w", encoding="utf-8") as f:
            json.dump(names, f, ensure_ascii=False, indent=2)
        write_dataset_yaml(dst_root, names)

    log_message(log_file, f"[{split}] done")


def main() -> None:
    args = parse_args()
    args.dst_root.mkdir(parents=True, exist_ok=True)
    if args.log_file:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        args.log_file.write_text("", encoding="utf-8")
    for split in args.splits:
        convert_split(args.src_root, args.dst_root, split, args.workers, args.max_images, args.label_subdir, args.log_file)


if __name__ == "__main__":
    main()
