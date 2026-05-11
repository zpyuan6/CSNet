import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert MVTec AD masks to YOLO instance segmentation format.")
    parser.add_argument("--src", type=Path, required=True, help="Path to mvtec_anomaly_detection root.")
    parser.add_argument("--dst", type=Path, required=True, help="Output YOLO-seg dataset root.")
    parser.add_argument(
        "--task",
        choices=["binary", "multiclass"],
        default="binary",
        help="binary: all defects -> class 0 anomaly. multiclass: each product__defect is a class.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument(
        "--min-points",
        type=int,
        default=3,
        help="Minimum polygon points required after contour simplification.",
    )
    parser.add_argument(
        "--approx-eps",
        type=float,
        default=1.5,
        help="cv2.approxPolyDP epsilon in pixels for mask contour simplification.",
    )
    parser.add_argument(
        "--include-train-good",
        action="store_true",
        help="Copy original train/good images into YOLO train split with empty labels.",
    )
    parser.add_argument(
        "--split-test-good",
        action="store_true",
        help="Also split original test/good images across val/test with empty labels.",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy images instead of hard-linking them.",
    )
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


def contour_to_yolo_polygon(contour: np.ndarray, width: int, height: int, min_points: int) -> list[float] | None:
    contour = contour.reshape(-1, 2)
    if contour.shape[0] < min_points:
        return None
    poly = []
    for x, y in contour:
        poly.extend([float(x) / width, float(y) / height])
    return poly


def mask_to_segments(mask_path: Path, approx_eps: float, min_points: int) -> list[list[float]]:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Failed to read mask: {mask_path}")
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    segments = []
    h, w = binary.shape[:2]
    for contour in contours:
        if cv2.contourArea(contour) <= 1.0:
            continue
        approx = cv2.approxPolyDP(contour, approx_eps, True)
        poly = contour_to_yolo_polygon(approx, w, h, min_points=min_points)
        if poly is not None:
            segments.append(poly)
    return segments


def write_yolo_seg_label(label_path: Path, class_id: int, segments: list[list[float]]) -> None:
    lines = []
    for seg in segments:
        coords = " ".join(f"{v:.6f}" for v in seg)
        lines.append(f"{class_id} {coords}")
    label_path.write_text("\n".join(lines), encoding="utf-8")


def write_empty_label(label_path: Path) -> None:
    label_path.write_text("", encoding="utf-8")


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
    n_test = n - n_train - n_val
    return {
        "train": items[:n_train],
        "val": items[n_train:n_train + n_val],
        "test": items[n_train + n_val:n_train + n_val + n_test],
    }


def collect_anomaly_items(src: Path, task: str) -> tuple[list[dict], dict[str, int]]:
    items = []
    class_names: dict[str, int] = {}

    for product_dir in sorted(p for p in src.iterdir() if p.is_dir()):
        gt_root = product_dir / "ground_truth"
        test_root = product_dir / "test"
        if not gt_root.exists() or not test_root.exists():
            continue
        for defect_dir in sorted(p for p in gt_root.iterdir() if p.is_dir()):
            defect_name = defect_dir.name
            image_dir = test_root / defect_name
            if not image_dir.exists():
                continue

            if task == "binary":
                class_name = "anomaly"
            else:
                class_name = f"{product_dir.name}__{defect_name}"

            if class_name not in class_names:
                class_names[class_name] = len(class_names)
            class_id = class_names[class_name]

            for mask_path in sorted(defect_dir.iterdir()):
                if mask_path.suffix.lower() not in IMAGE_EXTS:
                    continue
                stem = mask_path.stem.replace("_mask", "")
                image_path = image_dir / f"{stem}.png"
                if not image_path.exists():
                    continue
                items.append(
                    {
                        "product": product_dir.name,
                        "defect": defect_name,
                        "class_id": class_id,
                        "image_path": image_path,
                        "mask_path": mask_path,
                    }
                )
    return items, class_names


def collect_good_images(src: Path, include_train_good: bool, split_test_good: bool) -> dict[str, list[Path]]:
    good = defaultdict(list)
    for product_dir in sorted(p for p in src.iterdir() if p.is_dir()):
        if include_train_good:
            good_dir = product_dir / "train" / "good"
            if good_dir.exists():
                for image_path in sorted(good_dir.iterdir()):
                    if image_path.suffix.lower() in IMAGE_EXTS:
                        good["train"].append(image_path)
        if split_test_good:
            good_dir = product_dir / "test" / "good"
            if good_dir.exists():
                for image_path in sorted(good_dir.iterdir()):
                    if image_path.suffix.lower() in IMAGE_EXTS:
                        good["test_pool"].append(image_path)
    return good


def unique_name(image_path: Path, product: str, defect: str) -> str:
    return f"{product}__{defect}__{image_path.stem}{image_path.suffix.lower()}"


def save_item(item: dict, split: str, dst: Path, copy_files: bool, approx_eps: float, min_points: int) -> None:
    image_name = unique_name(item["image_path"], item["product"], item["defect"])
    label_name = Path(image_name).with_suffix(".txt").name
    out_image = dst / "images" / split / image_name
    out_label = dst / "labels" / split / label_name
    link_or_copy(item["image_path"], out_image, copy_files)
    segments = mask_to_segments(item["mask_path"], approx_eps=approx_eps, min_points=min_points)
    write_yolo_seg_label(out_label, item["class_id"], segments)


def save_good_image(image_path: Path, split: str, dst: Path, copy_files: bool) -> None:
    product = image_path.parents[2].name
    defect = image_path.parent.name
    image_name = unique_name(image_path, product, defect)
    label_name = Path(image_name).with_suffix(".txt").name
    out_image = dst / "images" / split / image_name
    out_label = dst / "labels" / split / label_name
    link_or_copy(image_path, out_image, copy_files)
    write_empty_label(out_label)


def write_yaml(dst: Path, class_names: dict[str, int]) -> Path:
    ordered_names = [name for name, _ in sorted(class_names.items(), key=lambda kv: kv[1])]
    yaml_path = dst / "mvtec_anomaly_seg.yaml"
    lines = [
        f"path: {dst.as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        f"nc: {len(ordered_names)}",
        f"names: {ordered_names}",
        "",
    ]
    yaml_path.write_text("\n".join(lines), encoding="utf-8")
    return yaml_path


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    ensure_dirs(args.dst)
    anomaly_items, class_names = collect_anomaly_items(args.src, args.task)
    if not anomaly_items:
        raise RuntimeError("No anomaly items found. Check --src path.")

    split_anomaly = split_items(
        anomaly_items,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        rng=rng,
    )

    for split, items in split_anomaly.items():
        for item in items:
            save_item(item, split, args.dst, args.copy, args.approx_eps, args.min_points)

    good = collect_good_images(args.src, args.include_train_good, args.split_test_good)
    for image_path in good["train"]:
        save_good_image(image_path, "train", args.dst, args.copy)

    if good["test_pool"]:
        pooled = list(good["test_pool"])
        rng.shuffle(pooled)
        split_good = split_items(
            [{"image_path": p} for p in pooled],
            train_ratio=0.0,
            val_ratio=args.val_ratio / (args.val_ratio + args.test_ratio),
            test_ratio=args.test_ratio / (args.val_ratio + args.test_ratio),
            rng=rng,
        )
        for item in split_good["val"]:
            save_good_image(item["image_path"], "val", args.dst, args.copy)
        for item in split_good["test"]:
            save_good_image(item["image_path"], "test", args.dst, args.copy)

    yaml_path = write_yaml(args.dst, class_names)

    print(f"Converted {len(anomaly_items)} anomaly images into YOLO-seg format.")
    print(f"Classes: {len(class_names)} -> {list(sorted(class_names, key=class_names.get))}")
    print(f"Dataset YAML: {yaml_path}")


if __name__ == "__main__":
    main()
