import argparse
import csv
import time
from pathlib import Path

import torch

from models import register_ultralytics_modules
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark multiple YOLO/custom models on a unified setup and export CSV."
    )
    parser.add_argument(
        "--models-csv",
        type=Path,
        required=True,
        help="CSV with columns: model,weights,series,highlight",
    )
    parser.add_argument("--data", type=str, required=True, help="Dataset yaml path.")
    parser.add_argument("--imgsz", type=int, default=640, help="Evaluation image size.")
    parser.add_argument("--batch", type=int, default=1, help="Batch size for val/predict.")
    parser.add_argument("--device", type=str, default="0", help="Device string.")
    parser.add_argument("--split", type=str, default="val", help="Dataset split for validation.")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations for speed test.")
    parser.add_argument("--iters", type=int, default=50, help="Timed iterations for speed test.")
    parser.add_argument(
        "--speed-image",
        type=str,
        default=None,
        help="Optional single image path for speed benchmarking. If omitted, speed uses first image from dataset split.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("runs/plots/benchmark_results.csv"),
        help="Output CSV path.",
    )
    return parser.parse_args()


def load_model_rows(csv_path: Path) -> list[dict]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"model", "weights", "series", "highlight"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing CSV columns: {sorted(missing)}")

        rows = []
        for row in reader:
            rows.append(
                {
                    "model": row["model"].strip(),
                    "weights": row["weights"].strip(),
                    "series": row["series"].strip(),
                    "highlight": row["highlight"].strip(),
                }
            )
    if not rows:
        raise ValueError("Model CSV is empty.")
    return rows


def resolve_speed_source(data_yaml: str, split: str, override: str | None) -> str:
    if override:
        return override

    from ultralytics.data.utils import check_det_dataset

    data = check_det_dataset(data_yaml)
    split_path = data.get(split)
    if not split_path:
        raise ValueError(f"Split '{split}' not found in dataset config.")

    split_dir = Path(split_path)
    if split_dir.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        for path in split_dir.rglob("*"):
            if path.suffix.lower() in exts:
                return str(path)
        raise ValueError(f"No images found under split directory: {split_dir}")

    raise ValueError("Speed source image could not be resolved. Pass --speed-image explicitly.")


def measure_speed_ms(model: YOLO, image_path: str, imgsz: int, device: str, batch: int, warmup: int, iters: int) -> float:
    predictor_kwargs = dict(imgsz=imgsz, device=device, batch=batch, verbose=False)

    for _ in range(warmup):
        model.predict(source=image_path, **predictor_kwargs)

    if str(device) != "cpu" and torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        model.predict(source=image_path, **predictor_kwargs)
    if str(device) != "cpu" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed * 1000.0 / iters


def evaluate_map(model: YOLO, data: str, imgsz: int, batch: int, device: str, split: str) -> float:
    metrics = model.val(data=data, imgsz=imgsz, batch=batch, device=device, split=split, verbose=False)
    if hasattr(metrics, "box") and hasattr(metrics.box, "map"):
        return float(metrics.box.map)
    if hasattr(metrics, "seg") and hasattr(metrics.box, "map"):
        return float(metrics.box.map)
    raise ValueError("Could not read box mAP50:95 from validation metrics.")


def write_results(rows: list[dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "map", "speed_ms", "series", "highlight", "weights"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    register_ultralytics_modules()
    model_rows = load_model_rows(args.models_csv)
    speed_source = resolve_speed_source(args.data, args.split, args.speed_image)

    results = []
    for row in model_rows:
        model = YOLO(row["weights"])
        map_5095 = evaluate_map(model, args.data, args.imgsz, args.batch, args.device, args.split)
        speed_ms = measure_speed_ms(
            model,
            speed_source,
            args.imgsz,
            args.device,
            args.batch,
            args.warmup,
            args.iters,
        )
        results.append(
            {
                "model": row["model"],
                "map": f"{map_5095:.6f}",
                "speed_ms": f"{speed_ms:.4f}",
                "series": row["series"],
                "highlight": row["highlight"],
                "weights": row["weights"],
            }
        )
        print(f"{row['model']}: map={map_5095:.6f}, speed_ms={speed_ms:.4f}")

    write_results(results, args.out_csv)
    print(f"Saved benchmark CSV to: {args.out_csv}")


if __name__ == "__main__":
    main()
