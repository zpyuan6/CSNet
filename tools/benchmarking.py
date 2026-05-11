import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import register_ultralytics_modules
from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils.torch_utils import get_flops, get_num_params


DEFAULT_MODELS = [
    {
        "name": "YOLO26n",
        "model": "yolo26n.pt",
        "weights": "",
    },
    {
        "name": "YOLO26s",
        "model": "yolo26s.pt",
        "weights": "",
    },
    {
        "name": "YOLO26m",
        "model": "yolo26m.pt",
        "weights": "",
    },
    {
        "name": "YOLO26l",
        "model": "yolo26l.pt",
        "weights": "",
    },
    {
        "name": "YOLO26x",
        "model": "yolo26x.pt",
        "weights": "",
    },
    {
        "name": "Capsule-YOLO26n",
        "model": r"configs/det_model/yolo26_capsneckhead_v8.yaml",
        "weights": r"runs\CapsNeckHeadv6_SegV2_AuxDet_Mild\weights\best.pt",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified local benchmarking for YOLO26 and custom capsule models.")
    parser.add_argument("--data", type=str, default="configs/data/coco.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--val-workers", type=int, default=0, help="Dataloader workers for val. Use 0 on Windows.")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--val-device", type=str, default="0", help="Device used for validation mAP.")
    parser.add_argument("--speed-image", type=str, required=True, help="Single image path used for local speed tests.")
    parser.add_argument("--speed-iters", type=int, default=5, help="Number of repeated speed runs.")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations inside each speed run.")
    parser.add_argument(
        "--predict-iters",
        type=int,
        default=10,
        help="Timed predict iterations inside each speed run.",
    )
    parser.add_argument(
        "--model-spec",
        action="append",
        default=[],
        help="Optional spec NAME=MODEL[,WEIGHTS]. If omitted, default YOLO26 n/s/m/l/x + capsule model are used.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("runs/benchmarking_results.csv"),
    )
    return parser.parse_args()


def parse_model_specs(specs: list[str]) -> list[dict]:
    rows = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Invalid --model-spec '{spec}'. Expected NAME=MODEL[,WEIGHTS].")
        name, rest = spec.split("=", 1)
        parts = [p.strip() for p in rest.split(",")]
        rows.append(
            {
                "name": name.strip(),
                "model": parts[0],
                "weights": parts[1] if len(parts) > 1 else "",
            }
        )
    return rows


def load_model(model_path: str, weights_path: str) -> YOLO:
    model = YOLO(model_path)
    if weights_path:
        model = model.load(weights_path)
    return model


def get_map_from_stats(stats: dict[str, Any], metrics: Any) -> tuple[float | None, float | None]:
    map50 = stats.get("metrics/mAP50(B)")
    map5095 = stats.get("metrics/mAP50-95(B)")
    if map50 is None:
        results_dict = getattr(metrics, "results_dict", {}) or {}
        map50 = results_dict.get("metrics/mAP50(B)")
    if map5095 is None:
        results_dict = getattr(metrics, "results_dict", {}) or {}
        map5095 = results_dict.get("metrics/mAP50-95(B)")
    if map50 is None and hasattr(metrics, "box"):
        map50 = float(metrics.box.map50)
    if map5095 is None and hasattr(metrics, "box"):
        map5095 = float(metrics.box.map)
    return (float(map50) if map50 is not None else None, float(map5095) if map5095 is not None else None)


def set_model_end2end(model: YOLO, enabled: bool) -> None:
    core = model.model
    if hasattr(core, "end2end"):
        core.end2end = enabled
    if hasattr(core, "set_head_attr"):
        try:
            core.set_head_attr()
        except TypeError:
            pass


def run_val_stats(
    model_path: str,
    weights_path: str,
    data: str,
    imgsz: int,
    batch: int,
    workers: int,
    device: str,
    split: str,
    end2end: bool,
) -> tuple[dict[str, Any], Any]:
    model = load_model(model_path, weights_path)
    set_model_end2end(model, end2end)
    custom = {"rect": True}
    args = {
        **model.overrides,
        **custom,
        "data": data,
        "imgsz": imgsz,
        "batch": batch,
        "workers": workers,
        "device": device,
        "split": split,
        "save_json": True,
        "verbose": False,
        "mode": "val",
    }
    validator = model._smart_load("validator")(args=args, _callbacks=model.callbacks)
    stats = validator(model=model.model)
    return stats or {}, validator.metrics


def evaluate_map_pair(
    model_path: str,
    weights_path: str,
    data: str,
    imgsz: int,
    batch: int,
    workers: int,
    device: str,
    split: str,
) -> tuple[float | None, float | None, float | None, float | None]:
    # standard path
    stats_std, metrics_std = run_val_stats(
        model_path, weights_path, data, imgsz, batch, workers, device, split, end2end=False
    )
    map50, map5095 = get_map_from_stats(stats_std, metrics_std)

    # e2e path
    stats_e2e, metrics_e2e = run_val_stats(
        model_path, weights_path, data, imgsz, batch, workers, device, split, end2end=True
    )
    e2e_map50, e2e_map5095 = get_map_from_stats(stats_e2e, metrics_e2e)
    return map50, map5095, e2e_map50, e2e_map5095


def measure_predict_ms(
    model: YOLO,
    image_path: str,
    device: str | int,
    imgsz: int,
    warmup: int,
    predict_iters: int,
) -> float:
    for _ in range(warmup):
        model.predict(image_path, imgsz=imgsz, device=device, verbose=False)

    use_cuda = str(device) != "cpu" and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(predict_iters):
        model.predict(image_path, imgsz=imgsz, device=device, verbose=False)

    if use_cuda:
        torch.cuda.synchronize()

    return (time.perf_counter() - start) * 1000.0 / predict_iters


def repeated_speed(
    model: YOLO,
    image_path: str,
    device: str | int,
    imgsz: int,
    warmup: int,
    predict_iters: int,
    speed_iters: int,
) -> tuple[float, float]:
    samples = [
        measure_predict_ms(model, image_path, device, imgsz, warmup, predict_iters)
        for _ in range(speed_iters)
    ]
    mean_ms = statistics.mean(samples)
    std_ms = statistics.pstdev(samples) if len(samples) > 1 else 0.0
    return mean_ms, std_ms


def main() -> None:
    args = parse_args()
    register_ultralytics_modules()

    models = parse_model_specs(args.model_spec) if args.model_spec else DEFAULT_MODELS
    rows = []

    for spec in models:
        print(f"=== {spec['name']} ===")
        model_for_meta = load_model(spec["model"], spec["weights"])

        params_m = get_num_params(model_for_meta.model) / 1e6
        flops_b = get_flops(model_for_meta.model, args.imgsz)

        map50, map5095, e2e_map50, e2e_map5095 = evaluate_map_pair(
            spec["model"],
            spec["weights"],
            args.data,
            args.imgsz,
            args.batch,
            args.val_workers,
            args.val_device,
            args.split,
        )

        gpu_mean = gpu_std = None
        if torch.cuda.is_available():
            model_for_gpu = load_model(spec["model"], spec["weights"])
            gpu_mean, gpu_std = repeated_speed(
                model_for_gpu,
                args.speed_image,
                0,
                args.imgsz,
                args.warmup,
                args.predict_iters,
                args.speed_iters,
            )

        model_for_cpu = load_model(spec["model"], spec["weights"])
        cpu_mean, cpu_std = repeated_speed(
            model_for_cpu,
            args.speed_image,
            "cpu",
            args.imgsz,
            args.warmup,
            args.predict_iters,
            args.speed_iters,
        )

        row = {
            "model": spec["name"],
            "size_pixels": args.imgsz,
            "map50": f"{map50:.6f}" if map50 is not None else "",
            "map50_95": f"{map5095:.6f}" if map5095 is not None else "",
            "e2e_map50": f"{e2e_map50:.6f}" if e2e_map50 is not None else "",
            "e2e_map50_95": f"{e2e_map5095:.6f}" if e2e_map5095 is not None else "",
            "local_gpu_speed_mean_ms": f"{gpu_mean:.4f}" if gpu_mean is not None else "",
            "local_gpu_speed_std_ms": f"{gpu_std:.4f}" if gpu_std is not None else "",
            "local_cpu_speed_mean_ms": f"{cpu_mean:.4f}",
            "local_cpu_speed_std_ms": f"{cpu_std:.4f}",
            "params_m": f"{params_m:.6f}",
            "flops_b": f"{flops_b:.1f}",
            "model_path": spec["model"],
            "weights_path": spec["weights"],
        }
        rows.append(row)

        print(row)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "size_pixels",
                "map50",
                "map50_95",
                "e2e_map50",
                "e2e_map50_95",
                "local_gpu_speed_mean_ms",
                "local_gpu_speed_std_ms",
                "local_cpu_speed_mean_ms",
                "local_cpu_speed_std_ms",
                "params_m",
                "flops_b",
                "model_path",
                "weights_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved CSV to {args.out_csv}")


if __name__ == "__main__":
    main()
