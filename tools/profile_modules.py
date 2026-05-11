import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import register_ultralytics_modules
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile per-module forward latency for a YOLO/custom model.")
    parser.add_argument("--weights", type=str, required=True, help="Model weights path.")
    parser.add_argument("--model", type=str, default=None, help="Optional model yaml path.")
    parser.add_argument("--image", type=str, required=True, help="Input image path.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--device", type=str, default="0", help="Device string.")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations.")
    parser.add_argument("--iters", type=int, default=50, help="Timed iterations.")
    parser.add_argument("--out-csv", type=Path, default=None, help="Optional output CSV path.")
    return parser.parse_args()


class ModuleProfiler:
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.starts = {}
        self.stats = defaultdict(lambda: {"total_ms": 0.0, "calls": 0, "type": ""})
        self.handles = []

    def _want_module(self, module: torch.nn.Module) -> bool:
        name = module.__class__.__name__
        wanted = {
            "CapsProj",
            "CapsAlign",
            "CapsRoute",
            "CapsRoutev2",
            "CapsuleDetectv5",
            "CapsuleDetectv6",
            "CapsuleDetectv7",
            "CapsuleSegmentv1",
            "CapsuleSegmentv2",
            "C3k2",
            "SPPF",
            "C2PSA",
            "Conv",
        }
        return name in wanted

    def _pre_hook(self, name: str):
        def hook(_module, _inputs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.starts[name] = time.perf_counter()

        return hook

    def _post_hook(self, name: str, cls_name: str):
        def hook(_module, _inputs, _output):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = self.starts.pop(name, None)
            if start is None:
                return
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self.stats[name]["total_ms"] += elapsed_ms
            self.stats[name]["calls"] += 1
            self.stats[name]["type"] = cls_name

        return hook

    def register(self) -> None:
        for name, module in self.model.named_modules():
            if not name or not self._want_module(module):
                continue
            cls_name = module.__class__.__name__
            self.handles.append(module.register_forward_pre_hook(self._pre_hook(name)))
            self.handles.append(module.register_forward_hook(self._post_hook(name, cls_name)))

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def summary(self) -> list[dict]:
        rows = []
        total = sum(v["total_ms"] for v in self.stats.values())
        for name, v in self.stats.items():
            calls = max(v["calls"], 1)
            rows.append(
                {
                    "module": name,
                    "type": v["type"],
                    "total_ms": v["total_ms"],
                    "calls": v["calls"],
                    "avg_ms": v["total_ms"] / calls,
                    "percent": (v["total_ms"] / total * 100.0) if total > 0 else 0.0,
                }
            )
        rows.sort(key=lambda x: x["total_ms"], reverse=True)
        return rows


def write_csv(rows: list[dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["module", "type", "total_ms", "calls", "avg_ms", "percent"])
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict]) -> None:
    print(f"{'Module':<40} {'Type':<22} {'Total ms':>10} {'Calls':>8} {'Avg ms':>10} {'Share':>8}")
    print("-" * 104)
    for row in rows:
        print(
            f"{row['module']:<40} {row['type']:<22} "
            f"{row['total_ms']:>10.3f} {row['calls']:>8} {row['avg_ms']:>10.3f} {row['percent']:>7.2f}%"
        )


def main() -> None:
    args = parse_args()
    register_ultralytics_modules()

    if args.model:
        yolo = YOLO(args.model).load(args.weights)
    else:
        yolo = YOLO(args.weights)

    predictor_args = dict(source=args.image, imgsz=args.imgsz, device=args.device, verbose=False)

    core_model = yolo.model
    core_model.eval()
    profiler = ModuleProfiler(core_model)
    profiler.register()

    for _ in range(args.warmup):
        yolo.predict(**predictor_args)

    for _ in range(args.iters):
        yolo.predict(**predictor_args)

    profiler.remove()
    rows = profiler.summary()
    print_summary(rows)

    if args.out_csv is not None:
        write_csv(rows, args.out_csv)
        print(f"\nSaved CSV to: {args.out_csv}")


if __name__ == "__main__":
    main()
