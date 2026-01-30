from __future__ import annotations

import argparse

from engine import evaluate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate custom YOLO-like model")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", default="configs/data/dataset.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="")
    parser.add_argument("--split", default="val", choices=["val", "test"])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    evaluate(
        weights=args.weights,
        data_cfg=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        split=args.split,
    )


if __name__ == "__main__":
    main()
