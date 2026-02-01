from __future__ import annotations

import argparse

from engine import train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train deformable capsule YOLO-like model")
    parser.add_argument("--model", default="configs/model/deformable_caps.yaml")
    parser.add_argument("--data", default="configs/data/nematode.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="")
    parser.add_argument("--project", default="runs/train")
    parser.add_argument("--name", default="deformable_caps")
    parser.add_argument("--exist-ok", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    train(
        model_cfg=args.model,
        data_cfg=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
    )


if __name__ == "__main__":
    main()
