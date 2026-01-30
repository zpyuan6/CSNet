from __future__ import annotations

import argparse

from engine import export


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export custom YOLO-like model")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--format", default="onnx")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--simplify", action="store_true")
    parser.add_argument("--opset", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    export(
        weights=args.weights,
        format=args.format,
        imgsz=args.imgsz,
        device=args.device,
        half=args.half,
        dynamic=args.dynamic,
        simplify=args.simplify,
        opset=args.opset,
    )


if __name__ == "__main__":
    main()
