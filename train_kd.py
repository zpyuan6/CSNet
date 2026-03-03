from __future__ import annotations

import argparse
from pathlib import Path

from engine import train_distill


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Two-stage KD training for capsule YOLO models")
    parser.add_argument("--model", default="configs/model/yolo26_capsneckhead_v3.yaml")
    parser.add_argument("--data", default="configs/data/coco.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--cache", action="store_true")

    parser.add_argument(
        "--pretrained",
        default="",
        help="Initial student weights path (recommended: best capsule checkpoint).",
    )

    # Stage schedule
    parser.add_argument("--epochs_s", type=int, default=80, help="Stage-1 epochs with smaller teacher.")
    parser.add_argument("--epochs_x", type=int, default=120, help="Stage-2 epochs with larger teacher.")
    parser.add_argument("--teacher_s", default="yolo26s.pt", help="Stage-1 teacher weights.")
    parser.add_argument("--teacher_x", default="yolo26x.pt", help="Stage-2 teacher weights.")

    # KD hypers
    parser.add_argument("--kd_cls_s", type=float, default=0.5, help="Stage-1 cls KD weight.")
    parser.add_argument("--kd_box_s", type=float, default=1.0, help="Stage-1 box KD weight.")
    parser.add_argument("--kd_cls_x", type=float, default=0.3, help="Stage-2 cls KD weight.")
    parser.add_argument("--kd_box_x", type=float, default=0.6, help="Stage-2 box KD weight.")
    parser.add_argument("--kd_temp", type=float, default=3.0, help="KD temperature.")
    parser.add_argument("--kd_warmup_epochs", type=int, default=10, help="KD warmup epochs per stage.")

    # Optimizer/scheduler
    parser.add_argument("--lr0", type=float, default=5e-4)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--warmup_epochs", type=float, default=3.0)
    parser.add_argument("--warmup_bias_lr", type=float, default=0.0)
    parser.add_argument("--weight_decay", type=float, default=3e-4)
    parser.add_argument(
        "--amp",
        type=lambda x: str(x).lower() in {"1", "true", "yes", "y"},
        default=True,
    )

    parser.add_argument("--project", default="runs/train")
    parser.add_argument("--name", default="caps_kd")
    parser.add_argument("--exist-ok", action="store_true")
    return parser


def _find_last_weight(stage_name: str) -> str:
    candidates = sorted(
        Path(".").glob(f"**/{stage_name}/weights/last.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"Cannot find stage checkpoint for '{stage_name}'.")
    return str(candidates[0])


def main() -> None:
    args = build_parser().parse_args()
    if not args.teacher_s or not args.teacher_x:
        raise ValueError("Both --teacher_s and --teacher_x are required.")

    common = dict(
        data_cfg=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        cache=args.cache,
        workers=args.workers,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        warmup_epochs=args.warmup_epochs,
        warmup_bias_lr=args.warmup_bias_lr,
        weight_decay=args.weight_decay,
        amp=args.amp,
        project=args.project,
        exist_ok=args.exist_ok,
    )

    stage1_name = f"{args.name}_kd_s"
    print(f"[KD stage1] teacher={args.teacher_s}, epochs={args.epochs_s}")
    train_distill(
        model_cfg=args.model,
        teacher=args.teacher_s,
        pretrained=args.pretrained or False,
        epochs=args.epochs_s,
        kd_cls=args.kd_cls_s,
        kd_box=args.kd_box_s,
        kd_temp=args.kd_temp,
        kd_warmup_epochs=args.kd_warmup_epochs,
        name=stage1_name,
        **common,
    )

    stage1_last = _find_last_weight(stage1_name)
    print(f"[KD stage2] teacher={args.teacher_x}, epochs={args.epochs_x}, from={stage1_last}")
    train_distill(
        model_cfg=stage1_last,
        teacher=args.teacher_x,
        pretrained=False,
        epochs=args.epochs_x,
        kd_cls=args.kd_cls_x,
        kd_box=args.kd_box_x,
        kd_temp=args.kd_temp,
        kd_warmup_epochs=args.kd_warmup_epochs,
        name=args.name,
        **common,
    )


if __name__ == "__main__":
    main()

