from __future__ import annotations

import argparse
from pathlib import Path

from engine import train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train YOLO26 with capsule modules (head/neck)")
    parser.add_argument("--model", default="configs/model/yolo26_capsneck.yaml")
    parser.add_argument("--data", default="configs/data/coco.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="")
    parser.add_argument(
        "--pretrained",
        default="",
        help="Path to pretrained weights (leave empty to train from scratch).",
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Cache images for faster training (requires RAM).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of dataloader worker processes.",
    )
    parser.add_argument(
        "--optimizer",
        default="auto",
        help="Optimizer name (e.g., SGD, AdamW, RMSProp, MuSGD, auto).",
    )

    # LR/scheduler stability knobs for capsule routing blocks.
    parser.add_argument("--lr0", type=float, default=5e-4, help="Initial learning rate.")
    parser.add_argument("--lrf", type=float, default=0.01, help="Final LR factor (lr_final = lr0 * lrf).")
    parser.add_argument("--warmup_epochs", type=float, default=3.0, help="Warmup epochs.")
    parser.add_argument("--warmup_bias_lr", type=float, default=0.0, help="Bias LR during warmup.")
    parser.add_argument("--weight_decay", type=float, default=3e-4, help="Weight decay.")
    parser.add_argument(
        "--amp",
        type=lambda x: str(x).lower() in {"1", "true", "yes", "y"},
        default=True,
        help="Enable AMP (mixed precision).",
    )

    # Optional two-stage freeze schedule.
    parser.add_argument(
        "--enable_freeze_pretrain",
        action="store_true",
        help="Enable two-stage training: freeze backbone first, then unfreeze.",
    )
    parser.add_argument(
        "--freeze_backbone_epochs",
        type=int,
        default=10,
        help="Backbone frozen epochs when --enable_freeze_pretrain is set.",
    )
    parser.add_argument(
        "--freeze_backbone_layers",
        type=int,
        default=11,
        help="Number of early model layers to freeze as backbone (Ultralytics freeze arg).",
    )

    parser.add_argument("--project", default="runs/train")
    parser.add_argument("--name", default="yolo26_capsneck")
    parser.add_argument("--exist-ok", action="store_true")

    return parser


def _find_last_weight(stage_name: str) -> str:
    candidates = sorted(
        Path(".").glob(f"**/{stage_name}/weights/last.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"Cannot find stage-1 checkpoint for '{stage_name}'.")
    return str(candidates[0])


def main() -> None:
    args = build_parser().parse_args()

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

    freeze_epochs = max(0, min(args.freeze_backbone_epochs, args.epochs)) if args.enable_freeze_pretrain else 0
    if freeze_epochs == 0:
        train(
            model_cfg=args.model,
            pretrained=args.pretrained or False,
            epochs=args.epochs,
            name=args.name,
            **common,
        )
        return

    stage1_name = f"{args.name}_freeze"
    print(f"[stage1] freeze backbone for {freeze_epochs} epochs (layers={args.freeze_backbone_layers})")
    train(
        model_cfg=args.model,
        pretrained=args.pretrained or False,
        epochs=freeze_epochs,
        freeze=args.freeze_backbone_layers,
        name=stage1_name,
        **common,
    )

    remain_epochs = args.epochs - freeze_epochs
    if remain_epochs <= 0:
        return

    stage1_last = _find_last_weight(stage1_name)
    print(f"[stage2] unfreeze and finetune for {remain_epochs} epochs from {stage1_last}")
    train(
        model_cfg=stage1_last,
        pretrained=False,
        epochs=remain_epochs,
        name=args.name,
        **common,
    )


if __name__ == "__main__":
    main()
