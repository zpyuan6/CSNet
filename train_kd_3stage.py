from __future__ import annotations

import argparse
from pathlib import Path

from engine import evaluate, train, train_distill


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Long-schedule two-stage KD + high-res fine-tune")
    parser.add_argument("--model", default="configs/model/yolo26_capsneckhead_v4.yaml")
    parser.add_argument("--data", default="configs/data/coco.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--device", default="")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--pretrained", default="", help="Initial student weights path.")

    # Stage 1: strong KD with smaller teacher.
    parser.add_argument("--teacher_s", default="yolo26s.pt")
    parser.add_argument("--epochs_s", type=int, default=80)
    parser.add_argument("--kd_cls_s", type=float, default=0.8)
    parser.add_argument("--kd_box_s", type=float, default=1.2)

    # Stage 2: weaker KD with larger teacher.
    parser.add_argument("--teacher_x", default="yolo26x.pt")
    parser.add_argument("--epochs_x", type=int, default=80)
    parser.add_argument("--kd_cls_x", type=float, default=0.35)
    parser.add_argument("--kd_box_x", type=float, default=0.55)

    # Shared KD controls.
    parser.add_argument("--kd_temp", type=float, default=3.0)
    parser.add_argument("--kd_warmup_epochs", type=int, default=10)

    # Stage 3: high-res fine-tune (no KD).
    parser.add_argument("--epochs_ft", type=int, default=30)
    parser.add_argument("--imgsz_ft", type=int, default=704)
    parser.add_argument("--batch_ft", type=int, default=48)

    # Optimizer/scheduler.
    parser.add_argument("--lr0", type=float, default=4e-4)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--warmup_epochs", type=float, default=5.0)
    parser.add_argument("--warmup_bias_lr", type=float, default=0.0)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument(
        "--amp",
        type=lambda x: str(x).lower() in {"1", "true", "yes", "y"},
        default=True,
    )

    parser.add_argument("--project", default="runs/train")
    parser.add_argument("--name", default="caps_longkd")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--val_imgsz", type=int, default=640, help="Unified validation image size.")
    parser.add_argument("--val_batch", type=int, default=0, help="Validation batch size. 0 means auto from train batch.")
    return parser


def _find_last_weight(project: str, stage_name: str) -> str:
    exact = Path(project) / stage_name / "weights" / "last.pt"
    if exact.exists():
        return str(exact)

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

    common = dict(
        data_cfg=args.data,
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

    stage1_name = f"{args.name}_s"
    print(f"[stage1 KD] teacher={args.teacher_s}, epochs={args.epochs_s}, imgsz={args.imgsz}, batch={args.batch}")
    train_distill(
        model_cfg=args.model,
        teacher=args.teacher_s,
        pretrained=args.pretrained or False,
        epochs=args.epochs_s,
        imgsz=args.imgsz,
        batch=args.batch,
        kd_cls=args.kd_cls_s,
        kd_box=args.kd_box_s,
        kd_temp=args.kd_temp,
        kd_warmup_epochs=args.kd_warmup_epochs,
        name=stage1_name,
        **common,
    )
    stage1_last = _find_last_weight(args.project, stage1_name)
    run_unified_val(stage1_last, "stage1", args.batch)

    stage2_name = f"{args.name}_x"
    print(f"[stage2 KD] teacher={args.teacher_x}, epochs={args.epochs_x}, from={stage1_last}")
    train_distill(
        model_cfg=stage1_last,
        teacher=args.teacher_x,
        pretrained=False,
        epochs=args.epochs_x,
        imgsz=args.imgsz,
        batch=args.batch,
        kd_cls=args.kd_cls_x,
        kd_box=args.kd_box_x,
        kd_temp=args.kd_temp,
        kd_warmup_epochs=args.kd_warmup_epochs,
        name=stage2_name,
        **common,
    )
    stage2_last = _find_last_weight(args.project, stage2_name)
    run_unified_val(stage2_last, "stage2", args.batch)

    if args.epochs_ft > 0:
        print(
            f"[stage3 FT] epochs={args.epochs_ft}, from={stage2_last}, "
            f"imgsz={args.imgsz_ft}, batch={args.batch_ft}"
        )
        train(
            model_cfg=stage2_last,
            data_cfg=args.data,
            pretrained=False,
            epochs=args.epochs_ft,
            imgsz=args.imgsz_ft,
            batch=args.batch_ft,
            device=args.device,
            cache=args.cache,
            workers=args.workers,
            optimizer=args.optimizer,
            lr0=max(args.lr0 * 0.6, 1e-5),
            lrf=args.lrf,
            warmup_epochs=max(1.0, args.warmup_epochs * 0.5),
            warmup_bias_lr=args.warmup_bias_lr,
            weight_decay=args.weight_decay,
            amp=args.amp,
            project=args.project,
            name=args.name,
            exist_ok=args.exist_ok,
        )
        final_last = _find_last_weight(args.project, args.name)
        run_unified_val(final_last, "stage3", args.batch_ft)
    else:
        run_unified_val(stage2_last, "final", args.batch)


if __name__ == "__main__":
    main()
    def run_unified_val(weights: str, tag: str, batch_fallback: int) -> None:
        val_batch = args.val_batch if args.val_batch > 0 else batch_fallback
        print(f"[{tag} val] weights={weights}, imgsz={args.val_imgsz}, batch={val_batch}")
        evaluate(
            weights=weights,
            data_cfg=args.data,
            imgsz=args.val_imgsz,
            batch=val_batch,
            device=args.device,
            workers=args.workers,
            project=args.project,
            name=f"{args.name}_{tag}_val640",
            exist_ok=True,
        )
