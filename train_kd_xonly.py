from __future__ import annotations

import argparse
from pathlib import Path

from engine import evaluate, train, train_distill


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Single-teacher YOLO26x KD + optional high-res fine-tune")
    parser.add_argument("--model", default="configs/model/yolo26_capsneckhead_v4.yaml")
    parser.add_argument("--data", default="configs/data/coco.yaml")
    parser.add_argument("--pretrained", default="", help="Initial student weights path.")

    parser.add_argument("--teacher_x", default="yolo26x.pt")
    parser.add_argument("--epochs_kd", type=int, default=140)
    parser.add_argument("--kd_cls", type=float, default=0.45)
    parser.add_argument("--kd_box", type=float, default=0.80)
    parser.add_argument("--kd_temp", type=float, default=3.0)
    parser.add_argument("--kd_warmup_epochs", type=int, default=12)

    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--epochs_ft", type=int, default=30, help="0 to disable stage2 fine-tune.")
    parser.add_argument("--imgsz_ft", type=int, default=704)
    parser.add_argument("--batch_ft", type=int, default=48)

    parser.add_argument("--device", default="")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--lr0", type=float, default=4e-4)
    parser.add_argument("--lrf", type=float, default=0.008)
    parser.add_argument("--warmup_epochs", type=float, default=5.0)
    parser.add_argument("--warmup_bias_lr", type=float, default=0.0)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument(
        "--amp",
        type=lambda x: str(x).lower() in {"1", "true", "yes", "y"},
        default=True,
    )

    parser.add_argument("--val_imgsz", type=int, default=640)
    parser.add_argument("--val_batch", type=int, default=0, help="0 means follow train batch.")
    parser.add_argument("--project", default="runs/train")
    parser.add_argument("--name", default="caps_kd_xonly")
    parser.add_argument("--exist-ok", action="store_true")
    return parser


def _find_last_weight(project: str, run_name: str) -> str:
    exact = Path(project) / run_name / "weights" / "last.pt"
    if exact.exists():
        return str(exact)
    candidates = sorted(
        Path(".").glob(f"**/{run_name}/weights/last.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"Cannot find stage checkpoint for '{run_name}'.")
    return str(candidates[0])


def main() -> None:
    args = build_parser().parse_args()

    def run_val640(weights: str, tag: str, batch_fallback: int) -> None:
        val_batch = args.val_batch if args.val_batch > 0 else batch_fallback
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

    kd_name = f"{args.name}_kd_x"
    train_distill(
        model_cfg=args.model,
        teacher=args.teacher_x,
        pretrained=args.pretrained or False,
        epochs=args.epochs_kd,
        imgsz=args.imgsz,
        batch=args.batch,
        kd_cls=args.kd_cls,
        kd_box=args.kd_box,
        kd_temp=args.kd_temp,
        kd_warmup_epochs=args.kd_warmup_epochs,
        name=kd_name,
        **common,
    )
    kd_last = _find_last_weight(args.project, kd_name)
    run_val640(kd_last, "kd", args.batch)

    if args.epochs_ft > 0:
        train(
            model_cfg=kd_last,
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
        run_val640(final_last, "final", args.batch_ft)


if __name__ == "__main__":
    main()
