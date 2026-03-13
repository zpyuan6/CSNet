from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from .dataset_analysis import analyze_dataset_with_probes
    from .batch import train_all_concepts
    from .probe import train_concept_probe
    from .report import save_supervised_overlay, save_supervised_report
    from .tcav import audit_single_detection_with_probes, save_tcav_json
    from .template import export_concept_annotation_template
except ImportError:
    from tools.audit_caps.supervised_tcav.dataset_analysis import analyze_dataset_with_probes
    from tools.audit_caps.supervised_tcav.batch import train_all_concepts
    from tools.audit_caps.supervised_tcav.probe import train_concept_probe
    from tools.audit_caps.supervised_tcav.report import save_supervised_overlay, save_supervised_report
    from tools.audit_caps.supervised_tcav.tcav import audit_single_detection_with_probes, save_tcav_json
    from tools.audit_caps.supervised_tcav.template import export_concept_annotation_template


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Supervised TCAV-style capsule auditing")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train-probe", help="Train a linear concept probe from supervised annotations")
    p_train.add_argument("--model", required=True)
    p_train.add_argument("--annotations", required=True)
    p_train.add_argument("--concept", required=True)
    p_train.add_argument("--layer", required=True)
    p_train.add_argument("--out", required=True)
    p_train.add_argument("--imgsz", type=int, default=640)
    p_train.add_argument("--device", default="")
    p_train.add_argument("--epochs", type=int, default=200)
    p_train.add_argument("--lr", type=float, default=1e-2)
    p_train.add_argument("--weight-decay", type=float, default=1e-4)

    p_train_all = sub.add_parser("train-all", help="Train probes for all concepts found in the annotation CSV")
    p_train_all.add_argument("--model", required=True)
    p_train_all.add_argument("--annotations", required=True)
    p_train_all.add_argument("--outdir", required=True)
    p_train_all.add_argument("--default-layer", default=None)
    p_train_all.add_argument("--imgsz", type=int, default=640)
    p_train_all.add_argument("--device", default="")
    p_train_all.add_argument("--epochs", type=int, default=200)
    p_train_all.add_argument("--lr", type=float, default=1e-2)
    p_train_all.add_argument("--weight-decay", type=float, default=1e-4)

    p_single = sub.add_parser("single", help="Audit a single detection with one or more trained probes")
    p_single.add_argument("--model", required=True)
    p_single.add_argument("--image", required=True)
    p_single.add_argument("--probes", nargs="+", required=True)
    p_single.add_argument("--out", required=True)
    p_single.add_argument("--imgsz", type=int, default=640)
    p_single.add_argument("--device", default="")
    p_single.add_argument("--class-id", type=int, default=None)
    p_single.add_argument("--det-index", type=int, default=None)

    p_ds = sub.add_parser("analyze-dataset", help="Run dataset-level supervised TCAV analysis")
    p_ds.add_argument("--model", required=True)
    p_ds.add_argument("--outdir", required=True)
    p_ds.add_argument("--annotations", default=None)
    p_ds.add_argument("--data", default=None)
    p_ds.add_argument("--split", default="val")
    p_ds.add_argument("--manifest", default=None)
    p_ds.add_argument("--probes", nargs="*", default=None)
    p_ds.add_argument("--imgsz", type=int, default=640)
    p_ds.add_argument("--device", default="")
    p_ds.add_argument("--class-id", type=int, default=None)
    p_ds.add_argument("--limit", type=int, default=None)

    p_template = sub.add_parser("template", help="Export a concept annotation CSV template")
    p_template.add_argument("--out-csv", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "train-probe":
        train_concept_probe(
            model_path=args.model,
            annotations_csv=args.annotations,
            concept=args.concept,
            layer=args.layer,
            out_path=args.out,
            imgsz=args.imgsz,
            device=args.device,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        return

    if args.command == "train-all":
        train_all_concepts(
            model_path=args.model,
            annotations_csv=args.annotations,
            outdir=args.outdir,
            default_layer=args.default_layer,
            imgsz=args.imgsz,
            device=args.device,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        return

    if args.command == "single":
        result = audit_single_detection_with_probes(
            model_path=args.model,
            image_path=args.image,
            probe_paths=args.probes,
            imgsz=args.imgsz,
            device=args.device,
            class_id=args.class_id,
            det_index=args.det_index,
        )
        save_tcav_json(result, args.out)
        report_dir = Path(args.out).with_suffix("")
        save_supervised_report(result, report_dir)
        save_supervised_overlay(result, report_dir)
        return

    if args.command == "analyze-dataset":
        analyze_dataset_with_probes(
            model_path=args.model,
            outdir=args.outdir,
            annotations_csv=args.annotations,
            data_yaml=args.data,
            split=args.split,
            manifest=args.manifest,
            probes=args.probes,
            imgsz=args.imgsz,
            device=args.device,
            class_id=args.class_id,
            limit=args.limit,
        )
        return

    if args.command == "template":
        export_concept_annotation_template(args.out_csv)


if __name__ == "__main__":
    main()
