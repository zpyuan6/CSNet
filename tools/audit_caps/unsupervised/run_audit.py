from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from .atlas import build_concept_atlas
    from .graph import build_concept_class_graph
    from .label_concepts import export_concept_label_template
    from .relevance import audit_single_image, save_audit_json
    from .report import save_audit_report
except ImportError:
    from tools.audit_caps.unsupervised.atlas import build_concept_atlas
    from tools.audit_caps.unsupervised.graph import build_concept_class_graph
    from tools.audit_caps.unsupervised.label_concepts import export_concept_label_template
    from tools.audit_caps.unsupervised.relevance import audit_single_image, save_audit_json
    from tools.audit_caps.unsupervised.report import save_audit_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit capsule concepts in custom YOLO models")
    sub = parser.add_subparsers(dest="command", required=True)

    p_single = sub.add_parser("single", help="Audit a single image")
    p_single.add_argument("--model", required=True)
    p_single.add_argument("--image", required=True)
    p_single.add_argument("--outdir", required=True)
    p_single.add_argument("--imgsz", type=int, default=640)
    p_single.add_argument("--device", default="")
    p_single.add_argument("--class-id", type=int, default=None)
    p_single.add_argument("--det-index", type=int, default=None)
    p_single.add_argument("--topk", type=int, default=5)

    p_atlas = sub.add_parser("atlas", help="Build a concept atlas over a dataset split")
    p_atlas.add_argument("--model", required=True)
    p_atlas.add_argument("--data", required=True)
    p_atlas.add_argument("--outdir", required=True)
    p_atlas.add_argument("--split", default="val")
    p_atlas.add_argument("--imgsz", type=int, default=640)
    p_atlas.add_argument("--device", default="")
    p_atlas.add_argument("--limit", type=int, default=100)
    p_atlas.add_argument("--topk-per-image", type=int, default=3)

    p_graph = sub.add_parser("graph", help="Build concept-class graph from atlas json")
    p_graph.add_argument("--atlas-json", required=True)
    p_graph.add_argument("--out", required=True)
    p_graph.add_argument("--topn", type=int, default=10)

    p_labels = sub.add_parser("labels", help="Export a label template CSV for concepts")
    p_labels.add_argument("--atlas-json", required=True)
    p_labels.add_argument("--out-csv", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "single":
        result = audit_single_image(
            model_path=args.model,
            image_path=args.image,
            imgsz=args.imgsz,
            device=args.device,
            class_id=args.class_id,
            det_index=args.det_index,
            topk=args.topk,
        )
        outdir = Path(args.outdir)
        save_audit_json(result, outdir / "audit.json")
        save_audit_report(result, outdir, topk=min(args.topk, 3))
        return

    if args.command == "atlas":
        build_concept_atlas(
            model_path=args.model,
            data_yaml=args.data,
            outdir=args.outdir,
            split=args.split,
            imgsz=args.imgsz,
            limit=args.limit,
            topk_per_image=args.topk_per_image,
            device=args.device,
        )
        return

    if args.command == "graph":
        build_concept_class_graph(args.atlas_json, args.out, topn=args.topn)
        return

    if args.command == "labels":
        export_concept_label_template(args.atlas_json, args.out_csv)


if __name__ == "__main__":
    main()
