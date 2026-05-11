from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from .atlas import build_concept_atlas
    from .atlas import relabel_concept_atlas
    from .baseline_adapter import load_baseline_sample_attribution
    from .graph import build_layered_concept_graph
    from .io import save_json
    from .labels import export_concept_labels_csv
    from .single import run_single_audit
    from .tree import build_inference_tree
except ImportError:
    from tools.audit_caps.unsupervised_audit.atlas import build_concept_atlas
    from tools.audit_caps.unsupervised_audit.atlas import relabel_concept_atlas
    from tools.audit_caps.unsupervised_audit.baseline_adapter import load_baseline_sample_attribution
    from tools.audit_caps.unsupervised_audit.graph import build_layered_concept_graph
    from tools.audit_caps.unsupervised_audit.io import save_json
    from tools.audit_caps.unsupervised_audit.labels import export_concept_labels_csv
    from tools.audit_caps.unsupervised_audit.single import run_single_audit
    from tools.audit_caps.unsupervised_audit.tree import build_inference_tree


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unsupervised capsule audit based on CRP/L-CRP-style design"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_single = sub.add_parser("single", help="Run sample-level concept attribution")
    p_single.add_argument("--model", required=True)
    p_single.add_argument("--image", required=True)
    p_single.add_argument("--outdir", required=True)
    p_single.add_argument("--imgsz", type=int, default=640)
    p_single.add_argument("--device", default="")
    p_single.add_argument("--class-id", type=int, default=None)
    p_single.add_argument("--det-index", type=int, default=None)
    p_single.add_argument("--topk", type=int, default=5)
    p_single.add_argument("--bbox-weight", type=float, default=0.05)
    p_single.add_argument("--attribution-method", default="capsule_lcrp")
    p_single.add_argument("--layers", nargs="+", default=None)

    p_atlas = sub.add_parser("atlas", help="Build a CRP-style concept reference atlas")
    p_atlas.add_argument("--model")
    p_atlas.add_argument("--data")
    p_atlas.add_argument("--outdir", required=True)
    p_atlas.add_argument("--split", default="val")
    p_atlas.add_argument("--imgsz", type=int, default=640)
    p_atlas.add_argument("--device", default="")
    p_atlas.add_argument("--limit", type=int, default=100)
    p_atlas.add_argument("--topk-per-image", type=int, default=3)
    p_atlas.add_argument("--attribution-method", default="capsule_lcrp")
    p_atlas.add_argument("--layers", nargs="+", default=None)
    p_atlas.add_argument("--labels-csv", default=None)
    p_atlas.add_argument("--no-refresh-visuals", action="store_true")

    p_graph = sub.add_parser("graph", help="Build a layered concept graph")
    p_graph.add_argument("--atlas-json", required=True)
    p_graph.add_argument("--out", required=True)

    p_tree = sub.add_parser("tree", help="Build a sample-level inference tree")
    p_tree.add_argument("--audit-json", required=True)
    p_tree.add_argument("--atlas-json", required=True)
    p_tree.add_argument("--out", required=True)

    p_labels = sub.add_parser("labels", help="Export concept label template CSV from atlas")
    p_labels.add_argument("--atlas-json", required=True)
    p_labels.add_argument("--out-csv", required=True)

    p_adapt = sub.add_parser(
        "adapt-baseline",
        help="Convert an existing unsupervised audit.json into the new shared schema",
    )
    p_adapt.add_argument("--audit-json", required=True)
    p_adapt.add_argument("--out", required=True)
    p_adapt.add_argument("--model", default="")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "single":
        run_single_audit(
            model_path=args.model,
            image_path=args.image,
            outdir=args.outdir,
            imgsz=args.imgsz,
            device=args.device,
            class_id=args.class_id,
            det_index=args.det_index,
            topk=args.topk,
            bbox_weight=args.bbox_weight,
            attribution_method=args.attribution_method,
            layers=args.layers,
        )
        return

    if args.command == "atlas":
        atlas_json = Path(args.outdir) / "concept_atlas.json"
        if args.labels_csv and atlas_json.exists():
            relabel_concept_atlas(
                atlas_json=atlas_json,
                labels_csv=args.labels_csv,
                outdir=args.outdir,
                refresh_visuals=not args.no_refresh_visuals,
            )
            return
        if not args.model or not args.data:
            raise SystemExit("atlas build requires --model and --data unless relabeling an existing atlas with --labels-csv")
        build_concept_atlas(
            model_path=args.model,
            data_yaml=args.data,
            outdir=args.outdir,
            split=args.split,
            imgsz=args.imgsz,
            limit=args.limit,
            topk_per_image=args.topk_per_image,
            device=args.device,
            attribution_method=args.attribution_method,
            layers=args.layers,
        )
        return

    if args.command == "graph":
        build_layered_concept_graph(args.atlas_json, args.out)
        return

    if args.command == "tree":
        build_inference_tree(args.audit_json, args.atlas_json, args.out)
        return

    if args.command == "labels":
        export_concept_labels_csv(args.atlas_json, args.out_csv)
        return

    if args.command == "adapt-baseline":
        record = load_baseline_sample_attribution(args.audit_json, model_path=args.model)
        save_json(record.to_dict(), args.out)
        return


if __name__ == "__main__":
    main()
