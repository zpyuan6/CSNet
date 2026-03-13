import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot COCO mAP50:95 vs inference speed (ms/image) for YOLO variants and custom models."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Input CSV with columns: model,map,speed_ms,series,highlight",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("runs/plots/speed_map_curve.png"),
        help="Output image path.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="COCO mAP50:95 vs Inference Speed",
        help="Plot title.",
    )
    parser.add_argument(
        "--annotate",
        action="store_true",
        help="Annotate every point with the model name.",
    )
    parser.add_argument(
        "--pareto",
        action="store_true",
        help="Draw Pareto front over all points.",
    )
    return parser.parse_args()


def load_rows(csv_path: Path) -> list[dict]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"model", "map", "speed_ms", "series", "highlight"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing CSV columns: {sorted(missing)}")

        rows = []
        for row in reader:
            rows.append(
                {
                    "model": row["model"].strip(),
                    "map": float(row["map"]),
                    "speed_ms": float(row["speed_ms"]),
                    "series": row["series"].strip(),
                    "highlight": row["highlight"].strip().lower() in {"1", "true", "yes", "y"},
                }
            )
    if not rows:
        raise ValueError("Input CSV is empty.")
    return rows


def plot(rows: list[dict], out_path: Path, title: str, annotate: bool, pareto: bool) -> None:
    import matplotlib.pyplot as plt

    colors = {
        "YOLOv6": "#6C6FF5",
        "YOLOv7": "#4C78A8",
        "YOLOv8": "#59A14F",
        "YOLOv9": "#F28E2B",
        "YOLOv10": "#B07AA1",
        "YOLO11": "#76B7B2",
        "YOLO26": "#9C755F",
        "Ours": "#D62728",
        "Other": "#7F7F7F",
    }
    markers = {
        "YOLOv6": "o",
        "YOLOv7": "s",
        "YOLOv8": "^",
        "YOLOv9": "D",
        "YOLOv10": "P",
        "YOLO11": "X",
        "YOLO26": "h",
        "Ours": "*",
        "Other": "o",
    }

    fig, ax = plt.subplots(figsize=(10, 6), dpi=180)

    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["series"], []).append(row)

    for series, points in grouped.items():
        points = sorted(points, key=lambda x: x["speed_ms"])
        color = colors.get(series, colors["Other"])
        marker = markers.get(series, markers["Other"])
        ax.plot(
            [p["speed_ms"] for p in points],
            [p["map"] for p in points],
            marker=marker,
            linewidth=2,
            markersize=5,
            color=color,
            label=series,
            alpha=0.9,
        )
        for p in points:
            is_ours = p["series"] == "Ours"
            size = 220 if is_ours else (90 if p["highlight"] else 42)
            edge = "black" if p["highlight"] or is_ours else color
            point_marker = "*" if is_ours else marker
            ax.scatter(
                p["speed_ms"],
                p["map"],
                s=size,
                color=color,
                marker=point_marker,
                edgecolors=edge,
                linewidths=1.4 if is_ours else 1.2,
                zorder=3,
            )
            if annotate or p["highlight"]:
                ax.annotate(
                    p["model"],
                    (p["speed_ms"], p["map"]),
                    textcoords="offset points",
                    xytext=(6, 6),
                    fontsize=8,
                )

    if pareto:
        pareto = compute_pareto_front(rows)
        if pareto:
            ax.plot(
                [p["speed_ms"] for p in pareto],
                [p["map"] for p in pareto],
                linestyle="--",
                linewidth=1.8,
                color="black",
                alpha=0.75,
                label="Pareto front",
            )

    ax.set_title(title)
    ax.set_xlabel("Inference Speed (ms/image)")
    ax.set_ylabel("COCO mAP50:95")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.4)
    ax.legend(frameon=True)

    xs = [row["speed_ms"] for row in rows]
    ys = [row["map"] for row in rows]
    x_pad = max(1.0, (max(xs) - min(xs)) * 0.08)
    y_pad = max(0.002, (max(ys) - min(ys)) * 0.12)
    ax.set_xlim(min(xs) - x_pad, max(xs) + x_pad)
    ax.set_ylim(min(ys) - y_pad, max(ys) + y_pad)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def compute_pareto_front(rows: list[dict]) -> list[dict]:
    points = sorted(rows, key=lambda x: (x["speed_ms"], -x["map"]))
    pareto = []
    best_map = -1.0
    for p in points:
        if p["map"] > best_map:
            pareto.append(p)
            best_map = p["map"]
    return pareto


def main() -> None:
    args = parse_args()
    rows = load_rows(args.csv)
    plot(rows, args.out, args.title, args.annotate, args.pareto)
    print(f"Saved plot to: {args.out}")


if __name__ == "__main__":
    main()
