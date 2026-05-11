import argparse
import csv
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot COCO mAP50:95 vs inference speed or FLOPs for YOLO variants and custom models."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Input CSV. Supports either {model,map,speed_ms,series,highlight} or {model,map50_95,flops_b,...}.",
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
        default=None,
        help="Plot title.",
    )
    parser.add_argument(
        "--x",
        type=str,
        default="speed_ms",
        choices=["speed_ms", "flops_b"],
        help="X-axis metric.",
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
    def _parse_float(value: str | None) -> float:
        text = (value or "").strip()
        if not text:
            return math.nan
        return float(text)

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        compact_format = {"model", "map", "speed_ms", "series", "highlight"}
        benchmark_format = {"model", "map50_95", "flops_b"}
        if not (compact_format <= fields or benchmark_format <= fields):
            raise ValueError(
                "Unsupported CSV columns. Need either compact format "
                "{model,map,speed_ms,series,highlight} or benchmarking format including {model,map50_95,flops_b}."
            )

        rows = []
        for row in reader:
            model = row["model"].strip()
            series = row.get("series", "").strip() or ("Ours" if "Capsule" in model or "Caps" in model else "Other")
            highlight = row.get("highlight", "").strip().lower() in {"1", "true", "yes", "y"}
            map_value = row.get("map", row.get("map50_95", ""))
            speed_value = row.get("speed_ms", row.get("local_gpu_speed_mean_ms", "nan"))
            flops_value = row.get("flops_b", "nan")
            rows.append(
                {
                    "model": model,
                    "map": _parse_float(map_value),
                    "speed_ms": _parse_float(speed_value),
                    "flops_b": _parse_float(flops_value),
                    "series": series,
                    "highlight": highlight,
                }
            )
    if not rows:
        raise ValueError("Input CSV is empty.")
    return rows


def plot(rows: list[dict], out_path: Path, title: str | None, annotate: bool, pareto: bool, x_key: str) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    title_fontsize = 18
    axis_label_fontsize = 16
    tick_fontsize = 14
    annotation_fontsize = 12
    legend_fontsize = 12

    series_labels = {
        "Ours": "_nolegend_",
    }
    colors = {
        "YOLOv6": "#6C6FF5",
        "YOLO6": "#6C6FF5",
        "YOLOv7": "#4C78A8",
        "YOLO7": "#4C78A8",
        "YOLOv8": "#59A14F",
        "YOLO8": "#59A14F",
        "YOLOv9": "#F28E2B",
        "YOLO9": "#F28E2B",
        "YOLOv10": "#B07AA1",
        "YOLO10": "#B07AA1",
        "YOLO11": "#76B7B2",
        "YOLO26": "#9C755F",
        "Ours": "#D62728",
        "Other": "#7F7F7F",
    }
    line_styles = {
        "YOLOv6": "--",
        "YOLO6": "--",
        "YOLOv7": "--",
        "YOLO7": "--",
        "YOLOv8": "--",
        "YOLO8": "--",
        "YOLOv9": "--",
        "YOLO9": "--",
        "YOLOv10": "--",
        "YOLO10": "--",
        "YOLO11": "--",
        "YOLO26": "--",
        "Ours": "-",
        "Other": "--",
    }
    markers = {
        "YOLOv6": "o",
        "YOLO6": "o",
        "YOLOv7": "s",
        "YOLO7": "s",
        "YOLOv8": "^",
        "YOLO8": "^",
        "YOLOv9": "D",
        "YOLO9": "D",
        "YOLOv10": "P",
        "YOLO10": "P",
        "YOLO11": "X",
        "YOLO26": "h",
        "Ours": "*",
        "Other": "o",
    }

    def _model_color(series: str, model: str) -> str:
        if model.startswith("SCN-") and model != "SCN-n":
            return "#9E9E9E"
        return colors.get(series, colors["Other"])

    def _model_marker(series: str, model: str) -> str:
        if model == "SCN-n":
            return "*"
        return markers.get(series, markers["Other"])

    fig, ax = plt.subplots(figsize=(10, 6), dpi=180)

    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["series"], []).append(row)

    for series, points in grouped.items():
        points = [p for p in points if str(p.get(x_key)) != "nan"]
        if not points:
            continue
        points = sorted(points, key=lambda x: x[x_key])
        color = colors.get(series, colors["Other"])
        marker = markers.get(series, markers["Other"])
        if series == "Ours":
            ax.plot(
                [p[x_key] for p in points],
                [p["map"] for p in points],
                marker="o",
                linewidth=2,
                markersize=4,
                color=colors["Ours"],
                linestyle=line_styles["Ours"],
                label=series_labels.get(series, series),
                alpha=0.9,
            )
        else:
            ax.plot(
                [p[x_key] for p in points],
                [p["map"] for p in points],
                marker=marker,
                linewidth=2,
                markersize=5,
                color=color,
                linestyle=line_styles.get(series, "--"),
                label=series_labels.get(series, series),
                alpha=0.9,
            )
        for p in points:
            is_ours = p["series"] == "Ours"
            display_name = f'{p["model"]} (Ours)' if is_ours else p["model"]
            point_color = _model_color(series, p["model"])
            size = 220 if is_ours else (90 if p["highlight"] else 42)
            edge = "black" if p["highlight"] or is_ours else point_color
            point_marker = _model_marker(series, p["model"])
            ax.scatter(
                p[x_key],
                p["map"],
                s=size,
                color=point_color,
                marker=point_marker,
                edgecolors=edge,
                linewidths=1.4 if is_ours else 1.2,
                zorder=3,
            )
            if annotate or p["highlight"]:
                if p["model"] == "YOLO26m":
                    xytext = (6, 11)
                    ha = "left"
                    va = "top"
                elif p["model"] in {"YOLO12n", "YOLO26x"}:
                    xytext = (6, 9)
                    ha = "left"
                    va = "top"
                elif p["model"] == "YOLO12m":
                    xytext = (6, 9)
                    ha = "left"
                    va = "top"
                elif p["model"] == "YOLO26l":
                    xytext = (6, 9)
                    ha = "left"
                    va = "top"
                elif p["model"] in {"YOLO8m", "YOLO10m", "YOLO12l"}:
                    xytext = (6, 0)
                    ha = "left"
                    va = "bottom"
                elif p["series"] in {"YOLOv6", "YOLO6"}:
                    xytext = (6, -3)
                    ha = "left"
                    va = "top"
                elif p["model"].startswith("SCN-"):
                    xytext = (0, 8)
                    ha = "center"
                    va = "bottom"
                else:
                    xytext = (6, 6)
                    ha = "left"
                    va = "bottom"
                if p["model"] in {"YOLO6l", "YOLO8l", "YOLO10l", "YOLO10x", "YOLO12x", "YOLO26x", "YOLO8x"}:
                    xytext = (xytext[0] - 12, xytext[1])
                if p["model"] == "YOLO6l":
                    xytext = (xytext[0], xytext[1] - 6)
                if p["model"] == "YOLO26x":
                    xytext = (xytext[0], xytext[1] + 3)
                ax.annotate(
                    display_name,
                    (p[x_key], p["map"]),
                    textcoords="offset points",
                    xytext=xytext,
                    ha=ha,
                    va=va,
                    fontsize=annotation_fontsize,
                    fontweight="bold" if is_ours else "normal",
                    zorder=6,
                )

    if pareto:
        pareto_rows = compute_pareto_front(rows, x_key=x_key)
        if pareto_rows:
            ax.plot(
                [p[x_key] for p in pareto_rows],
                [p["map"] for p in pareto_rows],
                linestyle="--",
                linewidth=1.8,
                color="black",
                alpha=0.75,
                label="Pareto front",
            )

    if title is None:
        title = "COCO mAP50:95 vs FLOPs" if x_key == "flops_b" else "COCO mAP50:95 vs Inference Speed"
    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel(
        "Computational Costs - FLOPs (B)" if x_key == "flops_b" else "Inference Speed (ms/image)",
        fontsize=axis_label_fontsize,
    )
    ax.set_ylabel("Detection Precision - COCO mAP50:95", fontsize=axis_label_fontsize)
    ax.tick_params(axis="both", labelsize=tick_fontsize)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.4)
    handles, labels = ax.get_legend_handles_labels()
    filtered = [(h, l) for h, l in zip(handles, labels) if l != "_nolegend_"]
    if filtered:
        handles, labels = zip(*filtered)
        handles = list(handles)
        labels = list(labels)
    else:
        handles, labels = [], []
    handles.extend(
        [
            Line2D(
                [0],
                [0],
                marker="*",
                linestyle="-",
                color=colors["Ours"],
                markerfacecolor=colors["Ours"],
                markeredgecolor="black",
                markersize=10,
                linewidth=2,
            ),
            Line2D(
                [0],
                [0],
                marker="*",
                linestyle="-",
                color=colors["Ours"],
                markerfacecolor="#9E9E9E",
                markeredgecolor="black",
                markersize=10,
                linewidth=2,
            ),
        ]
    )
    labels.extend(
        [
            "SCN (Our, Implemented)",
            "SCN (Our, Expected in Phase 2)",
        ]
    )
    legend = ax.legend(handles, labels, frameon=True, fontsize=legend_fontsize)
    if legend is not None:
        for text in legend.get_texts():
            if text.get_text().startswith("SCN-"):
                text.set_fontweight("bold")

    xs = [row[x_key] for row in rows if str(row.get(x_key)) != "nan"]
    ys = [row["map"] for row in rows]
    x_pad = max(1.0, (max(xs) - min(xs)) * 0.14)
    y_pad = max(0.002, (max(ys) - min(ys)) * 0.12)
    ax.set_xlim(min(xs) - x_pad, max(xs) + x_pad)
    ax.set_ylim(min(ys) - y_pad, max(ys) + y_pad)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def compute_pareto_front(rows: list[dict], x_key: str = "speed_ms") -> list[dict]:
    points = [row for row in rows if str(row.get(x_key)) != "nan"]
    points = sorted(points, key=lambda x: (x[x_key], -x["map"]))
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
    plot(rows, args.out, args.title, args.annotate, args.pareto, args.x)
    print(f"Saved plot to: {args.out}")


if __name__ == "__main__":
    main()
