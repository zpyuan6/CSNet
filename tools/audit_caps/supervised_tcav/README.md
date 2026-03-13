# Supervised TCAV-Style Capsule Audit

This folder contains the supervised concept auditing workflow for capsule features.

## Purpose

This method uses concept annotations to train a linear concept probe on intermediate capsule features.

It is used to:

- learn a supervised concept direction from annotated examples
- estimate whether a concept is present in a detection feature
- compute a TCAV-style directional sensitivity score for a detection result
- summarize concept support statistics at the dataset level

## Main Entry

Use:

```bash
python tools\audit_caps\supervised_tcav\run_supervised.py --help
```

Available commands:

- `template`: export a concept annotation csv template
- `train-probe`: train one concept probe
- `train-all`: train all concept probes from one annotation csv
- `single`: audit one detection with one or more trained probes
- `analyze-dataset`: run dataset-level concept analysis

## Required Data

### 1. Trained model weights

Example:

```text
runs\detect\runs\train\CapsNeckHeadv6_NoKD\weights\best.pt
```

### 2. Concept annotation CSV

Required columns:

- `image`
- `concept`
- `label`

Optional columns:

- `layer`
- `x1`
- `y1`
- `x2`
- `y2`
- `notes`

Meaning:

- `image`: image path, absolute or relative to the csv file
- `concept`: concept name such as `wheel`, `striped`, `rectangular`
- `label`: `1` for positive, `0` for negative
- `layer`: intermediate capsule layer name such as `model.15`
- `x1,y1,x2,y2`: optional box-level region for concept annotation

If box coordinates are provided, the concept probe is trained on ROI pooled capsule features.
If box coordinates are omitted, the concept probe is trained on globally pooled capsule features.

### 3. Optional dataset yaml

For dataset-level analysis, you may provide a dataset yaml such as:

```text
configs\data\coco-seg.yaml
```

This allows the tool to read images directly from a dataset split.

## Typical Workflow

### Step 1: export the annotation template

```bash
python tools\audit_caps\supervised_tcav\run_supervised.py template --out-csv runs\audit\supervised_tcav\concept_annotations.csv
```

### Step 2: train a single concept probe

```bash
python tools\audit_caps\supervised_tcav\run_supervised.py train-probe --model runs\detect\runs\train\CapsNeckHeadv6_NoKD\weights\best.pt --annotations runs\audit\supervised_tcav\concept_annotations.csv --concept wheel --layer model.15 --out runs\audit\supervised_tcav\wheel_probe.json --imgsz 640
```

### Step 3: train all concept probes

```bash
python tools\audit_caps\supervised_tcav\run_supervised.py train-all --model runs\detect\runs\train\CapsNeckHeadv6_NoKD\weights\best.pt --annotations runs\audit\supervised_tcav\concept_annotations.csv --outdir runs\audit\supervised_tcav\probes --default-layer model.15 --imgsz 640
```

Outputs:

- one probe json per concept
- `manifest.json`

### Step 4: audit a single image

```bash
python tools\audit_caps\supervised_tcav\run_supervised.py single --model runs\detect\runs\train\CapsNeckHeadv6_NoKD\weights\best.pt --image C:\Datasets\coco-yolo\images\val2017\000000000139.jpg --probes runs\audit\supervised_tcav\probes\wheel__model_15.json --out runs\audit\supervised_tcav\single_000139.json --imgsz 640
```

Outputs:

- `single_000139.json`
- `single_000139\report.md`
- `single_000139\overlay.png`

### Step 5: dataset-level analysis

Use annotation CSV as image source:

```bash
python tools\audit_caps\supervised_tcav\run_supervised.py analyze-dataset --model runs\detect\runs\train\CapsNeckHeadv6_NoKD\weights\best.pt --annotations runs\audit\supervised_tcav\concept_annotations.csv --manifest runs\audit\supervised_tcav\probes\manifest.json --outdir runs\audit\supervised_tcav\dataset_analysis --imgsz 640 --limit 50
```

Or read directly from a dataset yaml split:

```bash
python tools\audit_caps\supervised_tcav\run_supervised.py analyze-dataset --model runs\detect\runs\train\CapsNeckHeadv6_NoKD\weights\best.pt --data configs\data\coco-seg.yaml --split val --manifest runs\audit\supervised_tcav\probes\manifest.json --outdir runs\audit\supervised_tcav\dataset_analysis_val --imgsz 640 --limit 100
```

Outputs:

- `dataset_summary.json`
- `dataset_results.json`
- `concept_summary.csv`
- `concept_summary_by_class.csv`
- `summary.md`

## Interpretation

This method is supervised concept auditing.

- `probe_prob` measures whether the trained probe thinks the concept is present
- `tcav_score` measures whether the model output is sensitive to that concept direction
- positive and larger `tcav_score` indicates stronger support from the concept for the selected detection

## Recommended Annotation Strategy

Use box-level binary concept labels first.

Recommended first concepts:

- `wheel`
- `window`
- `head`
- `leg`
- `screen`
- `elongated`
- `round`
- `rectangular`
- `striped`
- `metallic`
