# Unsupervised Capsule Audit

This folder contains the unsupervised capsule concept auditing workflow.

## Purpose

This method does not require concept labels.

It is used to:

- find which capsule types contribute most to a detection result
- visualize what each capsule type responds to
- build a concept atlas from high-response image patches
- build a concept-to-class graph from atlas statistics

The implementation reuses the existing unsupervised audit pipeline in `tools/audit_caps`.

## Main Entry

Use:

```bash
python tools\audit_caps\unsupervised\run_unsupervised.py --help
```

Available commands:

- `single`: audit one image and one detection
- `atlas`: build concept atlas over a dataset split
- `graph`: build concept-class graph from atlas json
- `labels`: export a concept labeling template from atlas json

## Required Data

### 1. Trained model weights

Example:

```text
runs\detect\runs\train\CapsNeckHeadv6_NoKD\weights\best.pt
```

### 2. Input image or dataset yaml

For single-image audit:

- one image path

For atlas building:

- one dataset yaml such as `configs/data/coco.yaml`

No concept labels are required.

## Typical Workflow

### Step 1: audit a single image

```bash
python tools\audit_caps\unsupervised\run_unsupervised.py single --model runs\detect\runs\train\CapsNeckHeadv6_NoKD\weights\best.pt --image C:\Datasets\coco-yolo\images\val2017\000000000139.jpg --outdir runs\audit\unsupervised\single_000139 --imgsz 640 --topk 5
```

Outputs:

- `audit.json`
- report images/files in the output directory

### Step 2: build a concept atlas

```bash
python tools\audit_caps\unsupervised\run_unsupervised.py atlas --model runs\detect\runs\train\CapsNeckHeadv6_NoKD\weights\best.pt --data configs\data\coco.yaml --outdir runs\audit\unsupervised\atlas --split val --imgsz 640 --limit 200 --topk-per-image 3
```

Outputs:

- `concept_atlas.json`
- saved concept patches

### Step 3: build concept-class graph

```bash
python tools\audit_caps\unsupervised\run_unsupervised.py graph --atlas-json runs\audit\unsupervised\atlas\concept_atlas.json --out runs\audit\unsupervised\atlas\concept_graph.json --topn 10
```

### Step 4: export concept labeling template

```bash
python tools\audit_caps\unsupervised\run_unsupervised.py labels --atlas-json runs\audit\unsupervised\atlas\concept_atlas.json --out-csv runs\audit\unsupervised\atlas\concept_labels.csv
```

## Interpretation

This method is concept discovery, not concept supervision.

- `top capsule types` tell you which latent capsule units are most relevant
- `atlas patches` tell you what those units usually respond to
- `concept-class graph` summarizes which latent concepts are associated with which classes

This is suitable for exploratory analysis before manual concept naming.
