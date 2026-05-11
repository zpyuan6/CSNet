# Capsule Audit Toolkit

This directory contains the active capsule audit workflows for the capsule-based detector.

This new track is intended to rebuild the unsupervised workflow around CRP / L-CRP-style attribution, atlas construction, graph aggregation, and inference-tree export.

## Structure

### Unsupervised audit

Entry:

Use:

- `tools/audit_caps/unsupervised_audit/run_unsupervised.py`
- `tools/audit_caps/unsupervised_audit/run_audit.py`

Documentation:

- `tools/audit_caps/unsupervised_audit/README.md`

Implementation lives in:

- `tools/audit_caps/unsupervised_audit/schema.py`
- `tools/audit_caps/unsupervised_audit/io.py`
- `tools/audit_caps/unsupervised_audit/baseline_adapter.py`
- `tools/audit_caps/unsupervised_audit/run_audit.py`

Purpose:

- rebuild unsupervised audit around CRP / L-CRP-style logic
- share one schema across single-image attribution, atlas, graph, and tree
- support migration from earlier audit outputs

Use this first when:

- you do not yet have concept annotations
- you want the current unsupervised audit implementation

### Supervised TCAV-style audit

Entry:

- `tools/audit_caps/supervised_tcav/run_supervised.py`

Documentation:

- `tools/audit_caps/supervised_tcav/README.md`

Implementation lives in:

- `tools/audit_caps/supervised_tcav/data.py`
- `tools/audit_caps/supervised_tcav/features.py`
- `tools/audit_caps/supervised_tcav/probe.py`
- `tools/audit_caps/supervised_tcav/tcav.py`
- `tools/audit_caps/supervised_tcav/batch.py`
- `tools/audit_caps/supervised_tcav/dataset_analysis.py`
- `tools/audit_caps/supervised_tcav/report.py`
- `tools/audit_caps/supervised_tcav/template.py`

Purpose:

- train supervised concept probes
- estimate concept presence in a detection feature
- compute TCAV-style concept sensitivity
- summarize concept support on a dataset

Use this when:

- you already have concept annotations
- you want explicit, named concepts rather than discovered latent concepts

### Supervised concept ontology

Concept definitions live in:

- `configs/concept/concepts_coco.yaml`
- `configs/concept/concepts_inspection.yaml`
- `configs/concept/concepts_nematode.yaml`

These files use a minimal ontology format:

- `meta`: dataset-level metadata
- `levels`: hierarchy level definitions
- `level_concepts`: concept ids grouped by level, with each concept mapped to a scope
- `concepts`: concept definitions and `children` links

Example:

```yaml
level_concepts:
  L2:
    wheel: pixel
    window_pane: pixel
  L3:
    wheeled_transport_structure: instance
```

Valid concept scopes are:

- `image`: concept applies to the full image
- `instance`: concept applies to an object or defect instance
- `pixel`: concept expects a region or mask-level annotation

Ontology loading and validation lives in:

- `tools/audit_caps/ontology.py`

Use `load_concept_ontology(path)` in Python to build:

- `concepts`
- `children_map`
- `parents_map`
- `class_to_descendants()`
- `scopes_for(...)`

The same file can also run the first automated annotation workflow. It loads all concept names, scopes, and definitions from the provided yaml. No dataset-specific concept names should be hard-coded in the workflow.

Ontology-only auto expansion:

```powershell
python tools\audit_caps\ontology.py --concept-yaml configs\concept\concepts_coco.yaml --coco-json path\to\instances_val2017.json --out-json runs\audit\supervised_audit\concept_annotations\coco\concept_annotations.json
```

With VLM verification:

```powershell
$env:OPENAI_API_KEY="..."
python tools\audit_caps\ontology.py --concept-yaml configs\concept\concepts_coco.yaml --coco-json path\to\instances_val2017.json --image-root path\to\val2017 --out-json runs\audit\supervised_audit\concept_annotations\coco\concept_annotations.json --llm-provider openai --llm-model gpt-4o-mini
```

The OpenAI VLM path is implemented through LangChain and requires `langchain-openai` to be installed. The default `--llm-provider none` path does not import LangChain.

Pixel-scoped concepts are exported as review items by default. If the source COCO annotation has an instance segmentation and a coarse proxy is acceptable, use:

```powershell
python tools\audit_caps\ontology.py --concept-yaml configs\concept\concepts_inspection.yaml --coco-json path\to\annotations.json --out-json runs\audit\supervised_audit\concept_annotations\inspection\concept_annotations.json --pixel-policy instance_mask
```

For pixel-scoped concepts, a Hugging Face SAM2 bbox-prompted mask can also be generated:

```powershell
python tools\audit_caps\ontology.py --concept-yaml configs\concept\concepts_inspection.yaml --coco-json path\to\annotations.json --image-root path\to\images --out-json runs\audit\supervised_audit\concept_annotations\inspection\concept_annotations.json --pixel-policy sam2 --sam2-model facebook/sam2-hiera-large
```

The SAM2 path requires Hugging Face `transformers` with SAM2 support. SAM2 is used as a bbox-prompted mask generator; the concept candidate itself still comes from the ontology expansion and optional LLM verification.

Concept annotations should be exported as a COCO-compatible JSON extension rather than a custom annotation format. Keep the standard COCO keys:

- `images`
- `annotations`
- `categories`

Then add:

- `concepts`
- `concept_annotations`

Each `concept_annotations` entry should reference `image_id`, optionally `annotation_id`, and include `concept_id`, `scope`, `value`, `source`, and optional `score` / `segmentation` / `bbox` / `area`.

## Recommended Order

1. Start with the unsupervised workflow.
2. Inspect capsule atlases and graph statistics.
3. Decide which concepts are stable and worth naming.
4. Create concept annotations.
5. Train supervised probes and run TCAV-style analysis.

## Path Policy

Only the new paths are supported now.

Use:

- `tools/audit_caps/unsupervised_audit/...`
- `tools/audit_caps/supervised_tcav/...`

The previous `tools/audit_caps/unsupervised/...` implementation has been removed.
