# Capsule Audit Toolkit

This directory contains two concept-auditing workflows for the capsule-based detector.

## Structure

### Unsupervised audit

Entry:

- `tools/audit_caps/unsupervised/run_unsupervised.py`
- `tools/audit_caps/unsupervised/run_audit.py`

Documentation:

- `tools/audit_caps/unsupervised/README.md`

Implementation lives in:

- `tools/audit_caps/unsupervised/hooks.py`
- `tools/audit_caps/unsupervised/relevance.py`
- `tools/audit_caps/unsupervised/atlas.py`
- `tools/audit_caps/unsupervised/graph.py`
- `tools/audit_caps/unsupervised/report.py`
- `tools/audit_caps/unsupervised/label_concepts.py`
- `tools/audit_caps/unsupervised/run_audit.py`

Purpose:

- discover latent capsule concepts without labels
- build concept atlases
- build concept-class graphs

Use this first when:

- you do not yet have concept annotations
- you want to explore what capsule types represent

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

## Recommended Order

1. Start with the unsupervised workflow.
2. Inspect capsule atlases and graph statistics.
3. Decide which concepts are stable and worth naming.
4. Create concept annotations.
5. Train supervised probes and run TCAV-style analysis.

## Path Policy

Only the new paths are supported now.

Use:

- `tools/audit_caps/unsupervised/...`
- `tools/audit_caps/supervised_tcav/...`

The old unsupervised root-level files under `tools/audit_caps/` have been removed.
