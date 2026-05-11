# Unsupervised Audit

This package is the active unsupervised audit implementation for capsule auditing.

## Goal

Rebuild the audit workflow around the logic used by:

- CRP
- L-CRP

as closely as possible while adapting it to this repository's capsule detector.

## Scope

The final workflow should cover:

1. single-image, detection-conditioned concept attribution
2. dataset-level concept atlas / reference bank
3. global layered concept graph
4. sample-level inference tree

## Design Rules

- all stages must share one concept record schema
- sample attribution is the source of truth
- atlas, graph, and tree must all consume that same sample attribution format
- baseline JSON from older experiments can still be adapted into this schema during migration

## Planned Commands

- `single`
- `atlas`
- `graph`
- `tree`

These commands are exposed through:

```bash
python tools\audit_caps\unsupervised_audit\run_unsupervised.py --help
```

## Code Structure

- `single.py`
  - sample-level audit orchestration
- `atlas.py`
  - dataset-level concept atlas / concept references
- `graph.py`
  - global layered concept graph
- `tree.py`
  - sample-level inference tree
- `attribution.py`
  - attribution backend registry
- `crp_backend.py`, `crp_core.py`, `crp_rules.py`
  - current propagated-relevance backend

## Attribution Isolation

Upper-level visualization commands should remain stable:

- `single`
- `atlas`
- `graph`
- `tree`

Attribution methods should stay behind `attribution.py`.

Current backend:

- `capsule_lcrp`

Planned future backend:

- `activation_gradient_concept`

This allows multiple attribution methods to coexist while keeping the consumer stages unchanged.
