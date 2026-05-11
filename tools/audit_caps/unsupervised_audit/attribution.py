from __future__ import annotations

import inspect
from typing import Any

from .actgrad_backend import run_capsule_actgrad
from .crp_backend import run_capsule_lcrp


_ATTRIBUTION_BACKENDS: dict[str, dict[str, Any]] = {
    "capsule_lcrp": {
        "runner": run_capsule_lcrp,
        "sample_method": "capsule_lcrp_v3",
        "attribution_style": "capsule_lcrp_crp_core",
        "concept_mapping": "capsule_type",
        "implementation_note": "self_implemented_crp_core_grouped_channel_realization",
        "spatial_relevance_rule": "epsilon_lrp_local",
        "edge_semantics": "propagated_relevance",
        "requires_model_grad": False,
    },
    "activation_gradient_concept": {
        "runner": run_capsule_actgrad,
        "sample_method": "capsule_actgrad_v1",
        "attribution_style": "activation_gradient_concept",
        "concept_mapping": "capsule_type",
        "implementation_note": "grouped_channel_activation_times_gradient_positive_support",
        "spatial_relevance_rule": "activation_gradient_positive_sum",
        "edge_semantics": "concept_association",
        "requires_model_grad": True,
    },
}


def list_attribution_methods() -> list[str]:
    return sorted(_ATTRIBUTION_BACKENDS)


def get_attribution_backend(method: str) -> dict[str, Any]:
    try:
        return _ATTRIBUTION_BACKENDS[str(method)]
    except KeyError as exc:
        supported = ", ".join(list_attribution_methods())
        raise ValueError(f"Unsupported attribution method '{method}'. Supported methods: {supported}") from exc


def run_attribution(method: str, **kwargs: Any) -> Any:
    backend = get_attribution_backend(method)
    runner = backend["runner"]
    sig = inspect.signature(runner)
    filtered = {name: value for name, value in kwargs.items() if name in sig.parameters}
    return runner(**filtered)
