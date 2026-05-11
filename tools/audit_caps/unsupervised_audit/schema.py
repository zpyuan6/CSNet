from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class DetectionTarget:
    det_index: int
    class_id: int
    score: float
    bbox_xywh_image: list[float]
    bbox_xywh_model: list[float] | None = None
    class_name: str | None = None
    selector_branch: str | None = None
    gradient_branch: str | None = None
    raw_index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ConceptHeatmap:
    layer: str
    concept_id: str
    type_idx: int
    relevance_score: float
    peak_xy: list[int]
    heatmap: list[list[float]] | None = None
    sample_overlay_path: str | None = None
    crop_box_image: list[int] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AtlasReference:
    concept_id: str
    image_path: str
    crop_path: str | None = None
    score: float = 0.0
    class_id: int | None = None
    class_name: str | None = None
    bbox_xywh_image: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ConceptNode:
    id: str
    kind: str
    layer: str | None = None
    type_idx: int | None = None
    class_id: int | None = None
    class_name: str | None = None
    concept_name: str | None = None
    semantic_group: str | None = None
    confidence: str | None = None
    score: float = 0.0
    support: float | None = None
    heatmap: ConceptHeatmap | None = None
    atlas_refs: list[AtlasReference] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return data


@dataclass
class ConceptEdge:
    source: str
    target: str
    kind: str
    weight: float
    count: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SampleAttribution:
    method: str
    image_path: str
    image_size: list[int]
    model_path: str
    target: DetectionTarget
    nodes: list[ConceptNode]
    edges: list[ConceptEdge]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GlobalGraph:
    method: str
    graph_type: str
    nodes: list[ConceptNode]
    edges: list[ConceptEdge]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class InferenceTree:
    method: str
    target: DetectionTarget
    nodes: list[ConceptNode]
    edges: list[ConceptEdge]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
