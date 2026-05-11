from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OutputRelevanceSeed:
    class_relevance: float
    box_relevance: float
    total_relevance: float


@dataclass
class CapsuleLayerRelevance:
    layer: str
    type_idx: int
    score: float
    support: float
    heatmap: list[list[float]]
    peak_xy: list[int]

