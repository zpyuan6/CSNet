from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def build_concept_class_graph(atlas_json: str | Path, out_path: str | Path, topn: int = 10) -> dict[str, Any]:
    atlas = json.loads(Path(atlas_json).read_text(encoding="utf-8"))

    class_to_concepts: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    concept_to_classes: dict[str, dict[int, float]] = defaultdict(lambda: defaultdict(float))

    for concept_key, items in atlas.get("concepts", {}).items():
        for item in items:
            class_id = int(item["class_id"])
            score = float(item["score"])
            class_to_concepts[class_id][concept_key] += score
            concept_to_classes[concept_key][class_id] += score

    graph = {
        "class_to_concepts": {
            str(class_id): sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:topn]
            for class_id, scores in class_to_concepts.items()
        },
        "concept_to_classes": {
            concept: sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:topn]
            for concept, scores in concept_to_classes.items()
        },
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
    return graph
