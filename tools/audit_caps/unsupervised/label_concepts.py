from __future__ import annotations

import csv
import json
from pathlib import Path


def export_concept_label_template(atlas_json: str | Path, out_csv: str | Path) -> None:
    atlas = json.loads(Path(atlas_json).read_text(encoding="utf-8"))
    concepts = sorted(atlas.get("concepts", {}).keys())
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["concept_id", "concept_name", "semantic_group", "confidence", "notes"])
        for concept_id in concepts:
            writer.writerow([concept_id, "", "", "", ""])
