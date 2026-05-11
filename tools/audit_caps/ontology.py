from __future__ import annotations

import argparse
import base64
import json
import os
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import yaml


VALID_SCOPES = {"image", "instance", "pixel"}

CONCEPT_VERIFICATION_PROMPT = """You are validating visual concept labels for a dataset annotation.

Use only the image evidence and the concept definitions. Do not infer concepts that are not visually supported.

Object class: {class_name}
Candidate concepts:
{concept_list}

Return strict JSON with this schema:
{{
  "concepts": [
    {{"id": "concept_id", "present": true, "confidence": 0.0, "evidence": "short phrase"}}
  ]
}}
"""


@dataclass(frozen=True)
class Concept:
    id: str
    name: str
    level: str
    description: str
    children: tuple[str, ...]
    scope: str


@dataclass(frozen=True)
class ConceptOntology:
    path: Path
    meta: dict[str, Any]
    levels: tuple[dict[str, Any], ...]
    concepts: dict[str, Concept]
    level_concepts: dict[str, dict[str, str]]
    parents_map: dict[str, tuple[str, ...]]
    children_map: dict[str, tuple[str, ...]]

    def descendants(self, concept_id: str, include_self: bool = False) -> tuple[str, ...]:
        """Return descendants following concept.children edges."""
        if concept_id not in self.concepts:
            raise KeyError(f"Unknown concept id: {concept_id}")

        out: list[str] = []
        seen: set[str] = set()

        def visit(current: str) -> None:
            for child in self.children_map.get(current, ()):
                if child in seen:
                    continue
                seen.add(child)
                out.append(child)
                visit(child)

        if include_self:
            out.append(concept_id)
        visit(concept_id)
        return tuple(out)

    def class_to_descendants(self, include_self: bool = True) -> dict[str, tuple[str, ...]]:
        """Use L4 concepts as dataset class entries."""
        return {
            concept_id: self.descendants(concept_id, include_self=include_self)
            for concept_id, concept in self.concepts.items()
            if concept.level == "L4"
        }

    def scopes_for(self, concept_ids: list[str] | tuple[str, ...]) -> dict[str, str]:
        return {concept_id: self.concepts[concept_id].scope for concept_id in concept_ids}


def load_concept_ontology(path: str | Path, validate: bool = True) -> ConceptOntology:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Ontology file must contain a YAML mapping: {path}")

    level_concepts = _parse_level_concepts(data.get("level_concepts", {}))
    concepts_raw = data.get("concepts", [])
    if not isinstance(concepts_raw, list):
        raise ValueError("`concepts` must be a list.")

    scopes_by_concept = {
        concept_id: scope
        for concepts_in_level in level_concepts.values()
        for concept_id, scope in concepts_in_level.items()
    }

    concepts: dict[str, Concept] = {}
    for item in concepts_raw:
        if not isinstance(item, dict):
            raise ValueError("Every concept entry must be a mapping.")
        concept_id = str(item["id"])
        children = tuple(str(child) for child in item.get("children", []))
        concepts[concept_id] = Concept(
            id=concept_id,
            name=str(item.get("name", concept_id)),
            level=str(item["level"]),
            description=str(item.get("description", "")),
            children=children,
            scope=scopes_by_concept.get(concept_id, "instance"),
        )

    children_map = {concept_id: concept.children for concept_id, concept in concepts.items()}
    parents_mut: dict[str, list[str]] = {concept_id: [] for concept_id in concepts}
    for parent_id, children in children_map.items():
        for child_id in children:
            parents_mut.setdefault(child_id, []).append(parent_id)
    parents_map = {concept_id: tuple(parents) for concept_id, parents in parents_mut.items()}

    ontology = ConceptOntology(
        path=path,
        meta=dict(data.get("meta", {})),
        levels=tuple(data.get("levels", [])),
        concepts=concepts,
        level_concepts=level_concepts,
        parents_map=parents_map,
        children_map=children_map,
    )
    if validate:
        validate_ontology(ontology)
    return ontology


def validate_ontology(ontology: ConceptOntology) -> None:
    level_ids = {str(level["id"]) for level in ontology.levels}
    listed_concepts = {
        concept_id
        for concepts_in_level in ontology.level_concepts.values()
        for concept_id in concepts_in_level
    }

    missing_definitions = listed_concepts - set(ontology.concepts)
    if missing_definitions:
        raise ValueError(f"Concepts listed in level_concepts but missing definitions: {sorted(missing_definitions)}")

    missing_list_entries = set(ontology.concepts) - listed_concepts
    if missing_list_entries:
        raise ValueError(f"Concepts defined but missing from level_concepts: {sorted(missing_list_entries)}")

    for concept_id, concept in ontology.concepts.items():
        if concept.level not in level_ids:
            raise ValueError(f"{concept_id} uses unknown level: {concept.level}")
        if concept_id not in ontology.level_concepts.get(concept.level, {}):
            raise ValueError(f"{concept_id} is defined at {concept.level} but not listed under that level.")
        if concept.scope not in VALID_SCOPES:
            raise ValueError(f"{concept_id} uses invalid scope {concept.scope!r}; valid scopes: {sorted(VALID_SCOPES)}")
        for child_id in concept.children:
            if child_id not in ontology.concepts:
                raise ValueError(f"{concept_id} references missing child concept: {child_id}")


def _parse_level_concepts(raw: Any) -> dict[str, dict[str, str]]:
    if not isinstance(raw, dict):
        raise ValueError("`level_concepts` must be a mapping.")

    out: dict[str, dict[str, str]] = {}
    for level, concepts in raw.items():
        level_id = str(level)
        if isinstance(concepts, dict):
            out[level_id] = {str(concept_id): str(scope) for concept_id, scope in concepts.items()}
        elif isinstance(concepts, list):
            # Backward compatibility with the earlier scope-free format.
            out[level_id] = {str(concept_id): "instance" for concept_id in concepts}
        else:
            raise ValueError(f"level_concepts.{level_id} must be a mapping or list.")
    return out


def build_coco_concept_annotations(
    coco_json: str | Path,
    concept_yaml: str | Path,
    out_json: str | Path,
    image_root: str | Path | None = None,
    llm_provider: str = "none",
    llm_model: str = "gpt-4o-mini",
    max_annotations: int | None = None,
    pixel_policy: str = "needs_review",
    sam2_model: str = "facebook/sam2-hiera-large",
    sam2_device: str = "cuda",
    sam2_threshold: float = 0.0,
) -> dict[str, Any]:
    """Create a COCO-compatible concept annotation file from dataset annotations.

    The concept names and hierarchy are loaded only from `concept_yaml`. The code
    does not hard-code dataset-specific concept names.
    """
    ontology = load_concept_ontology(concept_yaml)
    coco_path = Path(coco_json)
    coco = json.loads(coco_path.read_text(encoding="utf-8"))

    images_by_id = {image["id"]: image for image in coco.get("images", [])}
    categories_by_id = {category["id"]: category for category in coco.get("categories", [])}
    class_to_concept = _build_class_to_concept_map(categories_by_id.values(), ontology)
    concept_categories, concept_numeric_ids = _build_concept_categories(ontology)

    concept_annotations: list[dict[str, Any]] = []
    next_id = 1
    annotations = coco.get("annotations", [])
    if max_annotations is not None:
        annotations = annotations[:max_annotations]

    llm_client = _build_llm_client(llm_provider=llm_provider, llm_model=llm_model)
    image_root_path = Path(image_root) if image_root is not None else None
    sam2_segmenter = (
        HuggingFaceSAM2Segmenter(model_name=sam2_model, device=sam2_device, threshold=sam2_threshold)
        if pixel_policy == "sam2"
        else None
    )

    for ann in annotations:
        category = categories_by_id.get(ann.get("category_id"))
        if category is None:
            continue
        class_name = str(category.get("name", ""))
        class_concept_id = class_to_concept.get(class_name)
        if class_concept_id is None:
            continue

        candidates = ontology.descendants(class_concept_id, include_self=True)
        candidates = tuple(concept_id for concept_id in candidates if concept_id in ontology.concepts)
        verified = _verify_candidates_with_llm(
            llm_client=llm_client,
            ontology=ontology,
            candidates=candidates,
            class_name=class_name,
            image=image_path_for_annotation(images_by_id.get(ann.get("image_id")), image_root_path),
            bbox=ann.get("bbox"),
        )

        for concept_id, result in verified.items():
            concept = ontology.concepts[concept_id]
            present = bool(result.get("present", True))
            if not present:
                continue

            entry = {
                "id": next_id,
                "image_id": ann.get("image_id"),
                "annotation_id": ann.get("id"),
                "concept_id": concept_numeric_ids[concept_id],
                "concept_name": concept_id,
                "scope": concept.scope,
                "value": 1,
                "source": result.get("source", "ontology_auto"),
                "score": float(result.get("confidence", 1.0)),
            }
            if result.get("evidence"):
                entry["evidence"] = result["evidence"]
            if concept.scope == "pixel":
                _attach_pixel_proxy(
                    entry,
                    ann,
                    pixel_policy=pixel_policy,
                    image_path=image_path_for_annotation(images_by_id.get(ann.get("image_id")), image_root_path),
                    sam2_segmenter=sam2_segmenter,
                )
            concept_annotations.append(entry)
            next_id += 1

    output = {
        "info": {
            **dict(coco.get("info", {})),
            "concept_ontology": str(concept_yaml),
            "concept_annotation_format": "coco_with_concept_extension",
        },
        "licenses": coco.get("licenses", []),
        "images": coco.get("images", []),
        "annotations": coco.get("annotations", []),
        "categories": coco.get("categories", []),
        "concepts": concept_categories,
        "concept_annotations": concept_annotations,
    }

    out_path = Path(out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def image_path_for_annotation(image: dict[str, Any] | None, image_root: Path | None) -> Path | None:
    if image is None or image_root is None:
        return None
    file_name = image.get("file_name")
    if not file_name:
        return None
    return image_root / str(file_name)


def _build_concept_categories(ontology: ConceptOntology) -> tuple[list[dict[str, Any]], dict[str, int]]:
    categories: list[dict[str, Any]] = []
    numeric_ids: dict[str, int] = {}
    for index, concept in enumerate(ontology.concepts.values(), start=1):
        numeric_ids[concept.id] = index
        categories.append(
            {
                "id": index,
                "name": concept.id,
                "display_name": concept.name,
                "level": concept.level,
                "scope": concept.scope,
                "description": concept.description,
            }
        )
    return categories, numeric_ids


def _build_class_to_concept_map(
    categories: Any,
    ontology: ConceptOntology,
) -> dict[str, str]:
    concept_lookup = {_normalize_name(concept_id): concept_id for concept_id in ontology.concepts}
    out: dict[str, str] = {}
    for category in categories:
        name = str(category.get("name", ""))
        concept_id = concept_lookup.get(_normalize_name(name))
        if concept_id is not None and ontology.concepts[concept_id].level == "L4":
            out[name] = concept_id
    return out


def _normalize_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def _attach_pixel_proxy(
    entry: dict[str, Any],
    ann: dict[str, Any],
    pixel_policy: str,
    image_path: Path | None = None,
    sam2_segmenter: "HuggingFaceSAM2Segmenter | None" = None,
) -> None:
    if pixel_policy == "skip":
        entry["needs_review"] = True
        return
    if pixel_policy == "instance_mask" and ann.get("segmentation"):
        entry["segmentation"] = ann.get("segmentation")
        entry["bbox"] = ann.get("bbox")
        entry["area"] = ann.get("area")
        entry["source"] = f"{entry['source']}+instance_mask_proxy"
        return
    if pixel_policy == "sam2":
        if sam2_segmenter is None or image_path is None or ann.get("bbox") is None:
            entry["needs_review"] = True
            entry["source"] = f"{entry['source']}+sam2_unavailable"
            return
        mask = sam2_segmenter.segment_bbox(image_path=image_path, bbox=ann["bbox"])
        if mask is None:
            entry["needs_review"] = True
            entry["source"] = f"{entry['source']}+sam2_empty"
            return
        entry["segmentation"] = binary_mask_to_uncompressed_rle(mask)
        entry["bbox"] = ann.get("bbox")
        entry["area"] = int(mask.sum())
        entry["source"] = f"{entry['source']}+sam2_bbox_prompt"
        return
    entry["needs_review"] = True


class HuggingFaceSAM2Segmenter:
    """BBox-prompted SAM2 segmenter using Hugging Face Transformers.

    This is concept-agnostic segmentation: the concept candidate comes from the
    ontology/LLM stage, and SAM2 converts the instance bbox into a pixel mask.
    """

    def __init__(self, model_name: str, device: str, threshold: float) -> None:
        try:
            import torch
            from transformers import Sam2Model, Sam2Processor
        except ImportError as exc:
            raise RuntimeError(
                "SAM2 pixel annotation requires Hugging Face Transformers with SAM2 support. "
                "Install `transformers` before using --pixel-policy sam2."
            ) from exc

        self.torch = torch
        self.processor = Sam2Processor.from_pretrained(model_name)
        self.model = Sam2Model.from_pretrained(model_name)
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        self.threshold = threshold
        self.model.to(self.device)
        self.model.eval()

    def segment_bbox(self, image_path: Path, bbox: list[float] | tuple[float, ...]) -> Any | None:
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        x, y, w, h = [float(v) for v in bbox]
        input_boxes = [[[x, y, x + w, y + h]]]
        inputs = self.processor(images=image, input_boxes=input_boxes, return_tensors="pt")
        inputs = {key: value.to(self.device) if hasattr(value, "to") else value for key, value in inputs.items()}
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        masks = self.processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )
        if not masks:
            return None
        mask = masks[0]
        while getattr(mask, "ndim", 0) > 2:
            mask = mask[0]
        return (mask > self.threshold).numpy().astype("uint8")


def binary_mask_to_uncompressed_rle(mask: Any) -> dict[str, Any]:
    height, width = mask.shape[:2]
    flat = mask.reshape(-1, order="F").astype("uint8")
    counts: list[int] = []
    last = 0
    run = 0
    for value in flat:
        value_int = int(value)
        if value_int == last:
            run += 1
        else:
            counts.append(run)
            run = 1
            last = value_int
    counts.append(run)
    return {"size": [int(height), int(width)], "counts": counts}


def _verify_candidates_with_llm(
    llm_client: "LLMClient | None",
    ontology: ConceptOntology,
    candidates: tuple[str, ...],
    class_name: str,
    image: Path | None,
    bbox: Any,
) -> dict[str, dict[str, Any]]:
    if llm_client is None:
        return {
            concept_id: {"present": True, "confidence": 1.0, "source": "ontology_auto"}
            for concept_id in candidates
        }

    prompt = build_concept_verification_prompt(ontology, candidates, class_name=class_name)
    try:
        response = llm_client.verify(prompt=prompt, image=image, bbox=bbox)
    except Exception as exc:  # Keep the automated workflow robust.
        return {
            concept_id: {
                "present": True,
                "confidence": 1.0,
                "source": "ontology_auto_llm_failed",
                "evidence": str(exc),
            }
            for concept_id in candidates
        }

    parsed = _parse_llm_concept_response(response)
    out: dict[str, dict[str, Any]] = {}
    for concept_id in candidates:
        out[concept_id] = parsed.get(
            concept_id,
            {"present": False, "confidence": 0.0, "source": "llm_missing"},
        )
        out[concept_id]["source"] = "llm_verified"
    return out


def build_concept_verification_prompt(
    ontology: ConceptOntology,
    candidates: tuple[str, ...],
    class_name: str,
) -> str:
    concept_lines = []
    for concept_id in candidates:
        concept = ontology.concepts[concept_id]
        concept_lines.append(
            f"- id: {concept.id}\n"
            f"  name: {concept.name}\n"
            f"  level: {concept.level}\n"
            f"  scope: {concept.scope}\n"
            f"  definition: {concept.description}"
        )
    return CONCEPT_VERIFICATION_PROMPT.format(
        class_name=class_name,
        concept_list="\n".join(concept_lines),
    )


def _parse_llm_concept_response(response: str) -> dict[str, dict[str, Any]]:
    payload = json.loads(response)
    concepts = payload.get("concepts", [])
    out: dict[str, dict[str, Any]] = {}
    if not isinstance(concepts, list):
        return out
    for item in concepts:
        if not isinstance(item, dict) or "id" not in item:
            continue
        out[str(item["id"])] = {
            "present": bool(item.get("present", False)),
            "confidence": float(item.get("confidence", 0.0)),
            "evidence": str(item.get("evidence", "")),
        }
    return out


class LLMClient:
    def verify(self, prompt: str, image: Path | None, bbox: Any) -> str:
        raise NotImplementedError


class LangChainOpenAIClient(LLMClient):
    def __init__(self, model: str) -> None:
        self.model = model
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is required for --llm-provider openai.")
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise RuntimeError(
                "LangChain OpenAI support requires `langchain-openai`. "
                "Install it before using --llm-provider openai."
            ) from exc
        self.llm = ChatOpenAI(model=self.model, temperature=0)

    def verify(self, prompt: str, image: Path | None, bbox: Any) -> str:
        try:
            from langchain_core.messages import HumanMessage
        except ImportError as exc:
            raise RuntimeError(
                "LangChain OpenAI support requires `langchain-core`. "
                "Install it before using --llm-provider openai."
            ) from exc

        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        image_url = _image_to_data_url(image, bbox=bbox) if image is not None and image.exists() else None
        if image_url is not None:
            content.append({"type": "image_url", "image_url": {"url": image_url}})
        response = self.llm.invoke([HumanMessage(content=content)])
        if isinstance(response.content, str):
            return response.content
        return json.dumps(response.content, ensure_ascii=False)


def _build_llm_client(llm_provider: str, llm_model: str) -> LLMClient | None:
    if llm_provider == "none":
        return None
    if llm_provider == "openai":
        return LangChainOpenAIClient(model=llm_model)
    raise ValueError(f"Unsupported llm provider: {llm_provider}")


def _image_to_data_url(image: Path, bbox: Any = None) -> str:
    if bbox is None:
        suffix = image.suffix.lower()
        mime = "image/png" if suffix == ".png" else "image/jpeg"
        encoded = base64.b64encode(image.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{encoded}"

    try:
        from PIL import Image

        with Image.open(image) as im:
            x, y, w, h = [float(v) for v in bbox]
            crop = im.crop((x, y, x + w, y + h)).convert("RGB")
            buf = BytesIO()
            crop.save(buf, format="JPEG", quality=92)
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"
    except Exception:
        encoded = base64.b64encode(image.read_bytes()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build COCO-compatible supervised concept annotations.")
    parser.add_argument("--concept-yaml", required=True, help="Concept ontology yaml.")
    parser.add_argument("--coco-json", required=True, help="Input COCO annotation json.")
    parser.add_argument("--out-json", required=True, help="Output COCO-compatible concept annotation json.")
    parser.add_argument("--image-root", default=None, help="Optional image root for VLM image inputs.")
    parser.add_argument("--llm-provider", default="none", choices=["none", "openai"])
    parser.add_argument("--llm-model", default="gpt-4o-mini")
    parser.add_argument("--max-annotations", type=int, default=None)
    parser.add_argument("--pixel-policy", default="needs_review", choices=["needs_review", "instance_mask", "sam2", "skip"])
    parser.add_argument("--sam2-model", default="facebook/sam2-hiera-large")
    parser.add_argument("--sam2-device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--sam2-threshold", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_coco_concept_annotations(
        coco_json=args.coco_json,
        concept_yaml=args.concept_yaml,
        out_json=args.out_json,
        image_root=args.image_root,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        max_annotations=args.max_annotations,
        pixel_policy=args.pixel_policy,
        sam2_model=args.sam2_model,
        sam2_device=args.sam2_device,
        sam2_threshold=args.sam2_threshold,
    )


if __name__ == "__main__":
    main()
