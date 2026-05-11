from __future__ import annotations

from pathlib import Path

import yaml


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _collect_images(root: Path) -> list[str]:
    if root.is_file() and root.suffix.lower() in IMAGE_SUFFIXES:
        return [str(root)]
    if root.is_file() and root.suffix.lower() == ".txt":
        image_paths: list[str] = []
        for line in root.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            candidate = Path(line)
            if not candidate.is_absolute():
                candidate = root.parent / candidate
            image_paths.append(str(candidate.resolve()))
        return image_paths
    if not root.exists():
        return []
    return sorted(str(p) for p in root.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)


def resolve_split_paths(data_yaml: str | Path, split: str = "val") -> list[str]:
    data_path = Path(data_yaml)
    payload = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    entry = payload.get(split)
    if entry is None:
        raise KeyError(f"Split '{split}' not found in {data_yaml}")

    base = Path(payload.get("path") or data_path.parent)
    if not base.is_absolute():
        base = (data_path.parent / base).resolve()
    roots: list[Path] = []
    if isinstance(entry, str):
        roots = [base / entry]
    elif isinstance(entry, list):
        roots = [base / str(item) for item in entry]
    else:
        raise TypeError(f"Unsupported split entry type for '{split}': {type(entry)}")

    image_paths: list[str] = []
    for root in roots:
        root = root.resolve()
        image_paths.extend(_collect_images(root))
    return sorted(dict.fromkeys(image_paths))
