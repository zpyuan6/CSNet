from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import load_annotations
from .features import extract_feature_dataset


class LinearConceptProbe(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


def _fit_standardizer(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = x.mean(dim=0)
    std = x.std(dim=0).clamp_min(1e-6)
    return mean, std


def _standardize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x - mean) / std


def train_concept_probe(
    model_path: str,
    annotations_csv: str,
    concept: str,
    layer: str,
    out_path: str,
    imgsz: int = 640,
    device: str = "",
    epochs: int = 200,
    lr: float = 1e-2,
    weight_decay: float = 1e-4,
) -> dict:
    annotations = load_annotations(annotations_csv, concept=concept, layer=layer)
    x, y = extract_feature_dataset(model_path, annotations, layer=layer, imgsz=imgsz, device=device)

    mean, std = _fit_standardizer(x)
    x_std = _standardize(x, mean, std)

    model = LinearConceptProbe(x_std.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = None
    best_loss = float("inf")
    for _ in range(epochs):
        logits = model(x_std)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if float(loss.item()) < best_loss:
            best_loss = float(loss.item())
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Probe training did not produce a valid state.")

    model.load_state_dict(best_state)
    with torch.no_grad():
        logits = model(x_std)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        acc = float((preds == y).float().mean().item())

    out = {
        "concept": concept,
        "layer": layer,
        "imgsz": int(imgsz),
        "feature_dim": int(x.shape[1]),
        "samples": int(x.shape[0]),
        "train_loss": float(best_loss),
        "train_acc": acc,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "state_dict": {k: v.tolist() for k, v in model.state_dict().items()},
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def load_probe(probe_path: str | Path) -> dict:
    return json.loads(Path(probe_path).read_text(encoding="utf-8"))
