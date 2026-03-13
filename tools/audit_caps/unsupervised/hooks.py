from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from modules.neck import CapsRoutev2


@dataclass
class CapsuleActivation:
    name: str
    k_out: int
    p_out: int
    output: torch.Tensor


class CapsuleHookManager:
    """Capture CapsRoutev2 outputs for concept-level auditing."""

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.handles: list[Any] = []
        self.activations: dict[str, CapsuleActivation] = {}

    def clear(self) -> None:
        self.activations.clear()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.activations.clear()

    def register(self) -> None:
        self.close()
        self.activations = {}
        for name, module in self.model.named_modules():
            if isinstance(module, CapsRoutev2):
                self.handles.append(module.register_forward_hook(self._make_hook(name, module)))

    def _make_hook(self, name: str, module: CapsRoutev2):
        def hook(_module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            if output.requires_grad:
                output.retain_grad()
            self.activations[name] = CapsuleActivation(
                name=name,
                k_out=int(module.K_out),
                p_out=int(module.P_out),
                output=output,
            )

        return hook
