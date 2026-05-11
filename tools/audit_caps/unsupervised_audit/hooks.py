from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from modules import CapsProj, CapsRoute, CapsRoutev2, CapsRoutev3, CapsRoutev4


@dataclass
class CapturedActivation:
    name: str
    module: nn.Module
    output: torch.Tensor
    k_out: int
    p_out: int
    inputs: tuple[torch.Tensor, ...]


class CapsuleHookManager:
    def __init__(self, model: nn.Module, target_layers: list[str] | None = None):
        self.model = model
        self.target_layers = None if not target_layers else {str(name) for name in target_layers}
        self.handles: list[torch.utils.hooks.RemovableHandle] = []
        self.activations: dict[str, CapturedActivation] = {}

    def clear(self) -> None:
        self.activations.clear()

    @staticmethod
    def _is_capsule_like(module: nn.Module) -> bool:
        return isinstance(module, (CapsProj, CapsRoute, CapsRoutev2, CapsRoutev3, CapsRoutev4))

    @staticmethod
    def _resolve_layout(module: nn.Module) -> tuple[int, int]:
        if isinstance(module, CapsProj):
            k_out = int(getattr(module, "K", 0))
            p_out = int(getattr(module, "D", 0))
        else:
            k_out = int(getattr(module, "K_out", getattr(module, "k_out", 0)))
            p_out = int(getattr(module, "P_out", getattr(module, "p_out", 0)))
        if k_out <= 0 or p_out <= 0:
            raise ValueError(f"Unable to resolve capsule layout from module {module.__class__.__name__}")
        return k_out, p_out

    @staticmethod
    def _resolve_generic_layout(output: torch.Tensor) -> tuple[int, int]:
        if output.ndim != 4:
            raise ValueError(f"Unsupported non-capsule output ndim={output.ndim}, expected BCHW.")
        return int(output.shape[1]), 0

    @staticmethod
    def _flatten_tensor_inputs(value) -> tuple[torch.Tensor, ...]:
        if isinstance(value, torch.Tensor):
            return (value,)
        if isinstance(value, (list, tuple)):
            tensors: list[torch.Tensor] = []
            for item in value:
                tensors.extend(CapsuleHookManager._flatten_tensor_inputs(item))
            return tuple(tensors)
        return ()

    def register(self) -> None:
        self.clear()
        for name, module in self.model.named_modules():
            if self.target_layers is None:
                if not self._is_capsule_like(module):
                    continue
            elif name not in self.target_layers:
                continue
            handle = module.register_forward_hook(self._make_hook(name))
            self.handles.append(handle)

    def _make_hook(self, name: str):
        def hook(module: nn.Module, inputs, output):
            if not isinstance(output, torch.Tensor):
                return
            if self.target_layers is not None and output.ndim != 4 and not self._is_capsule_like(module):
                return
            if output.requires_grad:
                output.retain_grad()
            tensor_inputs = self._flatten_tensor_inputs(inputs)
            if self._is_capsule_like(module):
                k_out, p_out = self._resolve_layout(module)
            else:
                k_out, p_out = self._resolve_generic_layout(output)
            self.activations[name] = CapturedActivation(
                name=name,
                module=module,
                output=output,
                k_out=k_out,
                p_out=p_out,
                inputs=tensor_inputs,
            )

        return hook

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.clear()
