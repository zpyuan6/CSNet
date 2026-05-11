from __future__ import annotations

import torch
import torch.nn as nn


def _match_source_span(module: nn.Module, child_k: int, child_p: int) -> tuple[int, int]:
    k_list = [int(v) for v in getattr(module, "K_in_list", [])]
    p_list = [int(v) for v in getattr(module, "P_in_list", [])]
    if k_list and p_list and len(k_list) == len(p_list):
        start = 0
        fallback: tuple[int, int] | None = None
        for k_val, p_val in zip(k_list, p_list):
            end = start + k_val
            if k_val == int(child_k) and p_val == int(child_p):
                return start, end
            if fallback is None and k_val == int(child_k):
                fallback = (start, end)
            start = end
        if fallback is not None:
            return fallback
    total_k = int(getattr(module, "K_cat", getattr(module, "K_in", child_k)))
    span = min(int(child_k), total_k)
    return 0, max(span, 0)


def _self_routing_affinity(route: nn.Module) -> torch.Tensor | None:
    if not hasattr(route, "W_pose") or not hasattr(route, "W_gate"):
        return None
    pose = route.W_pose.detach().abs().mean(dim=(-1, -2))
    gate = route.W_gate.detach().abs().mean(dim=-1)
    return 0.5 * (pose + gate)


def _hybrid_route_affinity(route: nn.Module, k_in: int, p_in: int) -> torch.Tensor | None:
    if not hasattr(route, "gate_proj") or not hasattr(route, "act_proj"):
        return None
    gate_weight = route.gate_proj.weight.detach().abs()
    if gate_weight.ndim != 4:
        return None
    gate_weight = gate_weight.view(route.K_out, k_in, p_in + 1).mean(dim=-1).transpose(0, 1)
    act_weight = route.act_proj.weight.detach().abs()
    if act_weight.ndim != 4:
        return gate_weight
    act_weight = act_weight.squeeze(-1).squeeze(-1).transpose(0, 1)
    return 0.5 * (gate_weight + act_weight)


def route_affinity_matrix(module: nn.Module, child_k: int, child_p: int) -> torch.Tensor | None:
    route = getattr(module, "route1", None)
    if route is None:
        return None
    full_affinity = _self_routing_affinity(route)
    if full_affinity is None:
        full_affinity = _hybrid_route_affinity(route, int(getattr(module, "K_cat", child_k)), int(getattr(module, "P_cat", child_p)))
    if full_affinity is None:
        return None
    start, end = _match_source_span(module, child_k=child_k, child_p=child_p)
    if end <= start or start >= full_affinity.shape[0]:
        return None
    return full_affinity[start:min(end, full_affinity.shape[0])]


def route_rule_weight(
    module: nn.Module,
    parent_type_idx: int,
    child_k: int,
    child_p: int,
    child_type_idx: int,
) -> float | None:
    affinity = route_affinity_matrix(module, child_k=child_k, child_p=child_p)
    if affinity is None:
        return None
    if child_type_idx < 0 or child_type_idx >= affinity.shape[0]:
        return None
    if parent_type_idx < 0 or parent_type_idx >= affinity.shape[1]:
        return None
    return float(affinity[child_type_idx, parent_type_idx].item())
