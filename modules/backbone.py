from __future__ import annotations

from typing import Iterable, Sequence

import torch
from torch import nn
from torch.nn import functional as F


def squash(x: torch.Tensor, dim: int = -1, eps: float = 1e-7) -> torch.Tensor:
    """Squash nonlinearity used by capsule networks."""
    squared_norm = (x * x).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1.0 + squared_norm)
    return scale * x / torch.sqrt(squared_norm + eps)


class ConvBNAct(nn.Module):
    """Convolution + BatchNorm + SiLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
    ) -> None:
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class PrimaryCaps2d(nn.Module):
    """Primary capsule layer for 2D feature maps."""

    def __init__(
        self,
        in_channels: int,
        num_caps: int,
        dim_caps: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int | None = None,
    ) -> None:
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        out_channels = num_caps * dim_caps
        self.num_caps = num_caps
        self.dim_caps = dim_caps
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn(self.conv(x)))
        bsz, _, h, w = x.shape
        x = x.view(bsz, self.num_caps, self.dim_caps, h, w)
        return squash(x, dim=2)


class RoutingCaps(nn.Module):
    """Dynamic routing between capsules."""

    def __init__(
        self,
        num_in_caps: int,
        dim_in: int,
        num_out_caps: int,
        dim_out: int,
        routing_iters: int = 3,
    ) -> None:
        super().__init__()
        self.num_in_caps = num_in_caps
        self.dim_in = dim_in
        self.num_out_caps = num_out_caps
        self.dim_out = dim_out
        self.routing_iters = routing_iters

        weight = torch.randn(1, num_in_caps, num_out_caps, dim_out, dim_in) * 0.01
        self.W = nn.Parameter(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"RoutingCaps expects [B, N, D], got {tuple(x.shape)}")
        bsz = x.shape[0]
        x = x.unsqueeze(2).unsqueeze(-1)  # [B, N, 1, D, 1]
        u_hat = torch.matmul(self.W, x).squeeze(-1)  # [B, N, M, Dout]
        b = x.new_zeros(bsz, self.num_in_caps, self.num_out_caps)
        for idx in range(self.routing_iters):
            c = F.softmax(b, dim=-1)
            s = (c.unsqueeze(-1) * u_hat).sum(dim=1)
            v = squash(s, dim=-1)
            if idx < self.routing_iters - 1:
                b = b + (u_hat * v.unsqueeze(1)).sum(dim=-1)
        return v


class DeformableCaps2d(nn.Module):
    """Deformable capsule layer with learned sampling offsets."""

    def __init__(
        self,
        in_channels: int,
        num_child_caps: int = 8,
        dim_child: int = 8,
        num_parent_caps: int = 8,
        dim_parent: int = 8,
        num_samples: int = 4,
        routing_iters: int = 3,
        offset_scale: float = 1.0,
        out_channels: int | None = None,
    ) -> None:
        super().__init__()
        self.num_child_caps = num_child_caps
        self.dim_child = dim_child
        self.num_parent_caps = num_parent_caps
        self.dim_parent = dim_parent
        self.num_samples = num_samples
        self.routing_iters = routing_iters
        self.offset_scale = offset_scale

        self.primary = PrimaryCaps2d(in_channels, num_child_caps, dim_child, kernel_size=1, stride=1, padding=0)
        self.offset = nn.Conv2d(in_channels, 2 * num_samples, kernel_size=3, stride=1, padding=1)
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)

        self.routing = RoutingCaps(
            num_in_caps=num_samples * num_child_caps,
            dim_in=dim_child,
            num_out_caps=num_parent_caps,
            dim_out=dim_parent,
            routing_iters=routing_iters,
        )

        caps_channels = num_parent_caps * dim_parent
        self.out_channels = out_channels or caps_channels
        self.project = None
        if self.out_channels != caps_channels:
            self.project = ConvBNAct(caps_channels, self.out_channels, kernel_size=1, stride=1, padding=0)

    @staticmethod
    def _base_grid(h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        ys = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack((xx, yy), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        child = self.primary(x)  # [B, Nc, Dc, H, W]
        bsz, _, _, h, w = child.shape

        child_flat = child.view(bsz, self.num_child_caps * self.dim_child, h, w)
        offsets = self.offset(x).view(bsz, self.num_samples, 2, h, w)
        offsets = torch.tanh(offsets) * self.offset_scale

        scale_x = max(w - 1, 1) / 2.0
        scale_y = max(h - 1, 1) / 2.0
        scale = offsets.new_tensor([scale_x, scale_y]).view(1, 1, 2, 1, 1)
        offsets = offsets / scale

        base = self._base_grid(h, w, x.device, x.dtype).view(1, 1, h, w, 2)
        grids = base + offsets.permute(0, 1, 3, 4, 2)

        sampled = []
        for idx in range(self.num_samples):
            grid = grids[:, idx]
            feat = F.grid_sample(
                child_flat,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )
            sampled.append(feat)

        sampled = torch.stack(sampled, dim=1)
        sampled = sampled.view(bsz, self.num_samples, self.num_child_caps, self.dim_child, h, w)
        sampled = sampled.permute(0, 4, 5, 1, 2, 3).contiguous()
        sampled = sampled.view(bsz * h * w, self.num_samples * self.num_child_caps, self.dim_child)

        routed = self.routing(sampled)
        routed = routed.view(bsz, h, w, self.num_parent_caps, self.dim_parent)
        routed = routed.permute(0, 3, 4, 1, 2).contiguous()
        out = routed.view(bsz, self.num_parent_caps * self.dim_parent, h, w)

        if self.project is not None:
            out = self.project(out)
        return out


class DeformableCapsBlock(nn.Module):
    """Backbone block: Conv downsample + deformable capsule routing."""

    def __init__(
        self,
        c1: int,
        c2: int,
        num_child_caps: int = 8,
        dim_child: int = 8,
        num_parent_caps: int = 8,
        dim_parent: int = 8,
        num_samples: int = 4,
        routing_iters: int = 3,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.down = ConvBNAct(c1, c2, kernel_size=3, stride=stride)
        self.caps = DeformableCaps2d(
            in_channels=c2,
            num_child_caps=num_child_caps,
            dim_child=dim_child,
            num_parent_caps=num_parent_caps,
            dim_parent=dim_parent,
            num_samples=num_samples,
            routing_iters=routing_iters,
            out_channels=c2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down(x)
        return self.caps(x)


class CapsuleBackbone(nn.Module):
    """Simple capsule-based backbone that returns multi-scale features."""

    def __init__(
        self,
        in_channels: int = 3,
        stem_channels: int = 64,
        stages: Sequence[int] = (128, 256, 512),
        capsule_cfgs: Iterable[dict] | None = None,
    ) -> None:
        super().__init__()
        self.stem = ConvBNAct(in_channels, stem_channels, kernel_size=3, stride=2)
        stage_cfgs = list(capsule_cfgs) if capsule_cfgs is not None else [{}] * len(stages)
        if len(stage_cfgs) != len(stages):
            raise ValueError("capsule_cfgs must match stages length")

        blocks = []
        in_ch = stem_channels
        for out_ch, cfg in zip(stages, stage_cfgs):
            blocks.append(
                nn.Sequential(
                    ConvBNAct(in_ch, out_ch, kernel_size=3, stride=2),
                    DeformableCaps2d(out_ch, out_channels=out_ch, **cfg),
                )
            )
            in_ch = out_ch
        self.stages = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        x = self.stem(x)
        outputs = []
        for stage in self.stages:
            x = stage(x)
            outputs.append(x)
        return tuple(outputs)


__all__ = [
    "CapsuleBackbone",
    "ConvBNAct",
    "DeformableCaps2d",
    "DeformableCapsBlock",
    "PrimaryCaps2d",
    "RoutingCaps",
    "squash",
]
