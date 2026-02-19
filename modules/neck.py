"""
CapsNeck: efficient capsule-style neck blocks for Ultralytics YAML models.

Design intent:
- Keep capsule semantics (type/channel grouping + routing-style fusion).
- Stay lightweight and export-friendly for detection training/inference.
- Avoid expensive iterative EM/dynamic routing inside the neck path.

This neck is "capsule-style" rather than a full matrix-capsule network:
1) CapsProj  : CNN feature -> packed capsules (K types * D dims)
2) CapsAlign : scale alignment between pyramid levels (no global context)
3) CapsRoute : efficient self-routing proxy across sources (softmax source gating)
4) CapsDecode: packed capsules -> standard feature map for Detect
5) CapsuleTap: optional pass-through cache hook for analysis/aux losses

Note:
- Routing here is source-level and single-step by default (iters=1), chosen for speed.
- If stronger capsule routing is needed, it should be added in the head where cost is lower.
"""


from __future__ import annotations

from typing import List, Optional, Tuple, Union

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules import C3k2, Conv


# -------------------------
# 1) CapsProj
# -------------------------

class CapsProj(nn.Module):
    """
    Project a standard feature map into packed capsule channels using one C3k2 block.

    Input:  x  [B, C, H, W]
    Output: u  [B, K*(D+1), H, W]

    Args:
      K: number of capsule types
      D: capsule pose dimension per type
      mix/mix_kernel: kept for backward YAML compatibility (unused)
    """

    def __init__(self, c1: int, K: int = 4, D: int = 16):
        super().__init__()
        self.K = int(K)
        self.D = int(D)
        self.c_out = self.K * (self.D + 1)

        # Use a single C3k2 block as the capsule projection operator.
        self.map = C3k2(c1, self.c_out, n=1, c3k=False, e=0.5, g=1, shortcut=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.map(x)


# -------------------------
# 2) CapsAlign (no context)
# -------------------------

class CapsAlign(nn.Module):
    """
    Align packed capsules across pyramid levels with YOLO-style ops.

    - Upsampling uses ``nn.Upsample(scale_factor=2, mode='nearest')``.
    - Downsampling uses stride-2 ``Conv`` blocks.

    Args:
      c1: input/output channel count.
      src_level: source pyramid level in {3,4,5}.
      tgt_level: target pyramid level in {3,4,5}.
      down_groups: groups for downsample Conv.
        Use capsule-type count K to keep each capsule block isolated.
    """

    def __init__(self, c1: int, src_level: int, tgt_level: int, down_groups: int = 1):
        super().__init__()
        self.c1 = int(c1)
        self.src_level = int(src_level)
        self.tgt_level = int(tgt_level)
        self.down_groups = int(down_groups)

        if self.src_level not in (3, 4, 5) or self.tgt_level not in (3, 4, 5):
            raise ValueError("CapsAlign levels must be in {3,4,5}.")

        if self.down_groups < 1 or self.c1 % self.down_groups != 0:
            raise ValueError(f"CapsAlign down_groups={self.down_groups} must divide c1={self.c1}.")

        steps = abs(self.src_level - self.tgt_level)
        if self.src_level == self.tgt_level:
            self.mode = 'identity'
            self.ops = nn.ModuleList()
        elif self.src_level > self.tgt_level:
            self.mode = 'up'
            # YOLO-style top-down path: nearest-neighbor upsample x2 per level.
            self.ops = nn.ModuleList(nn.Upsample(scale_factor=2, mode='nearest') for _ in range(steps))
        else:
            self.mode = 'down'
            # YOLO-style bottom-up path: stride-2 grouped Conv per level.
            self.ops = nn.ModuleList(Conv(self.c1, self.c1, 3, 2, g=self.down_groups) for _ in range(steps))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == 'identity':
            return x

        for op in self.ops:
            x = op(x)
        return x


# -------------------------
# 3) CapsRoute (light, parser-friendly)
# -------------------------

class ConvSelfRouting(nn.Module):
    """Grouped-conv self-routing over stacked capsule sources.

    Args:
      K_in: input capsule type count.
      P_in: input pose dimension.
      K_out: output capsule type count.
      P_out: output pose dimension.
      kernel_size: grouped conv kernel for local capsule mixing.
    """

    def __init__(self, K_in: int, P_in: int, K_out: int, P_out: int, kernel_size: int = 3):
        super().__init__()
        self.K_in = int(K_in)
        self.P_in = int(P_in)
        self.K_out = int(K_out)
        self.P_out = int(P_out)

        if min(self.K_in, self.P_in, self.K_out, self.P_out) <= 0:
            raise ValueError('ConvSelfRouting expects positive K/P values.')

        self.c_in = self.K_in * (self.P_in + 1)
        self.c_out = self.K_out * (self.P_out + 1)

        k = int(kernel_size)
        padding = k//2
        self.mix = nn.Conv2d(self.c_in, self.c_in, kernel_size=k, stride=1, padding=padding, groups=self.K_in, bias=False)
        self.gate = nn.Conv2d(self.c_in, self.K_in, kernel_size=1, stride=1, padding=0, groups=self.K_in, bias=True)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,H,W], C = K_in*(P_in+1)

        b, c, h, w = x.shape
        if c != self.c_in:
            raise ValueError(f'ConvSelfRouting expected C={self.c_in}, got C={c}')

        mixed = self.mix(x)
        logits = self.gate(mixed).reshape(b, self.K_in, h, w)
        weights = logits.softmax(dim=1)

        caps = mixed.reshape(b, self.K_in, self.P_in + 1, h, w)
        routed = weights.unsqueeze(2) * caps
        routed = routed.reshape(b, self.c_in, h, w)

        return routed


class SelfRouting(nn.Module):
    """Pose-transform self-routing on packed capsule tensor.

    Args:
      K_in: input capsule type count.
      P_in: input pose dimension.
      K_out: output capsule type count.
      P_out: output pose dimension.

    Input:
      x: [B, K_in*(P_in+1), H, W]

    Output:
      y: [B, K_out*(P_out+1), H, W]
    """

    def __init__(self, K_in: int, P_in: int, K_out: int, P_out: int):
        super().__init__()
        self.K_in = int(K_in)
        self.P_in = int(P_in)
        self.K_out = int(K_out)
        self.P_out = int(P_out)
        if min(self.K_in, self.P_in, self.K_out, self.P_out) <= 0:
            raise ValueError('SelfRouting expects positive K/P values.')

        self.c_in = self.K_in * (self.P_in + 1)
        self.c_out = self.K_out * (self.P_out + 1)
        self.eps = 1e-6

        self.W_pose = nn.Parameter(torch.empty(self.K_in, self.K_out, self.P_in, self.P_out))
        nn.init.kaiming_uniform_(self.W_pose, a=math.sqrt(5))
        self.W_gate = nn.Parameter(torch.zeros(self.K_in, self.K_out, self.P_in))
        self.b_gate = nn.Parameter(torch.zeros(1, self.K_in, self.K_out, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W], C = K_in*(P_in+1)
        if x.ndim != 4:
            raise TypeError(f'SelfRouting expects [B,C,H,W], got {tuple(x.shape)}')

        b, c, h, w = x.shape
        if c != self.c_in:
            raise ValueError(f'SelfRouting expected C={self.c_in}, got C={c}')

        # Packed capsule layout is interleaved per type: [pose(P), act(1)].
        # x_caps: [B, K_in, P_in+1, H, W]
        x_caps = x.reshape(b, self.K_in, self.P_in + 1, h, w)
        pose = x_caps[:, :, :self.P_in]  # [B, K_in, P_in, H, W]
        act = x_caps[:, :, self.P_in : self.P_in + 1].sigmoid()  # [B, K_in, 1, H, W]

        # votes: [B, K_in, K_out, H, W, P_out]
        votes = torch.einsum('bkphw,kopq->bkohwq', pose, self.W_pose)
        # logits/weights: [B, K_in, K_out, H, W]
        logits = torch.einsum('bkphw,kop->bkohw', pose, self.W_gate) + self.b_gate
        weights = logits.softmax(dim=2)

        ar = weights * act  # [B, K_in, K_out, H, W]
        ar_sum = ar.sum(dim=1, keepdim=True) + self.eps
        coeff = ar / ar_sum

        pose_out = (coeff.unsqueeze(-1) * votes).sum(dim=1)  # [B, K_out, H, W, P_out]
        pose_out = pose_out.permute(0, 1, 4, 2, 3)  # [B, K_out, P_out, H, W]
        act_out = ar_sum.squeeze(1).unsqueeze(2)  # [B, K_out, 1, H, W]

        # Keep interleaved packed output: [pose(P_out), act(1)] per capsule type.
        out = torch.cat([pose_out, act_out], dim=2).reshape(b, self.c_out, h, w)
        return out


class CapsRoute(nn.Module):
    """Capsule routing fusion by direct capsule concatenation.

    Args:
      K_in: list of input capsule type counts per source.
      P_in: list of input pose dimensions per source.
      K_out: target output capsule type count.
      P_out: target output pose dimension.
      kernel_size: grouped-conv kernel for ``ConvSelfRouting``.

    Notes:
      Inputs are concatenated directly (no pre-projection).
      For direct packed concat, all ``P_in`` must be identical.
    """

    def __init__(
        self,
        K_in: Union[List[int], Tuple[int, ...]],
        P_in: Union[List[int], Tuple[int, ...]],
        K_out: int,
        P_out: int,
        kernel_size: int = 3,
        pre_k: int = 3,
        post_k: int = 3,
        pre_groups: Optional[int] = None,
        post_groups: Optional[int] = None,
    ):
        super().__init__()
        self.K_in_list = [int(v) for v in K_in]
        self.P_in_list = [int(v) for v in P_in]
        if len(self.K_in_list) < 2 or len(self.K_in_list) != len(self.P_in_list):
            raise ValueError('CapsRoute expects K_in/P_in lists with same length >= 2.')
        if min(*self.K_in_list, *self.P_in_list) <= 0:
            raise ValueError('CapsRoute expects positive K_in/P_in values.')

        # Direct capsule concat requires a shared pose dimension.
        if len(set(self.P_in_list)) != 1:
            raise ValueError('CapsRoute direct concat requires all P_in to be identical.')

        self.num_sources = len(self.K_in_list)
        self.P_cat = int(self.P_in_list[0])
        self.K_cat = int(sum(self.K_in_list))
        self.c_cat = self.K_cat * (self.P_cat + 1)

        self.K_out = int(K_out)
        self.P_out = int(P_out)
        if min(self.K_out, self.P_out) <= 0:
            raise ValueError('CapsRoute expects positive K_out/P_out values.')
        self.c_out = self.K_out * (self.P_out + 1)

        # self.conv_route = ConvSelfRouting(
        #     K_in=self.K_cat,
        #     P_in=self.P_cat,
        #     K_out=self.K_cat,
        #     P_out=self.P_cat,
        #     kernel_size=kernel_size,
        # )
        # Grouped Conv before routing: C = K_cat * (P_cat + 1), groups = K_cat.
        self.conv_route = Conv(self.c_cat, self.c_cat, 3, 1, g=self.K_cat)
        self.route1 = SelfRouting(K_in=self.K_cat, P_in=self.P_cat, K_out=self.K_out, P_out=self.P_out)
        # Grouped Conv after routing: C = K_out * (P_out + 1), groups = K_out.
        self.spagg = Conv(self.c_out, self.c_out, 3, 1, g=self.K_out)
        # self.route2 = SelfRouting(K_in=self.K_out, P_in=self.P_out, K_out=self.K_out, P_out=self.P_out)

    def forward(self, xs: Union[List[torch.Tensor], Tuple[torch.Tensor, ...]]) -> torch.Tensor:
        if not isinstance(xs, (list, tuple)):
            raise TypeError(f'CapsRoute expects list/tuple inputs, got {type(xs)}')
        if len(xs) != self.num_sources:
            raise ValueError(f'CapsRoute expected {self.num_sources} sources, got {len(xs)}')

        h, w = int(xs[0].shape[-2]), int(xs[0].shape[-1])
        cat_parts = []
        for i, x in enumerate(xs):
            expected_c = self.K_in_list[i] * (self.P_in_list[i] + 1)
            if int(x.shape[1]) != expected_c:
                raise ValueError(f'CapsRoute source-{i} expected C={expected_c} from K_in/P_in, got C={int(x.shape[1])}')
            if int(x.shape[-2]) != h or int(x.shape[-1]) != w:
                raise ValueError('CapsRoute inputs must share H,W. Use CapsAlign before routing.')
            cat_parts.append(x)

        x_cat = torch.cat(cat_parts, dim=1)  # [B, K_cat*(P+1), H, W]
        routed = self.route1(self.conv_route(x_cat))
        routed = self.spagg(routed)
        return routed


# -------------------------
# 4) CapsDecode
# -------------------------

class CapsDecode(nn.Module):
    """
    Decode routed capsule features to standard feature map for Detect.

    Input:  y [B, C_in, H, W]  (often concat of weighted sources, so C_in = S*(K*D))
    Output: f [B, C_out, H, W]

    Args:
      c2: output channels (e.g., 256/512/1024)
    """

    def __init__(self, c1: int, c2: int):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


# -------------------------
# 5) CapsuleTap
# -------------------------

class CapsuleTap(nn.Module):
    """
    Pass-through hook to cache feature maps for explainability/aux loss.

    MUST NOT change tensor shape. Returns x unchanged.

    Args:
      tag: string identifier ("F3"/"F4"/"F5")
      K,D: capsule hyperparams (metadata only)
      cache_enabled: if True, cache during training (disabled in tracing/scripting)
    """

    def __init__(self, tag: str = "F", K: int = 4, D: int = 16, cache_enabled: bool = True):
        super().__init__()
        self.tag = str(tag)
        self.K = int(K)
        self.D = int(D)
        self.cache_enabled = bool(cache_enabled)
        self.last_x: Optional[torch.Tensor] = None

    def clear_cache(self) -> None:
        self.last_x = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
            self.cache_enabled
            and self.training
            and (not torch.jit.is_scripting())
            and (not torch.jit.is_tracing())
        ):
            self.last_x = x.detach()
        return x