from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules import Conv, DWConv, Detect, Segment
from ultralytics.nn.modules.block import Proto26


class PrimaryCaps(nn.Module):
    r"""Primary convolutional capsules.

    Outputs pose and activation, plus a concatenated NHWC capsule tensor.

    Args:
        A: Input feature channels.
        B: Number of capsule types.
        K: Convolution kernel size.
        P: Pose matrix side length (pose size is ``P*P``).
        stride: Convolution stride.

    Input shape:
        x: ``(N, A, H, W)``

    Output shape:
        a: ``(N, B, H_out, W_out)``
        p: ``(N, B*P*P, H_out, W_out)``
        out: ``(N, H_out, W_out, B*(P*P+1))``

    Parameter size:
        pose conv + act conv
        ``(K*K*A*B*P*P + B*P*P) + (K*K*A*B + B)``
    """

    def __init__(self, A: int = 32, B: int = 32, K: int = 1, P: int = 4, stride: int = 1):
        super().__init__()
        self.B = B
        self.P = P
        self.psize = P * P

        self.pose = nn.Conv2d(in_channels=A, out_channels=B * self.psize, kernel_size=K, stride=stride, bias=True)
        self.a = nn.Conv2d(in_channels=A, out_channels=B, kernel_size=K, stride=stride, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # p: (B, B*psize, H, W), a: (B, B, H, W)
        p = self.pose(x)
        a = self.sigmoid(self.a(x))
        out = torch.cat([p, a], dim=1).permute(0, 2, 3, 1).contiguous()  # (B, H, W, B*(psize+1))
        return a, p, out


class ConvCaps(nn.Module):
    r"""Convolutional capsules with EM routing.

    Args:
        B: Input capsule types.
        C: Output capsule types.
        K: Patch kernel size.
        P: Pose matrix side length (pose size is ``P*P``).
        stride: Spatial stride for patch extraction.
        iters: Number of EM routing iterations.
        coor_add: Add coordinate offsets (class-caps style option).
        w_shared: Share transform matrices across spatial positions.

    Input shape:
        x: ``(N, H, W, B*(P*P+1))``

    Output shape:
        p_out: ``(N, H_out, W_out, C*P*P)``
        a_out: ``(N, H_out, W_out, C)``
        out: ``(N, H_out, W_out, C*(P*P+1))``

    Parameter size:
        If ``w_shared=False``:
        ``weights: (K*K*B*C*P*P*P*P)``, ``beta_u: C``, ``beta_a: C``

        If ``w_shared=True``:
        ``weights: (B*C*P*P*P*P)``, ``beta_u: C``, ``beta_a: C``

        Total = ``weights + 2*C`` (excluding non-trainable buffers).
    """

    def __init__(
        self,
        B: int = 32,
        C: int = 32,
        K: int = 3,
        P: int = 4,
        stride: int = 1,
        iters: int = 3,
        coor_add: bool = False,
        w_shared: bool = False,
    ):
        super().__init__()
        self.B = B
        self.C = C
        self.K = K
        self.P = P
        self.psize = P * P
        self.stride = stride
        self.iters = iters
        self.coor_add = coor_add
        self.w_shared = w_shared

        self.eps = 1e-6
        self._lambda = 1e-3
        self.register_buffer("ln_2pi", torch.tensor(math.log(2 * math.pi), dtype=torch.float32), persistent=False)

        # Matrix-caps paper uses per-capsule beta scalars.
        self.beta_u = nn.Parameter(torch.zeros(C))
        self.beta_a = nn.Parameter(torch.zeros(C))

        # For non-shared conv-caps, input vote count is K*K*B. For shared mode it is B then repeated by HW.
        weight_in = B if w_shared else (K * K * B)
        self.weights = nn.Parameter(torch.randn(1, weight_in, C, self.psize, self.psize) * 0.02)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=2)

    def m_step(
        self,
        a_in: torch.Tensor,
        r: torch.Tensor,
        v: torch.Tensor,
        eps: float,
        b: int,
        B: int,
        C: int,
        psize: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # a_in: (b, B, 1) or (b, B, 1, 1), r: (b, B, C, 1), v: (b, B, C, psize)
        if a_in.ndim == 3:
            a_in = a_in.unsqueeze(2)
        r = r * a_in
        r = r / (r.sum(dim=2, keepdim=True) + eps)
        r_sum = r.sum(dim=1, keepdim=True)
        coeff = r / (r_sum + eps)

        mu = torch.sum(coeff * v, dim=1, keepdim=True)  # (b, 1, C, psize)
        sigma_sq = torch.sum(coeff * (v - mu).pow(2), dim=1, keepdim=True) + eps
        sigma_sq = sigma_sq.clamp_min(1e-4)

        r_sum_flat = r_sum.view(b, C, 1)
        sigma_sq_flat = sigma_sq.view(b, C, psize).clamp_min(1e-4)
        cost_h = (self.beta_u.view(1, C, 1) + torch.log(torch.sqrt(sigma_sq_flat))) * r_sum_flat
        a_out = self.sigmoid(self._lambda * (self.beta_a.view(1, C) - cost_h.sum(dim=2))).clamp(1e-4, 1.0 - 1e-4)

        mu = torch.nan_to_num(mu, nan=0.0, posinf=1e4, neginf=-1e4)
        sigma_sq = torch.nan_to_num(sigma_sq, nan=1e-4, posinf=1e4, neginf=1e-4)
        a_out = torch.nan_to_num(a_out, nan=0.5, posinf=1.0 - 1e-4, neginf=1e-4)
        return a_out, mu, sigma_sq

    def e_step(
        self,
        mu: torch.Tensor,
        sigma_sq: torch.Tensor,
        a_out: torch.Tensor,
        v: torch.Tensor,
        eps: float,
        b: int,
        C: int,
    ) -> torch.Tensor:
        # mu: (b,1,C,psize), sigma_sq: (b,1,C,psize), a_out: (b,C), v: (b,B,C,psize)
        sigma_sq = sigma_sq.clamp_min(1e-4)
        a_out = a_out.clamp(1e-4, 1.0 - 1e-4)
        ln_p_j_h = -1.0 * (v - mu).pow(2) / (2.0 * sigma_sq) - torch.log(torch.sqrt(sigma_sq)) - 0.5 * self.ln_2pi
        ln_ap = ln_p_j_h.sum(dim=3) + torch.log(a_out.view(b, 1, C) + eps)
        ln_ap = torch.nan_to_num(ln_ap, nan=0.0, posinf=50.0, neginf=-50.0)
        r = self.softmax(ln_ap).unsqueeze(-1)  # (b,B,C,1)
        r = torch.nan_to_num(r, nan=(1.0 / max(C, 1)), posinf=1.0, neginf=0.0)
        return r

    def caps_em_routing(self, v: torch.Tensor, a_in: torch.Tensor, C: int, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
        b, B, _, psize = v.shape
        r = v.new_full((b, B, C, 1), 1.0 / C)

        for t in range(self.iters):
            a_out, mu, sigma_sq = self.m_step(a_in, r, v, eps, b, B, C, psize)
            if t < self.iters - 1:
                r = self.e_step(mu, sigma_sq, a_out, v, eps, b, C)

        # p_out: (b, C, psize), a_out: (b, C)
        p_out = torch.nan_to_num(mu.squeeze(1), nan=0.0, posinf=1e4, neginf=-1e4)
        a_out = torch.nan_to_num(a_out, nan=0.5, posinf=1.0 - 1e-4, neginf=1e-4)
        return p_out, a_out

    def add_pathes(self, x: torch.Tensor, B: int, K: int, psize: int, stride: int) -> tuple[torch.Tensor, int, int]:
        # x: (b, h, w, B*(psize+1)) -> patches: (b, oh, ow, K*K, B*(psize+1))
        b, h, w, c = x.shape
        x_chw = x.permute(0, 3, 1, 2).contiguous()
        pad = K // 2
        patches = F.unfold(x_chw, kernel_size=K, padding=pad, stride=stride)

        oh = (h + 2 * pad - K) // stride + 1
        ow = (w + 2 * pad - K) // stride + 1
        patches = patches.transpose(1, 2).contiguous().view(b, oh, ow, K * K, c)
        return patches, oh, ow

    def transform_view(self, x: torch.Tensor, w: torch.Tensor, C: int, P: int, w_shared: bool = False) -> torch.Tensor:
        # x: (b, in_votes, psize), w: (1, in_votes_base, C, psize, psize)
        b, in_votes, psize = x.shape
        assert psize == P * P

        w0 = w[0]
        if w_shared:
            base = w0.size(0)
            reps = in_votes // base
            w0 = w0.repeat(reps, 1, 1, 1)

        # (b, in_votes, C, psize)
        v = torch.einsum("bip,icpq->bicq", x, w0)
        return v

    def add_coord(self, v: torch.Tensor, b: int, h: int, w: int, B: int, C: int, psize: int) -> torch.Tensor:
        # v: (b, h*w*B, C, psize)
        # Supports rectangular feature maps (h != w).
        v = v.view(b, h, w, B, C, psize)

        device = v.device
        dtype = v.dtype
        coor_h_vals = torch.arange(h, dtype=dtype, device=device) / float(max(h, 1))
        coor_w_vals = torch.arange(w, dtype=dtype, device=device) / float(max(w, 1))

        coor_h = torch.zeros(1, h, 1, 1, 1, psize, dtype=dtype, device=device)
        coor_w = torch.zeros(1, 1, w, 1, 1, psize, dtype=dtype, device=device)
        coor_h[0, :, 0, 0, 0, 0] = coor_h_vals
        coor_w[0, 0, :, 0, 0, 1] = coor_w_vals

        v = (v + coor_h + coor_w).view(b, h * w * B, C, psize)
        return v

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x shape: (b, h, w, B*(psize+1))
        b, h, w, c = x.shape

        if not self.w_shared:
            patches, oh, ow = self.add_pathes(x, self.B, self.K, self.psize, self.stride)

            p_in = patches[..., : self.B * self.psize].contiguous().view(b * oh * ow, self.K * self.K * self.B, self.psize)
            a_in = patches[..., self.B * self.psize :].contiguous().view(b * oh * ow, self.K * self.K * self.B, 1)

            v = self.transform_view(p_in, self.weights, self.C, self.P, w_shared=False)
            p_out, a_out = self.caps_em_routing(v, a_in, self.C, self.eps)

            p_out = p_out.view(b, oh, ow, self.C * self.psize)
            a_out = a_out.view(b, oh, ow, self.C)
            out = torch.cat([p_out, a_out], dim=3)
        else:
            assert c == self.B * (self.psize + 1)
            assert self.K == 1 
            assert self.stride == 1

            p_in = x[..., : self.B * self.psize].contiguous().view(b, h * w * self.B, self.psize)
            a_in = x[..., self.B * self.psize :].contiguous().view(b, h * w * self.B, 1)

            v = self.transform_view(p_in, self.weights, self.C, self.P, w_shared=True)
            if self.coor_add:
                v = self.add_coord(v, b, h, w, self.B, self.C, self.psize)

            p_cls, a_cls = self.caps_em_routing(v, a_in, self.C, self.eps)

            # Broadcast class capsules back to spatial map for Detect-style dense outputs.
            p_out = p_cls.reshape(b, 1, 1, self.C * self.psize).expand(b, h, w, self.C * self.psize)
            a_out = a_cls.unsqueeze(1).unsqueeze(1).expand(b, h, w, self.C)
            out = torch.cat([p_out, a_out], dim=3)

        return p_out, a_out, out



class DynamicConvCaps(nn.Module):
    r"""Convolutional capsules with Sabour-style dynamic routing.

    This layer keeps the same tensor interface as ``ConvCaps``:
        input:  (N, H, W, B*(P*P+1))
        output: p_out (N, H_out, W_out, C*P*P), a_out (N, H_out, W_out, C), out concat

    Args:
        B: Input capsule types.
        C: Output capsule types.
        K: Patch kernel size.
        P: Pose matrix side length.
        stride: Patch stride.
        iters: Routing iterations.
        coor_add: Add coordinates in shared mode.
        w_shared: Share transforms across spatial positions (requires K=1, stride=1).
    """

    def __init__(
        self,
        B: int = 32,
        C: int = 32,
        K: int = 3,
        P: int = 4,
        stride: int = 1,
        iters: int = 3,
        coor_add: bool = False,
        w_shared: bool = False,
    ):
        super().__init__()
        self.B = B
        self.C = C
        self.K = K
        self.P = P
        self.psize = P * P
        self.stride = stride
        self.iters = iters
        self.coor_add = coor_add
        self.w_shared = w_shared
        self.eps = 1e-6

        weight_in = B if w_shared else (K * K * B)
        self.weights = nn.Parameter(torch.randn(1, weight_in, C, self.psize, self.psize) * 0.02)

    @staticmethod
    def squash(s: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
        s2 = (s * s).sum(dim=dim, keepdim=True)
        scale = s2 / (1.0 + s2)
        return scale * s / torch.sqrt(s2 + eps)

    def add_pathes(self, x: torch.Tensor, K: int, stride: int) -> tuple[torch.Tensor, int, int]:
        b, h, w, c = x.shape
        x_chw = x.permute(0, 3, 1, 2).contiguous()
        pad = K // 2
        patches = F.unfold(x_chw, kernel_size=K, padding=pad, stride=stride)
        oh = (h + 2 * pad - K) // stride + 1
        ow = (w + 2 * pad - K) // stride + 1
        patches = patches.transpose(1, 2).contiguous().view(b, oh, ow, K * K, c)
        return patches, oh, ow

    def transform_view(self, x: torch.Tensor, w_shared: bool) -> torch.Tensor:
        # x: (b, in_votes, psize) -> votes: (b, in_votes, C, psize)
        b, in_votes, psize = x.shape
        if psize != self.psize:
            raise ValueError('Invalid pose size for DynamicConvCaps')

        w0 = self.weights[0]
        if w_shared:
            base = w0.size(0)
            reps = in_votes // base
            w0 = w0.repeat(reps, 1, 1, 1)

        return torch.einsum('bip,icpq->bicq', x, w0)

    def add_coord(self, v: torch.Tensor, b: int, h: int, w: int, B: int, C: int, psize: int) -> torch.Tensor:
        # v: (b, h*w*B, C, psize)
        v = v.view(b, h, w, B, C, psize)
        device, dtype = v.device, v.dtype
        coor_h_vals = torch.arange(h, dtype=dtype, device=device) / float(max(h, 1))
        coor_w_vals = torch.arange(w, dtype=dtype, device=device) / float(max(w, 1))

        coor_h = torch.zeros(1, h, 1, 1, 1, psize, dtype=dtype, device=device)
        coor_w = torch.zeros(1, 1, w, 1, 1, psize, dtype=dtype, device=device)
        coor_h[0, :, 0, 0, 0, 0] = coor_h_vals
        coor_w[0, 0, :, 0, 0, 1] = coor_w_vals

        return (v + coor_h + coor_w).view(b, h * w * B, C, psize)

    def dynamic_routing(self, v: torch.Tensor, a_in: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # v: (n, in_votes, C, psize), a_in: (n, in_votes, 1)
        n, in_votes, C, psize = v.shape
        b_ij = v.new_zeros(n, in_votes, C)

        a_in = a_in.clamp(1e-4, 1.0)
        for t in range(self.iters):
            c_ij = F.softmax(b_ij, dim=2)
            c_ij = c_ij * a_in
            c_ij = c_ij / (c_ij.sum(dim=2, keepdim=True) + self.eps)

            s_j = (c_ij.unsqueeze(-1) * v).sum(dim=1)
            v_j = self.squash(s_j, dim=-1, eps=self.eps)
            if t < self.iters - 1:
                agreement = (v * v_j.unsqueeze(1)).sum(dim=-1)
                b_ij = b_ij + agreement

        # activation from vector length in (0,1)
        a_out = torch.sqrt((v_j * v_j).sum(dim=-1) + self.eps).clamp(1e-4, 1.0 - 1e-4)
        return v_j, a_out

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, h, w, c = x.shape

        if not self.w_shared:
            patches, oh, ow = self.add_pathes(x, self.K, self.stride)
            p_in = patches[..., : self.B * self.psize].contiguous().view(b * oh * ow, self.K * self.K * self.B, self.psize)
            a_in = patches[..., self.B * self.psize :].contiguous().view(b * oh * ow, self.K * self.K * self.B, 1)

            votes = self.transform_view(p_in, w_shared=False)
            p_vec, a_vec = self.dynamic_routing(votes, a_in)

            p_out = p_vec.view(b, oh, ow, self.C * self.psize)
            a_out = a_vec.view(b, oh, ow, self.C)
            out = torch.cat([p_out, a_out], dim=3)
        else:
            if c != self.B * (self.psize + 1) or self.K != 1 or self.stride != 1:
                raise ValueError('DynamicConvCaps shared mode requires K=1, stride=1 and matching capsule channels')

            p_in = x[..., : self.B * self.psize].contiguous().view(b, h * w * self.B, self.psize)
            a_in = x[..., self.B * self.psize :].contiguous().view(b, h * w * self.B, 1)
            votes = self.transform_view(p_in, w_shared=True)
            if self.coor_add:
                votes = self.add_coord(votes, b, h, w, self.B, self.C, self.psize)

            p_vec, a_vec = self.dynamic_routing(votes, a_in)
            p_out = p_vec.reshape(b, 1, 1, self.C * self.psize).expand(b, h, w, self.C * self.psize)
            a_out = a_vec.unsqueeze(1).unsqueeze(1).expand(b, h, w, self.C)
            out = torch.cat([p_out, a_out], dim=3)

        p_out = torch.nan_to_num(p_out, nan=0.0, posinf=1e4, neginf=-1e4)
        a_out = torch.nan_to_num(a_out, nan=0.5, posinf=1.0 - 1e-4, neginf=1e-4)
        out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)
        return p_out, a_out, out


class SelfRoutingConvCaps(nn.Module):
    r"""Convolutional self-routing capsules.

    Keeps the same output contract as ``ConvCaps``/``DynamicConvCaps``:
        input:  (N, H, W, B*(P*P+1))
        output: p_out (N, H_out, W_out, C*P*P), a_out (N, H_out, W_out, C), out concat
    """

    def __init__(
        self,
        B: int = 32,
        C: int = 32,
        K: int = 3,
        P: int = 4,
        stride: int = 1,
        iters: int = 1,
        coor_add: bool = False,
        w_shared: bool = False,
    ):
        super().__init__()
        _ = (iters, w_shared)  # kept for API compatibility with other capsule layers.

        self.B = B
        self.C = C
        self.K = K
        self.P = P
        self.psize = P * P
        self.stride = stride
        self.coor_add = coor_add
        self.eps = 1e-6

        self.kk = K * K
        self.kkB = self.kk * B

        # Pose transform for each input capsule vote -> output capsule pose.
        self.W1 = nn.Parameter(torch.empty(self.kkB, C, self.psize, self.psize))
        nn.init.kaiming_uniform_(self.W1, a=math.sqrt(5))

        # Routing logits from local pose vectors.
        self.W2 = nn.Parameter(torch.zeros(self.kkB, C, self.psize))
        self.b2 = nn.Parameter(torch.zeros(1, 1, self.kkB, C))

    def _output_hw(self, h: int, w: int) -> tuple[int, int]:
        pad = self.K // 2
        oh = (h + 2 * pad - self.K) // self.stride + 1
        ow = (w + 2 * pad - self.K) // self.stride + 1
        return oh, ow

    def _add_coord(self, pose_unf: torch.Tensor, oh: int, ow: int) -> torch.Tensor:
        # pose_unf: (b, L, kkB, psize)
        if self.psize < 2:
            return pose_unf

        b, L, kkB, _ = pose_unf.shape
        device, dtype = pose_unf.device, pose_unf.dtype
        gy = torch.arange(oh, device=device, dtype=dtype) / float(max(oh, 1))
        gx = torch.arange(ow, device=device, dtype=dtype) / float(max(ow, 1))
        yy, xx = torch.meshgrid(gy, gx, indexing='ij')
        coords = torch.stack((yy, xx), dim=-1).view(1, L, 1, 2)

        pose_unf = pose_unf.clone()
        pose_unf[..., :2] = pose_unf[..., :2] + coords
        return pose_unf

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: (b, h, w, B*(psize+1))
        b, h, w, c = x.shape
        expected = self.B * (self.psize + 1)
        if c != expected:
            raise ValueError(f'SelfRoutingConvCaps expected {expected} channels, got {c}')

        pose = x[..., : self.B * self.psize]
        act = x[..., self.B * self.psize :]

        pose_chw = pose.permute(0, 3, 1, 2).contiguous()
        act_chw = act.permute(0, 3, 1, 2).contiguous()

        pad = self.K // 2
        pose_unf = F.unfold(pose_chw, kernel_size=self.K, stride=self.stride, padding=pad)
        act_unf = F.unfold(act_chw, kernel_size=self.K, stride=self.stride, padding=pad)

        oh, ow = self._output_hw(h, w)
        l = pose_unf.shape[-1]

        pose_unf = pose_unf.view(b, self.B, self.psize, self.kk, l).permute(0, 4, 3, 1, 2).contiguous()
        pose_unf = pose_unf.view(b, l, self.kkB, self.psize)

        act_unf = act_unf.view(b, self.B, self.kk, l).permute(0, 3, 2, 1).contiguous()
        act_unf = act_unf.view(b, l, self.kkB)

        if self.coor_add:
            pose_unf = self._add_coord(pose_unf, oh, ow)

        # Routing logits and couplings.
        logit = torch.einsum('blip,icp->blic', pose_unf, self.W2) + self.b2
        r = F.softmax(logit, dim=3)

        ar = act_unf.unsqueeze(-1) * r
        ar_sum = ar.sum(dim=2, keepdim=True) + self.eps
        coeff = ar / ar_sum

        a_norm = act_unf.sum(dim=2, keepdim=True) + self.eps
        a_out = (ar_sum.squeeze(2) / a_norm).clamp(1e-4, 1.0 - 1e-4)

        pose_votes = torch.einsum('blip,icpq->blicq', pose_unf, self.W1)
        pose_out = (coeff.unsqueeze(-1) * pose_votes).sum(dim=2)

        p_out = pose_out.view(b, oh, ow, self.C * self.psize)
        a_out = a_out.view(b, oh, ow, self.C)
        out = torch.cat([p_out, a_out], dim=3)

        p_out = torch.nan_to_num(p_out, nan=0.0, posinf=1e4, neginf=-1e4)
        a_out = torch.nan_to_num(a_out, nan=0.5, posinf=1.0 - 1e-4, neginf=1e-4)
        out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)
        return p_out, a_out, out



class CapsuleDualHead(nn.Module):
    """Capsule detection head for one feature level.

    Args:
        c_in: Input channels of this feature scale (from parser-provided ``ch``).
        nc: Number of classes (final activation capsule count in ``ConvCaps2``).
        reg_max: Detect DFL bins, box channels are ``4 * reg_max``.
        k: Number of capsule types in ``PrimaryCaps``.
        d: Requested pose descriptor size; internally mapped to square ``P*P``.

    Input shape:
        x: ``(N, c_in, H, W)``

    Output shape:
        boxes: ``(N, 4*reg_max, H, W)``
        scores: ``(N, nc, H, W)``
        aux: dict with final capsule activations when ``return_aux=True`` else ``None``

    Parameter size:
        ``PrimaryCaps(c_in,k) + ConvCaps(k,nc,w_shared=True) + box_bias(4*reg_max)``

    Structure:
        PrimaryCaps -> ConvCaps(class caps only, shared)
    """

    def __init__(self, c_in: int, nc: int, reg_max: int, k: int, d: int):
        super().__init__()
        # Matrix-caps pose is square; choose smallest square >= requested d.
        p = max(1, int(math.ceil(math.sqrt(d))))

        self.nc = nc
        self.reg_max = reg_max
        self.P = p
        self.psize = self.P * self.P

        # A=c_in, B=k, P controls pose channels as B*(P*P).
        self.primary = PrimaryCaps(A=c_in, B=k, K=1, P=self.P, stride=1)
        # Single class-caps layer with shared transforms for parameter reduction.
        self.conv_caps2 = ConvCaps(B=k, C=nc, K=1, P=self.P, stride=1, iters=1, coor_add=True, w_shared=True)

        # Detect-style localization prior set in CapsuleDetect.bias_init().
        self.box_bias = nn.Parameter(torch.zeros(4 * reg_max))

    def _pose_to_box(self, p_out: torch.Tensor, a_out: torch.Tensor) -> torch.Tensor:
        # p_out: (b,h,w,nc*psize), a_out is intentionally unused here.
        # Simple rule requested: use first 4*reg_max pose values as box channels.
        _ = a_out
        box_ch = 4 * self.reg_max

        if p_out.shape[-1] >= box_ch:
            box = p_out[..., :box_ch]
        else:
            # If pose channels are fewer than required box channels, repeat and trim.
            reps = math.ceil(box_ch / p_out.shape[-1])
            box = p_out.repeat(1, 1, 1, reps)[..., :box_ch]

        return box + self.box_bias.view(1, 1, 1, box_ch)

    def forward(self, x: torch.Tensor, return_aux: bool = False) -> tuple[torch.Tensor, torch.Tensor, dict | None]:
        _, _, caps0 = self.primary(x)
        p2, a2, _ = self.conv_caps2(caps0)

        boxes = self._pose_to_box(p2, a2).permute(0, 3, 1, 2).contiguous()  # (b,4*reg_max,h,w)
        a2_logits = torch.logit(a2.clamp(1e-4, 1.0 - 1e-4))
        scores = a2_logits.permute(0, 3, 1, 2).contiguous()  # (b,nc,h,w) logits

        aux = None
        if return_aux:
            aux = {
                "caps2_a": a2.permute(0, 3, 1, 2).contiguous(),
            }
        return boxes, scores, aux


class CapsuleClsHead(nn.Module):
    """Capsule classification branch used as a drop-in replacement for Detect.cv3."""

    def __init__(self, c_in: int, nc: int, k: int = 4, d: int = 16, iters: int = 1):
        super().__init__()
        p = max(1, int(math.ceil(math.sqrt(d))))
        self.primary = PrimaryCaps(A=c_in, B=k, K=1, P=p, stride=1)
        # Internal capsule refinement layer.
        self.mid_caps = SelfRoutingConvCaps(B=k, C=int((k+nc)/2), K=1, P=p, stride=1, iters=iters, coor_add=False, w_shared=True)
        self.class_caps = SelfRoutingConvCaps(B=int((k+nc)/2), C=nc, K=1, P=p, stride=1, iters=iters, coor_add=False, w_shared=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Output Detect-compatible class logits in BCHW.
        _, _, caps = self.primary(x)
        _, _, caps_mid = self.mid_caps(caps)
        _, a_out, _ = self.class_caps(caps_mid)
        logits = torch.logit(a_out.clamp(1e-4, 1.0 - 1e-4)).permute(0, 3, 1, 2).contiguous()
        return torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0).float()


class CapsuleDetect(Detect):
    """Detect head with capsule vote aggregation for both box and cls branches.

    Input feature of level i is packed as interleaved channels:
        [pose(d_i), act(1)] repeated k_i times -> C_i = k_i * (d_i + 1)

    In forward_head:
    - split pose/act per capsule type
    - run Detect box/cls heads on each type-specific pose tensor
    - aggregate type predictions with act-driven vote weights

    Detect decode/postprocess/end2end flow is reused unchanged.
    """

    def __init__(
        self,
        nc: int = 80,
        *args,
        reg_max: int = 16,
        end2end: bool = False,
        k: list[int] | tuple[int, ...] = (4, 8, 16),
        d: list[int] | tuple[int, ...] = (16, 16, 16),
        ch: tuple = (),
    ):
        parsed = list(args)
        if parsed and isinstance(parsed[-1], (list, tuple)):
            ch = tuple(parsed.pop(-1))

        # Parser layout: [k_list, d_list, reg_max, end2end, ch]
        if len(parsed) not in (2, 4):
            raise ValueError('CapsuleDetect expects [k_list, d_list, reg_max, end2end, ch].')

        k, d = parsed[0], parsed[1]
        if len(parsed) == 4:
            reg_max = int(parsed[2])
            end2end = bool(parsed[3])

        if not isinstance(k, (list, tuple)) or not isinstance(d, (list, tuple)):
            raise TypeError('CapsuleDetect requires list/tuple k and d (per-level settings).')

        ch = tuple(int(c) for c in ch)
        nl = len(ch)
        if len(k) != nl or len(d) != nl:
            raise ValueError(f'CapsuleDetect k/d length must equal number of levels ({nl}).')

        self.k_list = tuple(int(v) for v in k)
        self.d_list = tuple(int(v) for v in d)

        for i, c in enumerate(ch):
            expected = self.k_list[i] * (self.d_list[i] + 1)
            if c != expected:
                raise ValueError(
                    f'CapsuleDetect level-{i} channel mismatch: got {c}, expected {expected} from k={self.k_list[i]}, d={self.d_list[i]}.'
                )

        # Detect heads operate on per-type pose tensors (d_i channels).
        super().__init__(nc=nc, reg_max=reg_max, end2end=end2end, ch=self.d_list)

        # Vote weights from activation channels (K_i channels), separate for cls/box.
        self.box_vote = nn.ModuleList(
            nn.Sequential(Conv(k_i, k_i, 3), nn.Conv2d(k_i, k_i, 1, bias=True)) for k_i in self.k_list
        )
        self.cls_vote = nn.ModuleList(
            nn.Sequential(Conv(k_i, k_i, 3), nn.Conv2d(k_i, k_i, 1, bias=True)) for k_i in self.k_list
        )

    def _split_caps(self, x: list[torch.Tensor]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Split packed feature into pose and activation tensors per level.

        Returns:
            pose_caps: list of tensors, each (B, K, D, H, W)
            act_map:   list of tensors, each (B, K, H, W)
        """
        pose_caps, act_map = [], []
        for i, xi in enumerate(x):
            k_i = self.k_list[i]
            d_i = self.d_list[i]
            c = int(xi.shape[1])
            expected = k_i * (d_i + 1)
            if c != expected:
                raise ValueError(f'CapsuleDetect level-{i} channel mismatch: got {c}, expected {expected}.')

            b, _, h, w = xi.shape
            caps = xi.view(b, k_i, d_i + 1, h, w)
            pose_caps.append(caps[:, :, :d_i].contiguous())
            act_map.append(caps[:, :, d_i].contiguous())
        return pose_caps, act_map

    @staticmethod
    def _normalized_votes(raw: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
        # No softmax/sigmoid: use softplus + sum-normalization.
        w = F.softplus(raw) + eps
        return w / (w.sum(dim=1, keepdim=True) + eps)

    def _run_voted_head(
        self,
        pose: torch.Tensor,
        act: torch.Tensor,
        head: torch.nn.Module,
        vote_head: torch.nn.Module,
        out_ch: int,
    ) -> torch.Tensor:
        """Apply one Detect head per type and aggregate by vote weights.

        Args:
            pose: (B, K, D, H, W)
            act: (B, K, H, W)
            head: Detect box or cls head module for this level
            vote_head: vote logits module for this level
            out_ch: output channels of target prediction

        Returns:
            (B, out_ch, H, W)
        """
        b, k, d, h, w = pose.shape

        # No voting needed when there is only one capsule type.
        if k == 1:
            return head(pose[:, 0])

        pose_bt = pose.reshape(b * k, d, h, w)
        pred_bt = head(pose_bt).reshape(b, k, out_ch, h, w)

        vote_raw = vote_head(act)  # (B, K, H, W)
        vote = self._normalized_votes(vote_raw).unsqueeze(2)  # (B, K, 1, H, W)
        pred = (pred_bt * vote).sum(dim=1)
        return pred

    def forward_head(
        self, x: list[torch.Tensor], box_head: torch.nn.Module = None, cls_head: torch.nn.Module = None
    ) -> dict[str, torch.Tensor]:
        if box_head is None or cls_head is None:
            return dict()

        pose_caps, act_map = self._split_caps(x)
        bs = x[0].shape[0]

        box_list = []
        cls_list = []
        for i in range(self.nl):
            box_i = self._run_voted_head(
                pose_caps[i],
                act_map[i],
                box_head[i],
                self.box_vote[i],
                out_ch=4 * self.reg_max,
            )
            cls_i = self._run_voted_head(
                pose_caps[i],
                act_map[i],
                cls_head[i],
                self.cls_vote[i],
                out_ch=self.nc,
            )
            box_list.append(box_i.view(bs, 4 * self.reg_max, -1))
            cls_list.append(cls_i.view(bs, self.nc, -1))

        boxes = torch.cat(box_list, dim=-1)
        scores = torch.cat(cls_list, dim=-1)
        return dict(boxes=boxes, scores=scores, feats=x)
    

class CapsuleDetectv1(Detect):
    """Capsule Detect variant with activation-gated pose fusion.

    Per level:
    1) Split packed capsule channels into pose/activation (interleaved by type).
    2) Use a 2-layer 1x1 gate net on activation channels.
    3) Gate pose channels with residual scaling.
    4) Flatten to K*D channels and run original Detect cv2/cv3 heads.
    """

    def __init__(
        self,
        nc: int = 80,
        *args,
        reg_max: int = 16,
        end2end: bool = False,
        k: list[int] | tuple[int, ...] = (4, 8, 16),
        d: list[int] | tuple[int, ...] = (16, 16, 16),
        ch: tuple = (),
    ):
        parsed = list(args)
        if parsed and isinstance(parsed[-1], (list, tuple)):
            ch = tuple(parsed.pop(-1))

        # Parser layout: [k_list, d_list, reg_max, end2end, ch]
        if len(parsed) not in (2, 4):
            raise ValueError("CapsuleDetectv1 expects [k_list, d_list, reg_max, end2end, ch].")

        k, d = parsed[0], parsed[1]
        if len(parsed) == 4:
            reg_max = int(parsed[2])
            end2end = bool(parsed[3])

        if not isinstance(k, (list, tuple)) or not isinstance(d, (list, tuple)):
            raise TypeError("CapsuleDetectv1 requires list/tuple k and d (per-level settings).")

        ch = tuple(int(c) for c in ch)
        nl = len(ch)
        if len(k) != nl or len(d) != nl:
            raise ValueError(f"CapsuleDetectv1 k/d length must equal number of levels ({nl}).")

        self.k_list = tuple(int(v) for v in k)
        self.d_list = tuple(int(v) for v in d)

        # Input from neck is packed as K*(D+1): [pose(D), act(1)] repeated K types.
        for i, c in enumerate(ch):
            expected = self.k_list[i] * (self.d_list[i] + 1)
            if c != expected:
                raise ValueError(
                    f"CapsuleDetectv1 level-{i} channel mismatch: got {c}, "
                    f"expected {expected} from k={self.k_list[i]}, d={self.d_list[i]}."
                )

        # Detect heads consume merged pose channels: K*D.
        merged_ch = tuple(k_i * d_i for k_i, d_i in zip(self.k_list, self.d_list))
        super().__init__(nc=nc, reg_max=reg_max, end2end=end2end, ch=merged_ch)

        self.pose_gates = nn.ModuleList()
        self.gate_alpha = nn.ParameterList()
        for k_i, d_i in zip(self.k_list, self.d_list):
            out_ch = k_i * d_i
            hidden = max(8, k_i * 2)
            self.pose_gates.append(
                nn.Sequential(
                    nn.Conv2d(k_i, hidden, 1, bias=True),
                    nn.SiLU(inplace=True),
                    nn.Conv2d(hidden, out_ch, 1, bias=True),
                )
            )
            self.gate_alpha.append(nn.Parameter(torch.tensor(0.5)))

    def _split_pose_act(self, x: torch.Tensor, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Split one level packed tensor into pose and activation maps."""
        k_i = self.k_list[i]
        d_i = self.d_list[i]
        b, c, h, w = x.shape
        expected = k_i * (d_i + 1)
        if c != expected:
            raise ValueError(f"CapsuleDetectv1 level-{i} channel mismatch: got {c}, expected {expected}.")

        caps = x.view(b, k_i, d_i + 1, h, w)
        pose = caps[:, :, :d_i].reshape(b, k_i * d_i, h, w).contiguous()
        act = caps[:, :, d_i].contiguous()
        return pose, act

    def _merge_pose(self, x: list[torch.Tensor]) -> list[torch.Tensor]:
        merged = []
        for i, xi in enumerate(x):
            pose, act = self._split_pose_act(xi, i)
            gate = torch.sigmoid(self.pose_gates[i](act))
            # Residual gating keeps base pose information and improves stability.
            pose = pose * (1.0 + self.gate_alpha[i] * gate)
            merged.append(pose)
        return merged

    def forward_head(
        self, x: list[torch.Tensor], box_head: torch.nn.Module = None, cls_head: torch.nn.Module = None
    ) -> dict[str, torch.Tensor]:
        if box_head is None or cls_head is None:
            return dict()

        pose_feats = self._merge_pose(x)
        bs = pose_feats[0].shape[0]

        box_list = [box_head[i](pose_feats[i]).view(bs, 4 * self.reg_max, -1) for i in range(self.nl)]
        cls_list = [cls_head[i](pose_feats[i]).view(bs, self.nc, -1) for i in range(self.nl)]
        boxes = torch.cat(box_list, dim=-1)
        scores = torch.cat(cls_list, dim=-1)
        return dict(boxes=boxes, scores=scores, feats=x)


class CapsuleDetectv2(Detect):
    """Capsule Detect v2: activation-gated pose + activation bypass for classification."""

    def __init__(
        self,
        nc: int = 80,
        *args,
        reg_max: int = 16,
        end2end: bool = False,
        k: list[int] | tuple[int, ...] = (4, 8, 16),
        d: list[int] | tuple[int, ...] = (16, 16, 16),
        ch: tuple = (),
    ):
        parsed = list(args)
        if parsed and isinstance(parsed[-1], (list, tuple)):
            ch = tuple(parsed.pop(-1))

        # Parser layout: [k_list, d_list, reg_max, end2end, ch]
        if len(parsed) not in (2, 4):
            raise ValueError("CapsuleDetectv2 expects [k_list, d_list, reg_max, end2end, ch].")

        k, d = parsed[0], parsed[1]
        if len(parsed) == 4:
            reg_max = int(parsed[2])
            end2end = bool(parsed[3])

        if not isinstance(k, (list, tuple)) or not isinstance(d, (list, tuple)):
            raise TypeError("CapsuleDetectv2 requires list/tuple k and d (per-level settings).")

        ch = tuple(int(c) for c in ch)
        nl = len(ch)
        if len(k) != nl or len(d) != nl:
            raise ValueError(f"CapsuleDetectv2 k/d length must equal number of levels ({nl}).")

        self.k_list = tuple(int(v) for v in k)
        self.d_list = tuple(int(v) for v in d)

        # Input from neck is packed as K*(D+1): [pose(D), act(1)] repeated K types.
        for i, c in enumerate(ch):
            expected = self.k_list[i] * (self.d_list[i] + 1)
            if c != expected:
                raise ValueError(
                    f"CapsuleDetectv2 level-{i} channel mismatch: got {c}, "
                    f"expected {expected} from k={self.k_list[i]}, d={self.d_list[i]}."
                )

        # Detect heads consume merged pose channels: K*D.
        merged_ch = tuple(k_i * d_i for k_i, d_i in zip(self.k_list, self.d_list))
        super().__init__(nc=nc, reg_max=reg_max, end2end=end2end, ch=merged_ch)

        self.pose_gates = nn.ModuleList()
        self.gate_alpha = nn.ParameterList()
        self.cls_bypass = nn.ModuleList()
        self.cls_beta = nn.ParameterList()

        for k_i, d_i in zip(self.k_list, self.d_list):
            pose_ch = k_i * d_i
            gate_hidden = max(8, k_i * 2)
            self.pose_gates.append(
                nn.Sequential(
                    nn.Conv2d(k_i, gate_hidden, 1, bias=True),
                    nn.SiLU(inplace=True),
                    nn.Conv2d(gate_hidden, pose_ch, 1, bias=True),
                )
            )
            self.gate_alpha.append(nn.Parameter(torch.tensor(0.5)))

            cls_hidden = max(16, k_i * 2)
            self.cls_bypass.append(
                nn.Sequential(
                    nn.Conv2d(k_i, cls_hidden, 1, bias=True),
                    nn.SiLU(inplace=True),
                    nn.Conv2d(cls_hidden, pose_ch, 1, bias=True),
                )
            )
            self.cls_beta.append(nn.Parameter(torch.tensor(0.1)))

    def _split_pose_act(self, x: torch.Tensor, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Split one level packed tensor into pose and activation maps."""
        k_i = self.k_list[i]
        d_i = self.d_list[i]
        b, c, h, w = x.shape
        expected = k_i * (d_i + 1)
        if c != expected:
            raise ValueError(f"CapsuleDetectv2 level-{i} channel mismatch: got {c}, expected {expected}.")

        caps = x.view(b, k_i, d_i + 1, h, w)
        pose = caps[:, :, :d_i].reshape(b, k_i * d_i, h, w).contiguous()
        act = caps[:, :, d_i].contiguous()
        return pose, act

    def _fuse_pose(self, x: list[torch.Tensor]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        box_feats, cls_feats = [], []
        for i, xi in enumerate(x):
            pose, act = self._split_pose_act(xi, i)
            gate = torch.sigmoid(self.pose_gates[i](act))
            pose_g = pose * (1.0 + self.gate_alpha[i] * gate)

            # Classification bypass from activation channels.
            act_skip = self.cls_bypass[i](act)
            cls_in = pose_g + self.cls_beta[i] * act_skip

            box_feats.append(pose_g)
            cls_feats.append(cls_in)
        return box_feats, cls_feats

    def forward_head(
        self, x: list[torch.Tensor], box_head: torch.nn.Module = None, cls_head: torch.nn.Module = None
    ) -> dict[str, torch.Tensor]:
        if box_head is None or cls_head is None:
            return dict()

        box_feats, cls_feats = self._fuse_pose(x)
        bs = x[0].shape[0]

        box_list = [box_head[i](box_feats[i]).view(bs, 4 * self.reg_max, -1) for i in range(self.nl)]
        cls_list = [cls_head[i](cls_feats[i]).view(bs, self.nc, -1) for i in range(self.nl)]
        boxes = torch.cat(box_list, dim=-1)
        scores = torch.cat(cls_list, dim=-1)
        return dict(boxes=boxes, scores=scores, feats=x)


class CapsuleDetectv4(Detect):
    """Capsule Detect v4: box uses raw pose, cls uses act bypass + symbolic type prior."""

    def __init__(
        self,
        nc: int = 80,
        *args,
        reg_max: int = 16,
        end2end: bool = False,
        k: list[int] | tuple[int, ...] = (4, 8, 16),
        d: list[int] | tuple[int, ...] = (16, 16, 16),
        ch: tuple = (),
    ):
        parsed = list(args)
        if parsed and isinstance(parsed[-1], (list, tuple)):
            ch = tuple(parsed.pop(-1))

        if len(parsed) not in (2, 4):
            raise ValueError("CapsuleDetectv4 expects [k_list, d_list, reg_max, end2end, ch].")

        k, d = parsed[0], parsed[1]
        if len(parsed) == 4:
            reg_max = int(parsed[2])
            end2end = bool(parsed[3])

        if not isinstance(k, (list, tuple)) or not isinstance(d, (list, tuple)):
            raise TypeError("CapsuleDetectv4 requires list/tuple k and d (per-level settings).")

        ch = tuple(int(c) for c in ch)
        nl = len(ch)
        if len(k) != nl or len(d) != nl:
            raise ValueError(f"CapsuleDetectv4 k/d length must equal number of levels ({nl}).")

        self.k_list = tuple(int(v) for v in k)
        self.d_list = tuple(int(v) for v in d)

        for i, c in enumerate(ch):
            expected = self.k_list[i] * (self.d_list[i] + 1)
            if c != expected:
                raise ValueError(
                    f"CapsuleDetectv4 level-{i} channel mismatch: got {c}, "
                    f"expected {expected} from k={self.k_list[i]}, d={self.d_list[i]}."
                )

        merged_ch = tuple(k_i * d_i for k_i, d_i in zip(self.k_list, self.d_list))
        super().__init__(nc=nc, reg_max=reg_max, end2end=end2end, ch=merged_ch)

        self.cls_bypass = nn.ModuleList()
        self.cls_beta = nn.ParameterList()
        self.sym_prior = nn.ModuleList()
        self.sym_beta = nn.ParameterList()

        for k_i, d_i in zip(self.k_list, self.d_list):
            pose_ch = k_i * d_i
            cls_hidden = max(16, k_i * 2)
            self.cls_bypass.append(
                nn.Sequential(
                    nn.Conv2d(k_i, cls_hidden, 1, bias=True),
                    nn.SiLU(inplace=True),
                    nn.Conv2d(cls_hidden, pose_ch, 1, bias=True),
                )
            )
            self.cls_beta.append(nn.Parameter(torch.tensor(0.1)))
            self.sym_prior.append(nn.Conv2d(k_i, self.nc, 1, bias=False))
            self.sym_beta.append(nn.Parameter(torch.tensor(0.1)))

    def _split_pose_act(self, x: torch.Tensor, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        k_i = self.k_list[i]
        d_i = self.d_list[i]
        b, c, h, w = x.shape
        expected = k_i * (d_i + 1)
        if c != expected:
            raise ValueError(f"CapsuleDetectv4 level-{i} channel mismatch: got {c}, expected {expected}.")

        caps = x.view(b, k_i, d_i + 1, h, w)
        pose = caps[:, :, :d_i].reshape(b, k_i * d_i, h, w).contiguous()
        act = caps[:, :, d_i].contiguous()
        return pose, act

    def _build_feats(self, x: list[torch.Tensor]) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        box_feats, cls_feats, cls_priors = [], [], []
        for i, xi in enumerate(x):
            pose, act = self._split_pose_act(xi, i)
            cls_in = pose + self.cls_beta[i] * self.cls_bypass[i](act)
            cls_prior = self.sym_beta[i] * self.sym_prior[i](act)
            box_feats.append(pose)
            cls_feats.append(cls_in)
            cls_priors.append(cls_prior)
        return box_feats, cls_feats, cls_priors

    def forward_head(
        self, x: list[torch.Tensor], box_head: torch.nn.Module = None, cls_head: torch.nn.Module = None
    ) -> dict[str, torch.Tensor]:
        if box_head is None or cls_head is None:
            return dict()

        box_feats, cls_feats, cls_priors = self._build_feats(x)
        bs = x[0].shape[0]

        box_list = [box_head[i](box_feats[i]).view(bs, 4 * self.reg_max, -1) for i in range(self.nl)]
        cls_list = [
            (cls_head[i](cls_feats[i]) + cls_priors[i]).view(bs, self.nc, -1)
            for i in range(self.nl)
        ]
        boxes = torch.cat(box_list, dim=-1)
        scores = torch.cat(cls_list, dim=-1)
        return dict(boxes=boxes, scores=scores, feats=x)


def _setup_capsule_layout(
    k: list[int] | tuple[int, ...],
    d: list[int] | tuple[int, ...],
    ch: tuple,
    cls_name: str,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    if not isinstance(k, (list, tuple)) or not isinstance(d, (list, tuple)):
        raise TypeError(f"{cls_name} requires list/tuple k and d (per-level settings).")

    ch = tuple(int(c) for c in ch)
    nl = len(ch)
    if len(k) != nl or len(d) != nl:
        raise ValueError(f"{cls_name} k/d length must equal number of levels ({nl}).")

    k_list = tuple(int(v) for v in k)
    d_list = tuple(int(v) for v in d)
    for i, c in enumerate(ch):
        expected = k_list[i] * (d_list[i] + 1)
        if c != expected:
            raise ValueError(
                f"{cls_name} level-{i} channel mismatch: got {c}, "
                f"expected {expected} from k={k_list[i]}, d={d_list[i]}."
            )
    merged_ch = tuple(k_i * d_i for k_i, d_i in zip(k_list, d_list))
    return k_list, d_list, merged_ch


def _init_capsule_semantic_heads(obj: nn.Module) -> None:
    obj.cls_bypass = nn.ModuleList()
    obj.cls_beta = nn.ParameterList()
    obj.sym_prior = nn.ModuleList()
    obj.sym_norm = nn.ModuleList()
    obj.sym_dropout = nn.ModuleList()
    obj.sym_beta = nn.ParameterList()

    for k_i, d_i in zip(obj.k_list, obj.d_list):
        pose_ch = k_i * d_i
        cls_hidden = max(16, k_i * 2)
        obj.cls_bypass.append(
            nn.Sequential(
                nn.Conv2d(k_i, cls_hidden, 1, bias=True),
                nn.SiLU(inplace=True),
                nn.Conv2d(cls_hidden, pose_ch, 1, bias=True),
            )
        )
        obj.cls_beta.append(nn.Parameter(torch.tensor(0.1)))
        obj.sym_dropout.append(nn.Dropout2d(p=0.1))
        obj.sym_prior.append(nn.Conv2d(k_i, obj.nc, 1, bias=False))
        obj.sym_norm.append(nn.GroupNorm(1, obj.nc))
        obj.sym_beta.append(nn.Parameter(torch.tensor(0.1)))


def _capsule_split_pose_act(
    x: torch.Tensor,
    k_i: int,
    d_i: int,
    cls_name: str,
    level_i: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    b, c, h, w = x.shape
    expected = k_i * (d_i + 1)
    if c != expected:
        raise ValueError(f"{cls_name} level-{level_i} channel mismatch: got {c}, expected {expected}.")
    caps = x.view(b, k_i, d_i + 1, h, w)
    pose = caps[:, :, :d_i].reshape(b, k_i * d_i, h, w).contiguous()
    act = caps[:, :, d_i].contiguous()
    return pose, act


def _capsule_build_feats(obj: nn.Module, x: list[torch.Tensor]) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    box_feats, cls_feats, cls_priors = [], [], []
    cls_name = obj.__class__.__name__
    for i, xi in enumerate(x):
        pose, act = _capsule_split_pose_act(xi, obj.k_list[i], obj.d_list[i], cls_name, i)
        cls_scale = torch.tanh(obj.cls_beta[i])
        cls_in = pose + cls_scale * obj.cls_bypass[i](act)
        act_s = obj.sym_dropout[i](act)
        prior = obj.sym_prior[i](act_s)
        prior = obj.sym_norm[i](prior)
        prior = prior - prior.mean(dim=1, keepdim=True)
        sym_scale = torch.tanh(obj.sym_beta[i])
        cls_prior = sym_scale * prior
        box_feats.append(pose)
        cls_feats.append(cls_in)
        cls_priors.append(cls_prior)
    return box_feats, cls_feats, cls_priors


def _capsule_build_feats_gated(
    obj: nn.Module, x: list[torch.Tensor]
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    box_feats, cls_feats, cls_priors = [], [], []
    cls_name = obj.__class__.__name__
    for i, xi in enumerate(x):
        pose, act = _capsule_split_pose_act(xi, obj.k_list[i], obj.d_list[i], cls_name, i)
        cls_scale = torch.tanh(obj.cls_beta[i])
        gate = torch.sigmoid(obj.cls_bypass[i](act))
        cls_in = pose * (1.0 + cls_scale * gate)

        act_s = obj.sym_dropout[i](act)
        prior = obj.sym_prior[i](act_s)
        prior = obj.sym_norm[i](prior)
        prior = prior - prior.mean(dim=1, keepdim=True)
        sym_scale = torch.tanh(obj.sym_beta[i])
        cls_prior = sym_scale * prior

        box_feats.append(pose)
        cls_feats.append(cls_in)
        cls_priors.append(cls_prior)
    return box_feats, cls_feats, cls_priors


def _capsule_build_feats_boxcls(
    obj: nn.Module, x: list[torch.Tensor]
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    box_feats, cls_feats, cls_priors = [], [], []
    cls_name = obj.__class__.__name__
    for i, xi in enumerate(x):
        pose, act = _capsule_split_pose_act(xi, obj.k_list[i], obj.d_list[i], cls_name, i)
        act_s = obj.sym_dropout[i](act)
        prior = obj.sym_prior[i](act_s)
        prior = obj.sym_norm[i](prior)
        prior = prior - prior.mean(dim=1, keepdim=True)
        sym_scale = torch.tanh(obj.sym_beta[i])
        cls_prior = sym_scale * prior

        box_feats.append(pose)
        cls_feats.append(pose)
        cls_priors.append(cls_prior)
    return box_feats, cls_feats, cls_priors


class CapsuleDetectv5(Detect):
    """Capsule Detect v5: box uses raw pose, cls uses stabilized symbolic prior."""

    def __init__(
        self,
        nc: int = 80,
        *args,
        reg_max: int = 16,
        end2end: bool = False,
        k: list[int] | tuple[int, ...] = (4, 8, 16),
        d: list[int] | tuple[int, ...] = (16, 16, 16),
        ch: tuple = (),
    ):
        parsed = list(args)
        if parsed and isinstance(parsed[-1], (list, tuple)):
            ch = tuple(parsed.pop(-1))

        if len(parsed) not in (2, 4):
            raise ValueError("CapsuleDetectv5 expects [k_list, d_list, reg_max, end2end, ch].")

        k, d = parsed[0], parsed[1]
        if len(parsed) == 4:
            reg_max = int(parsed[2])
            end2end = bool(parsed[3])

        self.k_list, self.d_list, merged_ch = _setup_capsule_layout(k, d, ch, "CapsuleDetectv5")
        super().__init__(nc=nc, reg_max=reg_max, end2end=end2end, ch=merged_ch)
        _init_capsule_semantic_heads(self)

    def _split_pose_act(self, x: torch.Tensor, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return _capsule_split_pose_act(x, self.k_list[i], self.d_list[i], "CapsuleDetectv5", i)

    def _build_feats(self, x: list[torch.Tensor]) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        return _capsule_build_feats(self, x)

    def forward_head(
        self, x: list[torch.Tensor], box_head: torch.nn.Module = None, cls_head: torch.nn.Module = None
    ) -> dict[str, torch.Tensor]:
        if box_head is None or cls_head is None:
            return dict()

        box_feats, cls_feats, cls_priors = self._build_feats(x)
        bs = x[0].shape[0]

        box_list = [box_head[i](box_feats[i]).view(bs, 4 * self.reg_max, -1) for i in range(self.nl)]
        cls_list = [
            (cls_head[i](cls_feats[i]) + cls_priors[i]).view(bs, self.nc, -1)
            for i in range(self.nl)
        ]
        boxes = torch.cat(box_list, dim=-1)
        scores = torch.cat(cls_list, dim=-1)
        return dict(boxes=boxes, scores=scores, feats=x)


class CapsuleDetectv6(CapsuleDetectv5):
    """Capsule Detect v6: replace additive cls correction with multiplicative act gate."""

    def _build_feats(self, x: list[torch.Tensor]) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        return _capsule_build_feats_gated(self, x)


class CapsuleDetectv7(CapsuleDetectv5):
    """Capsule Detect v7: cls head consumes raw pose features plus symbolic priors only."""

    def _build_feats(self, x: list[torch.Tensor]) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        return _capsule_build_feats_boxcls(self, x)


class CapsuleSegmentv1(Segment):
    """Capsule-style Segment head aligned with CapsuleDetectv6 semantics."""

    def __init__(
        self,
        nc: int = 80,
        *args,
        nm: int = 32,
        npr: int = 256,
        reg_max: int = 16,
        end2end: bool = False,
        k: list[int] | tuple[int, ...] = (4, 8, 16),
        d: list[int] | tuple[int, ...] = (16, 16, 16),
        ch: tuple = (),
    ):
        parsed = list(args)
        if parsed and isinstance(parsed[-1], (list, tuple)):
            ch = tuple(parsed.pop(-1))

        if len(parsed) not in (2, 4, 6):
            raise ValueError("CapsuleSegmentv1 expects [k_list, d_list, (nm, npr), reg_max, end2end, ch].")

        k, d = parsed[0], parsed[1]
        if len(parsed) == 4:
            if isinstance(parsed[3], bool):
                reg_max = int(parsed[2])
                end2end = bool(parsed[3])
            else:
                nm = int(parsed[2])
                npr = int(parsed[3])
        elif len(parsed) == 6:
            nm = int(parsed[2])
            npr = int(parsed[3])
            reg_max = int(parsed[4])
            end2end = bool(parsed[5])

        self.k_list, self.d_list, merged_ch = _setup_capsule_layout(k, d, ch, "CapsuleSegmentv1")
        super().__init__(nc=nc, nm=nm, npr=npr, reg_max=reg_max, end2end=end2end, ch=merged_ch)
        _init_capsule_semantic_heads(self)
        self.proto = Proto26(merged_ch, self.npr, self.nm, nc)

    def _split_pose_act(self, x: torch.Tensor, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return _capsule_split_pose_act(x, self.k_list[i], self.d_list[i], "CapsuleSegmentv1", i)

    def _build_feats(self, x: list[torch.Tensor]) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        return _capsule_build_feats_gated(self, x)

    def forward_head(
        self,
        x: list[torch.Tensor],
        box_head: torch.nn.Module = None,
        cls_head: torch.nn.Module = None,
        mask_head: torch.nn.Module = None,
    ) -> dict[str, torch.Tensor]:
        if box_head is None or cls_head is None:
            return dict()

        box_feats, cls_feats, cls_priors = self._build_feats(x)
        bs = x[0].shape[0]
        boxes = torch.cat([box_head[i](box_feats[i]).view(bs, 4 * self.reg_max, -1) for i in range(self.nl)], dim=-1)
        scores = torch.cat(
            [(cls_head[i](cls_feats[i]) + cls_priors[i]).view(bs, self.nc, -1) for i in range(self.nl)],
            dim=-1,
        )
        preds = dict(boxes=boxes, scores=scores, feats=cls_feats)
        if mask_head is not None:
            preds["mask_coefficient"] = torch.cat(
                [mask_head[i](cls_feats[i]).view(bs, self.nm, -1) for i in range(self.nl)],
                dim=-1,
            )
        return preds

    def forward(self, x: list[torch.Tensor]) -> tuple | list[torch.Tensor] | dict[str, torch.Tensor]:
        _, cls_feats, _ = self._build_feats(x)
        outputs = Detect.forward(self, x)
        preds = outputs[1] if isinstance(outputs, tuple) else outputs
        proto_in = cls_feats
        proto = self.proto(proto_in)  # multi-level Proto26 over merged capsule features
        if isinstance(preds, dict):
            if self.end2end:
                preds["one2many"]["proto"] = proto
                preds["one2one"]["proto"] = tuple(p.detach() for p in proto) if isinstance(proto, tuple) else proto.detach()
            else:
                preds["proto"] = proto
        if self.training:
            return preds
        return (outputs, proto) if self.export else ((outputs[0], proto), preds)


class CapsuleSegmentv2(CapsuleSegmentv1):
    """Capsule Segment v2: cls head consumes raw pose features and symbolic priors only."""

    def _build_feats(self, x: list[torch.Tensor]) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        return _capsule_build_feats_boxcls(self, x)

