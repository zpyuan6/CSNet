"""Register custom modules and provide a parse_model with direct custom-layer support."""

from __future__ import annotations

import ast
import math
import contextlib
from typing import Any

import torch
import torch.nn as nn

from ultralytics.nn.modules import (
    AIFI,
    C1,
    C2,
    C2PSA,
    C3,
    C3TR,
    ELAN1,
    OBB,
    OBB26,
    PSA,
    SPP,
    SPPELAN,
    SPPF,
    A2C2f,
    AConv,
    ADown,
    Bottleneck,
    BottleneckCSP,
    C2f,
    C2fAttn,
    C2fCIB,
    C2fPSA,
    C3Ghost,
    C3k2,
    C3x,
    CBFuse,
    CBLinear,
    Classify,
    Concat,
    Conv,
    ConvTranspose,
    Detect,
    DWConv,
    DWConvTranspose2d,
    Focus,
    GhostBottleneck,
    GhostConv,
    HGBlock,
    HGStem,
    ImagePoolingAttn,
    Index,
    Pose,
    Pose26,
    RepC3,
    RepNCSPELAN4,
    ResNetLayer,
    RTDETRDecoder,
    SCDown,
    Segment,
    Segment26,
    TorchVision,
    WorldDetect,
    YOLOEDetect,
    YOLOESegment,
    YOLOESegment26,
    v10Detect,
)
from ultralytics.utils import LOGGER, colorstr
from ultralytics.utils.ops import make_divisible

from modules import (
    CapsAlign,
    CapsDecode,
    CapsProj,
    CapsRoute,
    CapsRoutev2,
    CapsuleDetect,
    CapsuleDetectv1,
    CapsuleDetectv2,
    CapsuleDetectv4,
    CapsuleDetectv5,
    CapsuleDetectv6,
    CapsuleDetectv7,
    CapsuleSegmentv1,
    CapsuleSegmentv2,
    CapsuleTap,
    DeformableCapsBlock,
)


CUSTOM_MODULES = {
    "DeformableCapsBlock": DeformableCapsBlock,
    "CapsuleDetect": CapsuleDetect,
    "CapsuleDetectv1": CapsuleDetectv1,
    "CapsuleDetectv2": CapsuleDetectv2,
    "CapsuleDetectv4": CapsuleDetectv4,
    "CapsuleDetectv5": CapsuleDetectv5,
    "CapsuleDetectv6": CapsuleDetectv6,
    "CapsuleDetectv7": CapsuleDetectv7,
    "CapsuleSegmentv1": CapsuleSegmentv1,
    "CapsuleSegmentv2": CapsuleSegmentv2,
    "CapsProj": CapsProj,
    "CapsAlign": CapsAlign,
    "CapsRoute": CapsRoute,
    "CapsRoutev2": CapsRoutev2,
    "CapsDecode": CapsDecode,
    "CapsuleTap": CapsuleTap,
}


def parse_model(d: dict[str, Any], ch: int, verbose: bool = True):
    """Parse a model.yaml dictionary into a PyTorch model with direct custom-module support."""
    legacy = True
    max_channels = float("inf")
    nc, act, scales, end2end = (d.get(x) for x in ("nc", "activation", "scales", "end2end"))
    reg_max = d.get("reg_max", 16)
    depth, width = (d.get(x, 1.0) for x in ("depth_multiple", "width_multiple"))
    scale = d.get("scale")
    if scales:
        if not scale:
            scale = next(iter(scales.keys()))
            LOGGER.warning(f"no model scale passed. Assuming scale='{scale}'.")
        depth, width, max_channels = scales[scale]

    if act:
        Conv.default_act = eval(act)
        if verbose:
            LOGGER.info(f"{colorstr('activation:')} {act}")

    if verbose:
        LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")

    ch = [ch]
    layers, save = [], []

    base_modules = frozenset(
        {
            Classify,
            Conv,
            ConvTranspose,
            GhostConv,
            Bottleneck,
            GhostBottleneck,
            SPP,
            SPPF,
            C2fPSA,
            C2PSA,
            DWConv,
            Focus,
            BottleneckCSP,
            C1,
            C2,
            C2f,
            C3k2,
            RepNCSPELAN4,
            ELAN1,
            ADown,
            AConv,
            SPPELAN,
            C2fAttn,
            C3,
            C3TR,
            C3Ghost,
            torch.nn.ConvTranspose2d,
            DWConvTranspose2d,
            C3x,
            RepC3,
            PSA,
            SCDown,
            C2fCIB,
            A2C2f,
        }
    )
    repeat_modules = frozenset(
        {
            BottleneckCSP,
            C1,
            C2,
            C2f,
            C3k2,
            C2fAttn,
            C3,
            C3TR,
            C3Ghost,
            C3x,
            RepC3,
            C2fPSA,
            C2fCIB,
            C2PSA,
            A2C2f,
        }
    )

    detect_modules = frozenset(
        {
            Detect,
            CapsuleDetect,
            CapsuleDetectv1,
            CapsuleDetectv2,
            CapsuleDetectv4,
            CapsuleDetectv5,
            CapsuleDetectv6,
            CapsuleDetectv7,
            CapsuleSegmentv1,
            CapsuleSegmentv2,
            WorldDetect,
            YOLOEDetect,
            Segment,
            Segment26,
            YOLOESegment,
            YOLOESegment26,
            Pose,
            Pose26,
            OBB,
            OBB26,
        }
    )

    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):
        m = (
            getattr(torch.nn, m[3:])
            if "nn." in m
            else getattr(__import__("torchvision").ops, m[16:])
            if "torchvision.ops." in m
            else globals()[m]
        )

        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)

        n = n_ = max(round(n * depth), 1) if n > 1 else n
        c2 = None

        if m in base_modules:
            c1, c2 = ch[f], args[0]
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            if m is C2fAttn:
                args[1] = make_divisible(min(args[1], max_channels // 2) * width, 8)
                args[2] = int(max(round(min(args[2], max_channels // 2 // 32)) * width, 1) if args[2] > 1 else args[2])

            args = [c1, c2, *args[1:]]
            if m in repeat_modules:
                args.insert(2, n)
                n = 1
            if m is C3k2:
                legacy = False
                if scale in "mlx":
                    args[3] = True
            if m is A2C2f:
                legacy = False
                if scale in "lx":
                    args.extend((True, 1.2))
            if m is C2fCIB:
                legacy = False

        elif m is CapsProj:
            c1 = ch[f]
            k_base = int(args[0]) if len(args) > 0 else 4
            # Width scaling for capsule type count keeps model-size behavior aligned with YOLO scales.
            k = max(int(round(k_base * width)), 1)
            d_caps = int(args[1]) if len(args) > 1 else 16
            args = [c1, k, d_caps]
            c2 = k * (d_caps + 1)

        elif m is CapsAlign:
            c1 = ch[f]
            if len(args) < 2:
                raise ValueError('CapsAlign args must be [src_level, tgt_level, (down_groups)].')
            src_level, tgt_level = int(args[0]), int(args[1])
            if len(args) > 2:
                dg_base = int(args[2])
                group_num = max(int(round(dg_base * width)), 1)
                # keep grouped-conv valid: down_groups must divide channels
            else:
                group_num = 1
            args = [c1, src_level, tgt_level, group_num]
            c2 = c1

        elif m in {CapsRoute, CapsRoutev2}:
            num_src = len(f) if isinstance(f, (list, tuple)) else 1

            # Preferred YAML args:
            #   [K_in_list, P_in_list, K_out, P_out]
            #   [K_in_list, P_in_list, K_out, P_out, kernel_size, pre_k, post_k, pre_groups, post_groups]
            # Legacy support:
            #   [P_in, K_out, P_out]
            #   [P_in]
            if len(args) >= 4:
                K_in_raw, P_in_raw, K_out, P_out = args[0], args[1], int(args[2]), int(args[3])
                kernel_size = int(args[4]) if len(args) > 4 else 3
                pre_k = int(args[5]) if len(args) > 5 else 3
                post_k = int(args[6]) if len(args) > 6 else 3
                pre_groups_raw = int(args[7]) if len(args) > 7 else 0
                post_groups_raw = int(args[8]) if len(args) > 8 else 0
            elif len(args) == 3:
                P_in_raw, K_out, P_out = int(args[0]), int(args[1]), int(args[2])
                K_in_raw = 1
                kernel_size, pre_k, post_k = 3, 3, 3
                pre_groups_raw, post_groups_raw = 0, 0
            elif len(args) == 1:
                P_in_raw = int(args[0])
                K_in_raw, K_out, P_out = 1, 1, int(P_in_raw)
                kernel_size, pre_k, post_k = 3, 3, 3
                pre_groups_raw, post_groups_raw = 0, 0
            else:
                raise ValueError('CapsRoute/CapsRoutev2 args must be [K_in,P_in,K_out,P_out,(kernel_size,pre_k,post_k,pre_groups,post_groups)] or legacy [P_in,(K_out,P_out)].')

            if isinstance(K_in_raw, (list, tuple)):
                K_in_base = [int(v) for v in K_in_raw]
            else:
                K_in_base = [int(K_in_raw)] * num_src

            if isinstance(P_in_raw, (list, tuple)):
                P_in = [int(v) for v in P_in_raw]
            else:
                P_in = [int(P_in_raw)] * num_src

            if len(K_in_base) != num_src or len(P_in) != num_src:
                raise ValueError('CapsRoute/CapsRoutev2 K_in/P_in lists must match number of sources.')

            # Width scaling follows Ultralytics scale.width behavior.
            K_in = [max(int(round(k * width)), 1) for k in K_in_base]
            K_out = max(int(round(K_out * width)), 1)

            pre_groups = None
            if pre_groups_raw > 0:
                pre_groups = max(int(round(pre_groups_raw * width)), 1)
            post_groups = None
            if post_groups_raw > 0:
                post_groups = max(int(round(post_groups_raw * width)), 1)

            args = [K_in, P_in, K_out, int(P_out), kernel_size, pre_k, post_k, pre_groups, post_groups]
            c2 = K_out * (int(P_out) + 1)

        elif m is CapsDecode:
            c1 = ch[f]
            c2 = int(args[0]) if len(args) else c1
            c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [c1, c2]

        elif m is CapsuleTap:
            c2 = ch[f]

        elif m is AIFI:
            args = [ch[f], *args]
        elif m in frozenset({HGStem, HGBlock}):
            c1, cm, c2 = ch[f], args[0], args[1]
            args = [c1, cm, c2, *args[2:]]
            if m is HGBlock:
                args.insert(4, n)
                n = 1
        elif m is ResNetLayer:
            c2 = args[1] if args[3] else args[1] * 4
        elif m is torch.nn.BatchNorm2d:
            args = [ch[f]]
            c2 = ch[f]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m in detect_modules:
            if m in {CapsuleDetect, CapsuleDetectv1, CapsuleDetectv2, CapsuleDetectv4, CapsuleDetectv5, CapsuleDetectv6, CapsuleDetectv7, CapsuleSegmentv1, CapsuleSegmentv2}:
                if len(args) < 3:
                    raise ValueError('CapsuleDetect/CapsuleDetectv1/CapsuleDetectv2/CapsuleDetectv4/CapsuleDetectv5/CapsuleDetectv6/CapsuleDetectv7/CapsuleSegmentv1/CapsuleSegmentv2 args must include [nc, k_list, d_list].')
                if not isinstance(args[1], (list, tuple)) or not isinstance(args[2], (list, tuple)):
                    raise TypeError('CapsuleDetect/CapsuleDetectv1/CapsuleDetectv2/CapsuleDetectv4/CapsuleDetectv5/CapsuleDetectv6/CapsuleDetectv7/CapsuleSegmentv1/CapsuleSegmentv2 requires k_list and d_list in YAML.')
                # Width-scale capsule type counts per level; keep pose dims as provided.
                args[1] = [max(int(round(int(v) * width)), 1) for v in args[1]]
                args[2] = [int(v) for v in args[2]]

            args.extend([reg_max, end2end, [ch[x] for x in f]])
            if m in {Segment, Segment26, YOLOESegment, YOLOESegment26}:
                args[2] = make_divisible(min(args[2], max_channels) * width, 8)
            if m in {
                Detect,
                CapsuleDetect,
                CapsuleDetectv1,
                CapsuleDetectv2,
                CapsuleDetectv4,
                CapsuleDetectv5,
                CapsuleDetectv6,
                CapsuleDetectv7,
                CapsuleSegmentv1,
                CapsuleSegmentv2,
                YOLOEDetect,
                Segment,
                Segment26,
                YOLOESegment,
                YOLOESegment26,
                Pose,
                Pose26,
                OBB,
                OBB26,
            }:
                m.legacy = legacy
            c2 = ch[f[-1]] if isinstance(f, (list, tuple)) else ch[f]

        elif m is v10Detect:
            args.append([ch[x] for x in f])
            c2 = ch[f[-1]] if isinstance(f, (list, tuple)) else ch[f]
        elif m is ImagePoolingAttn:
            args.insert(1, [ch[x] for x in f])
            c2 = ch[f[-1]] if isinstance(f, (list, tuple)) else ch[f]
        elif m is RTDETRDecoder:
            args.insert(1, [ch[x] for x in f])
            c2 = ch[f[-1]] if isinstance(f, (list, tuple)) else ch[f]
        elif m is CBLinear:
            c2 = args[0]
            c1 = ch[f]
            args = [c1, c2, *args[1:]]
        elif m is CBFuse:
            c2 = ch[f[-1]]
        elif m in frozenset({TorchVision, Index}):
            c2 = args[0]
            args = [*args[1:]]
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)

        if m in {CapsRoute, CapsRoutev2}:
            c2 = int(getattr(m_, "c_out", c2))

        t = str(m)[8:-2].replace("__main__.", "")
        m_.np = sum(x.numel() for x in m_.parameters())
        m_.i, m_.f, m_.type = i, f, t
        if verbose:
            LOGGER.info(f"{i:>3}{f!s:>20}{n_:>3}{m_.np:10.0f}  {t:<45}{args!s:<30}")

        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)

    # Keep all intermediate outputs to avoid None entries for custom multi-source routing blocks.
    return nn.Sequential(*layers), sorted(set(save + list(range(len(layers)))))


def register_ultralytics_modules() -> None:
    """Register custom modules and replace Ultralytics parse_model with this custom parse_model."""
    import ultralytics.nn.modules as nn_modules
    import ultralytics.nn.tasks as nn_tasks

    for name, cls in CUSTOM_MODULES.items():
        setattr(nn_tasks, name, cls)
        setattr(nn_modules, name, cls)

    if getattr(nn_tasks, "_capsule_parse_patched", False):
        return

    nn_tasks.parse_model = parse_model
    nn_tasks._capsule_parse_patched = True
