from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ultralytics.models import yolo
from ultralytics.nn.autobackend import check_class_names

from modules import CapsuleOpenVocabDetectV2
from .capsule_ovd import (
    _build_text_model,
    CapsuleOVD,
    CapsuleOVDTrainer,
    CapsuleOVDValidator,
)
from .capsule_ovd import register_ultralytics_modules
from ultralytics.nn.tasks import DetectionModel


class CapsuleOVDv2Model(DetectionModel):
    """Capsule detector with stronger YOLOE-style text adaptation."""

    def __init__(self, cfg="configs/ovd_model/yolo26_caps_ovd_v2.yaml", ch=3, nc=None, verbose=True):
        register_ultralytics_modules()
        self.text_model = "mobileclip:blt"
        self.pe = None
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        self.text_model = self.yaml.get("text_model", "mobileclip:blt")
        head = self.model[-1]
        if not isinstance(head, CapsuleOpenVocabDetectV2):
            raise TypeError(f"CapsuleOVDv2Model requires CapsuleOpenVocabDetectV2 head, got {type(head).__name__}.")

    @torch.inference_mode()
    def get_text_pe(self, text, batch=80, cache_clip_model=False, without_projection=False):
        del without_projection
        device = next(self.model.parameters()).device
        if not getattr(self, "clip_model", None) and cache_clip_model:
            self.clip_model = _build_text_model(self.text_model, device=device)

        model = self.clip_model if cache_clip_model else _build_text_model(self.text_model, device=device)
        text_token = model.tokenize(text)
        txt_feats = [model.encode_text(token).detach() for token in text_token.split(batch)]
        txt_feats = txt_feats[0] if len(txt_feats) == 1 else torch.cat(txt_feats, dim=0)
        return txt_feats.reshape(-1, len(text), txt_feats.shape[-1])

    def set_vocab(self, vocab, names):
        raise NotImplementedError("Prompt-free vocabulary is not implemented for CapsuleOVDv2Model yet.")

    def get_vocab(self, names):
        raise NotImplementedError("Prompt-free vocabulary is not implemented for CapsuleOVDv2Model yet.")

    def set_classes(self, names, embeddings):
        head = self.model[-1]
        assert isinstance(head, CapsuleOpenVocabDetectV2)
        if embeddings.ndim != 3:
            raise ValueError(f"Expected embeddings with shape [1, num_classes, dim], got {tuple(embeddings.shape)}.")
        self.pe = head.get_tpe(embeddings.to(device=next(self.parameters()).device, dtype=torch.float32))
        head.nc = len(names)
        head.set_text_embeddings(embeddings.squeeze(0))
        self.names = check_class_names(names)

    def get_cls_pe(self, tpe=None, vpe=None):
        if vpe is not None:
            raise NotImplementedError("Visual prompts are not implemented for CapsuleOVDv2Model yet.")
        head = self.model[-1]
        assert isinstance(head, CapsuleOpenVocabDetectV2)
        if tpe is None:
            return self.pe
        tpe = tpe.to(device=next(self.parameters()).device, dtype=torch.float32)
        return head.get_tpe(tpe)

    def predict(self, x, profile=False, visualize=False, tpe=None, augment=False, embed=None, vpe=None):
        if vpe is not None:
            raise NotImplementedError("Visual prompts are not implemented for CapsuleOVDv2Model yet.")
        if augment:
            return self._predict_augment(x)

        y, dt, embeddings = [], [], []
        embed = frozenset(embed) if embed is not None else {-1}
        max_idx = max(embed)
        cls_pe = self.get_cls_pe(tpe)

        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x, text_embs=cls_pe) if isinstance(m, CapsuleOpenVocabDetectV2) else m(x)
            y.append(x if m.i in self.save else None)
            if visualize:
                from ultralytics.utils.plotting import feature_visualization

                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if m.i in embed:
                pooled = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1)
                embeddings.append(pooled)
                if m.i == max_idx:
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x

    def loss(self, batch, preds=None):
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()
        if preds is None:
            preds = self.forward(batch["img"], tpe=batch.get("txt_feats"))
        return self.criterion(preds, batch)


class CapsuleOVDv2Trainer(CapsuleOVDTrainer):
    def get_model(self, cfg=None, weights=None, verbose: bool = True):
        model = CapsuleOVDv2Model(
            cfg["yaml_file"] if isinstance(cfg, dict) else cfg,
            ch=self.data["channels"],
            nc=min(self.data["nc"], 80),
            verbose=verbose,
        )
        if weights:
            model.load(weights)
        return model


class CapsuleOVDv2(CapsuleOVD):
    """Ultralytics-style wrapper for Capsule OVD v2."""

    def __init__(
        self,
        model: str | Path = "configs/ovd_model/yolo26_caps_ovd_v2.yaml",
        task: str | None = "detect",
        verbose: bool = False,
    ):
        super().__init__(model=model, task=task, verbose=verbose)

    @property
    def task_map(self) -> dict[str, dict[str, Any]]:
        return {
            "detect": {
                "model": CapsuleOVDv2Model,
                "trainer": CapsuleOVDv2Trainer,
                "validator": CapsuleOVDValidator,
                "predictor": yolo.detect.DetectionPredictor,
            }
        }
