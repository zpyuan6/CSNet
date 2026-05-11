from __future__ import annotations

import itertools
from copy import copy
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ultralytics import download
from ultralytics.data import YOLOConcatDataset, build_grounding, build_yolo_dataset
from ultralytics.engine.model import Model
from ultralytics.models import yolo
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator
from ultralytics.nn.autobackend import check_class_names
from ultralytics.nn.tasks import DetectionModel
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import DATASETS_DIR, DEFAULT_CFG, LOGGER, RANK, YAML
from ultralytics.utils import checks
from ultralytics.utils.checks import check_file
from ultralytics.utils.torch_utils import select_device, smart_inference_mode, unwrap_model

from modules import CapsuleOpenVocabDetect
from .custom_yolo import register_ultralytics_modules


class _MobileCLIPTextEncoder(nn.Module):
    """Minimal MobileCLIP wrapper that avoids importing the CLIP package at module import time."""

    _config_size_map = {"s0": "s0", "s1": "s1", "s2": "s2", "b": "b", "blt": "b"}

    def __init__(self, size: str, device: torch.device) -> None:
        super().__init__()
        try:
            import mobileclip
        except ImportError:
            checks.check_requirements("git+https://github.com/ultralytics/mobileclip.git")
            import mobileclip

        config = self._config_size_map[size]
        weight = f"mobileclip_{size}.pt"
        if not Path(weight).is_file():
            download(f"https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/{weight}")
        self.model = mobileclip.create_model_and_transforms(f"mobileclip_{config}", pretrained=weight, device=device)[0]
        self.tokenizer = mobileclip.get_tokenizer(f"mobileclip_{config}")
        self.device = device
        self.eval()

    def tokenize(self, texts):
        return self.tokenizer(texts).to(self.device)

    @smart_inference_mode()
    def encode_text(self, texts: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        text_features = self.model.encode_text(texts).to(dtype)
        text_features /= text_features.norm(p=2, dim=-1, keepdim=True)
        return text_features


def _build_text_model(variant: str, device: torch.device):
    base, size = variant.split(":")
    if base == "mobileclip":
        return _MobileCLIPTextEncoder(size, device)
    if base == "clip":
        raise ImportError("clip text encoder is not available in this environment; use a mobileclip variant instead.")
    raise ValueError(f"Unsupported text model variant '{variant}'.")


def _get_text_output_dim(variant: str) -> int:
    base, _size = variant.split(":")
    if base == "mobileclip":
        return 512
    raise ValueError(f"Unsupported text model variant '{variant}'.")


class CapsuleOVDModel(DetectionModel):
    """Capsule detector with YOLOE-style text-prompt open-vocabulary classification."""

    def __init__(self, cfg="configs/ovd_model/yolo26_caps_ovd_v1.yaml", ch=3, nc=None, verbose=True):
        register_ultralytics_modules()
        self.text_model = "mobileclip:blt"
        self.pe = None
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        self.text_model = self.yaml.get("text_model", "mobileclip:blt")
        head = self.model[-1]
        if not isinstance(head, CapsuleOpenVocabDetect):
            raise TypeError(f"CapsuleOVDModel requires CapsuleOpenVocabDetect head, got {type(head).__name__}.")
        self.text_proj = nn.Sequential(nn.Linear(_get_text_output_dim(self.text_model), head.embed), nn.LayerNorm(head.embed))

    @smart_inference_mode()
    def get_text_pe(self, text, batch=80, cache_clip_model=False, without_projection=False):
        """Encode class names into raw or projected text embeddings."""
        device = next(self.model.parameters()).device
        if not getattr(self, "clip_model", None) and cache_clip_model:
            self.clip_model = _build_text_model(self.text_model, device=device)

        model = self.clip_model if cache_clip_model else _build_text_model(self.text_model, device=device)
        text_token = model.tokenize(text)
        txt_feats = [model.encode_text(token).detach() for token in text_token.split(batch)]
        txt_feats = txt_feats[0] if len(txt_feats) == 1 else torch.cat(txt_feats, dim=0)
        txt_feats = txt_feats.reshape(-1, len(text), txt_feats.shape[-1])
        return txt_feats if without_projection else self.project_text_pe(txt_feats)

    def project_text_pe(self, text_embs: torch.Tensor | None) -> torch.Tensor | None:
        """Project raw text encoder features into the detector embedding space."""
        if text_embs is None:
            return None
        projected = self.text_proj(text_embs.to(device=next(self.parameters()).device, dtype=torch.float32))
        return nn.functional.normalize(projected, dim=-1)

    def _normalize_or_project_text_pe(self, text_embs: torch.Tensor | None) -> torch.Tensor | None:
        """Accept either raw text encoder features or already-projected detector embeddings."""
        if text_embs is None:
            return None
        head = self.model[-1]
        assert isinstance(head, CapsuleOpenVocabDetect)
        text_embs = text_embs.to(device=next(self.parameters()).device, dtype=torch.float32)
        if text_embs.shape[-1] == head.embed:
            return nn.functional.normalize(text_embs, dim=-1)
        return self.project_text_pe(text_embs)

    def set_vocab(self, vocab, names):
        raise NotImplementedError("Prompt-free vocabulary is not implemented for CapsuleOVDModel yet.")

    def get_vocab(self, names):
        raise NotImplementedError("Prompt-free vocabulary is not implemented for CapsuleOVDModel yet.")

    def set_classes(self, names, embeddings):
        """Cache projected text embeddings for offline text-prompt inference."""
        head = self.model[-1]
        assert isinstance(head, CapsuleOpenVocabDetect)
        if embeddings.ndim != 3:
            raise ValueError(f"Expected embeddings with shape [1, num_classes, dim], got {tuple(embeddings.shape)}.")
        self.pe = self._normalize_or_project_text_pe(embeddings)
        head.nc = len(names)
        head.set_text_embeddings(self.pe.squeeze(0))
        self.names = check_class_names(names)

    def get_cls_pe(self, tpe=None, vpe=None):
        """YOLOE-compatible class embedding accessor."""
        if vpe is not None:
            raise NotImplementedError("Visual prompts are not implemented for CapsuleOVDModel yet.")
        return self._normalize_or_project_text_pe(tpe) if tpe is not None else self.pe

    def predict(self, x, profile=False, visualize=False, tpe=None, augment=False, embed=None, vpe=None):
        """Run a forward pass while injecting text embeddings into the capsule OVD head."""
        if vpe is not None:
            raise NotImplementedError("Visual prompts are not implemented for CapsuleOVDModel yet.")
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
            x = m(x, text_embs=cls_pe) if isinstance(m, CapsuleOpenVocabDetect) else m(x)
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
        """Compute detection loss with raw cached text features projected on the fly."""
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()
        if preds is None:
            preds = self.forward(batch["img"], tpe=batch.get("txt_feats"))
        return self.criterion(preds, batch)


class CapsuleOVDValidator(DetectionValidator):
    """Detection validator that refreshes text prompts from class names before evaluation."""

    @smart_inference_mode()
    def __call__(self, trainer: Any | None = None, model: CapsuleOVDModel | str | None = None):
        if trainer is not None:
            self.device = trainer.device
            model = trainer.ema.ema
            names = [name.split("/", 1)[0] for name in list(self.dataloader.dataset.data["names"].values())]
            tpe = model.get_text_pe(names)
            model.set_classes(names, tpe)
            return super().__call__(trainer, model)

        self.device = select_device(self.args.device, verbose=False)
        if isinstance(model, (str, Path)):
            from ultralytics.nn.tasks import load_checkpoint

            model, _ = load_checkpoint(model, device=self.device)
        model.eval().to(self.device)
        names = [name.split("/", 1)[0] for name in list(self.dataloader.dataset.data["names"].values())]
        tpe = model.get_text_pe(names)
        model.set_classes(names, tpe)
        return super().__call__(model=model)


class CapsuleOVDTrainer(DetectionTrainer):
    """Trainer for the Stage-1 capsule open-vocabulary baseline."""

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict | None = None, _callbacks=None):
        overrides = dict(overrides or {})
        assert not overrides.get("compile"), f"Training with 'model={overrides.get('model')}' requires 'compile=False'"
        super().__init__(cfg, overrides, _callbacks)
        self.text_embeddings = None

    def get_model(self, cfg=None, weights=None, verbose: bool = True):
        model = CapsuleOVDModel(
            cfg["yaml_file"] if isinstance(cfg, dict) else cfg,
            ch=self.data["channels"],
            nc=min(self.data["nc"], 80),
            verbose=verbose and RANK == -1,
        )
        if weights:
            model.load(weights)
        return model

    def build_dataset(self, img_path: list[str] | str, mode: str = "train", batch: int | None = None):
        gs = max(int(unwrap_model(self.model).stride.max() if self.model else 0), 32)
        if mode != "train":
            return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=False, stride=gs)

        datasets = [
            build_yolo_dataset(self.args, im_path, batch, self.training_data[im_path], stride=gs, multi_modal=True)
            if isinstance(im_path, str)
            else build_grounding(
                self.args,
                im_path["img_path"],
                im_path["json_file"],
                batch,
                stride=gs,
                max_samples=self.data["nc"],
            )
            for im_path in img_path
        ]
        self.set_text_embeddings(datasets, batch)
        return YOLOConcatDataset(datasets) if len(datasets) > 1 else datasets[0]

    def get_validator(self):
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return CapsuleOVDValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    @staticmethod
    def check_data_config(data: dict | str | Path) -> dict:
        """Load a mixed OVD data config from dict or YAML path."""
        if not isinstance(data, dict):
            return YAML.load(check_file(data))
        return data

    def get_dataset(self):
        """Support both standard detect YAMLs and YOLO-World-style mixed OVD configs."""
        data_arg = self.args.data
        if not isinstance(data_arg, dict):
            data_yaml = self.check_data_config(data_arg)
        else:
            data_yaml = data_arg

        if isinstance(data_yaml.get("train"), dict) and isinstance(data_yaml.get("val"), dict):
            final_data = {}
            self.args.data = data_yaml
            assert data_yaml.get("train", False), "train dataset not found"
            assert data_yaml.get("val", False), "validation dataset not found"

            data = {k: [check_det_dataset(d) for d in v.get("yolo_data", [])] for k, v in data_yaml.items()}
            assert len(data["val"]) == 1, f"Only support validating on 1 dataset for now, but got {len(data['val'])}."
            val_split = "minival" if "lvis" in data["val"][0]["val"].lower() else "val"
            for d in data["val"]:
                if d.get("minival") is not None:
                    d["minival"] = str(d["path"] / d["minival"])

            for split in {"train", "val"}:
                final_data[split] = [d["train" if split == "train" else val_split] for d in data[split]]
                grounding_data = data_yaml[split].get("grounding_data")
                if grounding_data is None:
                    continue
                grounding_items = grounding_data if isinstance(grounding_data, list) else [grounding_data]
                for item in grounding_items:
                    assert isinstance(item, dict), f"Grounding data should be dict, but got {type(item)}"
                    for key in {"img_path", "json_file"}:
                        path = Path(item[key])
                        if not path.exists() and not path.is_absolute():
                            item[key] = str((DATASETS_DIR / item[key]).resolve())
                final_data[split] += grounding_items

            data["val"] = data["val"][0]
            final_data["val"] = final_data["val"][0]
            final_data["nc"] = data["val"]["nc"]
            final_data["names"] = data["val"]["names"]
            final_data["path"] = data["val"]["path"]
            final_data["channels"] = data["val"]["channels"]

            self.data = final_data
            if self.args.single_cls:
                LOGGER.info("Overriding class names with single class.")
                self.data["names"] = {0: "object"}
                self.data["nc"] = 1

            self.training_data = {}
            for d in data["train"]:
                if self.args.single_cls:
                    d["names"] = {0: "object"}
                    d["nc"] = 1
                self.training_data[d["train"]] = d
            return final_data

        return super().get_dataset()

    def set_text_embeddings(self, datasets: list[Any], batch: int | None) -> None:
        text_embeddings = {}
        for dataset in datasets:
            if not hasattr(dataset, "category_names"):
                continue
            text_embeddings.update(
                self.generate_text_embeddings(
                    list(dataset.category_names), batch or self.args.batch, cache_dir=Path(dataset.img_path).parent
                )
            )
        self.text_embeddings = text_embeddings

    def generate_text_embeddings(self, texts: list[str], batch: int, cache_dir: Path) -> dict[str, torch.Tensor]:
        model = unwrap_model(self.model)
        cache_name = getattr(model, "text_model", "mobileclip:blt").replace(":", "_").replace("/", "_")
        cache_path = cache_dir / f"text_embeddings_{cache_name}.pt"
        if cache_path.exists():
            LOGGER.info(f"Reading existed cache from '{cache_path}'")
            txt_map = torch.load(cache_path, map_location=self.device)
            if sorted(txt_map.keys()) == sorted(texts):
                return txt_map

        LOGGER.info(f"Caching text embeddings to '{cache_path}'")
        txt_feats = model.get_text_pe(texts, batch=batch, cache_clip_model=False, without_projection=True)
        txt_map = dict(zip(texts, txt_feats.squeeze(0)))
        torch.save(txt_map, cache_path)
        return txt_map

    def preprocess_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        batch = super().preprocess_batch(batch)
        texts = list(itertools.chain(*batch["texts"]))
        txt_feats = torch.stack([self.text_embeddings[text] for text in texts]).to(
            self.device, non_blocking=self.device.type == "cuda"
        )
        batch["txt_feats"] = txt_feats.reshape(len(batch["texts"]), -1, txt_feats.shape[-1])
        return batch

    def plot_training_labels(self):
        """Skip static label plots for mixed OVD datasets."""
        pass

    def final_eval(self):
        """Point final eval back to the validation detect YAML when using mixed OVD data."""
        if isinstance(self.args.data, dict) and isinstance(self.args.data.get("val"), dict):
            val = self.args.data["val"]["yolo_data"][0]
            self.validator.args.data = val
            self.validator.args.split = "minival" if isinstance(val, str) and "lvis" in val.lower() else "val"
        return super().final_eval()


class CapsuleOVD(Model):
    """Ultralytics-style wrapper for the capsule text-prompt OVD baseline."""

    def __init__(
        self,
        model: str | Path = "configs/ovd_model/yolo26_caps_ovd_v1.yaml",
        task: str | None = "detect",
        verbose: bool = False,
    ):
        super().__init__(model=model, task=task, verbose=verbose)

    @property
    def task_map(self) -> dict[str, dict[str, Any]]:
        return {
            "detect": {
                "model": CapsuleOVDModel,
                "trainer": CapsuleOVDTrainer,
                "validator": CapsuleOVDValidator,
                "predictor": yolo.detect.DetectionPredictor,
            }
        }

    def get_text_pe(self, texts):
        assert isinstance(self.model, CapsuleOVDModel)
        return self.model.get_text_pe(texts)

    def set_classes(self, classes: list[str], embeddings: torch.Tensor | None = None) -> None:
        assert isinstance(self.model, CapsuleOVDModel)
        if embeddings is None:
            embeddings = self.get_text_pe(classes)
        self.model.set_classes(classes, embeddings)
        assert " " not in classes
        self.model.names = classes
        if self.predictor:
            self.predictor.model.names = classes
