from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.models import yolo
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.loss import E2ELoss, v8SegmentationLoss

from models import register_ultralytics_modules


def train(model_cfg: str, data_cfg: str, **kwargs):
    """Train a custom YOLO-like model using Ultralytics."""
    register_ultralytics_modules()
    model = YOLO(model_cfg)
    task = kwargs.get("task", "")
    trainer = WeightedSegmentationTrainer if task == "segment" else None
    if trainer is not None:
        return model.train(data=data_cfg, trainer=trainer, **kwargs)
    return model.train(data=data_cfg, **kwargs)


class WeightedSegmentationLoss(v8SegmentationLoss):
    """Local segmentation loss with independent seg/sem gains."""

    def __init__(
        self,
        model,
        tal_topk: int = 10,
        tal_topk2: int | None = None,
        seg_gain: float = 7.5,
        sem_gain: float = 7.5,
    ):
        super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
        self.seg_gain = float(seg_gain)
        self.sem_gain = float(sem_gain)

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        pred_masks, proto = preds["mask_coefficient"].permute(0, 2, 1).contiguous(), preds["proto"]
        loss = torch.zeros(5, device=self.device)  # box, seg, cls, dfl, sem
        if isinstance(proto, tuple) and len(proto) == 2:
            proto, pred_semseg = proto
        else:
            pred_semseg = None
        (fg_mask, target_gt_idx, target_bboxes, _, _), det_loss, _ = self.get_assigned_targets_and_loss(preds, batch)
        loss[0], loss[2], loss[3] = det_loss[0], det_loss[1], det_loss[2]

        batch_size, _, mask_h, mask_w = proto.shape
        if fg_mask.sum():
            masks = batch["masks"].to(self.device).float()
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):
                proto = F.interpolate(proto, masks.shape[-2:], mode="bilinear", align_corners=False)

            imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=pred_masks.dtype) * self.stride[0]
            loss[1] = self.calculate_segmentation_loss(
                fg_mask,
                masks,
                target_gt_idx,
                target_bboxes,
                batch["batch_idx"].view(-1, 1),
                proto,
                pred_masks,
                imgsz,
            )
            if pred_semseg is not None:
                sem_masks = batch["sem_masks"].to(self.device)
                sem_masks = F.one_hot(sem_masks.long(), num_classes=self.nc).permute(0, 3, 1, 2).float()

                if self.overlap:
                    mask_zero = masks == 0
                    sem_masks[mask_zero.unsqueeze(1).expand_as(sem_masks)] = 0
                else:
                    batch_idx = batch["batch_idx"].view(-1)
                    for i in range(batch_size):
                        instance_mask_i = masks[batch_idx == i]
                        if len(instance_mask_i) == 0:
                            continue
                        sem_masks[i, :, instance_mask_i.sum(dim=0) == 0] = 0

                loss[4] = self.bcedice_loss(pred_semseg, sem_masks)
                loss[4] *= self.sem_gain
        else:
            loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()
            if pred_semseg is not None:
                loss[4] += (pred_semseg * 0).sum()

        loss[1] *= self.seg_gain
        return loss * batch_size, loss.detach()


class WeightedSegmentationModel(SegmentationModel):
    """Segmentation model using local weighted segmentation criterion."""

    def init_criterion(self):
        seg_gain = float(getattr(self, "seg_gain", self.args.box))
        sem_gain = float(getattr(self, "sem_gain", self.args.box))
        factory = lambda model, tal_topk=10, tal_topk2=None: WeightedSegmentationLoss(
            model,
            tal_topk=tal_topk,
            tal_topk2=tal_topk2,
            seg_gain=seg_gain,
            sem_gain=sem_gain,
        )
        return E2ELoss(self, factory) if getattr(self, "end2end", False) else factory(self)


class WeightedSegmentationTrainer(yolo.segment.SegmentationTrainer):
    """Local segmentation trainer exposing independent seg/sem gains."""

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict | None = None, _callbacks=None):
        o = dict(overrides or {})
        self.seg_gain = float(o.pop("seg", o.get("box", 7.5)))
        self.sem_gain = float(o.pop("sem", o.get("box", 7.5)))
        super().__init__(cfg, o, _callbacks)

    def get_model(self, cfg: dict | str | None = None, weights: str | None = None, verbose: bool = True):
        model = WeightedSegmentationModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose)
        model.seg_gain = self.seg_gain
        model.sem_gain = self.sem_gain
        if weights:
            model.load(weights)
        return model


@dataclass
class KDConfig:
    teacher: str
    kd_cls: float = 0.5
    kd_box: float = 1.0
    kd_temp: float = 3.0
    kd_warmup_epochs: int = 10


def _extract_head_outputs(preds: Any):
    """Extract Detect-style outputs (boxes, scores) from nested prediction objects."""
    if isinstance(preds, dict):
        if "boxes" in preds and "scores" in preds:
            return preds["boxes"], preds["scores"]
        if "one2many" in preds:
            return _extract_head_outputs(preds["one2many"])
    if isinstance(preds, (list, tuple)) and preds:
        for p in preds:
            out = _extract_head_outputs(p)
            if out is not None:
                return out
    return None


def _align_last_dim(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Align final dim (anchor axis) by cropping to min length."""
    n = min(a.shape[-1], b.shape[-1])
    return a[..., :n], b[..., :n]


def _build_kd_loss(
    student_preds: Any,
    teacher_preds: Any,
    kd_cls: float,
    kd_box: float,
    kd_temp: float,
) -> torch.Tensor:
    """Compute lightweight output-layer KD loss."""
    s = _extract_head_outputs(student_preds)
    t = _extract_head_outputs(teacher_preds)
    if s is None or t is None:
        # Return zero if prediction structure is not distillable.
        dev = None
        if isinstance(student_preds, dict):
            for v in student_preds.values():
                if isinstance(v, torch.Tensor):
                    dev = v.device
                    break
        return torch.zeros((), device=dev or "cpu")

    s_box, s_cls = s
    t_box, t_cls = t

    kd = torch.zeros((), device=s_box.device)

    # cls KD: temperature-scaled KL on sigmoid probabilities.
    if kd_cls > 0:
        s_cls, t_cls = _align_last_dim(s_cls, t_cls)
        t_prob = torch.sigmoid(t_cls / kd_temp).detach()
        s_log_prob = torch.log(torch.sigmoid(s_cls / kd_temp).clamp_min(1e-6))
        # Bernoulli KL(p_t || p_s)
        kl = t_prob * (torch.log(t_prob.clamp_min(1e-6)) - s_log_prob)
        kl += (1.0 - t_prob) * (
            torch.log((1.0 - t_prob).clamp_min(1e-6))
            - torch.log((1.0 - torch.sigmoid(s_cls / kd_temp)).clamp_min(1e-6))
        )
        kd = kd + kd_cls * kl.mean() * (kd_temp * kd_temp)

    # box KD: only when channel count matches (e.g., same reg_max).
    if kd_box > 0 and s_box.shape[1] == t_box.shape[1]:
        s_box, t_box = _align_last_dim(s_box, t_box)
        kd = kd + kd_box * F.smooth_l1_loss(s_box, t_box.detach())

    return kd


class DistillDetectionTrainer(DetectionTrainer):
    """Detection trainer with output-layer knowledge distillation."""

    train_loss_names = ("box_loss", "cls_loss", "dfl_loss", "kd_loss")
    val_loss_names = ("box_loss", "cls_loss", "dfl_loss")

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks=None):
        # Remove custom KD keys before Ultralytics config alignment.
        o = dict(overrides or {})
        teacher = o.pop("teacher", "")
        kd_cls = float(o.pop("kd_cls", 0.5))
        kd_box = float(o.pop("kd_box", 1.0))
        kd_temp = float(o.pop("kd_temp", 3.0))
        kd_warmup_epochs = int(o.pop("kd_warmup_epochs", 10))

        super().__init__(cfg, o, _callbacks)
        self.kd_cfg = KDConfig(
            teacher=teacher,
            kd_cls=kd_cls,
            kd_box=kd_box,
            kd_temp=kd_temp,
            kd_warmup_epochs=kd_warmup_epochs,
        )
        self.teacher_model = None
        if not self.kd_cfg.teacher:
            raise ValueError("Distillation requires --teacher path.")

    def _setup_train(self):
        super()._setup_train()
        if self.teacher_model is None:
            self.teacher_model = YOLO(self.kd_cfg.teacher).model
            for p in self.teacher_model.parameters():
                p.requires_grad = False
        self.teacher_model.to(self.device)
        # Use train-mode forward to keep Detect raw output structure.
        self.teacher_model.train()

    def get_validator(self):
        # Keep validator on the standard 3-term detection loss.
        return super().get_validator()

    def label_loss_items(self, loss_items=None, prefix="train"):
        if prefix == "val":
            names = self.val_loss_names
        elif loss_items is not None and len(loss_items) == len(self.val_loss_names):
            names = self.val_loss_names
        else:
            names = self.train_loss_names

        keys = [f"{prefix}/{x}" for x in names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]
            return dict(zip(keys, loss_items))
        return keys

    def progress_string(self):
        return ("\n" + "%11s" * (4 + len(self.train_loss_names))) % (
            "Epoch",
            "GPU_mem",
            *self.train_loss_names,
            "Instances",
            "Size",
        )

    def validate(self):
        # Ultralytics validator accumulates model.loss(...)[1], which remains the standard 3-term detection loss.
        # Temporarily expose only those 3 terms to keep validator loss buffers aligned.
        orig_loss_items = self.loss_items
        if isinstance(self.loss_items, torch.Tensor) and self.loss_items.numel() > len(self.val_loss_names):
            self.loss_items = self.loss_items[: len(self.val_loss_names)].clone()
        try:
            return super().validate()
        finally:
            self.loss_items = orig_loss_items

    def _kd_weight(self):
        if self.kd_cfg.kd_warmup_epochs <= 0:
            return 1.0
        return min(1.0, (self.epoch + 1) / float(self.kd_cfg.kd_warmup_epochs))

    def _do_train(self):
        # Copy parent loop and inject KD term at forward stage.
        # Keeping explicit copy avoids touching upstream trainer implementation.
        import math
        import time
        import warnings
        import numpy as np
        import torch.distributed as dist
        from ultralytics.utils import LOGGER, RANK, TQDM, colorstr
        from ultralytics.utils.torch_utils import autocast, unwrap_model

        if self.world_size > 1:
            self._setup_ddp()
        self._setup_train()

        nb = len(self.train_loader)
        nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1
        last_opt_step = -1
        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        self.run_callbacks("on_train_start")
        LOGGER.info(
            f"Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n"
            f"Using {self.train_loader.num_workers * (self.world_size or 1)} dataloader workers\n"
            f"Logging results to {colorstr('bold', self.save_dir)}\n"
            f"Starting training for " + (f"{self.args.time} hours..." if self.args.time else f"{self.epochs} epochs...")
        )
        if self.args.close_mosaic:
            base_idx = (self.epochs - self.args.close_mosaic) * nb
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])
        epoch = self.start_epoch
        self.optimizer.zero_grad()
        while True:
            self.epoch = epoch
            self.run_callbacks("on_train_epoch_start")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.scheduler.step()

            self._model_train()
            if RANK != -1:
                self.train_loader.sampler.set_epoch(epoch)
            pbar = enumerate(self.train_loader)
            if epoch == (self.epochs - self.args.close_mosaic):
                self._close_dataloader_mosaic()
                self.train_loader.reset()

            if RANK in {-1, 0}:
                LOGGER.info(self.progress_string())
                pbar = TQDM(enumerate(self.train_loader), total=nb)
            self.tloss = None
            for i, batch in pbar:
                self.run_callbacks("on_train_batch_start")
                ni = i + nb * epoch
                if ni <= nw:
                    xi = [0, nw]
                    self.accumulate = max(1, int(np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round()))
                    for x in self.optimizer.param_groups:
                        x["lr"] = np.interp(
                            ni,
                            xi,
                            [
                                self.args.warmup_bias_lr if x.get("param_group") == "bias" else 0.0,
                                x["initial_lr"] * self.lf(epoch),
                            ],
                        )
                        if "momentum" in x:
                            x["momentum"] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])

                with autocast(self.amp):
                    batch = self.preprocess_batch(batch)
                    preds = self.model(batch["img"])
                    loss, self.loss_items = unwrap_model(self.model).loss(batch, preds)
                    with torch.no_grad():
                        t_preds = self.teacher_model(batch["img"])
                    kd = _build_kd_loss(
                        preds,
                        t_preds,
                        kd_cls=self.kd_cfg.kd_cls,
                        kd_box=self.kd_cfg.kd_box,
                        kd_temp=self.kd_cfg.kd_temp,
                    )
                    kd = kd * self._kd_weight()
                    self.loss = loss.sum() + kd
                    if isinstance(self.loss_items, torch.Tensor):
                        self.loss_items = torch.cat([self.loss_items, kd.detach().reshape(1)])
                    if RANK != -1:
                        self.loss *= self.world_size
                    self.tloss = self.loss_items if self.tloss is None else (self.tloss * i + self.loss_items) / (i + 1)

                self.scaler.scale(self.loss).backward()
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = ni

                    if self.args.time:
                        self.stop = (time.time() - self.train_time_start) > (self.args.time * 3600)
                        if RANK != -1:
                            broadcast_list = [self.stop if RANK == 0 else None]
                            dist.broadcast_object_list(broadcast_list, 0)
                            self.stop = broadcast_list[0]
                        if self.stop:
                            break

                if RANK in {-1, 0}:
                    loss_length = self.tloss.shape[0] if len(self.tloss.shape) else 1
                    pbar.set_description(
                        ("%11s" * 2 + "%11.4g" * (2 + loss_length))
                        % (
                            f"{epoch + 1}/{self.epochs}",
                            f"{self._get_memory():.3g}G",
                            *(self.tloss if loss_length > 1 else torch.unsqueeze(self.tloss, 0)),
                            batch["cls"].shape[0],
                            batch["img"].shape[-1],
                        )
                    )
                    self.run_callbacks("on_batch_end")
                    if self.args.plots and ni in self.plot_idx:
                        self.plot_training_samples(batch, ni)
                self.run_callbacks("on_train_batch_end")

            if hasattr(unwrap_model(self.model).criterion, "update"):
                unwrap_model(self.model).criterion.update()
            self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}
            self.run_callbacks("on_train_epoch_end")
            if RANK in {-1, 0}:
                self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])

            final_epoch = epoch + 1 >= self.epochs
            if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
                self._clear_memory(threshold=0.5)
                self.metrics, self.fitness = self.validate()
            if self._handle_nan_recovery(epoch):
                continue

            self.nan_recovery_attempts = 0
            if RANK in {-1, 0}:
                self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
                self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch
                if self.args.time:
                    self.stop |= (time.time() - self.train_time_start) > (self.args.time * 3600)
                if self.args.save or final_epoch:
                    self.save_model()
                    self.run_callbacks("on_model_save")

            t = time.time()
            self.epoch_time = t - self.epoch_time_start
            self.epoch_time_start = t
            if self.args.time:
                mean_epoch_time = (t - self.train_time_start) / (epoch - self.start_epoch + 1)
                self.epochs = self.args.epochs = math.ceil(self.args.time * 3600 / mean_epoch_time)
                self._setup_scheduler()
                self.scheduler.last_epoch = self.epoch
                self.stop |= epoch >= self.epochs
            self.run_callbacks("on_fit_epoch_end")
            self._clear_memory(0.5)

            if RANK != -1:
                broadcast_list = [self.stop if RANK == 0 else None]
                dist.broadcast_object_list(broadcast_list, 0)
                self.stop = broadcast_list[0]
            if self.stop:
                break
            epoch += 1

        if RANK in {-1, 0}:
            seconds = time.time() - self.train_time_start
            LOGGER.info(f"\n{epoch - self.start_epoch + 1} epochs completed in {seconds / 3600:.3f} hours.")
            self.final_eval()
            if self.args.plots:
                self.plot_metrics()
            self.run_callbacks("on_train_end")
        self._clear_memory()
        self.run_callbacks("teardown")


def train_distill(model_cfg: str, data_cfg: str, teacher: str, **kwargs):
    """Train with output-layer knowledge distillation using a frozen teacher."""
    register_ultralytics_modules()
    model = YOLO(model_cfg)
    return model.train(data=data_cfg, trainer=DistillDetectionTrainer, teacher=teacher, **kwargs)
