# hislr/training/trainer.py
# ─────────────────────────────────────────────────────────────────────────────
# HiSLR Training Engine
#
# Features:
#   - Two-stage training (Stage 1: PhAP pre-train, Stage 2: full + TCR)
#   - Multi-GPU training via torch.nn.parallel.DistributedDataParallel (DDP)
#   - Mixed-precision (AMP) with gradient clipping
#   - Cosine LR scheduler with linear warmup
#   - Square-root class-balanced sampling
#   - Checkpoint saving (top-K by val accuracy)
#   - Per-instance and per-class Top-1 / Top-5 accuracy evaluation
# ─────────────────────────────────────────────────────────────────────────────

import os
import time
import math
import json
import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR

from ..models.hislr import HiSLR
from ..training.losses import HiSLRLoss
from ..data.dataset import ISLRDataset, build_weighted_sampler
from ..configs.default import HiSLRConfig
from ..utils.metrics import compute_accuracy, AverageMeter

logger = logging.getLogger(__name__)


# ── LR Schedule ───────────────────────────────────────────────────────────────

def cosine_schedule_with_warmup(optimizer, warmup_epochs: int, total_epochs: int):
    """
    Cosine annealing LR schedule with linear warmup.
    """
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch) / max(1, warmup_epochs)
        progress = float(epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


# ── Checkpoint Manager ─────────────────────────────────────────────────────────

class CheckpointManager:
    """Keeps the top-K checkpoints ranked by validation accuracy."""

    def __init__(self, checkpoint_dir: str, save_top_k: int = 3):
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.checkpoint_dir = checkpoint_dir
        self.save_top_k = save_top_k
        self.best_scores = []   # list of (score, path)

    def save(self, model: nn.Module, optimizer, scheduler, epoch: int, score: float, cfg: HiSLRConfig):
        path = os.path.join(self.checkpoint_dir, f"epoch_{epoch:04d}_acc{score:.4f}.pth")
        torch.save({
            "epoch": epoch,
            "score": score,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler else None,
        }, path)

        self.best_scores.append((score, path))
        self.best_scores.sort(key=lambda x: x[0], reverse=True)

        # Remove excess checkpoints
        while len(self.best_scores) > self.save_top_k:
            _, old_path = self.best_scores.pop()
            if os.path.exists(old_path):
                os.remove(old_path)

        return path

    @property
    def best_checkpoint(self) -> Optional[str]:
        return self.best_scores[0][1] if self.best_scores else None


# ── Trainer ────────────────────────────────────────────────────────────────────

class HiSLRTrainer:
    """
    Full training and evaluation engine for HiSLR.
    """

    def __init__(self, cfg: HiSLRConfig, rank: int = 0, world_size: int = 1):
        self.cfg = cfg
        self.rank = rank
        self.world_size = world_size
        self.is_main = (rank == 0)

        os.makedirs(cfg.training.checkpoint_dir, exist_ok=True)
        os.makedirs(cfg.training.log_dir, exist_ok=True)

        # Logging
        if self.is_main:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s [%(levelname)s] %(message)s",
                handlers=[
                    logging.StreamHandler(),
                    logging.FileHandler(os.path.join(cfg.training.log_dir, "train.log")),
                ],
            )

        self._build_model()
        self._build_dataloaders()
        self._build_optimizer()
        self._build_loss()

        self.checkpoint_mgr = CheckpointManager(
            cfg.training.checkpoint_dir, cfg.training.save_top_k
        )
        self.scaler = GradScaler(enabled=cfg.training.use_amp)

    def _build_model(self):
        mc = self.cfg.model
        dc = self.cfg.data

        self.model = HiSLR(
            num_classes=dc.num_classes,
            swin_variant=mc.swin_variant,
            swin_pretrained=mc.swin_pretrained,
            swin_pretrain_path=mc.swin_pretrain_path,
            swin_drop_path_rate=mc.swin_drop_path_rate,
            num_joints=dc.num_joints,
            gcn_hidden_channels=mc.gcn_hidden_channels,
            gcn_num_scales=mc.gcn_num_scales,
            gcn_dropout=mc.gcn_dropout,
            fusion_stages=mc.fusion_stages,
            fusion_heads=mc.fusion_heads,
            fusion_embed_dim=mc.fusion_embed_dim,
            fusion_dropout=mc.fusion_dropout,
            joint_embed_dim=mc.joint_embed_dim,
            cls_hidden_dim=mc.cls_hidden_dim,
            cls_dropout=mc.cls_dropout,
            use_phap=mc.use_phap,
            phon_attributes=mc.phon_attributes if mc.use_phap else None,
            phon_num_classes=mc.phon_num_classes if mc.use_phap else None,
            use_tcr=mc.use_tcr,
            tcr_proj_dim=mc.tcr_proj_dim,
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        if self.world_size > 1:
            self.model = nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.rank],
                find_unused_parameters=True,
            )

        if self.is_main:
            total_params = sum(p.numel() for p in self.model.parameters())
            logger.info(f"HiSLR model: {total_params / 1e6:.1f}M parameters")

    def _build_dataloaders(self):
        dc = self.cfg.data
        tc = self.cfg.training
        mc = self.cfg.model

        train_dataset = ISLRDataset(
            annotation_file=os.path.join(dc.data_root, dc.annotation_file),
            num_frames=dc.num_frames,
            frame_size=dc.frame_size,
            num_joints=dc.num_joints,
            is_train=True,
            normalize_skeleton=dc.normalize_skeleton,
            random_flip=dc.random_flip,
            color_jitter=dc.color_jitter,
            joint_noise_sigma=dc.joint_noise_sigma,
            joint_dropout_prob=dc.joint_dropout_prob,
            speed_aug_range=tuple(dc.speed_aug_range),
            skeleton_scale_range=tuple(dc.skeleton_scale_range),
            skeleton_rotate_range=dc.skeleton_rotate_range,
            tcr_slow_factor=mc.tcr_slow_factor if mc.use_tcr else 0.5,
            tcr_fast_factor=mc.tcr_fast_factor if mc.use_tcr else 2.0,
            generate_tcr_views=mc.use_tcr,
        )

        val_dataset = ISLRDataset(
            annotation_file=os.path.join(dc.data_root, "val_annotations.json"),
            num_frames=dc.num_frames,
            frame_size=dc.frame_size,
            num_joints=dc.num_joints,
            is_train=False,
            normalize_skeleton=dc.normalize_skeleton,
            generate_tcr_views=False,
        )

        sampler = build_weighted_sampler(train_dataset, tc.sampling_strategy)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=tc.batch_size,
            sampler=sampler,
            num_workers=dc.num_workers,
            pin_memory=dc.pin_memory,
            drop_last=True,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=tc.batch_size * 2,
            shuffle=False,
            num_workers=dc.num_workers,
            pin_memory=dc.pin_memory,
        )

        self.num_classes = dc.num_classes

    def _build_optimizer(self):
        tc = self.cfg.training
        mc = self.cfg.model

        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        param_groups = raw_model.get_param_groups(
            lr_pretrained=tc.lr_pretrained,
            lr_new=tc.lr_new_layers,
        )

        self.optimizer = optim.AdamW(param_groups, weight_decay=tc.weight_decay)
        self.scheduler = cosine_schedule_with_warmup(
            self.optimizer,
            warmup_epochs=tc.warmup_epochs,
            total_epochs=tc.total_epochs,
        )

    def _build_loss(self):
        mc = self.cfg.model
        tc = self.cfg.training
        dc = self.cfg.data

        self.criterion = HiSLRLoss(
            num_classes=dc.num_classes,
            label_smoothing=mc.label_smoothing,
            phon_num_classes=mc.phon_num_classes if mc.use_phap else None,
            lambda_phon=tc.lambda_phon,
            lambda_tcr=tc.lambda_tcr,
            tcr_temperature=mc.tcr_temperature,
        )

    def _current_stage(self, epoch: int) -> int:
        """Return 1 (pre-train) or 2 (full training) based on epoch."""
        return 1 if epoch < self.cfg.training.stage1_epochs else 2

    def train_one_epoch(self, epoch: int) -> dict:
        self.model.train()
        tc = self.cfg.training
        stage = self._current_stage(epoch)

        meters = {k: AverageMeter() for k in ["total", "ce", "phon", "tcr", "top1", "top5"]}
        t0 = time.time()

        for step, batch in enumerate(self.train_loader):
            rgb       = batch["rgb"].to(self.device, non_blocking=True)
            skeleton  = batch["skeleton"].to(self.device, non_blocking=True)
            gloss_lbl = batch["gloss_label"].to(self.device, non_blocking=True)
            phon_lbl  = batch["phon_labels"].to(self.device, non_blocking=True)

            # TCR views (only in Stage 2)
            rgb_slow = skeleton_slow = rgb_fast = skeleton_fast = None
            if stage == 2 and "rgb_slow" in batch:
                rgb_slow      = batch["rgb_slow"].to(self.device, non_blocking=True)
                skeleton_slow = batch["skeleton_slow"].to(self.device, non_blocking=True)
                rgb_fast      = batch["rgb_fast"].to(self.device, non_blocking=True)
                skeleton_fast = batch["skeleton_fast"].to(self.device, non_blocking=True)

            with autocast(enabled=tc.use_amp):
                out = self.model(
                    rgb, skeleton,
                    rgb_slow=rgb_slow, skeleton_slow=skeleton_slow,
                    rgb_fast=rgb_fast, skeleton_fast=skeleton_fast,
                )
                losses = self.criterion(out, gloss_lbl, phon_lbl, stage=stage)

            self.optimizer.zero_grad()
            self.scaler.scale(losses["total"]).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), tc.grad_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Metrics
            B = rgb.shape[0]
            top1, top5 = compute_accuracy(out["logits"], gloss_lbl, topk=(1, 5))
            for k, v in losses.items():
                if k in meters:
                    meters[k].update(v.item(), B)
            meters["top1"].update(top1, B)
            meters["top5"].update(top5, B)

            if self.is_main and step % tc.log_interval == 0:
                elapsed = time.time() - t0
                logger.info(
                    f"[Stage {stage}] Ep {epoch:3d} Step {step:4d}/{len(self.train_loader)} | "
                    f"Loss {meters['total'].avg:.4f} (CE:{meters['ce'].avg:.3f} "
                    f"Ph:{meters['phon'].avg:.3f} TCR:{meters['tcr'].avg:.3f}) | "
                    f"Top1 {meters['top1'].avg:.2f}% | {elapsed:.1f}s"
                )

        self.scheduler.step()

        return {k: m.avg for k, m in meters.items()}

    @torch.no_grad()
    def evaluate(self, epoch: int) -> dict:
        self.model.eval()

        all_logits  = []
        all_labels  = []

        for batch in self.val_loader:
            rgb      = batch["rgb"].to(self.device, non_blocking=True)
            skeleton = batch["skeleton"].to(self.device, non_blocking=True)
            labels   = batch["gloss_label"].to(self.device, non_blocking=True)

            with autocast(enabled=self.cfg.training.use_amp):
                out = self.model(rgb, skeleton)

            all_logits.append(out["logits"].cpu())
            all_labels.append(labels.cpu())

        all_logits = torch.cat(all_logits)   # (N, C)
        all_labels = torch.cat(all_labels)   # (N,)

        # Per-instance accuracy
        pi_top1, pi_top5 = compute_accuracy(all_logits, all_labels, topk=(1, 5))

        # Per-class top-1 accuracy
        pc_top1 = compute_per_class_accuracy(all_logits, all_labels, self.num_classes)

        if self.is_main:
            logger.info(
                f"[Eval] Epoch {epoch:3d} | "
                f"P-I Top1: {pi_top1:.2f}%  Top5: {pi_top5:.2f}%  |  "
                f"P-C Top1: {pc_top1:.2f}%"
            )

        return {"pi_top1": pi_top1, "pi_top5": pi_top5, "pc_top1": pc_top1}

    def train(self):
        tc = self.cfg.training
        best_pi_top1 = 0.0

        logger.info(f"Starting HiSLR training | {tc.total_epochs} epochs | "
                    f"Stage 1: {tc.stage1_epochs} | Stage 2: {tc.stage2_epochs}")

        for epoch in range(tc.total_epochs):
            train_metrics = self.train_one_epoch(epoch)

            if (epoch + 1) % tc.val_interval == 0:
                val_metrics = self.evaluate(epoch)
                pi_top1 = val_metrics["pi_top1"]

                if self.is_main and pi_top1 > best_pi_top1:
                    best_pi_top1 = pi_top1
                    raw_model = self.model.module if hasattr(self.model, "module") else self.model
                    path = self.checkpoint_mgr.save(
                        raw_model, self.optimizer, self.scheduler, epoch, pi_top1, self.cfg
                    )
                    logger.info(f"New best P-I Top1: {pi_top1:.2f}% — saved to {path}")

        if self.is_main:
            logger.info(f"Training complete. Best P-I Top1: {best_pi_top1:.2f}%")
            logger.info(f"Best checkpoint: {self.checkpoint_mgr.best_checkpoint}")


def compute_per_class_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> float:
    """Compute per-class Top-1 accuracy (macro-average over classes)."""
    preds = logits.argmax(dim=1)
    class_correct = torch.zeros(num_classes)
    class_total   = torch.zeros(num_classes)

    for c in range(num_classes):
        mask = labels == c
        if mask.sum() > 0:
            class_correct[c] = (preds[mask] == labels[mask]).float().sum()
            class_total[c]   = mask.sum()

    valid = class_total > 0
    if valid.sum() == 0:
        return 0.0
    return (class_correct[valid] / class_total[valid]).mean().item() * 100.0
