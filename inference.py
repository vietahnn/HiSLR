# hislr/inference.py
# ─────────────────────────────────────────────────────────────────────────────
# HiSLR Inference — load checkpoint, run prediction on a single clip or
# evaluate on a full dataset split.
# ─────────────────────────────────────────────────────────────────────────────

import os
import json
import torch
import numpy as np
from typing import Optional, List, Tuple
from torch.cuda.amp import autocast

from .models.hislr import HiSLR
from .data.dataset import ISLRDataset, normalize_skeleton
from .configs.default import HiSLRConfig
from .utils.metrics import compute_accuracy


def load_model(
    checkpoint_path: str,
    cfg: HiSLRConfig,
    device: str = "cuda",
) -> HiSLR:
    """
    Load a HiSLR model from a checkpoint.

    Args:
        checkpoint_path: path to .pth checkpoint file
        cfg:             HiSLRConfig used during training
        device:          "cuda" or "cpu"

    Returns:
        model in eval mode on the specified device
    """
    mc = cfg.model
    dc = cfg.data

    model = HiSLR(
        num_classes=dc.num_classes,
        swin_variant=mc.swin_variant,
        swin_pretrained=False,       # weights come from checkpoint
        num_joints=dc.num_joints,
        gcn_hidden_channels=mc.gcn_hidden_channels,
        gcn_num_scales=mc.gcn_num_scales,
        gcn_dropout=mc.gcn_dropout,
        fusion_stages=mc.fusion_stages,
        fusion_heads=mc.fusion_heads,
        fusion_embed_dim=mc.fusion_embed_dim,
        joint_embed_dim=mc.joint_embed_dim,
        cls_hidden_dim=mc.cls_hidden_dim,
        use_phap=False,              # not needed at inference
        use_tcr=False,
    )

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state, strict=False)

    model = model.to(device).eval()
    print(f"Loaded HiSLR from {checkpoint_path}")
    return model


class HiSLRInference:
    """
    High-level inference API for HiSLR.

    Example usage:
        engine = HiSLRInference("checkpoint.pth", cfg, class_names)
        result = engine.predict_video("video.mp4", "skeleton.npy")
        print(result["predicted_sign"], result["confidence"])
    """

    def __init__(
        self,
        checkpoint_path: str,
        cfg: HiSLRConfig,
        class_names: Optional[List[str]] = None,
        device: str = "cuda",
    ):
        self.cfg = cfg
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = load_model(checkpoint_path, cfg, str(self.device))
        self.class_names = class_names or []

        # RGB normalization
        self.rgb_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.rgb_std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)

    def _preprocess_frames(self, frames: np.ndarray) -> torch.Tensor:
        """
        (T, H, W, 3) uint8 -> (1, T, 3, H, W) float32 normalized tensor on device.
        """
        t = torch.from_numpy(frames).float() / 255.0    # (T, H, W, 3)
        t = t.permute(0, 3, 1, 2).unsqueeze(0)          # (1, T, 3, H, W)
        t = t.to(self.device)
        return (t - self.rgb_mean.unsqueeze(2)) / self.rgb_std.unsqueeze(2)

    def _preprocess_skeleton(self, joints: np.ndarray) -> torch.Tensor:
        """
        (T, J, 4) float32 -> (1, T, J, 4) tensor on device, normalized.
        """
        joints = normalize_skeleton(joints)
        return torch.from_numpy(joints).float().unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict(
        self,
        frames: np.ndarray,
        joints: np.ndarray,
        top_k: int = 5,
    ) -> dict:
        """
        Run inference on pre-loaded frames and skeleton joints.

        Args:
            frames: (T, H, W, 3) uint8 numpy array
            joints: (T, J, 4) float32 numpy array (raw, unnormalized)
            top_k:  number of top predictions to return

        Returns:
            dict with:
                'predicted_class':  int
                'predicted_sign':   str (if class_names available)
                'confidence':       float
                'top_k_classes':    list of int
                'top_k_signs':      list of str
                'top_k_probs':      list of float
        """
        rgb_tensor  = self._preprocess_frames(frames)    # (1, T, 3, H, W)
        skel_tensor = self._preprocess_skeleton(joints)  # (1, T, J, 4)

        with autocast(enabled=True):
            out = self.model(rgb_tensor, skel_tensor)

        logits = out["logits"][0]                        # (N,)
        probs  = torch.softmax(logits, dim=0)

        top_probs, top_indices = probs.topk(top_k)
        top_probs   = top_probs.cpu().tolist()
        top_indices = top_indices.cpu().tolist()

        top_signs = [
            self.class_names[i] if i < len(self.class_names) else f"class_{i}"
            for i in top_indices
        ]

        return {
            "predicted_class": top_indices[0],
            "predicted_sign":  top_signs[0],
            "confidence":      top_probs[0],
            "top_k_classes":   top_indices,
            "top_k_signs":     top_signs,
            "top_k_probs":     top_probs,
        }

    @torch.no_grad()
    def evaluate_dataset(
        self,
        annotation_file: str,
        batch_size: int = 32,
    ) -> dict:
        """
        Run full evaluation on a dataset split and return accuracy metrics.
        """
        from torch.utils.data import DataLoader
        from .utils.metrics import compute_accuracy

        dc = self.cfg.data

        dataset = ISLRDataset(
            annotation_file=annotation_file,
            num_frames=dc.num_frames,
            frame_size=dc.frame_size,
            num_joints=dc.num_joints,
            is_train=False,
            normalize_skeleton=True,
        )

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        all_logits = []
        all_labels = []

        for batch in loader:
            rgb  = batch["rgb"].to(self.device)
            skel = batch["skeleton"].to(self.device)
            lbl  = batch["gloss_label"]

            with autocast(enabled=True):
                out = self.model(rgb, skel)

            all_logits.append(out["logits"].cpu())
            all_labels.append(lbl)

        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)

        top1, top5 = compute_accuracy(all_logits, all_labels, topk=(1, 5))

        print(f"Evaluation results:")
        print(f"  P-I Top-1: {top1:.2f}%")
        print(f"  P-I Top-5: {top5:.2f}%")

        return {"pi_top1": top1, "pi_top5": top5}
