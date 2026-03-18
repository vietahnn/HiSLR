# hislr/models/hislr.py
# ─────────────────────────────────────────────────────────────────────────────
# HiSLR — Full Model
#
# Integrates:
#   1. Swin-V2 RGB Encoder       (Video Swin Transformer V2)
#   2. MS-GCN Skeleton Encoder   (Multi-Scale Spatio-Temporal GCN)
#   3. HiCMF Fusion Backbone     (3-stage bidirectional cross-attention)
#   4. Classification Head       (MLP + label smoothing CE)
#   5. PhAP Heads                (16 phonological attribute predictors)
#   6. TCR Module                (Temporal Contrastive Regularization)
# ─────────────────────────────────────────────────────────────────────────────

import torch
import torch.nn as nn
import torch.nn.functional as F

from .msgcn import MSGCNEncoder
from .hicmf import HiCMF
from .swin_wrapper import SwinV2Encoder


class PhonologicalHead(nn.Module):
    """
    Single phonological attribute prediction head.
    A lightweight linear classifier attached to z_joint.
    """

    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc(z)  # (B, num_classes)


class TCRProjectionHead(nn.Module):
    """
    2-layer MLP projection head for Temporal Contrastive Regularization.
    Projects z_joint into a lower-dimensional space for contrastive loss.
    """

    def __init__(self, in_dim: int = 2048, proj_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(z), dim=-1)  # (B, proj_dim) — L2 normalized


class ClassificationHead(nn.Module):
    """
    Two-layer MLP classification head.
    z_joint (2048) -> hidden (1024) -> num_classes
    """

    def __init__(
        self,
        in_dim: int = 2048,
        hidden_dim: int = 1024,
        num_classes: int = 2000,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)  # (B, num_classes)


class HiSLR(nn.Module):
    """
    HiSLR: Hierarchical Sign Language Recognition Model.

    Forward pass (training):
        Given RGB clip (B, T, 3, H, W) and skeleton (B, T, J, 4),
        returns a dict with:
            'logits'       : (B, N)         — gloss classification logits
            'phon_logits'  : list of tensors — one (B, K_i) per phonological attribute
            'tcr_proj'     : (B, proj_dim)   — projected embedding for TCR loss
                             (None if use_tcr=False or slow/fast view not provided)

    Forward pass (inference):
        Returns only 'logits'.
    """

    def __init__(
        self,
        num_classes: int = 2000,
        # Swin-V2
        swin_variant: str = "swin_v2_b",
        swin_pretrained: bool = True,
        swin_pretrain_path: str = "",
        swin_drop_path_rate: float = 0.2,
        # MS-GCN
        num_joints: int = 75,
        gcn_hidden_channels: list = None,
        gcn_num_scales: int = 3,
        gcn_dropout: float = 0.25,
        # HiCMF
        fusion_stages: int = 3,
        fusion_heads: list = None,
        fusion_embed_dim: int = 256,
        fusion_dropout: float = 0.1,
        joint_embed_dim: int = 2048,
        # Classification head
        cls_hidden_dim: int = 1024,
        cls_dropout: float = 0.3,
        # PhAP
        use_phap: bool = True,
        phon_attributes: list = None,
        phon_num_classes: list = None,
        # TCR
        use_tcr: bool = True,
        tcr_proj_dim: int = 256,
    ):
        super().__init__()
        if gcn_hidden_channels is None:
            gcn_hidden_channels = [128, 256, 512]
        if fusion_heads is None:
            fusion_heads = [8, 8, 16]

        self.num_classes = num_classes
        self.use_phap = use_phap
        self.use_tcr = use_tcr
        self.joint_embed_dim = joint_embed_dim

        # ── 1. RGB Encoder ────────────────────────────────────────────────────
        self.rgb_encoder = SwinV2Encoder(
            variant=swin_variant,
            pretrained=swin_pretrained,
            pretrain_path=swin_pretrain_path,
            drop_path_rate=swin_drop_path_rate,
        )
        swin_channels = self.rgb_encoder.out_channels  # [128, 256, 512, 1024]

        # ── 2. Skeleton Encoder ───────────────────────────────────────────────
        self.skeleton_encoder = MSGCNEncoder(
            in_channels=4,
            hidden_channels=gcn_hidden_channels,
            num_joints=num_joints,
            num_scales=gcn_num_scales,
            adaptive=True,
            dropout=gcn_dropout,
        )

        # ── 3. HiCMF Fusion Backbone ──────────────────────────────────────────
        self.hicmf = HiCMF(
            swin_channels=swin_channels,
            gcn_channels=gcn_hidden_channels,
            fusion_stages=fusion_stages,
            fusion_heads=fusion_heads,
            embed_dim=fusion_embed_dim,
            dropout=fusion_dropout,
            joint_embed_dim=joint_embed_dim,
        )

        # ── 4. Classification Head ─────────────────────────────────────────────
        self.cls_head = ClassificationHead(
            in_dim=joint_embed_dim,
            hidden_dim=cls_hidden_dim,
            num_classes=num_classes,
            dropout=cls_dropout,
        )

        # ── 5. PhAP Auxiliary Heads ───────────────────────────────────────────
        if use_phap:
            assert phon_attributes is not None and phon_num_classes is not None
            assert len(phon_attributes) == len(phon_num_classes)
            self.phon_attr_names = phon_attributes
            self.phon_heads = nn.ModuleList([
                PhonologicalHead(joint_embed_dim, k)
                for k in phon_num_classes
            ])
        else:
            self.phon_heads = None
            self.phon_attr_names = []

        # ── 6. TCR Projection Head ─────────────────────────────────────────────
        if use_tcr:
            self.tcr_proj = TCRProjectionHead(joint_embed_dim, tcr_proj_dim)
        else:
            self.tcr_proj = None

    def encode(
        self,
        rgb: torch.Tensor,
        skeleton: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode a (rgb, skeleton) pair to a joint embedding z_joint.

        Args:
            rgb:      (B, T, 3, H, W)
            skeleton: (B, T, J, 4)

        Returns:
            z_joint: (B, joint_embed_dim)
        """
        # RGB encoding: returns dict {R1, R2, R3, R4}
        swin_feats = self.rgb_encoder(rgb)

        # Skeleton encoding: returns S1, S2, S3, S1_pool, S2_pool, S3_pool
        S1, S2, S3, S1_pool, S2_pool, S3_pool = self.skeleton_encoder(skeleton)

        gcn_feats = {
            "S1_pool": S1_pool,
            "S2_pool": S2_pool,
            "S3_pool": S3_pool,
        }

        # Hierarchical fusion
        z_joint = self.hicmf(swin_feats, gcn_feats)
        return z_joint

    def forward(
        self,
        rgb: torch.Tensor,
        skeleton: torch.Tensor,
        rgb_slow: torch.Tensor = None,
        skeleton_slow: torch.Tensor = None,
        rgb_fast: torch.Tensor = None,
        skeleton_fast: torch.Tensor = None,
        return_embedding: bool = False,
    ) -> dict:
        """
        Args:
            rgb:           (B, T, 3, H, W)  — main RGB clip
            skeleton:      (B, T, J, 4)     — main skeleton clip
            rgb_slow:      (B, T, 3, H, W)  — temporally slowed clip (for TCR)
            skeleton_slow: (B, T, J, 4)     — slowed skeleton
            rgb_fast:      (B, T, 3, H, W)  — temporally sped-up clip (for TCR)
            skeleton_fast: (B, T, J, 4)     — fast skeleton
            return_embedding: bool          — also return z_joint in output dict

        Returns:
            dict with:
                'logits'        : (B, N)
                'phon_logits'   : list[(B, K_i)]  (None if not use_phap)
                'tcr_proj_slow' : (B, proj_dim)   (None if slow not provided)
                'tcr_proj_fast' : (B, proj_dim)   (None if fast not provided)
                'z_joint'       : (B, joint_embed_dim)  (if return_embedding)
        """
        # ── Main clip ─────────────────────────────────────────────────────────
        z_joint = self.encode(rgb, skeleton)

        out = {}

        # Classification
        logits = self.cls_head(z_joint)
        out["logits"] = logits

        # Phonological auxiliary predictions
        if self.use_phap and self.phon_heads is not None:
            phon_logits = [head(z_joint) for head in self.phon_heads]
            out["phon_logits"] = phon_logits
        else:
            out["phon_logits"] = None

        # TCR: encode slow and fast views
        if self.use_tcr and self.tcr_proj is not None:
            if rgb_slow is not None and skeleton_slow is not None:
                z_slow = self.encode(rgb_slow, skeleton_slow)
                out["tcr_proj_slow"] = self.tcr_proj(z_slow)
            else:
                out["tcr_proj_slow"] = None

            if rgb_fast is not None and skeleton_fast is not None:
                z_fast = self.encode(rgb_fast, skeleton_fast)
                out["tcr_proj_fast"] = self.tcr_proj(z_fast)
            else:
                out["tcr_proj_fast"] = None
        else:
            out["tcr_proj_slow"] = None
            out["tcr_proj_fast"] = None

        if return_embedding:
            out["z_joint"] = z_joint

        return out

    def get_param_groups(
        self,
        lr_pretrained: float = 1e-5,
        lr_new: float = 1e-4,
    ) -> list:
        """
        Separate parameter groups for pretrained (Swin-V2) vs new layers,
        applying different learning rates.
        """
        pretrained_params = list(self.rgb_encoder.parameters())
        pretrained_ids = set(id(p) for p in pretrained_params)

        new_params = [p for p in self.parameters() if id(p) not in pretrained_ids]

        return [
            {"params": pretrained_params, "lr": lr_pretrained, "name": "pretrained_swin"},
            {"params": new_params, "lr": lr_new, "name": "new_layers"},
        ]
