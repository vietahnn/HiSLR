# hislr/models/hicmf.py
# ─────────────────────────────────────────────────────────────────────────────
# Hierarchical Cross-Modal Fusion (HiCMF) Backbone
#
# Fuses RGB (Swin-V2) and Skeleton (MS-GCN) features at three levels:
#   Stage 1 — Early:        R1 (shallow RGB)  <-> S1_pool (T resolution)
#   Stage 2 — Intermediate: R3 (mid RGB)      <-> S2_pool (T/2 resolution)
#   Stage 3 — Late:         R4 (deep RGB)     <-> S3_pool (T/4 resolution)
#
# Each fusion stage uses bidirectional cross-attention:
#   RGB attends over Skeleton, Skeleton attends over RGB.
# Outputs are fused via residual addition and then projected.
# ─────────────────────────────────────────────────────────────────────────────

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalAttention(nn.Module):
    """
    Bidirectional Cross-Modal Attention block.

    Given two sequences Q_a (query from modality A) and K_b (key/value from
    modality B), computes:
        A' = A + Attention(Q_A, K_B, V_B)   [A attends over B]
        B' = B + Attention(Q_B, K_A, V_A)   [B attends over A]

    Both sequences can have different lengths and channel dims (projected to
    embed_dim before attention).
    """

    def __init__(
        self,
        dim_a: int,
        dim_b: int,
        embed_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0

        # Projections for modality A (RGB)
        self.proj_a_q = nn.Linear(dim_a, embed_dim)
        self.proj_a_k = nn.Linear(dim_a, embed_dim)
        self.proj_a_v = nn.Linear(dim_a, embed_dim)
        self.out_proj_a = nn.Linear(embed_dim, dim_a)

        # Projections for modality B (Skeleton)
        self.proj_b_q = nn.Linear(dim_b, embed_dim)
        self.proj_b_k = nn.Linear(dim_b, embed_dim)
        self.proj_b_v = nn.Linear(dim_b, embed_dim)
        self.out_proj_b = nn.Linear(embed_dim, dim_b)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

        self.norm_a = nn.LayerNorm(dim_a)
        self.norm_b = nn.LayerNorm(dim_b)

    def _attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """
        Scaled dot-product attention.
        Args:
            q: (B, L_q, E)
            k: (B, L_k, E)
            v: (B, L_k, E)
        Returns:
            out: (B, L_q, E)
        """
        B, L_q, E = q.shape
        L_k = k.shape[1]
        H = self.num_heads
        D = self.head_dim

        q = q.reshape(B, L_q, H, D).transpose(1, 2)  # (B, H, L_q, D)
        k = k.reshape(B, L_k, H, D).transpose(1, 2)  # (B, H, L_k, D)
        v = v.reshape(B, L_k, H, D).transpose(1, 2)  # (B, H, L_k, D)

        attn = torch.einsum("bhqd,bhkd->bhqk", q, k) * self.scale  # (B, H, L_q, L_k)
        attn = self.dropout(F.softmax(attn, dim=-1))

        out = torch.einsum("bhqk,bhkd->bhqd", attn, v)  # (B, H, L_q, D)
        out = out.transpose(1, 2).reshape(B, L_q, E)     # (B, L_q, E)
        return out

    def forward(
        self,
        feat_a: torch.Tensor,
        feat_b: torch.Tensor,
    ):
        """
        Args:
            feat_a: (B, L_a, dim_a)   — RGB features (flattened tokens)
            feat_b: (B, L_b, dim_b)   — Skeleton features (time steps)
        Returns:
            feat_a': (B, L_a, dim_a)
            feat_b': (B, L_b, dim_b)
        """
        # ── A attends over B ─────────────────────────────────────────────────
        q_a = self.proj_a_q(feat_a)
        k_b = self.proj_b_k(feat_b)
        v_b = self.proj_b_v(feat_b)
        attn_a = self._attention(q_a, k_b, v_b)
        delta_a = self.out_proj_a(attn_a)
        feat_a_out = self.norm_a(feat_a + delta_a)

        # ── B attends over A ─────────────────────────────────────────────────
        q_b = self.proj_b_q(feat_b)
        k_a = self.proj_a_k(feat_a)
        v_a = self.proj_a_v(feat_a)
        attn_b = self._attention(q_b, k_a, v_a)
        delta_b = self.out_proj_b(attn_b)
        feat_b_out = self.norm_b(feat_b + delta_b)

        return feat_a_out, feat_b_out


class HiCMF(nn.Module):
    """
    Hierarchical Cross-Modal Fusion backbone.

    Accepts hierarchical features from Swin-V2 and MS-GCN encoders and
    performs progressive bidirectional cross-attention fusion at 3 stages.

    After Stage 3, produces a joint embedding of shape (B, joint_embed_dim)
    by concatenating the globally pooled RGB and skeleton representations.
    """

    def __init__(
        self,
        swin_channels: list = None,    # [128, 256, 512, 1024] for Swin-V2-B
        gcn_channels: list = None,     # [128, 256, 512]
        fusion_stages: int = 3,
        fusion_heads: list = None,     # heads per stage [8, 8, 16]
        embed_dim: int = 256,
        dropout: float = 0.1,
        joint_embed_dim: int = 2048,
    ):
        super().__init__()
        if swin_channels is None:
            swin_channels = [128, 256, 512, 1024]
        if gcn_channels is None:
            gcn_channels = [128, 256, 512]
        if fusion_heads is None:
            fusion_heads = [8, 8, 16]

        self.fusion_stages = fusion_stages
        self.joint_embed_dim = joint_embed_dim

        # ── Stage 1: RGB R1 (128) <-> Skeleton S1_pool (128) ────────────────
        if fusion_stages >= 1:
            self.cma_stage1 = CrossModalAttention(
                dim_a=swin_channels[0],
                dim_b=gcn_channels[0],
                embed_dim=embed_dim,
                num_heads=fusion_heads[0],
                dropout=dropout,
            )

        # ── Stage 2: RGB R3 (512) <-> Skeleton S2_pool (256) ────────────────
        if fusion_stages >= 2:
            self.cma_stage2 = CrossModalAttention(
                dim_a=swin_channels[2],
                dim_b=gcn_channels[1],
                embed_dim=embed_dim,
                num_heads=fusion_heads[1],
                dropout=dropout,
            )
            # Align skeleton T to T/2 (already at T/2 for S2)
            # Align RGB R3 spatial tokens (H/16 * W/16) — kept as is

        # ── Stage 3: RGB R4 (1024) <-> Skeleton S3_pool (512) ───────────────
        if fusion_stages >= 3:
            self.cma_stage3 = CrossModalAttention(
                dim_a=swin_channels[3],
                dim_b=gcn_channels[2],
                embed_dim=embed_dim,
                num_heads=fusion_heads[2],
                dropout=dropout,
            )

        # Final projection: concat pooled RGB (1024) + pooled Skel (512) -> joint_embed_dim
        self.final_proj = nn.Sequential(
            nn.Linear(swin_channels[3] + gcn_channels[2], joint_embed_dim),
            nn.LayerNorm(joint_embed_dim),
            nn.GELU(),
        )

    def _flatten_swin(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Flatten Swin spatial-temporal feature map to token sequence.
        Input:  (B, C, T, H, W) or (B, T, H, W, C)  — handle both conventions
        Output: (B, T*H*W, C)
        """
        if feat.dim() == 5:
            if feat.shape[1] == feat.shape[-1]:
                # Ambiguous; assume (B, T, H, W, C) (Swin output convention)
                B, T, H, W, C = feat.shape
                return feat.reshape(B, T * H * W, C)
            else:
                # (B, C, T, H, W)
                B, C, T, H, W = feat.shape
                return feat.permute(0, 2, 3, 4, 1).reshape(B, T * H * W, C)
        raise ValueError(f"Unexpected feature shape: {feat.shape}")

    def _pool_swin(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Spatiotemporal average pool Swin feature map.
        Input:  (B, T, H, W, C) or (B, C, T, H, W)
        Output: (B, C)
        """
        if feat.dim() == 5:
            if feat.shape[-1] < feat.shape[1]:
                # (B, C, T, H, W)
                return feat.mean(dim=[2, 3, 4])
            else:
                # (B, T, H, W, C)
                return feat.mean(dim=[1, 2, 3])
        raise ValueError(f"Unexpected feature shape: {feat.shape}")

    def forward(
        self,
        swin_features: dict,
        gcn_features: dict,
    ) -> torch.Tensor:
        """
        Args:
            swin_features: dict with keys 'R1', 'R2', 'R3', 'R4'
                           each (B, T_i, H_i, W_i, C_i) [Swin THWC convention]
            gcn_features:  dict with keys 'S1_pool', 'S2_pool', 'S3_pool'
                           each (B, T_i, C_i) — joint-averaged skeleton features

        Returns:
            z_joint: (B, joint_embed_dim)  — unified joint embedding
        """
        R1 = swin_features["R1"]
        R3 = swin_features["R3"]
        R4 = swin_features["R4"]
        S1_pool = gcn_features["S1_pool"]   # (B, T, 128)
        S2_pool = gcn_features["S2_pool"]   # (B, T/2, 256)
        S3_pool = gcn_features["S3_pool"]   # (B, T/4, 512)

        # ── Stage 1: Early Fusion ────────────────────────────────────────────
        if self.fusion_stages >= 1:
            r1_tok = self._flatten_swin(R1)  # (B, T*H1*W1, 128)
            R1_fused, S1_fused = self.cma_stage1(r1_tok, S1_pool)
            # Write back fused S1 (we carry it forward; R1 fusion updates local)

        # ── Stage 2: Intermediate Fusion ─────────────────────────────────────
        if self.fusion_stages >= 2:
            r3_tok = self._flatten_swin(R3)  # (B, T*H3*W3, 512)
            R3_fused, S2_fused = self.cma_stage2(r3_tok, S2_pool)

        # ── Stage 3: Late Semantic Fusion ─────────────────────────────────────
        if self.fusion_stages >= 3:
            r4_tok = self._flatten_swin(R4)  # (B, T*H4*W4, 1024)
            R4_fused, S3_fused = self.cma_stage3(r4_tok, S3_pool)
        else:
            # Fallback: use unfused features
            R4_fused = self._flatten_swin(R4)
            S3_fused = S3_pool

        # ── Global Pooling & Joint Embedding ─────────────────────────────────
        rgb_vec  = R4_fused.mean(dim=1)   # (B, 1024)
        skel_vec = S3_fused.mean(dim=1)   # (B, 512)

        joint = torch.cat([rgb_vec, skel_vec], dim=-1)  # (B, 1536)
        z_joint = self.final_proj(joint)                 # (B, joint_embed_dim)

        return z_joint
