#!/usr/bin/env python3
# tests/test_hislr.py
# ─────────────────────────────────────────────────────────────────────────────
# Unit tests for all HiSLR components.
# Run with: python -m pytest tests/test_hislr.py -v
# (Does NOT require GPU or real data — all tests use random tensors.)
# ─────────────────────────────────────────────────────────────────────────────

import pytest
import torch
import numpy as np
import sys
import os

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

B  = 2      # batch size
T  = 16     # frames
H  = 224    # height
W  = 224    # width
J  = 75     # joints
C  = 4      # joint channels (x, y, z, vis)
N  = 100    # number of classes (small for test speed)
D  = 256    # embed dim


@pytest.fixture
def rgb_batch():
    return torch.randn(B, T, 3, H, W)


@pytest.fixture
def skel_batch():
    return torch.randn(B, T, J, C)


@pytest.fixture
def gloss_labels():
    return torch.randint(0, N, (B,))


@pytest.fixture
def phon_labels():
    # 16 attributes; some -1 (unannotated)
    labels = torch.randint(0, 5, (B, 16))
    labels[:, 2] = -1   # simulate missing annotation
    labels[:, 7] = -1
    return labels


# ─────────────────────────────────────────────────────────────────────────────
# 1. MS-GCN Skeleton Encoder
# ─────────────────────────────────────────────────────────────────────────────

class TestMSGCN:

    def test_adjacency_matrix_shape(self):
        from hislr.models.msgcn import build_adjacency_matrix
        A = build_adjacency_matrix(75)
        assert A.shape == (75, 75), "Adjacency matrix must be (J, J)"

    def test_adjacency_matrix_symmetric(self):
        from hislr.models.msgcn import build_adjacency_matrix
        A = build_adjacency_matrix(75)
        assert np.allclose(A, A.T, atol=1e-5), "Adjacency must be symmetric"

    def test_adjacency_diagonal_ones(self):
        from hislr.models.msgcn import build_adjacency_matrix
        A = build_adjacency_matrix(75)
        # After normalization diagonal won't be exactly 1, but > 0
        assert np.all(np.diag(A) > 0), "Diagonal (self-connections) must be positive"

    def test_spatial_graph_conv_forward(self):
        from hislr.models.msgcn import SpatialGraphConv
        layer = SpatialGraphConv(in_channels=4, out_channels=64, num_joints=J)
        x = torch.randn(B, T, J, 4)
        out = layer(x)
        assert out.shape == (B, T, J, 64), f"Expected ({B},{T},{J},64), got {out.shape}"

    def test_msgcn_encoder_output_shapes(self, skel_batch):
        from hislr.models.msgcn import MSGCNEncoder
        encoder = MSGCNEncoder(
            in_channels=C,
            hidden_channels=[32, 64, 128],  # small for test
            num_joints=J,
            num_scales=2,
        )
        S1, S2, S3, S1_pool, S2_pool, S3_pool = encoder(skel_batch)

        assert S1.shape[0] == B
        assert S2.shape[0] == B
        assert S3.shape[0] == B
        # Temporal strides: S2 = T/2, S3 = T/4
        assert S2_pool.shape[1] == T // 2, f"S2 temporal dim should be {T//2}"
        assert S3_pool.shape[1] == T // 4, f"S3 temporal dim should be {T//4}"
        # Channel dims
        assert S1_pool.shape[2] == 32
        assert S2_pool.shape[2] == 64
        assert S3_pool.shape[2] == 128

    def test_msgcn_gradient_flow(self, skel_batch):
        from hislr.models.msgcn import MSGCNEncoder
        encoder = MSGCNEncoder(in_channels=C, hidden_channels=[32, 64, 128], num_joints=J, num_scales=2)
        skel_batch.requires_grad_(False)
        S1, S2, S3, S1_p, S2_p, S3_p = encoder(skel_batch)
        loss = S3_p.mean()
        loss.backward()
        # Check that at least one parameter received a gradient
        has_grad = any(p.grad is not None for p in encoder.parameters())
        assert has_grad, "No gradients flowed through MS-GCN"


# ─────────────────────────────────────────────────────────────────────────────
# 2. HiCMF Fusion Backbone
# ─────────────────────────────────────────────────────────────────────────────

class TestHiCMF:

    def _make_swin_feats(self, channels):
        """Create mock Swin-V2 hierarchical feature maps in (B,T,H,W,C) format."""
        return {
            "R1": torch.randn(B, T,    H//4,  W//4,  channels[0]),
            "R2": torch.randn(B, T,    H//8,  W//8,  channels[1]),
            "R3": torch.randn(B, T,    H//16, W//16, channels[2]),
            "R4": torch.randn(B, T,    H//32, W//32, channels[3]),
        }

    def _make_gcn_feats(self, channels):
        """Create mock GCN pooled features."""
        return {
            "S1_pool": torch.randn(B, T,    channels[0]),
            "S2_pool": torch.randn(B, T//2, channels[1]),
            "S3_pool": torch.randn(B, T//4, channels[2]),
        }

    def test_cross_modal_attention_shapes(self):
        from hislr.models.hicmf import CrossModalAttention
        cma = CrossModalAttention(dim_a=64, dim_b=32, embed_dim=64, num_heads=4)
        feat_a = torch.randn(B, 10, 64)
        feat_b = torch.randn(B, 8,  32)
        out_a, out_b = cma(feat_a, feat_b)
        assert out_a.shape == (B, 10, 64), f"RGB output shape wrong: {out_a.shape}"
        assert out_b.shape == (B, 8,  32), f"Skel output shape wrong: {out_b.shape}"

    def test_hicmf_3stage_output(self):
        from hislr.models.hicmf import HiCMF
        swin_ch = [32, 64, 128, 256]
        gcn_ch  = [32, 64, 128]
        hicmf = HiCMF(
            swin_channels=swin_ch,
            gcn_channels=gcn_ch,
            fusion_stages=3,
            fusion_heads=[4, 4, 4],
            embed_dim=64,
            joint_embed_dim=512,
        )
        swin_feats = self._make_swin_feats(swin_ch)
        gcn_feats  = self._make_gcn_feats(gcn_ch)
        z = hicmf(swin_feats, gcn_feats)
        assert z.shape == (B, 512), f"Expected (B, 512), got {z.shape}"

    def test_hicmf_1stage_fallback(self):
        from hislr.models.hicmf import HiCMF
        swin_ch = [32, 64, 128, 256]
        gcn_ch  = [32, 64, 128]
        hicmf = HiCMF(
            swin_channels=swin_ch,
            gcn_channels=gcn_ch,
            fusion_stages=1,
            fusion_heads=[4, 4, 4],
            embed_dim=64,
            joint_embed_dim=512,
        )
        swin_feats = self._make_swin_feats(swin_ch)
        gcn_feats  = self._make_gcn_feats(gcn_ch)
        z = hicmf(swin_feats, gcn_feats)
        assert z.shape == (B, 512)

    def test_hicmf_gradient_flow(self):
        from hislr.models.hicmf import HiCMF
        swin_ch = [32, 64, 128, 256]
        gcn_ch  = [32, 64, 128]
        hicmf = HiCMF(swin_channels=swin_ch, gcn_channels=gcn_ch,
                       fusion_stages=3, fusion_heads=[4, 4, 4],
                       embed_dim=64, joint_embed_dim=512)
        swin_feats = self._make_swin_feats(swin_ch)
        gcn_feats  = self._make_gcn_feats(gcn_ch)
        z = hicmf(swin_feats, gcn_feats)
        z.mean().backward()
        has_grad = any(p.grad is not None for p in hicmf.parameters())
        assert has_grad


# ─────────────────────────────────────────────────────────────────────────────
# 3. Loss Functions
# ─────────────────────────────────────────────────────────────────────────────

class TestLosses:

    def test_label_smooth_ce_scalar(self):
        from hislr.training.losses import LabelSmoothCE
        criterion = LabelSmoothCE(num_classes=N, epsilon=0.1)
        logits  = torch.randn(B, N)
        targets = torch.randint(0, N, (B,))
        loss = criterion(logits, targets)
        assert loss.ndim == 0,      "Loss must be a scalar"
        assert loss.item() > 0,     "Loss must be positive"
        assert not torch.isnan(loss), "Loss must not be NaN"

    def test_label_smooth_ce_perfect_prediction(self):
        from hislr.training.losses import LabelSmoothCE
        criterion = LabelSmoothCE(num_classes=N, epsilon=0.1)
        targets = torch.zeros(B, dtype=torch.long)
        logits  = torch.full((B, N), -100.0)
        logits[:, 0] = 100.0   # perfect prediction
        loss = criterion(logits, targets)
        assert loss.item() < 0.5, "Loss for perfect prediction should be low"

    def test_phonological_loss_with_mask(self, phon_labels):
        from hislr.training.losses import PhonologicalLoss
        phon_num_classes = [87, 87, 9, 32, 4, 17, 6, 4, 6, 2, 2, 2, 2, 2, 2, 2]
        criterion = PhonologicalLoss(phon_num_classes)
        phon_logits = [torch.randn(B, k) for k in phon_num_classes]
        loss = criterion(phon_logits, phon_labels)
        assert not torch.isnan(loss), "PhAP loss must not be NaN"
        assert loss.item() >= 0

    def test_phonological_loss_all_missing(self):
        from hislr.training.losses import PhonologicalLoss
        phon_num_classes = [4] * 16
        criterion = PhonologicalLoss(phon_num_classes)
        phon_logits = [torch.randn(B, 4) for _ in range(16)]
        all_missing = torch.full((B, 16), -1, dtype=torch.long)
        loss = criterion(phon_logits, all_missing)
        assert loss.item() == 0.0, "All-missing labels should produce zero loss"

    def test_tcr_loss_positive_pairs(self):
        from hislr.training.losses import TCRLoss
        import torch.nn.functional as F
        criterion = TCRLoss(temperature=0.07)
        z = torch.randn(B, D)
        z_slow = F.normalize(z + 0.01 * torch.randn_like(z), dim=-1)
        z_fast = F.normalize(z + 0.01 * torch.randn_like(z), dim=-1)
        loss = criterion(z_slow, z_fast)
        assert not torch.isnan(loss)
        assert loss.item() >= 0

    def test_tcr_loss_identical_views_is_low(self):
        """Identical slow/fast views should give near-zero cross-entropy diagonal."""
        from hislr.training.losses import TCRLoss
        import torch.nn.functional as F
        criterion = TCRLoss(temperature=0.07)
        z = F.normalize(torch.randn(B, D), dim=-1)
        loss = criterion(z, z.clone())
        # With B=2 identical pairs, loss should be close to log(1/1) = 0
        assert loss.item() < 0.5, f"Identical views loss too high: {loss.item()}"

    def test_hislr_loss_combined(self, gloss_labels, phon_labels):
        from hislr.training.losses import HiSLRLoss
        phon_num_classes = [87, 87, 9, 32, 4, 17, 6, 4, 6, 2, 2, 2, 2, 2, 2, 2]
        import torch.nn.functional as F

        criterion = HiSLRLoss(
            num_classes=N,
            label_smoothing=0.1,
            phon_num_classes=phon_num_classes,
            lambda_phon=0.3,
            lambda_tcr=0.1,
        )

        z = F.normalize(torch.randn(B, D), dim=-1)
        model_out = {
            "logits":         torch.randn(B, N),
            "phon_logits":    [torch.randn(B, k) for k in phon_num_classes],
            "tcr_proj_slow":  z,
            "tcr_proj_fast":  F.normalize(torch.randn(B, D), dim=-1),
        }

        # Stage 2: all losses active
        losses = criterion(model_out, gloss_labels, phon_labels, stage=2)
        for k in ["total", "ce", "phon", "tcr"]:
            assert k in losses
            assert not torch.isnan(losses[k]), f"Loss '{k}' is NaN"

        # Stage 1: TCR should be zero
        losses_s1 = criterion(model_out, gloss_labels, phon_labels, stage=1)
        assert losses_s1["tcr"].item() == 0.0, "Stage 1 TCR should be 0"

    def test_total_loss_backward(self, gloss_labels, phon_labels):
        from hislr.training.losses import HiSLRLoss
        import torch.nn.functional as F
        phon_num_classes = [4] * 16
        criterion = HiSLRLoss(num_classes=N, phon_num_classes=phon_num_classes)

        logits = torch.randn(B, N, requires_grad=True)
        z      = F.normalize(torch.randn(B, D, requires_grad=True), dim=-1)
        model_out = {
            "logits":         logits,
            "phon_logits":    [torch.randn(B, 4, requires_grad=True) for _ in range(16)],
            "tcr_proj_slow":  z,
            "tcr_proj_fast":  F.normalize(torch.randn(B, D), dim=-1),
        }
        losses = criterion(model_out, gloss_labels, phon_labels, stage=2)
        losses["total"].backward()
        assert logits.grad is not None, "Gradients must flow to logits"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Full HiSLR Model (using stub Swin encoder — no timm required)
# ─────────────────────────────────────────────────────────────────────────────

class TestHiSLRModel:

    @pytest.fixture
    def small_model(self):
        """HiSLR with reduced channels for fast testing."""
        from hislr.models.hislr import HiSLR
        return HiSLR(
            num_classes=N,
            swin_variant="swin_v2_t",       # smallest variant
            swin_pretrained=False,
            num_joints=J,
            gcn_hidden_channels=[32, 64, 128],
            gcn_num_scales=2,
            gcn_dropout=0.0,
            fusion_stages=3,
            fusion_heads=[4, 4, 4],
            fusion_embed_dim=64,
            joint_embed_dim=256,
            cls_hidden_dim=128,
            cls_dropout=0.0,
            use_phap=True,
            phon_attributes=[
                "dominant_handshape", "nondominant_handshape", "major_location",
                "minor_location", "contact", "dominant_movement", "path_movement",
                "wrist_twist", "palm_orientation", "nmm_brows", "nmm_eyes",
                "nmm_cheeks", "nmm_mouth", "nmm_tongue", "nmm_head", "nmm_shoulders",
            ],
            phon_num_classes=[87, 87, 9, 32, 4, 17, 6, 4, 6, 2, 2, 2, 2, 2, 2, 2],
            use_tcr=True,
            tcr_proj_dim=64,
        )

    def test_model_builds(self, small_model):
        assert small_model is not None

    def test_model_param_count(self, small_model):
        total = sum(p.numel() for p in small_model.parameters())
        assert total > 0
        print(f"\n  Small HiSLR: {total/1e6:.2f}M parameters")

    def test_encode_shape(self, small_model, rgb_batch, skel_batch):
        small_model.eval()
        with torch.no_grad():
            z = small_model.encode(rgb_batch, skel_batch)
        assert z.shape == (B, 256), f"Expected (B, 256), got {z.shape}"

    def test_forward_inference_shapes(self, small_model, rgb_batch, skel_batch):
        small_model.eval()
        with torch.no_grad():
            out = small_model(rgb_batch, skel_batch)
        assert "logits" in out
        assert out["logits"].shape == (B, N), f"Logits shape wrong: {out['logits'].shape}"

    def test_forward_training_shapes(self, small_model, rgb_batch, skel_batch):
        import torch.nn.functional as F
        small_model.train()
        z_slow = F.normalize(torch.randn(B, 64), dim=-1)
        z_fast = F.normalize(torch.randn(B, 64), dim=-1)
        out = small_model(
            rgb_batch, skel_batch,
            rgb_slow=rgb_batch, skeleton_slow=skel_batch,
            rgb_fast=rgb_batch, skeleton_fast=skel_batch,
        )
        assert out["logits"].shape == (B, N)
        assert out["phon_logits"] is not None
        assert len(out["phon_logits"]) == 16
        assert out["tcr_proj_slow"] is not None
        assert out["tcr_proj_fast"] is not None
        assert out["tcr_proj_slow"].shape == (B, 64)

    def test_forward_backward(self, small_model, rgb_batch, skel_batch, gloss_labels, phon_labels):
        from hislr.training.losses import HiSLRLoss
        import torch.nn.functional as F

        phon_num_classes = [87, 87, 9, 32, 4, 17, 6, 4, 6, 2, 2, 2, 2, 2, 2, 2]
        criterion = HiSLRLoss(
            num_classes=N,
            phon_num_classes=phon_num_classes,
            lambda_phon=0.3,
            lambda_tcr=0.1,
        )

        small_model.train()
        out = small_model(
            rgb_batch, skel_batch,
            rgb_slow=rgb_batch, skeleton_slow=skel_batch,
            rgb_fast=rgb_batch, skeleton_fast=skel_batch,
        )
        losses = criterion(out, gloss_labels, phon_labels, stage=2)
        losses["total"].backward()

        # Verify gradients exist in at least some parameters
        grad_count = sum(1 for p in small_model.parameters() if p.grad is not None)
        assert grad_count > 0, "No gradients computed in full backward pass"

    def test_param_groups(self, small_model):
        groups = small_model.get_param_groups(lr_pretrained=1e-5, lr_new=1e-4)
        assert len(groups) == 2
        assert groups[0]["lr"] == 1e-5
        assert groups[1]["lr"] == 1e-4
        # Verify no parameter appears in both groups
        ids_0 = set(id(p) for p in groups[0]["params"])
        ids_1 = set(id(p) for p in groups[1]["params"])
        assert len(ids_0 & ids_1) == 0, "Parameter appears in both LR groups"

    def test_no_phap_model(self, rgb_batch, skel_batch):
        from hislr.models.hislr import HiSLR
        model = HiSLR(
            num_classes=N,
            swin_variant="swin_v2_t",
            swin_pretrained=False,
            gcn_hidden_channels=[32, 64, 128],
            use_phap=False,
            use_tcr=False,
            joint_embed_dim=256,
            cls_hidden_dim=128,
        )
        model.eval()
        with torch.no_grad():
            out = model(rgb_batch, skel_batch)
        assert out["phon_logits"] is None
        assert out["tcr_proj_slow"] is None
        assert out["logits"].shape == (B, N)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Data utilities
# ─────────────────────────────────────────────────────────────────────────────

class TestDataUtils:

    def test_normalize_skeleton(self):
        from hislr.data.dataset import normalize_skeleton
        joints = np.random.randn(T, J, 4).astype(np.float32)
        joints[:, :, 3] = 1.0   # visibility = 1
        norm = normalize_skeleton(joints)
        # Anchor (shoulder midpoint) should be near zero after normalization
        LEFT_SHOULDER  = 11
        RIGHT_SHOULDER = 12
        anchor = (norm[:, LEFT_SHOULDER, :3] + norm[:, RIGHT_SHOULDER, :3]) / 2
        assert np.allclose(anchor, 0.0, atol=0.05), "Anchor should be near zero after normalization"

    def test_normalize_skeleton_scale(self):
        from hislr.data.dataset import normalize_skeleton
        joints = np.zeros((T, J, 4), dtype=np.float32)
        # Set shoulder width to 2 (left at x=-1, right at x=1)
        joints[:, 11, 0] = -1.0
        joints[:, 12, 0] =  1.0
        norm = normalize_skeleton(joints)
        # After normalization, shoulder width should be ~1
        width = np.abs(norm[:, 11, 0] - norm[:, 12, 0])
        assert np.allclose(width, 1.0, atol=0.01)

    def test_temporal_subsample_slow(self):
        from hislr.data.dataset import temporal_subsample
        frames = np.random.randint(0, 255, (T, 64, 64, 3), dtype=np.uint8)
        joints = np.random.randn(T, J, 4).astype(np.float32)
        f_slow, j_slow = temporal_subsample(frames, joints, factor=0.5)
        assert f_slow.shape == frames.shape, "Slow view must have same T as input"
        assert j_slow.shape == joints.shape

    def test_temporal_subsample_fast(self):
        from hislr.data.dataset import temporal_subsample
        frames = np.random.randint(0, 255, (T, 64, 64, 3), dtype=np.uint8)
        joints = np.random.randn(T, J, 4).astype(np.float32)
        f_fast, j_fast = temporal_subsample(frames, joints, factor=2.0)
        assert f_fast.shape == frames.shape
        assert j_fast.shape == joints.shape

    def test_augment_skeleton_flip(self):
        from hislr.data.dataset import augment_skeleton
        joints = np.zeros((T, J, 4), dtype=np.float32)
        joints[:, :, 0] = 0.5   # positive x
        augmented = augment_skeleton(joints, noise_sigma=0, dropout_prob=0, flip=True)
        # X should be flipped to negative
        assert np.all(augmented[:, 0, 0] <= 0), "Flip should negate X coordinates"

    def test_augment_skeleton_no_nan(self):
        from hislr.data.dataset import augment_skeleton
        joints = np.random.randn(T, J, 4).astype(np.float32)
        augmented = augment_skeleton(
            joints, noise_sigma=0.005, dropout_prob=0.1,
            scale_range=(0.9, 1.1), rotate_range=15.0, flip=False,
        )
        assert not np.any(np.isnan(augmented)), "Augmented skeleton must not contain NaN"


# ─────────────────────────────────────────────────────────────────────────────
# 6. Metrics
# ─────────────────────────────────────────────────────────────────────────────

class TestMetrics:

    def test_compute_accuracy_top1(self):
        from hislr.utils.metrics import compute_accuracy
        # Perfect predictions
        logits = torch.eye(B) * 100  # (B, B)
        targets = torch.arange(B)
        top1, top5 = compute_accuracy(logits, targets, topk=(1, min(5, B)))
        assert abs(top1 - 100.0) < 1e-3, "Perfect predictions should yield 100% Top-1"

    def test_compute_accuracy_random(self):
        from hislr.utils.metrics import compute_accuracy
        logits = torch.randn(100, N)
        targets = torch.randint(0, N, (100,))
        top1, top5 = compute_accuracy(logits, targets, topk=(1, 5))
        assert 0 <= top1 <= 100
        assert top1 <= top5, "Top-5 accuracy must be >= Top-1"

    def test_average_meter(self):
        from hislr.utils.metrics import AverageMeter
        meter = AverageMeter()
        meter.update(10.0, 2)
        meter.update(20.0, 2)
        assert abs(meter.avg - 15.0) < 1e-6, f"Expected avg 15.0, got {meter.avg}"
        meter.reset()
        assert meter.avg == 0.0 and meter.count == 0

    def test_confusion_matrix(self):
        from hislr.utils.metrics import compute_confusion_matrix
        logits = torch.eye(5) * 100
        targets = torch.arange(5)
        cm = compute_confusion_matrix(logits, targets, num_classes=5)
        assert cm.shape == (5, 5)
        assert cm.trace().item() == 5, "Perfect predictions: all on diagonal"


# ─────────────────────────────────────────────────────────────────────────────
# 7. Integration test: one training step
# ─────────────────────────────────────────────────────────────────────────────

class TestIntegration:

    def test_one_training_step(self, rgb_batch, skel_batch, gloss_labels, phon_labels):
        """Full forward + loss + backward + optimizer step."""
        import torch.optim as optim
        from hislr.models.hislr import HiSLR
        from hislr.training.losses import HiSLRLoss
        import torch.nn.functional as F

        phon_num_classes = [87, 87, 9, 32, 4, 17, 6, 4, 6, 2, 2, 2, 2, 2, 2, 2]

        model = HiSLR(
            num_classes=N,
            swin_variant="swin_v2_t",
            swin_pretrained=False,
            gcn_hidden_channels=[32, 64, 128],
            gcn_num_scales=2,
            use_phap=True,
            phon_attributes=["dominant_handshape"] * 16,   # simplified
            phon_num_classes=phon_num_classes,
            use_tcr=True,
            joint_embed_dim=256,
            cls_hidden_dim=128,
            tcr_proj_dim=64,
        )
        criterion = HiSLRLoss(
            num_classes=N,
            phon_num_classes=phon_num_classes,
        )
        optimizer = optim.AdamW(model.parameters(), lr=1e-4)

        model.train()
        optimizer.zero_grad()

        out = model(
            rgb_batch, skel_batch,
            rgb_slow=rgb_batch, skeleton_slow=skel_batch,
            rgb_fast=rgb_batch, skeleton_fast=skel_batch,
        )
        losses = criterion(out, gloss_labels, phon_labels, stage=2)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # If we got here without error, the full training step works
        assert losses["total"].item() > 0


# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
