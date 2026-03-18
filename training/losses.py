# hislr/training/losses.py
# ─────────────────────────────────────────────────────────────────────────────
# HiSLR Loss Functions
#
#   1. LabelSmoothCE      — cross-entropy with label smoothing for gloss head
#   2. PhonologicalLoss   — masked multi-task loss for 16 PhAP attribute heads
#   3. TCRLoss            — InfoNCE temporal contrastive regularization
#   4. HiSLRLoss          — combined loss orchestrator
# ─────────────────────────────────────────────────────────────────────────────

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class LabelSmoothCE(nn.Module):
    """
    Cross-entropy loss with label smoothing.
    Smoothing factor epsilon distributes (epsilon) probability mass
    uniformly across all classes, reducing overconfidence on tail classes.
    """

    def __init__(self, num_classes: int, epsilon: float = 0.1, reduction: str = "mean"):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B, N)  — raw class logits
            targets: (B,)    — integer class labels in [0, N-1]
        Returns:
            scalar loss
        """
        B, N = logits.shape
        log_probs = F.log_softmax(logits, dim=-1)  # (B, N)

        # One-hot smooth targets
        with torch.no_grad():
            smooth_targets = torch.full_like(log_probs, self.epsilon / (N - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.epsilon)

        loss = -(smooth_targets * log_probs).sum(dim=-1)  # (B,)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class PhonologicalLoss(nn.Module):
    """
    Multi-task phonological attribute loss for PhAP pre-training.

    For each of 16 attributes, computes cross-entropy between the
    predicted logits and ground-truth attribute label.

    Signs without annotation (label == -1) are masked out — the head
    receives no gradient for unannotated examples.

    The total PhAP loss is the mean across annotated (attribute, sample) pairs.
    """

    def __init__(self, phon_num_classes: List[int]):
        super().__init__()
        self.phon_num_classes = phon_num_classes
        self.num_attributes = len(phon_num_classes)

    def forward(
        self,
        phon_logits: List[torch.Tensor],
        phon_targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            phon_logits:  list of (B, K_i) — one per attribute head
            phon_targets: (B, 16)           — integer labels, -1 = missing

        Returns:
            scalar loss (mean over annotated pairs)
        """
        assert len(phon_logits) == self.num_attributes

        total_loss = torch.tensor(0.0, device=phon_logits[0].device, requires_grad=True)
        num_valid = 0

        for i, (logits_i, n_cls) in enumerate(zip(phon_logits, self.phon_num_classes)):
            targets_i = phon_targets[:, i]          # (B,)
            mask = targets_i >= 0                    # annotated samples

            if mask.sum() == 0:
                continue

            logits_masked  = logits_i[mask]          # (M, K_i)
            targets_masked = targets_i[mask]         # (M,)

            loss_i = F.cross_entropy(logits_masked, targets_masked)
            total_loss = total_loss + loss_i
            num_valid += 1

        if num_valid > 0:
            return total_loss / num_valid
        return total_loss


class TCRLoss(nn.Module):
    """
    Temporal Contrastive Regularization (TCR) loss.

    InfoNCE loss between L2-normalized projected embeddings of slow and
    fast temporal augmentations of the same sign clip.

    Positives: (z_slow_i, z_fast_i) — same clip, different speeds.
    Negatives: all other clips in the mini-batch.

    L_TCR = -log [ exp(sim(z_slow, z_fast) / tau)
                   / sum_k exp(sim(z_slow, z_k) / tau) ]
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        z_slow: torch.Tensor,
        z_fast: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            z_slow: (B, D) — L2-normalized projection of slow views
            z_fast: (B, D) — L2-normalized projection of fast views
        Returns:
            scalar InfoNCE loss (symmetric: slow->fast + fast->slow, averaged)
        """
        B = z_slow.shape[0]

        # Similarity matrix (B, B)
        sim_matrix = torch.einsum("id,jd->ij", z_slow, z_fast) / self.temperature

        # Labels: diagonal is positive pair
        labels = torch.arange(B, device=z_slow.device)

        # Symmetric loss
        loss_sf = F.cross_entropy(sim_matrix, labels)         # slow -> fast
        loss_fs = F.cross_entropy(sim_matrix.t(), labels)     # fast -> slow

        return (loss_sf + loss_fs) / 2


class HiSLRLoss(nn.Module):
    """
    Combined HiSLR training loss orchestrator.

    L_total = L_CE + lambda_phon * L_phon + lambda_tcr * L_TCR

    During Stage 1 (PhAP pre-training), lambda_tcr is forced to 0.
    During Stage 2 (full training), all terms are active.
    """

    def __init__(
        self,
        num_classes: int = 2000,
        label_smoothing: float = 0.1,
        phon_num_classes: List[int] = None,
        lambda_phon: float = 0.3,
        lambda_tcr: float = 0.1,
        tcr_temperature: float = 0.07,
    ):
        super().__init__()

        self.lambda_phon = lambda_phon
        self.lambda_tcr  = lambda_tcr

        self.ce_loss = LabelSmoothCE(num_classes, epsilon=label_smoothing)
        self.phon_loss = PhonologicalLoss(phon_num_classes) if phon_num_classes else None
        self.tcr_loss  = TCRLoss(temperature=tcr_temperature)

    def forward(
        self,
        model_out: dict,
        gloss_targets: torch.Tensor,
        phon_targets: Optional[torch.Tensor] = None,
        stage: int = 2,
    ) -> dict:
        """
        Args:
            model_out:     output dict from HiSLR.forward()
            gloss_targets: (B,) integer gloss labels
            phon_targets:  (B, 16) phonological labels, -1 for missing
            stage:         1 = no TCR,  2 = full

        Returns:
            dict with:
                'total'     : scalar total loss
                'ce'        : gloss classification loss
                'phon'      : phonological auxiliary loss (0 if disabled)
                'tcr'       : temporal contrastive loss (0 if stage 1)
        """
        losses = {}

        # ── 1. Gloss CE Loss ──────────────────────────────────────────────────
        l_ce = self.ce_loss(model_out["logits"], gloss_targets)
        losses["ce"] = l_ce

        # ── 2. Phonological Loss ──────────────────────────────────────────────
        l_phon = torch.tensor(0.0, device=l_ce.device)
        if (
            self.phon_loss is not None
            and model_out.get("phon_logits") is not None
            and phon_targets is not None
        ):
            l_phon = self.phon_loss(model_out["phon_logits"], phon_targets)
        losses["phon"] = l_phon

        # ── 3. TCR Loss (Stage 2 only) ─────────────────────────────────────────
        l_tcr = torch.tensor(0.0, device=l_ce.device)
        if (
            stage == 2
            and model_out.get("tcr_proj_slow") is not None
            and model_out.get("tcr_proj_fast") is not None
        ):
            l_tcr = self.tcr_loss(model_out["tcr_proj_slow"], model_out["tcr_proj_fast"])
        losses["tcr"] = l_tcr

        # ── Total ─────────────────────────────────────────────────────────────
        total = l_ce + self.lambda_phon * l_phon + self.lambda_tcr * l_tcr
        losses["total"] = total

        return losses
