# hislr/utils/metrics.py
# ─────────────────────────────────────────────────────────────────────────────
# Evaluation utilities for HiSLR.
# ─────────────────────────────────────────────────────────────────────────────

import torch
from typing import Tuple


class AverageMeter:
    """Computes and stores a running average."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val   = 0.0
        self.avg   = 0.0
        self.sum   = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / max(self.count, 1)


def compute_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    topk: Tuple[int, ...] = (1, 5),
) -> list:
    """
    Compute Top-K accuracy.

    Args:
        logits:  (B, N) raw logits
        targets: (B,)   ground-truth integer labels
        topk:    tuple of K values

    Returns:
        list of float accuracy values (%) in the same order as topk
    """
    with torch.no_grad():
        maxk = max(topk)
        B = targets.shape[0]

        _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)  # (B, maxk)
        pred = pred.t()                                                   # (maxk, B)
        correct = pred.eq(targets.view(1, -1).expand_as(pred))           # (maxk, B)

        results = []
        for k in topk:
            correct_k = correct[:k].any(dim=0).float().sum()
            results.append((correct_k / B * 100).item())

    return results


def compute_confusion_matrix(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """
    Compute confusion matrix.

    Returns:
        cm: (num_classes, num_classes) tensor where cm[i, j] = number of
            samples with true label i predicted as j.
    """
    preds = logits.argmax(dim=1)
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for t, p in zip(targets, preds):
        cm[t.item(), p.item()] += 1
    return cm
