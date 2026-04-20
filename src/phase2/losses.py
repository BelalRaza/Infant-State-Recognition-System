"""LDAM loss with Deferred Re-Weighting (DRW) for class imbalance.

Cao et al. "Learning Imbalanced Datasets with Label-Distribution-Aware
Margin Loss", NeurIPS 2019.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LDAMLoss(nn.Module):
    """Label-Distribution-Aware Margin loss.

    Enforces larger margins for minority classes so the decision boundary
    is pushed further from under-represented regions.
    """

    def __init__(self, cls_num_list, max_m=0.5, s=30.0, label_smoothing=0.0):
        super().__init__()
        cls_num = np.array(cls_num_list, dtype=np.float32)
        margins = 1.0 / np.power(cls_num, 0.25)
        margins = margins * (max_m / np.max(margins))
        self.register_buffer("margins", torch.tensor(margins, dtype=torch.float32))
        self.s = s
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets, reduction='mean'):
        index = torch.zeros_like(logits, dtype=torch.bool)
        index.scatter_(1, targets.unsqueeze(1), True)

        margin_logits = logits.clone()
        margin_logits[index] -= self.margins[targets]
        margin_logits *= self.s

        if self.label_smoothing > 0:
            n_classes = logits.size(1)
            smooth = self.label_smoothing / n_classes
            targets_smooth = torch.full_like(logits, smooth)
            targets_smooth.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing + smooth)
            log_probs = F.log_softmax(margin_logits, dim=1)
            per_sample = -(targets_smooth * log_probs).sum(dim=1)
            if reduction == 'none':
                return per_sample
            return per_sample.mean()

        return F.cross_entropy(margin_logits, targets, reduction=reduction)


def compute_drw_weights(cls_num_list, beta=0.9999):
    """Compute class-balanced re-weighting factors (Cui et al. 2019)."""
    effective_num = 1.0 - np.power(beta, np.array(cls_num_list, dtype=np.float64))
    weights = (1.0 - beta) / effective_num
    weights = weights / np.sum(weights) * len(cls_num_list)
    return torch.tensor(weights, dtype=torch.float32)


class DRWScheduler:
    """Deferred Re-Weighting: normal training for first N epochs,
    then apply class-balanced weights."""

    def __init__(self, cls_num_list, switch_epoch=60):
        self.switch_epoch = switch_epoch
        self.uniform_weights = torch.ones(len(cls_num_list))
        self.cb_weights = compute_drw_weights(cls_num_list)

    def get_weights(self, epoch):
        if epoch < self.switch_epoch:
            return self.uniform_weights
        return self.cb_weights
