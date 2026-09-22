"""Supervised Contrastive Loss (SupCon, Khosla et al. NeurIPS 2020,
arXiv:2004.11362), single-view form: one feature vector per sample per
batch (no multi-crop/multi-augmentation views), used alongside
class-weighted cross-entropy in train_cross_attention_supcon.py.

For anchor i with L2-normalized feature z_i and label y_i, over the
in-batch positives P(i) = {p != i : y_p == y_i}:

    L_i = -1/|P(i)| * sum_{p in P(i)} log( exp(z_i . z_p / T) / sum_{a != i} exp(z_i . z_a / T) )

Anchors with no in-batch positive (|P(i)| == 0 - e.g. Melanoma, PAD-UFES-20's
smallest train class at 38 images, can land alone in a batch of 32) are
excluded from the mean rather than forced to contribute 0 or divide by
zero, per the standard reference implementation's convention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = features.device
        batch_size = features.shape[0]

        features = F.normalize(features, dim=1)
        similarity = torch.matmul(features, features.T) / self.temperature  # [B, B]

        # Numerical stability: subtract per-row max before exponentiating.
        similarity = similarity - similarity.max(dim=1, keepdim=True).values.detach()

        self_mask = torch.eye(batch_size, dtype=torch.bool, device=device)
        positive_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~self_mask  # [B, B]

        exp_sim = torch.exp(similarity).masked_fill(self_mask, 0.0)
        log_prob = similarity - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

        num_positives = positive_mask.sum(dim=1)  # [B]
        has_positive = num_positives > 0
        if not has_positive.any():
            return features.new_zeros(())

        mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1)[has_positive] / num_positives[has_positive]
        return -mean_log_prob_pos.mean()
