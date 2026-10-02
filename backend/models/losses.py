"""
Loss functions for deep metric learning in person re-identification:
1. Margin-based losses: ArcFace loss and Circle loss
2. Batch-Hard Triplet loss with hard positive and hard negative mining.
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceLoss(nn.Module):
    """
    ArcFace: Additive Angular Margin Loss.
    cos(theta + m) for the target class.
    Forces tighter angular intra-class clustering and larger inter-class separation.
    """
    def __init__(self, scale: float = 30.0, margin: float = 0.35):
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, cosine: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        # cosine: (B, num_classes)
        sine = torch.sqrt(1.0 - torch.clamp(cosine ** 2, 0.0, 1.0))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1.0)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale
        return F.cross_entropy(output, label)


class CircleLoss(nn.Module):
    """
    Circle Loss (Sun et al., CVPR 2020):
    A unified perspective of pair similarity optimization.
    Dynamically weights positive and negative similarity pairs based on their optimization status.
    """
    def __init__(self, m: float = 0.25, gamma: float = 64.0):
        super().__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()

    def forward(self, sp: torch.Tensor, sn: torch.Tensor) -> torch.Tensor:
        """
        sp: similarities of positive pairs
        sn: similarities of negative pairs
        """
        ap = torch.clamp_min(-sp.detach() + 1 + self.m, min=0.0)
        an = torch.clamp_min(sn.detach() + self.m, min=0.0)

        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = -ap * (sp - delta_p) * self.gamma
        logit_n = an * (sn - delta_n) * self.gamma

        loss = self.soft_plus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))
        return loss


class BatchHardTripletLoss(nn.Module):
    """
    Batch-Hard Triplet Loss with Euclidean or Cosine distance.
    For each anchor sample in the batch, mines:
    - The hardest positive: sample of same identity with largest distance
    - The hardest negative: sample of different identity with smallest distance
    """
    def __init__(self, margin: float = 0.3, distance: str = "cosine"):
        super().__init__()
        self.margin = margin
        self.distance = distance

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        embeddings: (B, D) normalized features
        labels: (B,) identity labels
        """
        B = embeddings.size(0)

        if self.distance == "cosine":
            # Similarity matrix: (B, B)
            sim_mat = torch.matmul(embeddings, embeddings.t())
            dist_mat = 1.0 - sim_mat
        else:
            # Squared Euclidean distance: ||u - v||^2
            dist_mat = torch.cdist(embeddings, embeddings, p=2)

        # Mask for positives and negatives
        labels = labels.view(-1, 1)
        is_pos = torch.eq(labels, labels.t())
        is_neg = torch.ne(labels, labels.t())

        # For each anchor, hardest positive has maximum distance
        pos_dist = dist_mat * is_pos.float()
        hardest_pos, _ = torch.max(pos_dist, dim=1)

        # For each anchor, hardest negative has minimum distance
        # Add a large constant to non-negative pairs before finding min
        max_dist = dist_mat.max().detach()
        neg_dist = dist_mat + max_dist * (~is_neg).float()
        hardest_neg, _ = torch.min(neg_dist, dim=1)

        # Triplet margin loss: max(0, d(a,p) - d(a,n) + margin)
        losses = F.relu(hardest_pos - hardest_neg + self.margin)
        return losses.mean()
