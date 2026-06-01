"""Prototype-based classifier (distance head instead of a linear head).

Instead of a dot-product linear layer, each class is represented by a learnable
prototype vector, and an image is scored by its (negative squared) Euclidean
distance to every prototype. Distances are unaffected by per-class weight-norm
growth, which is exactly the bias that hurts the linear baseline on long-tail
data (see docs/02_prototype.md).

The same ``pairwise_sq_distance`` / ``compute_prototypes`` helpers are reused by
the episodic meta-learning trainer, so the distance logic lives in one place.
"""

from __future__ import annotations

import torch
from torch import nn

from src.models.backbone import FEATURE_DIM, build_encoder


def pairwise_sq_distance(features: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    """Squared Euclidean distance between each feature and each prototype.

    Args:
        features: Tensor of shape ``(N, D)``.
        prototypes: Tensor of shape ``(C, D)``.

    Returns:
        Tensor of shape ``(N, C)`` of squared distances.
    """
    return torch.cdist(features, prototypes, p=2) ** 2


def compute_prototypes(features: torch.Tensor, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Mean feature vector per class (the classic prototype definition).

    Used by the episodic trainer to build prototypes from a support set.
    """
    dim = features.size(1)
    prototypes = torch.zeros(num_classes, dim, device=features.device)
    for class_id in range(num_classes):
        mask = labels == class_id
        if mask.any():
            prototypes[class_id] = features[mask].mean(dim=0)
    return prototypes


class PrototypeClassifier(nn.Module):
    """ResNet-18 encoder + one learnable prototype per class.

    Logits are the negative squared distance to each prototype, so a standard
    cross-entropy loss pulls a sample's features towards its class prototype.
    """

    def __init__(self, num_classes: int = 100, pretrained: bool = True) -> None:
        super().__init__()
        self.encoder = build_encoder(pretrained=pretrained)
        self.prototypes = nn.Parameter(torch.randn(num_classes, FEATURE_DIM))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.encoder(images)
        return -pairwise_sq_distance(features, self.prototypes)

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        return self.encoder(images)
