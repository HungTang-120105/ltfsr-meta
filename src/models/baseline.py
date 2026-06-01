"""Baseline classifier: ResNet-18 encoder + a linear softmax head.

This is the standard supervised recipe and the reference point every other
method is compared against. On long-tail data its weakness is well known: the
linear head's weight norms grow with class frequency, so head classes dominate
and tail classes are under-predicted (see docs/01_baseline.md).
"""

from __future__ import annotations

import torch
from torch import nn

from src.models.backbone import FEATURE_DIM, build_encoder


class BaselineClassifier(nn.Module):
    """ResNet-18 features followed by a single linear classification layer."""

    def __init__(self, num_classes: int = 100, pretrained: bool = True) -> None:
        super().__init__()
        self.encoder = build_encoder(pretrained=pretrained)
        self.classifier = nn.Linear(FEATURE_DIM, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.encoder(images)
        return self.classifier(features)

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Return the 512-d encoder features (used for t-SNE visualisation)."""
        return self.encoder(images)
