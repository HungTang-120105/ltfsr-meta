"""Shared feature extractor used by every method in the project.

All four methods (baseline, prototype, contrastive, meta-learning) embed images
with the same ResNet-18 encoder, so the only thing that changes between methods
is what sits on top of these features. Keeping one encoder makes the comparison
fair and the code small.
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

FEATURE_DIM = 512  # ResNet-18 output dimension after global average pooling


def build_encoder(pretrained: bool = True) -> nn.Module:
    """Return a ResNet-18 with its classification head removed.

    The returned module maps an image batch ``(B, 3, H, W)`` to feature vectors
    ``(B, 512)``. ImageNet-pretrained weights are used when available; if the
    download fails (e.g. offline Kaggle) we silently fall back to random init.
    """
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    try:
        encoder = resnet18(weights=weights)
    except Exception:
        encoder = resnet18(weights=None)

    encoder.fc = nn.Identity()  # expose the 512-d features directly
    return encoder
