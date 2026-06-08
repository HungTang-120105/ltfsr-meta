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


def build_encoder(pretrained: bool = False) -> nn.Module:
    """Return a ResNet-18 with its classification head removed.

    Maps an image batch ``(B, 3, H, W)`` to feature vectors ``(B, 512)``.

    Two mutually exclusive setups (chosen by ``pretrained``):

    * ``pretrained=False`` (default, the **main** setup): swap in the standard
      **CIFAR stem** — a 3x3 stride-1 conv and *no* max-pool — and train from
      scratch on 32x32 images. The original ImageNet stem (7x7 stride-2 + maxpool)
      would shrink a 32x32 image to 4x4 before any real features form, which is
      why it must be replaced. This is the convention used by every CIFAR-LT
      paper, so the numbers are comparable.
    * ``pretrained=True`` (optional "pretrained" reference table): keep the
      ImageNet stem **and** weights. This only makes sense if the inputs are
      resized to 224 (see ``build_transforms(image_size=224)``); the pretrained
      7x7 stem is meaningless at 32x32.
    """
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    try:
        encoder = resnet18(weights=weights)
    except Exception:
        encoder = resnet18(weights=None)

    if not pretrained:
        encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        encoder.maxpool = nn.Identity()

    encoder.fc = nn.Identity()  # expose the 512-d features directly
    return encoder
