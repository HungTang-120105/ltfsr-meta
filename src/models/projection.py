"""Projection head for Supervised Contrastive (SupCon) pre-training.

SupCon does not classify directly; it learns a representation by pulling
same-class images together in a small normalised embedding space. A 2-layer MLP
projects the 512-d encoder features down to this space (typically 128-d). After
contrastive pre-training the projection head is discarded and a classifier is
trained on the frozen encoder (see docs/03_contrastive.md).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from src.models.backbone import FEATURE_DIM, build_encoder


class ProjectionHead(nn.Module):
    """2-layer MLP producing L2-normalised embeddings for the SupCon loss."""

    def __init__(self, in_dim: int = FEATURE_DIM, hidden_dim: int = 512, out_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(features), dim=1)


class ContrastiveModel(nn.Module):
    """Encoder + projection head used only during contrastive pre-training."""

    def __init__(self, pretrained: bool = True, embedding_dim: int = 128) -> None:
        super().__init__()
        self.encoder = build_encoder(pretrained=pretrained)
        self.projection = ProjectionHead(out_dim=embedding_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.projection(self.encoder(images))
