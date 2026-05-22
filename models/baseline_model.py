from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


@dataclass(frozen=True)
class BaselineModelConfig:
    num_classes: int = 100
    pretrained: bool = True


class BaselineClassifier(nn.Module):
    def __init__(self, num_classes: int = 100, pretrained: bool = True) -> None:
        super().__init__()
        backbone = None
        if pretrained:
            try:
                backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
            except Exception:
                backbone = None

        if backbone is None:
            backbone = resnet18(weights=None)

        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()

        self.backbone = backbone
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)