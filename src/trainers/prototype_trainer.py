"""Method 2 - Prototype: ResNet-18 + learnable per-class prototypes.

Same training loop as the baseline, but the linear head is replaced by a
distance-based prototype head. Because distances ignore weight-norm growth, this
typically lifts tail-class accuracy. See docs/02_prototype.md.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.models.prototype import PrototypeClassifier
from src.trainers.classifier import fit_classifier


def train_prototype(
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int,
    device: torch.device,
    run_dir: Path,
    epochs: int = 100,
    learning_rate: float = 0.1,
    pretrained: bool = True,
) -> tuple[PrototypeClassifier, pd.DataFrame]:
    """Train the prototype classifier and return (best model, history)."""
    model = PrototypeClassifier(num_classes=num_classes, pretrained=pretrained).to(device)
    return fit_classifier(
        model, train_loader, val_loader, device, run_dir,
        epochs=epochs, learning_rate=learning_rate,
    )
