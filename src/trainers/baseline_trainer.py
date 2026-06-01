"""Method 1 - Baseline: ResNet-18 + linear softmax head + cross-entropy.

This is the reference method. See docs/01_baseline.md for the intuition and the
expected long-tail failure mode (tail classes under-predicted).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.models.baseline import BaselineClassifier
from src.trainers.classifier import fit_classifier


def train_baseline(
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int,
    device: torch.device,
    run_dir: Path,
    epochs: int = 100,
    learning_rate: float = 0.1,
    pretrained: bool = True,
) -> tuple[BaselineClassifier, pd.DataFrame]:
    """Train the baseline classifier and return (best model, history)."""
    model = BaselineClassifier(num_classes=num_classes, pretrained=pretrained).to(device)
    return fit_classifier(
        model, train_loader, val_loader, device, run_dir,
        epochs=epochs, learning_rate=learning_rate,
    )
