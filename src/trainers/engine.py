"""Shared train/evaluate loops for standard (non-episodic) classification.

The baseline, prototype and the linear-probe stage of the contrastive method are
all ordinary classifiers: ``model(images) -> logits`` trained with cross-entropy.
They therefore share these two functions, which keeps each trainer file tiny and
guarantees they are evaluated identically.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Run one training epoch; return sample-weighted loss and accuracy."""
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_samples += images.size(0)

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module | None = None,
    collect_features: bool = False,
) -> dict:
    """Evaluate a classifier over a full loader.

    Returns a dict with ``y_true``, ``y_pred``, ``loss``, ``accuracy`` and,
    when ``collect_features`` is set, the encoder ``features`` for t-SNE.
    Metrics are computed over the whole dataset, not averaged per batch.
    """
    model.eval()
    all_true, all_pred, all_features = [], [], []
    total_loss, total_samples = 0.0, 0

    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        logits = model(images)

        if criterion is not None:
            total_loss += criterion(logits, targets).item() * images.size(0)
        total_samples += images.size(0)

        all_true.append(targets.cpu().numpy())
        all_pred.append(logits.argmax(dim=1).cpu().numpy())
        if collect_features:
            all_features.append(model.extract_features(images).cpu().numpy())

    result = {
        "y_true": np.concatenate(all_true),
        "y_pred": np.concatenate(all_pred),
        "loss": total_loss / total_samples if criterion is not None else float("nan"),
    }
    result["accuracy"] = float((result["y_true"] == result["y_pred"]).mean())
    if collect_features:
        result["features"] = np.concatenate(all_features)
    return result
