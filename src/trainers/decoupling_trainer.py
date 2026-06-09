"""Method 3 - Decoupling (cRT): learn features first, then re-balance the head.

Key long-tail insight (Kang et al., ICLR 2020, "Decoupling Representation and
Classifier"): a plain model already learns good *features* even on long-tail
data — the part that is biased toward head classes is the linear *classifier*.
So we train in two stages:

  Stage 1 (representation): train the whole network normally with cross-entropy
           on the natural long-tail distribution.
  Stage 2 (cRT = classifier Re-Training): freeze the encoder, throw away the
           biased linear head, and re-train a fresh head for a few epochs on a
           CLASS-BALANCED sampler. The features never move; only the decision
           boundary is rebalanced.

This is where the big tail-accuracy gain comes from, and it directly demonstrates
the report's thesis: on long-tail data, the bottleneck is the classifier.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.datasets.cifar_lt import make_balanced_loader
from src.models.backbone import FEATURE_DIM
from src.models.baseline import BaselineClassifier
from src.trainers.classifier import fit_classifier


def rebalance_classifier(
    model: nn.Module,
    train_dataset,
    val_loader: DataLoader,
    num_classes: int,
    device: torch.device,
    run_dir: Path,
    epochs: int = 10,
    learning_rate: float = 0.1,
    batch_size: int = 128,
    num_workers: int = 2,
    checkpoint_name: str = "best_model.pt",
) -> tuple[nn.Module, pd.DataFrame]:
    """cRT stage: freeze the encoder, reset the head, retrain it class-balanced.

    Shared by Method 3 (decoupling) and Method 4 (SupCon + cRT) — the only
    difference between them is how the frozen encoder was trained.
    """
    for parameter in model.encoder.parameters():
        parameter.requires_grad = False
    model.classifier = nn.Linear(FEATURE_DIM, num_classes).to(device)  # fresh, unbiased head

    balanced_loader = make_balanced_loader(train_dataset, batch_size, num_workers)
    return fit_classifier(
        model, balanced_loader, val_loader, device, run_dir,
        epochs=epochs, learning_rate=learning_rate, checkpoint_name=checkpoint_name,
        eval_encoder=True,  # freeze encoder BN: only the classifier is being retrained
    )


def train_decoupling(
    train_loader: DataLoader,
    train_dataset,
    val_loader: DataLoader,
    num_classes: int,
    device: torch.device,
    run_dir: Path,
    epochs: int = 200,
    learning_rate: float = 0.1,
    crt_epochs: int = 10,
    crt_lr: float = 0.1,
    batch_size: int = 128,
    num_workers: int = 2,
    pretrained: bool = False,
) -> tuple[BaselineClassifier, pd.DataFrame]:
    """Two-stage decoupled training; returns (cRT model, combined history)."""
    # Stage 1: ordinary representation learning on the long-tail distribution.
    model = BaselineClassifier(num_classes=num_classes, pretrained=pretrained).to(device)
    model, repr_history = fit_classifier(
        model, train_loader, val_loader, device, run_dir,
        epochs=epochs, learning_rate=learning_rate, checkpoint_name="stage1_repr.pt",
    )

    # Stage 2: cRT — rebalance the classifier on the frozen features.
    model, crt_history = rebalance_classifier(
        model, train_dataset, val_loader, num_classes, device, run_dir,
        epochs=crt_epochs, learning_rate=crt_lr, batch_size=batch_size, num_workers=num_workers,
    )

    repr_history["stage"] = "representation"
    crt_history["stage"] = "crt"
    history = pd.concat([repr_history, crt_history], ignore_index=True)
    return model, history
