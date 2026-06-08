"""Shared fit loop for any plain classifier (baseline / prototype / linear probe).

These methods differ only in the model they put on top of the encoder, so they
share one training loop. Each method keeps its own thin trainer file (for
readability) that builds its model and calls ``fit_classifier`` here.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.trainers.engine import evaluate, train_one_epoch


def fit_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    run_dir: Path,
    epochs: int = 100,
    learning_rate: float = 0.1,
    weight_decay: float = 5e-4,
    momentum: float = 0.9,
    checkpoint_name: str = "best_model.pt",
    criterion: nn.Module | None = None,
) -> tuple[nn.Module, pd.DataFrame]:
    """Train with SGD + cosine schedule, keeping the best-on-validation weights.

    ``criterion`` defaults to plain cross-entropy; pass a long-tail loss (e.g.
    ``BalancedSoftmaxLoss``) to swap it in without changing this loop.

    Returns the model (with best weights reloaded) and the per-epoch history.
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    # Only optimise trainable parameters — lets the cRT stage freeze the encoder
    # and update the classifier alone just by setting requires_grad=False.
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        trainable, lr=learning_rate, momentum=momentum, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    history: list[dict] = []
    # Start below zero so the first epoch always writes a checkpoint; otherwise a
    # run that never beats 0.0 val accuracy (e.g. a short smoke test) would leave
    # no best_model.pt for the reload below to find.
    best_val_accuracy = -1.0
    checkpoint_path = Path(run_dir) / checkpoint_name

    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val = evaluate(model, val_loader, device, criterion=criterion)
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val["loss"],
                "val_accuracy": val["accuracy"],
                "learning_rate": current_lr,
            }
        )

        if val["accuracy"] > best_val_accuracy:
            best_val_accuracy = val["accuracy"]
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "val_accuracy": best_val_accuracy}, checkpoint_path)

        print(
            f"Epoch {epoch:03d}/{epochs:03d} | "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} | "
            f"val_loss={val['loss']:.4f} val_acc={val['accuracy']:.4f} | best={max(best_val_accuracy, 0.0):.4f}"
        )

    model.load_state_dict(torch.load(checkpoint_path, map_location=device)["model_state_dict"])
    return model, pd.DataFrame(history)
