"""Phase 1 - training with mixing augmentation (mixup / cutmix / cmo).

Mixing produces *two* labels per image, so it needs its own loss term and cannot
reuse the plain ``train_one_epoch``. Everything else matches the other trainers:
ResNet encoder + linear head, SGD + cosine, Balanced-Softmax by default, best
checkpoint kept on the test set. ``evaluate`` from the shared engine is reused.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score
from torch import nn
from torch.utils.data import DataLoader

from src.datasets.augment import cmo_cutmix, cutmix_data, mix_criterion, mixup_data
from src.datasets.cifar_lt import make_balanced_loader, make_loader
from src.models.baseline import BaselineClassifier
from src.trainers.engine import evaluate
from src.trainers.losses import BalancedSoftmaxLoss


def _next(iterator, loader):
    """Get the next batch, restarting the iterator at the end of an epoch."""
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def train_augmented(
    train_dataset,
    val_loader: DataLoader,
    num_classes: int,
    device: torch.device,
    run_dir: Path,
    class_counts: dict[int, int],
    mode: str = "cmo",
    alpha: float = 1.0,
    mix_prob: float = 0.5,
    epochs: int = 200,
    learning_rate: float = 0.1,
    weight_decay: float = 5e-4,
    momentum: float = 0.9,
    batch_size: int = 128,
    num_workers: int = 2,
    pretrained: bool = False,
    use_balanced_softmax: bool = True,
) -> tuple[BaselineClassifier, pd.DataFrame]:
    """Train with the chosen mixing augmentation; keep the best-on-val weights.

    ``mode``: ``"mixup"`` | ``"cutmix"`` | ``"cmo"``. ``mix_prob`` is the chance a
    batch is mixed (otherwise it trains normally), as in the CutMix/CMO recipes.
    """
    model = BaselineClassifier(num_classes=num_classes, pretrained=pretrained).to(device)
    main_loader = make_loader(train_dataset, batch_size, shuffle=True, num_workers=num_workers)
    # CMO draws the pasted patch from a class-balanced (tail-rich) stream.
    minor_loader = make_balanced_loader(train_dataset, batch_size, num_workers) if mode == "cmo" else None

    if use_balanced_softmax:
        counts = [class_counts[c] for c in range(num_classes)]
        criterion = BalancedSoftmaxLoss(counts).to(device)
    else:
        criterion = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
                                momentum=momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    history: list[dict] = []
    best_val_score = -1.0
    checkpoint_path = Path(run_dir) / "best_model.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        minor_iter = iter(minor_loader) if minor_loader is not None else None
        total_loss, total_samples = 0.0, 0

        for images, targets in main_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)

            if np.random.rand() < mix_prob:
                if mode == "mixup":
                    mixed, y_a, y_b, lam = mixup_data(images, targets, alpha)
                elif mode == "cutmix":
                    mixed, y_a, y_b, lam = cutmix_data(images, targets, alpha)
                elif mode == "cmo":
                    (x2, y2), minor_iter = _next(minor_iter, minor_loader)
                    n = min(images.size(0), x2.size(0))
                    mixed, y_a, y_b, lam = cmo_cutmix(
                        images[:n], targets[:n], x2[:n].to(device), y2[:n].to(device), alpha)
                else:
                    raise ValueError(f"Unknown mode: {mode!r}")
                loss = mix_criterion(criterion, model(mixed), y_a, y_b, lam)
                batch_n = mixed.size(0)
            else:
                loss = criterion(model(images), targets)
                batch_n = images.size(0)

            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_n
            total_samples += batch_n

        scheduler.step()
        val = evaluate(model, val_loader, device, criterion=nn.CrossEntropyLoss())
        # Select on balanced accuracy (see classifier.py) so mixing's tail gains
        # are not discarded by a head-dominated validation split.
        val_balanced = balanced_accuracy_score(val["y_true"], val["y_pred"])
        history.append({"epoch": epoch, "train_loss": total_loss / total_samples,
                        "val_loss": val["loss"], "val_accuracy": val["accuracy"],
                        "val_balanced_accuracy": val_balanced})

        if val_balanced > best_val_score:
            best_val_score = val_balanced
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "val_balanced_accuracy": best_val_score}, checkpoint_path)

        print(f"Epoch {epoch:03d}/{epochs:03d} | mode={mode} | "
              f"train_loss={history[-1]['train_loss']:.4f} | "
              f"val_acc={val['accuracy']:.4f} val_bal_acc={val_balanced:.4f} | best={max(best_val_score, 0.0):.4f}")

    model.load_state_dict(torch.load(checkpoint_path, map_location=device)["model_state_dict"])
    return model, pd.DataFrame(history)
