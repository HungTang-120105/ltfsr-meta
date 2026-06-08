"""Method 4 - Meta-learning: episodic Prototypical Networks.

The encoder is trained on thousands of small N-way K-shot episodes (see
docs/04_meta_learning.md). Within each episode, class prototypes are the mean of
the support features, and query images are classified by distance to those
prototypes. This explicitly rehearses learning from few examples, which is what
tail classes need.

For a fair comparison with the other methods, final evaluation is a full 100-way
test: prototypes are computed from the whole training set and every test image is
assigned to its nearest prototype.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.datasets.episodic import build_class_index, sample_episode
from src.models.backbone import FEATURE_DIM, build_encoder
from src.models.prototype import compute_prototypes, pairwise_sq_distance


def load_images(dataset, indices: list[int], device: torch.device) -> torch.Tensor:
    """Stack the transformed images at ``indices`` into one batch tensor."""
    images = torch.stack([dataset[index][0] for index in indices])
    return images.to(device)


def run_episode(encoder: nn.Module, dataset, class_index, n_way, k_shot, n_query,
                rng: random.Random, device: torch.device, train: bool) -> tuple[torch.Tensor, float]:
    """Run one episode; return the query loss and query accuracy."""
    support_idx, support_labels, query_idx, query_labels = sample_episode(
        class_index, n_way, k_shot, n_query, rng
    )
    support_images = load_images(dataset, support_idx, device)
    query_images = load_images(dataset, query_idx, device)
    support_y = torch.tensor(support_labels, device=device)
    query_y = torch.tensor(query_labels, device=device)

    with torch.set_grad_enabled(train):
        support_features = encoder(support_images)
        query_features = encoder(query_images)
        prototypes = compute_prototypes(support_features, support_y, num_classes=n_way)
        logits = -pairwise_sq_distance(query_features, prototypes)
        loss = nn.functional.cross_entropy(logits, query_y)

    accuracy = (logits.argmax(dim=1) == query_y).float().mean().item()
    return loss, accuracy


def train_meta(
    train_dataset,
    monitor_dataset,
    device: torch.device,
    run_dir: Path,
    epochs: int = 100,
    episodes_per_epoch: int = 100,
    n_way: int = 5,
    k_shot: int = 5,
    n_query: int = 15,
    learning_rate: float = 0.001,
    pretrained: bool = False,
    seed: int = 42,
) -> tuple[nn.Module, pd.DataFrame]:
    """Episodically train the encoder; keep the best weights by monitor accuracy."""
    encoder = build_encoder(pretrained=pretrained).to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    rng = random.Random(seed)

    train_index = build_class_index([label for _, label in train_dataset.samples])
    monitor_index = build_class_index([label for _, label in monitor_dataset.samples])

    history: list[dict] = []
    # Below zero so the first epoch always writes a checkpoint (see classifier.py).
    best_accuracy = -1.0
    checkpoint_path = Path(run_dir) / "best_model.pt"

    for epoch in range(1, epochs + 1):
        encoder.train()
        train_losses, train_accuracies = [], []
        for _ in range(episodes_per_epoch):
            optimizer.zero_grad(set_to_none=True)
            loss, accuracy = run_episode(encoder, train_dataset, train_index,
                                         n_way, k_shot, n_query, rng, device, train=True)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            train_accuracies.append(accuracy)

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        val_loss, val_accuracy = evaluate_episodes(encoder, monitor_dataset, monitor_index,
                                                    n_way, k_shot, n_query, rng, device)

        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "train_accuracy": float(np.mean(train_accuracies)),
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "learning_rate": current_lr,
        })
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save({"epoch": epoch, "model_state_dict": encoder.state_dict(),
                        "val_accuracy": best_accuracy}, checkpoint_path)

        print(f"Epoch {epoch:03d}/{epochs:03d} | "
              f"train_loss={history[-1]['train_loss']:.4f} train_acc={history[-1]['train_accuracy']:.4f} | "
              f"val_acc={val_accuracy:.4f} | best={max(best_accuracy, 0.0):.4f}")

    encoder.load_state_dict(torch.load(checkpoint_path, map_location=device)["model_state_dict"])
    return encoder, pd.DataFrame(history)


@torch.no_grad()
def evaluate_episodes(encoder, dataset, class_index, n_way, k_shot, n_query,
                      rng: random.Random, device, num_episodes: int = 50) -> tuple[float, float]:
    """Mean loss and accuracy over a batch of monitoring episodes."""
    encoder.eval()
    losses, accuracies = [], []
    for _ in range(num_episodes):
        loss, accuracy = run_episode(encoder, dataset, class_index,
                                     n_way, k_shot, n_query, rng, device, train=False)
        losses.append(loss.item())
        accuracies.append(accuracy)
    return float(np.mean(losses)), float(np.mean(accuracies))


@torch.no_grad()
def compute_global_prototypes(encoder, loader: DataLoader, num_classes: int, device) -> torch.Tensor:
    """Mean feature per class over a full loader (the 100-way classifier)."""
    encoder.eval()
    sums = torch.zeros(num_classes, FEATURE_DIM, device=device)
    counts = torch.zeros(num_classes, device=device)
    for images, labels in loader:
        features = encoder(images.to(device))
        labels = labels.to(device)
        sums.index_add_(0, labels, features)
        counts.index_add_(0, labels, torch.ones_like(labels, dtype=torch.float))
    return sums / counts.clamp(min=1).unsqueeze(1)


@torch.no_grad()
def evaluate_meta(encoder, train_eval_loader: DataLoader, test_loader: DataLoader,
                  num_classes: int, device, collect_features: bool = True) -> dict:
    """Full 100-way evaluation by nearest global prototype (engine-compatible dict)."""
    encoder.eval()
    prototypes = compute_global_prototypes(encoder, train_eval_loader, num_classes, device)

    all_true, all_pred, all_features = [], [], []
    for images, labels in test_loader:
        features = encoder(images.to(device))
        logits = -pairwise_sq_distance(features, prototypes)
        all_true.append(labels.numpy())
        all_pred.append(logits.argmax(dim=1).cpu().numpy())
        if collect_features:
            all_features.append(features.cpu().numpy())

    result = {"y_true": np.concatenate(all_true), "y_pred": np.concatenate(all_pred), "loss": float("nan")}
    result["accuracy"] = float((result["y_true"] == result["y_pred"]).mean())
    if collect_features:
        result["features"] = np.concatenate(all_features)
    return result
