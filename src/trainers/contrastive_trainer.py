"""Method 3 - Supervised Contrastive (SupCon) pre-training + linear probe.

Two stages (see docs/03_contrastive.md):
  1. Pre-train the encoder so that two augmented views of same-class images are
     pulled together and different classes pushed apart (SupCon loss). This
     produces a well-structured feature space without using a softmax head.
  2. Freeze the encoder and train a small linear classifier on top, then
     evaluate it like any other classifier.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.datasets.cifar_lt import TwoCropTransform, build_transforms, make_loader
from src.models.baseline import BaselineClassifier
from src.models.projection import ContrastiveModel
from src.trainers.decoupling_trainer import rebalance_classifier


def supcon_loss(embeddings: torch.Tensor, labels: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """Supervised Contrastive loss (Khosla et al., 2020), all-pairs form.

    Args:
        embeddings: L2-normalised embeddings, shape ``(N, D)`` (N = 2 * batch).
        labels: Class labels, shape ``(N,)``.
    """
    device = embeddings.device
    similarity = embeddings @ embeddings.T / temperature
    similarity = similarity - similarity.max(dim=1, keepdim=True).values.detach()

    self_mask = ~torch.eye(len(labels), dtype=torch.bool, device=device)
    positive_mask = (labels[:, None] == labels[None, :]) & self_mask

    exp_similarity = torch.exp(similarity) * self_mask
    log_prob = similarity - torch.log(exp_similarity.sum(dim=1, keepdim=True) + 1e-12)

    positive_count = positive_mask.sum(dim=1)
    mean_log_prob = (positive_mask * log_prob).sum(dim=1) / positive_count.clamp(min=1)
    return -mean_log_prob[positive_count > 0].mean()


def pretrain_encoder(
    train_dataset,
    device: torch.device,
    epochs: int,
    batch_size: int,
    num_workers: int,
    learning_rate: float,
    temperature: float,
    pretrained: bool,
    image_size: int = 32,
) -> tuple[ContrastiveModel, pd.DataFrame]:
    """Stage 1: contrastive pre-training with two augmented views per image.

    Pre-training runs on exactly ``train_dataset`` (the training split), so it never
    sees the validation images used for model selection.
    """
    two_view = copy.copy(train_dataset)
    two_view.transform = TwoCropTransform(build_transforms(train=True, image_size=image_size))
    loader = make_loader(two_view, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    model = ContrastiveModel(pretrained=pretrained).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    history: list[dict] = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_batches = 0.0, 0
        for views, labels in loader:
            images = torch.cat([views[0], views[1]], dim=0).to(device)
            labels = labels.repeat(2).to(device)

            optimizer.zero_grad(set_to_none=True)
            loss = supcon_loss(model(images), labels, temperature)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_batches += 1
        scheduler.step()

        epoch_loss = total_loss / max(total_batches, 1)
        history.append({"epoch": epoch, "supcon_loss": epoch_loss})
        print(f"[SupCon pretrain] Epoch {epoch:03d}/{epochs:03d} | loss={epoch_loss:.4f}")

    return model, pd.DataFrame(history)


def train_contrastive(
    train_dataset,
    val_loader: DataLoader,
    num_classes: int,
    device: torch.device,
    run_dir: Path,
    pretrain_epochs: int = 200,
    probe_epochs: int = 10,
    batch_size: int = 128,
    num_workers: int = 2,
    pretrain_lr: float = 0.5,
    probe_lr: float = 0.1,
    temperature: float = 0.07,
    pretrained: bool = False,
    image_size: int = 32,
) -> tuple[BaselineClassifier, pd.DataFrame, pd.DataFrame]:
    """Run both stages; return (cRT model, classifier history, pretrain history).

    Stage 1 learns the representation with SupCon on ``train_dataset``. Stage 2 is
    **cRT** (the same class-balanced classifier re-training used by Method 3), not a
    plain linear probe — that is what lets the strong contrastive features actually
    help the tail instead of inheriting the head bias.
    """
    encoder_model, pretrain_history = pretrain_encoder(
        train_dataset, device, pretrain_epochs, batch_size, num_workers,
        pretrain_lr, temperature, pretrained, image_size=image_size,
    )

    # Stage 2: cRT on the frozen contrastive encoder.
    classifier = BaselineClassifier(num_classes=num_classes, pretrained=False).to(device)
    classifier.encoder.load_state_dict(encoder_model.encoder.state_dict())

    classifier, probe_history = rebalance_classifier(
        classifier, train_dataset, val_loader, num_classes, device, run_dir,
        epochs=probe_epochs, learning_rate=probe_lr, batch_size=batch_size, num_workers=num_workers,
    )
    return classifier, probe_history, pretrain_history
