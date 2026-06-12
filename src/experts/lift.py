"""LIFT-style lightweight fine-tuning of a frozen CLIP for long-tail recognition.

Inspired by Shi et al., *Long-tail Learning with Foundation Models: Heavy
Fine-tuning Hurts* (LIFT, ICML 2024). The lesson: with a strong frozen backbone,
fully fine-tuning *hurts* the tail — a tiny adapter plus the right loss wins. We
keep the spirit while staying Kaggle-cheap and readable by training on the
**cached** CLIP features (the backbone stays frozen and is never re-run):

* **Residual adapter** — a small bottleneck MLP with a learnable scalar gate,
  initialised to zero so the model *starts exactly at zero-shot CLIP* and only
  departs from it as training improves the tail. This is the ``<1% trainable
  params`` story for the slides.
* **Cosine classifier initialised from the text features** — the head begins with
  CLIP's own language knowledge of every class name (semantic init), so even the
  5-image tail classes start from a meaningful prototype instead of noise.
* **Logit-adjusted loss** — reuse ``BalancedSoftmaxLoss`` so the rare classes are
  not drowned out by the head during training.

Model selection is the project standard: keep the epoch with the best **balanced**
accuracy on the long-tail validation split.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.evaluation.metrics import compute_metrics
from src.trainers.losses import BalancedSoftmaxLoss


class ResidualAdapter(nn.Module):
    """``x + gate * up(relu(down(x)))`` — a bottleneck adapter, gate starts at 0."""

    def __init__(self, dim: int, bottleneck: int = 64) -> None:
        super().__init__()
        self.down = nn.Linear(dim, bottleneck)
        self.up = nn.Linear(bottleneck, dim)
        # gate=0 -> adapter is the identity at init, so the model == zero-shot CLIP.
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.gate * self.up(F.relu(self.down(x)))


class CosineClassifier(nn.Module):
    """Scaled cosine similarity to per-class weights, initialised from text features."""

    def __init__(self, text_features: torch.Tensor, logit_scale: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(text_features.clone())          # (C, D), unit-norm init
        self.logit_scale = nn.Parameter(torch.tensor(math.log(logit_scale)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.logit_scale.exp().clamp(max=100.0)
        return scale * F.normalize(x, dim=-1) @ F.normalize(self.weight, dim=-1).t()


class LIFTModel(nn.Module):
    """Frozen-feature adapter + text-initialised cosine head."""

    def __init__(self, text_features: torch.Tensor, logit_scale: float,
                 bottleneck: int = 64) -> None:
        super().__init__()
        self.adapter = ResidualAdapter(text_features.shape[1], bottleneck)
        self.classifier = CosineClassifier(text_features, logit_scale)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.adapter(features))


def _balanced_acc(model: LIFTModel, features: torch.Tensor, labels: torch.Tensor,
                  num_classes: int, device) -> float:
    model.eval()
    with torch.no_grad():
        pred = model(features.to(device)).argmax(dim=1).cpu().numpy()
    return compute_metrics(labels.numpy(), pred, num_classes)["balanced_accuracy"]


def train_lift(train_features: torch.Tensor, train_labels: torch.Tensor,
               val_features: torch.Tensor, val_labels: torch.Tensor,
               text_features: torch.Tensor, logit_scale: float,
               class_counts: list[int], num_classes: int, device,
               epochs: int = 50, lr: float = 1e-3, weight_decay: float = 1e-2,
               batch_size: int = 256, bottleneck: int = 64,
               mixup_alpha: float = 0.0) -> tuple[LIFTModel, dict]:
    """Train the adapter + cosine head; return the best-by-val-balanced-acc model.

    Only the adapter and the cosine head are trainable (the backbone is frozen and
    already baked into the cached features), so this is a few seconds per epoch.

    ``mixup_alpha > 0`` turns on **tail-aware feature mixup** (see
    ``src.experts.feature_mixup``): each batch is convex-mixed with a class-balanced,
    tail-rich partner and trained with a soft two-label loss — the CMO idea in feature
    space. ``0`` (default) keeps the plain hard-label training, so existing callers are
    unchanged.
    """
    model = LIFTModel(text_features.to(device), logit_scale, bottleneck).to(device)
    criterion = BalancedSoftmaxLoss(class_counts).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    dataset = torch.utils.data.TensorDataset(train_features, train_labels.long())
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    if mixup_alpha > 0:
        from src.experts.feature_mixup import mixup_batch, tail_rich_weights
        partner_w = tail_rich_weights(train_labels, num_classes)

    best_state, best_val, history = copy.deepcopy(model.state_dict()), -1.0, []
    for epoch in range(epochs):
        model.train()
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            if mixup_alpha > 0:
                features, y_a, y_b, lam = mixup_batch(features, labels, train_features,
                                                      train_labels.long(), partner_w, mixup_alpha)
                logits = model(features)
                loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
            else:
                loss = criterion(model(features), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        val_acc = _balanced_acc(model, val_features, val_labels, num_classes, device)
        history.append({"epoch": epoch, "val_balanced_accuracy": val_acc})
        if val_acc > best_val:
            best_val, best_state = val_acc, copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    model.eval()
    return model, {"best_val_balanced_accuracy": best_val, "history": history}


@torch.no_grad()
def predict(model: LIFTModel, features: torch.Tensor, labels: torch.Tensor, device) -> dict:
    """Engine-style ``{y_true, y_pred}`` from the trained LIFT model."""
    model.eval()
    pred = model(features.to(device)).argmax(dim=1).cpu().numpy()
    return {"y_true": labels.numpy(), "y_pred": pred}
