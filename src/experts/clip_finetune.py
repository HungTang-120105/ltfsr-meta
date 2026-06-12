"""Fine-tuning-depth ablation: how much of CLIP should we actually train?

The rest of the project keeps CLIP **frozen** and trains tiny heads/adapters on cached
features (LIFT, Tip-Adapter). A fair question is whether *fine-tuning the backbone*
would do better. LIFT's thesis (ICML 2024) says **heavy fine-tuning hurts the tail** —
this module lets us show that on our own data by sweeping the adaptation depth:

    zero-shot  <  linear_probe  <  last_block  ?  full_ft
    (none)        (head only)     (head + last  (whole visual
                                   transformer    encoder
                                   block)         unfrozen)

Each depth trains a linear head (+ optionally part of the visual encoder) on the
long-tail **images** with Balanced Softmax, selecting the best epoch on balanced val
accuracy — same protocol as everything else. Unlike the frozen track this needs
backprop through the ViT (no feature caching), so it is the one GPU-heavy extra; keep
``epochs`` small. The expected curve (tail accuracy peaks at light adaptation, then
drops for full fine-tuning) is exactly what justifies the frozen design.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from src.evaluation.metrics import balanced_accuracy
from src.trainers.losses import BalancedSoftmaxLoss

DEPTHS = ("linear_probe", "last_block", "full_ft")


def set_finetune_depth(visual: nn.Module, depth: str) -> None:
    """Unfreeze the visual-encoder parameters that ``depth`` makes trainable."""
    visual.requires_grad_(False)
    if depth == "linear_probe":
        return                                   # head only; backbone fully frozen
    if depth == "full_ft":
        visual.requires_grad_(True)
        return
    if depth == "last_block":                    # head + last transformer block + final LN
        visual.transformer.resblocks[-1].requires_grad_(True)
        if hasattr(visual, "ln_post"):
            visual.ln_post.requires_grad_(True)
        return
    raise ValueError(f"unknown depth: {depth!r} (use one of {DEPTHS})")


class _ClipWithHead(nn.Module):
    """CLIP visual tower + a linear classification head on the (normalized) features."""

    def __init__(self, clip_model: nn.Module, num_classes: int, feat_dim: int) -> None:
        super().__init__()
        self.clip = clip_model
        self.head = nn.Linear(feat_dim, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(F.normalize(self.clip.encode_image(images), dim=-1))


def _loader(dataset, preprocess, batch_size, num_workers, shuffle):
    view = copy.copy(dataset)
    view.transform = preprocess
    return DataLoader(view, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=torch.cuda.is_available())


@torch.no_grad()
def _eval(net, loader, device):
    net.eval()
    y_true, y_pred = [], []
    for images, targets in loader:
        y_pred.append(net(images.to(device)).argmax(1).cpu().numpy())
        y_true.append(targets.numpy())
    return np.concatenate(y_true), np.concatenate(y_pred)


def train_clip_finetune(clip_model, preprocess, train_dataset, val_dataset, num_classes: int,
                        class_counts: list[int], feat_dim: int, device, depth: str,
                        epochs: int = 10, lr: float = 1e-4, batch_size: int = 128,
                        num_workers: int = 2) -> tuple[nn.Module, dict]:
    """Fine-tune CLIP at the given ``depth`` (fresh copy); return (net, info).

    A **deepcopy** of ``clip_model`` is trained so every depth starts from the same
    pretrained weights and the caller's model is left frozen/intact.
    """
    model = copy.deepcopy(clip_model).to(device)
    model.requires_grad_(False)                  # freeze everything (incl. text tower)
    set_finetune_depth(model.visual, depth)      # then unfreeze the visual part for this depth

    net = _ClipWithHead(model, num_classes, feat_dim).to(device)
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr)
    criterion = BalancedSoftmaxLoss(class_counts).to(device)
    train_loader = _loader(train_dataset, preprocess, batch_size, num_workers, True)
    val_loader = _loader(val_dataset, preprocess, batch_size, num_workers, False)

    best_state, best_val = copy.deepcopy(net.state_dict()), -1.0
    for _ in range(epochs):
        net.train()
        for images, targets in train_loader:
            loss = criterion(net(images.to(device)), targets.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        y_true, y_pred = _eval(net, val_loader, device)
        val_acc = balanced_accuracy(y_true, y_pred)
        if val_acc > best_val:
            best_val, best_state = val_acc, copy.deepcopy(net.state_dict())

    net.load_state_dict(best_state)
    net.eval()
    return net, {"best_val_balanced_accuracy": float(best_val),
                 "trainable_params": int(sum(p.numel() for p in params))}


def predict(net, dataset, preprocess, device, batch_size: int = 128, num_workers: int = 2) -> dict:
    """Engine-style ``{y_true, y_pred}`` for a fine-tuned CLIP classifier."""
    y_true, y_pred = _eval(net, _loader(dataset, preprocess, batch_size, num_workers, False), device)
    return {"y_true": y_true, "y_pred": y_pred}
