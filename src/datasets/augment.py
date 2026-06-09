"""Mixing augmentations, including the long-tail-specific CMO.

Three batch-level mixes that regularise training and, for CMO, specifically add
variety to tail classes:

* ``mixup``  — blend two images and their labels (Zhang et al., 2018).
* ``cutmix`` — paste a random box from one image onto another (Yun et al., 2019).
* ``cmo``    — Context-rich Minority Oversampling (Park et al., 2022): CutMix where
  the pasted box comes from a **class-balanced (tail-rich)** stream, so rare
  classes are pasted onto the diverse backgrounds of frequent classes. This is
  the right way to "augment the tail": create new tail *contexts* from head data,
  rather than heavily distorting the few real tail images.

Each function returns ``(mixed_x, y_a, y_b, lam)``; train with ``mix_criterion``.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def rand_bbox(height: int, width: int, lam: float) -> tuple[int, int, int, int]:
    """A random box whose area is ``(1 - lam)`` of the image (CutMix convention)."""
    ratio = np.sqrt(1.0 - lam)
    cut_h, cut_w = int(height * ratio), int(width * ratio)
    cy, cx = np.random.randint(height), np.random.randint(width)
    y1, y2 = np.clip(cy - cut_h // 2, 0, height), np.clip(cy + cut_h // 2, 0, height)
    x1, x2 = np.clip(cx - cut_w // 2, 0, width), np.clip(cx + cut_w // 2, 0, width)
    return y1, y2, x1, x2


def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0):
    """Convex blend of the batch with a shuffled copy of itself."""
    lam = float(np.random.beta(alpha, alpha))
    index = torch.randperm(x.size(0), device=x.device)
    mixed = lam * x + (1 - lam) * x[index]
    return mixed, y, y[index], lam


def cutmix_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0):
    """Paste a random box from a shuffled copy of the batch onto each image."""
    lam = float(np.random.beta(alpha, alpha))
    index = torch.randperm(x.size(0), device=x.device)
    y1, y2, x1, x2 = rand_bbox(x.size(2), x.size(3), lam)
    x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    lam = 1 - (y2 - y1) * (x2 - x1) / (x.size(2) * x.size(3))  # exact area used
    return x, y, y[index], lam


def cmo_cutmix(x: torch.Tensor, y: torch.Tensor,
               x_minor: torch.Tensor, y_minor: torch.Tensor, alpha: float = 1.0):
    """CutMix where the pasted box comes from a (tail-rich) second stream."""
    lam = float(np.random.beta(alpha, alpha))
    y1, y2, x1, x2 = rand_bbox(x.size(2), x.size(3), lam)
    x[:, :, y1:y2, x1:x2] = x_minor[:, :, y1:y2, x1:x2]
    lam = 1 - (y2 - y1) * (x2 - x1) / (x.size(2) * x.size(3))
    return x, y, y_minor, lam


def mix_criterion(criterion: nn.Module, output: torch.Tensor,
                  y_a: torch.Tensor, y_b: torch.Tensor, lam: float) -> torch.Tensor:
    """Loss for a mixed batch: weight each label by its area/blend fraction."""
    return lam * criterion(output, y_a) + (1 - lam) * criterion(output, y_b)
