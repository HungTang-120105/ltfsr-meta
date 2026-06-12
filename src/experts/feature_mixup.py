"""Tail-aware feature mixup — the CMO idea, moved into the frozen feature space.

CMO (Method 6) pastes a **tail** object onto a head image so rare classes appear in
many contexts. We can't paste boxes onto a frozen CLIP/DINOv2 feature vector, but we
can do the feature-space analogue: convex-mix each sample with a partner drawn from a
**class-balanced (tail-rich) stream**, and mix the labels the same way. Over an epoch
the tail features get blended into many head contexts, widening their decision region
— exactly CMO's effect, computed in seconds on cached features.

Used as an optional augmentation inside ``lift.train_lift`` (``mixup_alpha > 0``); the
mixed batch trains with a soft, two-label loss (``lam * loss(y_a) + (1-lam) * loss(y_b)``),
reusing the same ``BalancedSoftmaxLoss``.
"""

from __future__ import annotations

import numpy as np
import torch


def tail_rich_weights(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Per-sample sampling weights ``1 / class_count`` (draws every class equally).

    Sampling partners with these weights makes tail samples appear as the mixed-in
    minority far more often than their raw frequency — the "tail-rich stream" of CMO.
    """
    counts = torch.bincount(labels.long(), minlength=num_classes).clamp_min(1)
    return 1.0 / counts[labels.long()].float()


def mixup_batch(x_a: torch.Tensor, y_a: torch.Tensor, x_pool: torch.Tensor,
                y_pool: torch.Tensor, partner_weights: torch.Tensor, alpha: float):
    """Mix a batch with tail-rich partners -> ``(mixed_x, y_a, y_b, lam)``.

    ``lam ~ Beta(alpha, alpha)`` is the weight on the original batch; the partner
    contributes ``1 - lam``. Mixing is linear (the cosine head re-normalises the
    adapted feature downstream, so no explicit renormalisation is needed here).
    """
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.multinomial(partner_weights, x_a.size(0), replacement=True)
    x_b, y_b = x_pool[idx].to(x_a.device), y_pool[idx].to(x_a.device)
    mixed = lam * x_a + (1.0 - lam) * x_b
    return mixed, y_a, y_b, lam
