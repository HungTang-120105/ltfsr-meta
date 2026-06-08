"""Long-tail-aware loss functions.

Only one is needed: **Balanced Softmax** (Ren et al., NeurIPS 2020), also known
as the logit-adjustment loss. It is the simplest strong fix for class imbalance
— two lines of real logic — and is competitive with heavier methods like
LDAM-DRW.

Idea: a plain softmax implicitly assumes a uniform class prior, but the training
set is long-tailed, so the model is biased toward frequent (head) classes. We
correct this by adding the log of the class prior to the logits *during training*
only. At test time the raw logits are used (the prior is uniform on the balanced
test set), so evaluation code does not change at all.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class BalancedSoftmaxLoss(nn.Module):
    """Cross-entropy with a class-prior correction added to the logits."""

    def __init__(self, class_counts: list[int]) -> None:
        super().__init__()
        freq = torch.tensor(class_counts, dtype=torch.float)
        # log prior, shape (num_classes,); registered so it moves with .to(device)
        self.register_buffer("log_prior", torch.log(freq / freq.sum()))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits + self.log_prior, target)
