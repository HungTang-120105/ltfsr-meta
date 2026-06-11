"""Module B — Generalized Logit Adjustment (debias the foundation model itself).

Balanced Softmax (our Method 2) corrects the bias from *our* long-tailed training
set. But a foundation model like CLIP was pre-trained on a web-scale set that is
**also** highly imbalanced, so its zero-shot / fine-tuned logits carry a second,
hidden label bias toward web-frequent classes. GLA (Zhu et al., NeurIPS 2023)
removes that bias too — so it is literally a *generalization* of logit adjustment:

    balanced_softmax : logits - log(our training prior)
    GLA              : logits - log(our training prior) - log(foundation-model prior)

We do not know the foundation model's prior in closed form, so we estimate it the
standard way: the model's **average predicted distribution** over a pool of images
is what it *tends* to predict; on a balanced test set the truth is uniform, so that
average reveals the bias. We subtract ``strength * log(estimated_prior)`` and pick
``strength`` on the balanced validation accuracy.

Cheap (a single ``(C,)`` vector, no training) and stacks on top of any expert's
logits — zero-shot CLIP, LIFT, Tip-Adapter, or a fused score.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def estimate_log_bias(logits: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Estimate a model's implicit log-prior from its predictions, shape ``(C,)``.

    ``logits`` are this expert's scores on a pool of images (use the **test** features
    without their labels — this is transductive prior estimation, not label leakage,
    since only the predicted probabilities are read).
    """
    prior = F.softmax(logits, dim=1).mean(dim=0).clamp_min(eps)
    return prior.log()


def gla_adjust(logits: torch.Tensor, log_bias: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
    """Remove the estimated label bias: ``logits - strength * log_bias``."""
    return logits - strength * log_bias.to(logits.device)


def tune_strength(val_logits: torch.Tensor, val_labels: np.ndarray, log_bias: torch.Tensor,
                  strengths=None) -> tuple[float, float]:
    """Pick the debiasing ``strength`` by balanced accuracy on validation.

    Returns ``(best_strength, best_val_balanced_acc)``. ``strength=0`` (no debiasing)
    is always in the grid, so GLA can never hurt the selected result.
    """
    from src.evaluation.metrics import balanced_accuracy

    strengths = np.linspace(0.0, 1.5, 16) if strengths is None else strengths
    best = (0.0, -1.0)
    for s in strengths:
        pred = gla_adjust(val_logits, log_bias, float(s)).argmax(dim=1).cpu().numpy()
        score = balanced_accuracy(val_labels, pred)
        if score > best[1]:
            best = (float(s), float(score))
    return best
