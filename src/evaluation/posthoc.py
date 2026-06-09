"""Post-hoc, no-retrain classifiers that reuse a trained encoder.

Two long-tail techniques that need only a finished checkpoint:

* ``tier_fusion`` — a parametric linear head is data-hungry (great on head classes)
  while a nearest-class-mean / prototype head is data-efficient (better on the
  rare tail). We blend the two *per class*, trusting prototypes more for the
  few-shot classes. This is the practical way to harvest meta-learning's strength
  (good prototypes from few examples) on the full 100-way problem.

* ``tau_normalized_predict`` — τ-normalization (Kang et al., 2020): a linear head's
  weight norm grows with class frequency, biasing toward head classes. Dividing
  each class weight by ``||w_c|| ** tau`` removes that bias with no retraining.

Both return an engine-style ``{y_true, y_pred}`` dict for ``compute_metrics``.
"""

from __future__ import annotations

import numpy as np
import torch

from src.models.prototype import pairwise_sq_distance
from src.trainers.meta_trainer import compute_global_prototypes


@torch.no_grad()
def tier_fusion(model, train_eval_loader, test_loader, num_classes: int,
                shot_groups: dict, device, weights: dict | None = None) -> dict:
    """Blend parametric and prototype probabilities, weighted per shot-group.

    ``weights`` maps ``many/medium/few -> prototype weight in [0, 1]``. The default
    leans on the linear head for head classes and on prototypes for the tail.
    """
    if weights is None:
        weights = {"many": 0.0, "medium": 0.3, "few": 0.8}
    model.eval()

    prototypes = compute_global_prototypes(model.encoder, train_eval_loader, num_classes, device)

    proto_weight = torch.zeros(num_classes, device=device)
    for group, value in weights.items():
        for class_id in shot_groups.get(group, []):
            proto_weight[class_id] = value

    all_true, all_pred = [], []
    for images, targets in test_loader:
        features = model.encoder(images.to(device))
        p_param = model.classifier(features).softmax(dim=1)
        p_proto = (-pairwise_sq_distance(features, prototypes)).softmax(dim=1)
        blended = (1 - proto_weight) * p_param + proto_weight * p_proto
        all_pred.append(blended.argmax(dim=1).cpu().numpy())
        all_true.append(targets.numpy())
    return {"y_true": np.concatenate(all_true), "y_pred": np.concatenate(all_pred)}


@torch.no_grad()
def tau_normalized_predict(model, test_loader, device, tau: float = 1.0) -> dict:
    """Classify with τ-normalized classifier weights (no bias, no retraining)."""
    model.eval()
    weight = model.classifier.weight.data
    norms = weight.norm(dim=1, keepdim=True).clamp_min(1e-12)
    weight_tau = weight / (norms ** tau)

    all_true, all_pred = [], []
    for images, targets in test_loader:
        features = model.encoder(images.to(device))
        logits = features @ weight_tau.t()
        all_pred.append(logits.argmax(dim=1).cpu().numpy())
        all_true.append(targets.numpy())
    return {"y_true": np.concatenate(all_true), "y_pred": np.concatenate(all_pred)}
