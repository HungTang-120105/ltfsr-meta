"""Evaluation metrics for imbalanced / long-tail classification.

Accuracy alone is misleading on long-tail data because it is dominated by the
head classes. This module reports a fuller picture: balanced accuracy, macro and
weighted F1, the standard Many/Medium/Few-shot accuracy split, plus G-Mean and
Matthews Correlation Coefficient. All heavy lifting uses scikit-learn (already a
dependency) so the code stays short and trustworthy.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)


def per_class_recall(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """Recall (a.k.a. per-class accuracy) for every class id, NaN if unseen."""
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    support = matrix.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.diag(matrix) / support
    recall[support == 0] = np.nan
    return recall


def g_mean(recall: np.ndarray) -> float:
    """Geometric mean of per-class recall (balances head vs tail performance)."""
    valid = recall[~np.isnan(recall)]
    if valid.size == 0:
        return 0.0
    # Geometric mean via logs; epsilon keeps a single zero-recall class finite.
    return float(np.exp(np.mean(np.log(np.clip(valid, 1e-12, 1.0)))))


def shot_group_accuracy(recall: np.ndarray, shot_groups: dict[str, list[int]]) -> dict[str, float]:
    """Mean per-class accuracy within the many / medium / few-shot groups."""
    results: dict[str, float] = {}
    for group_name, class_ids in shot_groups.items():
        values = recall[class_ids]
        values = values[~np.isnan(values)]
        results[f"{group_name}_shot_accuracy"] = float(values.mean()) if values.size else 0.0
    return results


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
    shot_groups: dict[str, list[int]] | None = None,
) -> dict[str, float]:
    """Compute the full metric suite for one set of predictions.

    Args:
        y_true: Ground-truth class ids, shape ``(N,)``.
        y_pred: Predicted class ids, shape ``(N,)``.
        num_classes: Total number of classes.
        shot_groups: Optional ``{"many"/"medium"/"few": [class ids]}`` mapping;
            when given, adds the per-group accuracies.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    recall = per_class_recall(y_true, y_pred, num_classes)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "g_mean": g_mean(recall),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }
    if shot_groups is not None:
        metrics.update(shot_group_accuracy(recall, shot_groups))
    return metrics


def format_metrics(metrics: dict[str, float]) -> str:
    """Pretty multi-line string for printing a metrics dict."""
    return "\n".join(f"  {name:>20}: {value:.4f}" for name, value in metrics.items())
