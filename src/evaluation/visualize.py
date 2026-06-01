"""Plotting helpers. Every function saves a PNG and returns its path.

Figures are written to the run directory so each experiment is self-documenting.
matplotlib uses the non-interactive 'Agg' backend so the code runs headless on
Kaggle without a display.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def _save(fig, out_dir: Path, name: str) -> Path:
    path = Path(out_dir) / name
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_training_curves(history: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Loss and accuracy curves (train vs validation) over epochs."""
    paths = []
    for metric, ylabel in [("loss", "Loss"), ("accuracy", "Accuracy")]:
        fig, ax = plt.subplots(figsize=(7, 5))
        if f"train_{metric}" in history:
            ax.plot(history["epoch"], history[f"train_{metric}"], label=f"train {metric}")
        if f"val_{metric}" in history:
            ax.plot(history["epoch"], history[f"val_{metric}"], label=f"val {metric}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} curve")
        ax.legend()
        ax.grid(True, alpha=0.3)
        paths.append(_save(fig, out_dir, f"curve_{metric}.png"))
    return paths


def plot_confusion_matrices(y_true, y_pred, num_classes: int, out_dir: Path) -> list[Path]:
    """Raw and row-normalised confusion matrices."""
    labels = list(range(num_classes))
    raw = confusion_matrix(y_true, y_pred, labels=labels)
    with np.errstate(divide="ignore", invalid="ignore"):
        norm = raw / raw.sum(axis=1, keepdims=True)
    norm = np.nan_to_num(norm)

    paths = []
    for matrix, name, title in [(raw, "confusion_matrix.png", "Confusion matrix"),
                                (norm, "confusion_matrix_normalized.png", "Normalized confusion matrix")]:
        fig, ax = plt.subplots(figsize=(8, 7))
        image = ax.imshow(matrix, cmap="viridis", aspect="auto")
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.set_title(title)
        fig.colorbar(image, ax=ax)
        paths.append(_save(fig, out_dir, name))
    return paths


def plot_class_frequency(class_counts: dict[int, int], out_dir: Path) -> Path:
    """Bar chart of training images per class, sorted high to low."""
    ordered = sorted(class_counts.values(), reverse=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(range(len(ordered)), ordered, color="steelblue")
    ax.set_xlabel("Class (sorted by frequency)")
    ax.set_ylabel("Number of training images")
    ax.set_title("Class frequency (long-tail profile)")
    ax.grid(True, axis="y", alpha=0.3)
    return _save(fig, out_dir, "class_frequency.png")


def plot_shot_distribution(shot_groups: dict[str, list[int]], out_dir: Path) -> Path:
    """How many classes fall into each many/medium/few-shot group."""
    names = ["many", "medium", "few"]
    sizes = [len(shot_groups.get(name, [])) for name in names]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(names, sizes, color=["#2ca02c", "#ff7f0e", "#d62728"])
    ax.set_ylabel("Number of classes")
    ax.set_title("Head / medium / tail group sizes")
    for index, size in enumerate(sizes):
        ax.text(index, size, str(size), ha="center", va="bottom")
    return _save(fig, out_dir, "shot_distribution.png")


def plot_tsne(features: np.ndarray, labels: np.ndarray, out_dir: Path, max_points: int = 2000) -> Path | None:
    """2-D t-SNE scatter of learned features (skipped if scikit-learn lacks it)."""
    try:
        from sklearn.manifold import TSNE
    except Exception:
        return None

    if len(features) > max_points:
        keep = np.random.choice(len(features), size=max_points, replace=False)
        features, labels = features[keep], labels[keep]

    embedding = TSNE(n_components=2, init="pca", learning_rate="auto").fit_transform(features)
    fig, ax = plt.subplots(figsize=(8, 7))
    scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap="tab20", s=6, alpha=0.7)
    ax.set_title("t-SNE of learned features")
    fig.colorbar(scatter, ax=ax, label="class id")
    return _save(fig, out_dir, "tsne.png")
