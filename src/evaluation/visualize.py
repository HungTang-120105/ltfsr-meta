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


def plot_metric_comparison(comparison: pd.DataFrame, out_dir: Path,
                           metrics: list[str] | None = None) -> Path:
    """Grouped bar chart: for each metric, one bar per method (all in one figure).

    ``comparison`` is the table returned by ``experiment.compare_runs`` (a
    ``method`` column plus one column per metric).
    """
    if metrics is None:
        metrics = ["accuracy", "balanced_accuracy", "macro_f1", "g_mean",
                   "many_shot_accuracy", "medium_shot_accuracy", "few_shot_accuracy"]
    metrics = [m for m in metrics if m in comparison.columns]

    x = np.arange(len(metrics))
    n_methods = len(comparison)
    width = 0.8 / max(n_methods, 1)

    fig, ax = plt.subplots(figsize=(max(8, len(metrics) * 1.6), 5))
    for i, (_, row) in enumerate(comparison.iterrows()):
        offset = (i - (n_methods - 1) / 2) * width
        ax.bar(x + offset, [row[m] for m in metrics], width, label=row["method"])
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=30, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Method comparison across metrics")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    return _save(fig, out_dir, "comparison_metrics.png")


def plot_curves_overlay(histories: dict, out_dir: Path,
                        columns: list[str] | None = None) -> list[Path]:
    """One figure per measure, with every method overlaid as a line.

    ``histories`` maps ``method_name -> per-epoch history DataFrame``. Each
    requested ``column`` (e.g. ``val_accuracy``) becomes a single plot containing
    all methods. The x-axis is the epoch index (1..N) so methods with different
    schedule lengths still line up at the start.
    """
    if columns is None:
        columns = ["val_accuracy", "val_loss", "train_accuracy", "train_loss"]
    paths = []
    for column in columns:
        fig, ax = plt.subplots(figsize=(7, 5))
        plotted = False
        for method, history in histories.items():
            if column in history.columns:
                ax.plot(range(1, len(history) + 1), history[column], label=method)
                plotted = True
        if not plotted:
            plt.close(fig)
            continue
        ax.set_xlabel("Epoch")
        ax.set_ylabel(column)
        ax.set_title(f"{column} — all methods")
        ax.legend()
        ax.grid(True, alpha=0.3)
        paths.append(_save(fig, out_dir, f"overlay_{column}.png"))
    return paths


def plot_tsne(features: np.ndarray, labels: np.ndarray, out_dir: Path, max_points: int = 2000) -> Path | None:
    """2-D t-SNE scatter of learned features (skipped if scikit-learn lacks it)."""
    try:
        from sklearn.manifold import TSNE
    except Exception:
        return None

    if len(features) > max_points:
        keep = np.random.choice(len(features), size=max_points, replace=False)
        features, labels = features[keep], labels[keep]

    # t-SNE needs perplexity < n_samples; cap it so small (e.g. smoke-test) sets
    # don't crash, and skip entirely when there are too few points to embed.
    n_samples = len(features)
    if n_samples < 3:
        return None
    perplexity = min(30, n_samples - 1)
    embedding = TSNE(n_components=2, init="pca", learning_rate="auto",
                     perplexity=perplexity).fit_transform(features)
    fig, ax = plt.subplots(figsize=(8, 7))
    scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap="tab20", s=6, alpha=0.7)
    ax.set_title("t-SNE of learned features")
    fig.colorbar(scatter, ax=ax, label="class id")
    return _save(fig, out_dir, "tsne.png")
