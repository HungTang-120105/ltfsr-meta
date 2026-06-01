"""Episode sampling for episodic (meta-learning) training.

A single episode is a small N-way classification task: pick ``n_way`` classes,
show the model ``k_shot`` labelled *support* images per class, then ask it to
classify ``n_query`` held-out *query* images of those same classes. Training on
thousands of such episodes teaches the encoder to form good prototypes from very
few examples — exactly the few-shot skill needed for tail classes
(see docs/04_meta_learning.md).
"""

from __future__ import annotations

import random
from collections import defaultdict


def build_class_index(targets: list[int]) -> dict[int, list[int]]:
    """Map each class id to the list of dataset indices belonging to it."""
    class_index: dict[int, list[int]] = defaultdict(list)
    for sample_index, label in enumerate(targets):
        class_index[int(label)].append(sample_index)
    return dict(class_index)


def eligible_classes(class_index: dict[int, list[int]], min_images: int = 2) -> list[int]:
    """Classes with at least ``min_images`` samples (need >=1 support + 1 query)."""
    return [class_id for class_id, indices in class_index.items() if len(indices) >= min_images]


def sample_episode(
    class_index: dict[int, list[int]],
    n_way: int,
    k_shot: int,
    n_query: int,
    rng: random.Random,
) -> tuple[list[int], list[int], list[int], list[int]]:
    """Sample one N-way K-shot episode.

    Tail classes with fewer than ``k_shot + 1`` images still take part with a
    smaller support set, so the long-tail nature of the data is preserved rather
    than hidden by dropping rare classes. Classes are relabelled to 0..N-1 for
    the episode.

    Returns:
        support_indices, support_labels, query_indices, query_labels — where the
        labels are episode-local (0..n_way-1) and aligned with their index lists.
    """
    pool = eligible_classes(class_index, min_images=2)
    chosen_classes = rng.sample(pool, k=min(n_way, len(pool)))

    support_indices: list[int] = []
    support_labels: list[int] = []
    query_indices: list[int] = []
    query_labels: list[int] = []

    for episode_label, class_id in enumerate(chosen_classes):
        indices = class_index[class_id][:]
        rng.shuffle(indices)

        support_k = min(k_shot, len(indices) - 1)
        support_indices.extend(indices[:support_k])
        support_labels.extend([episode_label] * support_k)

        remaining = indices[support_k:]
        query_k = min(n_query, len(remaining))
        query_indices.extend(remaining[:query_k])
        query_labels.extend([episode_label] * query_k)

    return support_indices, support_labels, query_indices, query_labels
