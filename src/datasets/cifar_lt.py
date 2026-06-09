"""Loading and grouping for the prepared CIFAR-100-LT dataset.

The dataset on disk follows an ``ImageFolder`` layout produced by
``data/prepare_datasets.py``::

    CIFAR-100-LT/
      train/class_000/ ... class_099/
      test/class_000/  ... class_099/
      class_counts.json   # train images kept per class (the long-tail profile)

Folder names sort to 0..99, so the ImageFolder label equals the class id.
"""

from __future__ import annotations

import copy
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder

# CIFAR-100 channel statistics, kept identical to the original baseline.
CIFAR_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR_STD = (0.2673, 0.2564, 0.2762)


def build_transforms(train: bool, image_size: int = 32) -> transforms.Compose:
    """CIFAR augmentation for training; plain normalisation for eval.

    ``image_size`` stays 32 for the main from-scratch setup. Set it to 224 for the
    optional pretrained-ImageNet setup, which needs the larger input
    (``build_encoder(pretrained=True)``).
    """
    resize = [] if image_size == 32 else [transforms.Resize(image_size)]
    if train:
        steps = resize + [
            transforms.RandomCrop(image_size, padding=image_size // 8),
            transforms.RandomHorizontalFlip(),
        ]
    else:
        steps = resize
    steps += [transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)]
    return transforms.Compose(steps)


class TwoCropTransform:
    """Return two independent augmentations of one image (for SupCon)."""

    def __init__(self, base_transform: Callable) -> None:
        self.base_transform = base_transform

    def __call__(self, image) -> list:
        return [self.base_transform(image), self.base_transform(image)]


def load_split(data_dir: Path, split: str, transform: transforms.Compose) -> ImageFolder:
    """Load the ``train`` or ``test`` split as an ImageFolder dataset."""
    return ImageFolder(Path(data_dir) / split, transform=transform)


def subset(dataset, indices: list[int]):
    """Shallow copy of an ImageFolder keeping only ``indices``.

    The transform and all other attributes are shared with the original; only the
    sample list is narrowed, so ``.samples`` / ``.targets`` (needed by the episodic
    sampler and the balanced sampler) stay consistent.
    """
    new = copy.copy(dataset)
    new.samples = [dataset.samples[i] for i in indices]
    new.targets = [dataset.targets[i] for i in indices]
    if hasattr(dataset, "imgs"):
        new.imgs = new.samples
    return new


def split_indices_by_class(targets: list[int], val_fraction: float = 0.1,
                           seed: int = 42) -> tuple[list[int], list[int]]:
    """Stratified per-class split -> (train_indices, val_indices).

    A fraction of each class is held out for validation (model selection). Every
    class keeps at least one training sample, so the rarest tail classes may
    contribute 0 images to validation rather than losing their only example.
    """
    rng = random.Random(seed)
    by_class: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(targets):
        by_class[int(label)].append(index)

    train_indices, val_indices = [], []
    for indices in by_class.values():
        indices = indices[:]
        rng.shuffle(indices)
        n_val = min(int(len(indices) * val_fraction), len(indices) - 1)  # keep >=1 for train
        val_indices.extend(indices[:n_val])
        train_indices.extend(indices[n_val:])
    return sorted(train_indices), sorted(val_indices)


def make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int = 2) -> DataLoader:
    """Create a DataLoader with sensible defaults for CPU/GPU and Kaggle."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def make_balanced_loader(dataset, batch_size: int, num_workers: int = 2) -> DataLoader:
    """A loader that draws every CLASS with equal probability (class-balanced).

    On long-tail data a normal shuffled loader is *instance*-balanced: head
    classes dominate every batch. This sampler weights each image by ``1 / (count
    of its class)``, so over an epoch every class is seen about equally often.
    This is exactly what the decoupling / cRT stage needs to de-bias the
    classifier without touching the encoder.
    """
    targets = np.array([label for _, label in dataset.samples])
    class_counts = np.bincount(targets)
    weights = 1.0 / class_counts[targets]
    sampler = WeightedRandomSampler(weights, num_samples=len(targets), replacement=True)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def limit_dataset(dataset, max_samples: int | None):
    """Optionally keep only the first ``max_samples`` items (smoke testing)."""
    if max_samples is None or max_samples >= len(dataset):
        return dataset
    return Subset(dataset, list(range(max_samples)))


def load_class_counts(data_dir: Path) -> dict[int, int]:
    """Read ``class_counts.json`` and return an ``{int class_id: count}`` map."""
    raw = json.loads((Path(data_dir) / "class_counts.json").read_text(encoding="utf-8"))
    return {int(class_id): int(count) for class_id, count in raw.items()}


def split_shot_groups(
    class_counts: dict[int, int],
    many_threshold: int = 100,
    few_threshold: int = 20,
) -> dict[str, list[int]]:
    """Partition class ids into many / medium / few-shot groups by train count.

    Convention follows the long-tail literature: ``many`` = more than
    ``many_threshold`` training images, ``few`` = fewer than ``few_threshold``,
    everything else is ``medium``.
    """
    groups: dict[str, list[int]] = {"many": [], "medium": [], "few": []}
    for class_id, count in class_counts.items():
        if count > many_threshold:
            groups["many"].append(class_id)
        elif count < few_threshold:
            groups["few"].append(class_id)
        else:
            groups["medium"].append(class_id)
    return groups
