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

import json
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

# CIFAR-100 channel statistics, kept identical to the original baseline.
CIFAR_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR_STD = (0.2673, 0.2564, 0.2762)


def build_transforms(train: bool) -> transforms.Compose:
    """Standard CIFAR augmentation for training; plain normalisation for eval."""
    if train:
        return transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )


class TwoCropTransform:
    """Return two independent augmentations of one image (for SupCon)."""

    def __init__(self, base_transform: Callable) -> None:
        self.base_transform = base_transform

    def __call__(self, image) -> list:
        return [self.base_transform(image), self.base_transform(image)]


def load_split(data_dir: Path, split: str, transform: transforms.Compose) -> ImageFolder:
    """Load the ``train`` or ``test`` split as an ImageFolder dataset."""
    return ImageFolder(Path(data_dir) / split, transform=transform)


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
