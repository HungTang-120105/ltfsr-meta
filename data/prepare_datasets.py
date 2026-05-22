"""Prepare CIFAR datasets for ImageFolder-based training.

The repository training code expects a directory layout like:

    dataset_root/
    ├── train/
    │   ├── class_000/
    │   └── ...
    └── test/
        ├── class_000/
        └── ...

This script downloads CIFAR-10 or CIFAR-100 via torchvision, exports the
images into that layout, and writes simple metadata files so the split can be
validated later.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from torchvision.datasets import CIFAR10, CIFAR100


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    torch_dataset: type
    num_classes: int
    default_root_name: str
    description: str


DATASETS: dict[str, DatasetSpec] = {
    "cifar10": DatasetSpec(
        name="CIFAR-10",
        torch_dataset=CIFAR10,
        num_classes=10,
        default_root_name="CIFAR-10",
        description="Standard CIFAR-10 exported to an ImageFolder layout.",
    ),
    "cifar100": DatasetSpec(
        name="CIFAR-100",
        torch_dataset=CIFAR100,
        num_classes=100,
        default_root_name="CIFAR-100",
        description="Standard CIFAR-100 exported to an ImageFolder layout.",
    ),
    "cifar100-lt": DatasetSpec(
        name="CIFAR-100-LT",
        torch_dataset=CIFAR100,
        num_classes=100,
        default_root_name="CIFAR-100-LT",
        description="CIFAR-100 long-tail split exported to an ImageFolder layout.",
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare CIFAR datasets for training.")
    parser.add_argument("--dataset", choices=sorted(DATASETS.keys()), default="cifar100-lt")
    parser.add_argument("--data_dir", type=Path, required=True, help="Where to write the prepared dataset.")
    parser.add_argument(
        "--raw_dir",
        type=Path,
        default=None,
        help="Optional cache directory for the raw torchvision dataset.",
    )
    parser.add_argument(
        "--imbalance_factor",
        type=float,
        default=100.0,
        help="Long-tail imbalance factor used only for cifar100-lt.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for long-tail shuffling.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove any existing prepared dataset directory before writing a new one.",
    )
    return parser


def ensure_clean_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and overwrite:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def save_image(image_array, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image_array).save(destination)


def write_manifest(manifest_path: Path, rows: list[tuple[int, int, str]]) -> None:
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_index", "label", "relative_path"])
        writer.writerows(rows)


def export_split(dataset, indices: list[int], split_dir: Path) -> list[tuple[int, int, str]]:
    rows: list[tuple[int, int, str]] = []
    for output_index, dataset_index in enumerate(indices):
        label = int(dataset.targets[dataset_index])
        class_dir = split_dir / f"class_{label:03d}"
        file_name = f"{output_index:06d}.png"
        image_path = class_dir / file_name
        save_image(dataset.data[dataset_index], image_path)
        rows.append((dataset_index, label, f"class_{label:03d}/{file_name}"))
    return rows


def build_long_tail_indices(targets: list[int], num_classes: int, imbalance_factor: float, seed: int) -> tuple[list[int], dict[str, int]]:
    class_indices: dict[int, list[int]] = {class_id: [] for class_id in range(num_classes)}
    for sample_index, label in enumerate(targets):
        class_indices[int(label)].append(sample_index)

    rng = random.Random(seed)
    selected_indices: list[int] = []
    class_counts: dict[str, int] = {}
    max_images = max(len(indices) for indices in class_indices.values())

    for class_id in range(num_classes):
        shuffled = list(class_indices[class_id])
        rng.shuffle(shuffled)
        if num_classes == 1:
            class_size = max_images
        else:
            exponent = class_id / float(num_classes - 1)
            class_size = int(max_images * (imbalance_factor ** (-exponent)))
        class_size = max(1, class_size)
        class_counts[str(class_id)] = class_size
        selected_indices.extend(shuffled[:class_size])

    rng.shuffle(selected_indices)
    return selected_indices, class_counts


def export_cifar_dataset(spec: DatasetSpec, data_dir: Path, raw_dir: Path, imbalance_factor: float, seed: int, overwrite: bool) -> Path:
    dataset_root = data_dir / spec.default_root_name
    ensure_clean_dir(dataset_root, overwrite)

    train_dir = dataset_root / "train"
    test_dir = dataset_root / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = spec.torch_dataset(root=str(raw_dir), train=True, download=True)
    test_dataset = spec.torch_dataset(root=str(raw_dir), train=False, download=True)

    if spec.name == "CIFAR-100-LT":
        train_indices, class_counts = build_long_tail_indices(
            targets=list(train_dataset.targets),
            num_classes=spec.num_classes,
            imbalance_factor=imbalance_factor,
            seed=seed,
        )
    else:
        train_indices = list(range(len(train_dataset.targets)))
        class_counts = {str(class_id): 0 for class_id in range(spec.num_classes)}
        for label in train_dataset.targets:
            class_counts[str(int(label))] += 1

    train_rows = export_split(train_dataset, train_indices, train_dir)
    test_rows = export_split(test_dataset, list(range(len(test_dataset.targets))), test_dir)

    write_manifest(dataset_root / "train_manifest.csv", train_rows)
    write_manifest(dataset_root / "test_manifest.csv", test_rows)

    metadata = {
        "dataset": spec.name,
        "num_classes": spec.num_classes,
        "imbalance_factor": imbalance_factor if spec.name == "CIFAR-100-LT" else None,
        "seed": seed if spec.name == "CIFAR-100-LT" else None,
        "train_images": len(train_rows),
        "test_images": len(test_rows),
        "class_counts": class_counts,
    }
    with (dataset_root / "class_counts.json").open("w", encoding="utf-8") as handle:
        json.dump(class_counts, handle, indent=2, ensure_ascii=False)
    with (dataset_root / "dataset_info.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)

    return dataset_root


def main() -> None:
    args = build_parser().parse_args()
    spec = DATASETS[args.dataset]

    data_dir = args.data_dir.expanduser().resolve()
    raw_dir = (args.raw_dir or (data_dir / "raw")).expanduser().resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    dataset_root = export_cifar_dataset(
        spec=spec,
        data_dir=data_dir,
        raw_dir=raw_dir,
        imbalance_factor=args.imbalance_factor,
        seed=args.seed,
        overwrite=args.overwrite,
    )

    print(f"Prepared {spec.name} at: {dataset_root}")
    print(spec.description)
    print("Training scripts can point to the root above; it contains train/, test/, train_manifest.csv, test_manifest.csv, and class_counts.json.")


if __name__ == "__main__":
    main()