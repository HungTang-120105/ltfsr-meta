"""Validate the prepared CIFAR-100-LT dataset layout.

Checks:
- expected root folders exist
- train/test class folders exist
- manifest row counts match actual image counts
- class_count metadata is internally consistent
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate CIFAR-100-LT dataset layout.")
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("./data/CIFAR-100-LT"),
        help="Path to the prepared CIFAR-100-LT root directory.",
    )
    return parser


def read_manifest(manifest_path: Path) -> list[tuple[int, int]]:
    rows: list[tuple[int, int]] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append((int(row["image_index"]), int(row["label"])))
    return rows


def count_images_by_class(split_dir: Path) -> Counter[int]:
    counts: Counter[int] = Counter()
    for class_dir in split_dir.iterdir():
        if not class_dir.is_dir() or not class_dir.name.startswith("class_"):
            continue
        label = int(class_dir.name.split("_", maxsplit=1)[1])
        counts[label] = sum(1 for path in class_dir.iterdir() if path.is_file())
    return counts


def main() -> None:
    args = build_parser().parse_args()
    root = args.data_dir.expanduser().resolve()

    train_dir = root / "train"
    test_dir = root / "test"
    train_manifest = root / "train_manifest.csv"
    test_manifest = root / "test_manifest.csv"
    class_counts_file = root / "class_counts.json"

    missing = [path for path in [train_dir, test_dir, train_manifest, test_manifest, class_counts_file] if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required dataset paths: " + ", ".join(str(path) for path in missing))

    train_rows = read_manifest(train_manifest)
    test_rows = read_manifest(test_manifest)
    class_counts = json.loads(class_counts_file.read_text(encoding="utf-8"))

    train_counts = count_images_by_class(train_dir)
    test_counts = count_images_by_class(test_dir)

    expected_train_total = sum(int(value) for value in class_counts.values())
    actual_train_total = sum(train_counts.values())
    actual_test_total = sum(test_counts.values())

    if expected_train_total != actual_train_total:
        raise ValueError(f"Train total mismatch: manifest expects {expected_train_total}, filesystem has {actual_train_total}")

    if expected_train_total != len(train_rows):
        raise ValueError(f"Train manifest row count mismatch: {len(train_rows)} vs expected {expected_train_total}")

    if actual_test_total != len(test_rows):
        raise ValueError(f"Test manifest row count mismatch: {len(test_rows)} vs filesystem {actual_test_total}")

    if len(train_counts) != 100 or len(test_counts) != 100:
        raise ValueError("Expected 100 class folders in both train and test splits")

    print("CIFAR-100-LT validation passed")
    print(f"Train images: {actual_train_total}")
    print(f"Test images: {actual_test_total}")
    print(f"Head class count: {class_counts['0']}")
    print(f"Tail class count: {class_counts['99']}")


if __name__ == "__main__":
    main()