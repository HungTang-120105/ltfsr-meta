"""Convert CUB-200-2011 into a long-tail ImageFolder layout identical to CIFAR-100-LT.

CUB-200-2011 has ~60 images/class total but its official train split is only ~30/class.
To get a richer long-tail train set we use **all** images: hold out a small **balanced**
test set (``--test_per_class``) and subsample the remaining pool into an exponential
long-tail train (same formula as CIFAR-100-LT). A balanced test means
``accuracy == balanced_accuracy`` just like CIFAR.

    CUB-200-LT/
      train/class_000/ ... class_199/    # long-tail (subsampled)
      test/class_000/  ... class_199/    # balanced (test_per_class each)
      class_counts.json   # train images per class (the long-tail profile)
      class_names.json    # readable bird names in label order (for CLIP / LLM)

CUB's pool caps the imbalance: with ~50 train/class available, ``max_images=50`` & IF=10
gives head 50 / tail 5. Pass ``--imbalance_factor`` / ``--max_images`` to tune.

Usage:
    python data/prepare_cub_lt.py --imbalance_factor 10 --overwrite
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

NUM_CLASSES = 200


def find_cub_root(start: Path) -> Path:
    """Locate the folder that directly contains ``images.txt`` (handles nesting)."""
    if (start / "images.txt").exists():
        return start
    hits = list(start.rglob("images.txt"))
    if not hits:
        raise SystemExit(f"images.txt not found under {start} — is CUB_200_2011 there?")
    return hits[0].parent


def read_pairs(path: Path) -> dict[str, str]:
    pairs = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            key, value = line.split(maxsplit=1)
            pairs[key] = value.strip()
    return pairs


def readable_name(raw: str) -> str:
    name = raw.split(".", 1)[1] if "." in raw else raw
    return name.replace("_", " ").strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cub_root", default="data/CUB_200_2011")
    parser.add_argument("--out_dir", default="data/CUB-200-LT")
    parser.add_argument("--imbalance_factor", type=float, default=10.0)
    parser.add_argument("--max_images", type=int, default=50, help="head-class train size (capped at available)")
    parser.add_argument("--test_per_class", type=int, default=10, help="balanced held-out test per class")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = find_cub_root(Path(args.cub_root))
    images_dir = root / "images"
    out_dir = Path(args.out_dir)
    if out_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"{out_dir} exists; pass --overwrite to rebuild.")
        shutil.rmtree(out_dir)

    class_raw = read_pairs(root / "classes.txt")
    img_path = read_pairs(root / "images.txt")
    img_label = read_pairs(root / "image_class_labels.txt")
    class_names = [readable_name(class_raw[str(c)]) for c in range(1, NUM_CLASSES + 1)]

    # all image paths grouped by 0-based class id (ignore the official split)
    by_class: dict[int, list[str]] = defaultdict(list)
    for image_id, rel in img_path.items():
        by_class[int(img_label[image_id]) - 1].append(rel)

    rng = random.Random(args.seed)
    class_counts: dict[str, int] = {}
    test_total = 0

    for cls0 in range(NUM_CLASSES):
        pool = by_class[cls0][:]
        rng.shuffle(pool)
        n_test = min(args.test_per_class, len(pool) - 1)          # keep >=1 for train
        test_imgs, train_pool = pool[:n_test], pool[n_test:]

        exponent = cls0 / float(NUM_CLASSES - 1)
        target = int(round(args.max_images * (args.imbalance_factor ** (-exponent))))
        keep = min(max(target, 1), len(train_pool))
        train_imgs = train_pool[:keep]
        class_counts[str(cls0)] = keep

        dtr = out_dir / "train" / f"class_{cls0:03d}"; dtr.mkdir(parents=True, exist_ok=True)
        for rel in train_imgs:
            shutil.copy2(images_dir / rel, dtr / Path(rel).name)
        dte = out_dir / "test" / f"class_{cls0:03d}"; dte.mkdir(parents=True, exist_ok=True)
        for rel in test_imgs:
            shutil.copy2(images_dir / rel, dte / Path(rel).name)
            test_total += 1

    (out_dir / "class_counts.json").write_text(json.dumps(class_counts, indent=2), encoding="utf-8")
    (out_dir / "class_names.json").write_text(
        json.dumps(class_names, indent=2, ensure_ascii=False), encoding="utf-8")

    counts = list(class_counts.values())
    print(f"CUB-200-LT written to {out_dir}  (source: {root})")
    print(f"  classes: {NUM_CLASSES} | IF: {args.imbalance_factor}")
    print(f"  train: {sum(counts)} imgs (head={max(counts)}, tail={min(counts)})")
    print(f"  test:  {test_total} imgs (balanced ~{args.test_per_class}/class)")
    print(f"  many(>15): {sum(c > 15 for c in counts)} | medium(6-15): {sum(6 <= c <= 15 for c in counts)} "
          f"| few(<6): {sum(c < 6 for c in counts)}")


if __name__ == "__main__":
    main()
