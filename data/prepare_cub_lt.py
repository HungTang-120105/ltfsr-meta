"""Convert CUB-200-2011 into a long-tail ImageFolder layout identical to CIFAR-100-LT.

CUB-200-2011 is roughly balanced (~30 train images per class). We subsample the
**train** split into an exponential long-tail profile (same formula as CIFAR-100-LT),
keep the **test** split intact, and export to::

    CUB-200-LT/
      train/class_000/ ... class_199/
      test/class_000/  ... class_199/
      class_counts.json    # {"0": head_count, ..., "199": tail_count}  (train profile)
      class_names.json     # readable class names in label order (for CLIP / LLM prompts)

So every notebook can run on it by only changing DATA_DIR / NUM_CLASSES / class names —
the data-loading code (ImageFolder) is unchanged.

CUB's ~30 images/class caps the achievable imbalance: with max=30 and IF=10 the tail has
3 images (IF=20 -> ~1-2). Pass --imbalance_factor to choose; default 10.

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


def read_pairs(path: Path) -> dict[str, str]:
    """Read a CUB `<id> <value>` file into {id: value}."""
    pairs = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        key, value = line.split(maxsplit=1)
        pairs[key] = value.strip()
    return pairs


def readable_name(raw: str) -> str:
    """`001.Black_footed_Albatross` -> `Black footed Albatross`."""
    name = raw.split(".", 1)[1] if "." in raw else raw
    return name.replace("_", " ").strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cub_root", default="data/CUB_200_2011/CUB_200_2011",
                        help="Folder containing classes.txt / images.txt / images/ ...")
    parser.add_argument("--out_dir", default="data/CUB-200-LT")
    parser.add_argument("--imbalance_factor", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cub_root = Path(args.cub_root)
    out_dir = Path(args.out_dir)
    images_dir = cub_root / "images"
    if out_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"{out_dir} exists; pass --overwrite to rebuild.")
        shutil.rmtree(out_dir)

    # --- parse CUB metadata ---
    class_raw = read_pairs(cub_root / "classes.txt")          # class_id(1-200) -> "001.Name"
    img_path = read_pairs(cub_root / "images.txt")            # image_id -> rel path
    img_label = read_pairs(cub_root / "image_class_labels.txt")  # image_id -> class_id
    img_split = read_pairs(cub_root / "train_test_split.txt")    # image_id -> "1"(train)/"0"(test)

    class_names = [readable_name(class_raw[str(c)]) for c in range(1, NUM_CLASSES + 1)]

    # group image paths by 0-based class id, per split
    train_by_class: dict[int, list[str]] = defaultdict(list)
    test_by_class: dict[int, list[str]] = defaultdict(list)
    for image_id, rel in img_path.items():
        cls0 = int(img_label[image_id]) - 1                   # 0-based label
        (train_by_class if img_split[image_id] == "1" else test_by_class)[cls0].append(rel)

    # --- exponential long-tail profile on train (same formula as CIFAR-100-LT) ---
    rng = random.Random(args.seed)
    max_images = max(len(v) for v in train_by_class.values())  # ~30
    class_counts: dict[str, int] = {}

    for cls0 in range(NUM_CLASSES):
        exponent = cls0 / float(NUM_CLASSES - 1)
        target = int(round(max_images * (args.imbalance_factor ** (-exponent))))
        available = train_by_class[cls0]
        keep = min(max(target, 1), len(available))             # >=1, capped at available
        chosen = rng.sample(available, keep)
        class_counts[str(cls0)] = keep

        dst = out_dir / "train" / f"class_{cls0:03d}"
        dst.mkdir(parents=True, exist_ok=True)
        for rel in chosen:
            shutil.copy2(images_dir / rel, dst / Path(rel).name)

    # --- test split: keep all images (evaluation set) ---
    test_total = 0
    for cls0 in range(NUM_CLASSES):
        dst = out_dir / "test" / f"class_{cls0:03d}"
        dst.mkdir(parents=True, exist_ok=True)
        for rel in test_by_class[cls0]:
            shutil.copy2(images_dir / rel, dst / Path(rel).name)
            test_total += 1

    (out_dir / "class_counts.json").write_text(
        json.dumps(class_counts, indent=2), encoding="utf-8")
    (out_dir / "class_names.json").write_text(
        json.dumps(class_names, indent=2, ensure_ascii=False), encoding="utf-8")

    counts = list(class_counts.values())
    print(f"CUB-200-LT written to {out_dir}")
    print(f"  classes: {NUM_CLASSES} | IF: {args.imbalance_factor}")
    print(f"  train: {sum(counts)} images (head={max(counts)}, tail={min(counts)})")
    print(f"  test:  {test_total} images (kept all)")
    print(f"  many(>15): {sum(c > 15 for c in counts)} | "
          f"medium(6-15): {sum(6 <= c <= 15 for c in counts)} | few(<6): {sum(c < 6 for c in counts)}")


if __name__ == "__main__":
    main()
