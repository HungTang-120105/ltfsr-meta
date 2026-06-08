# LTFSR-Meta — Long-Tail Few-Shot Recognition on CIFAR-100-LT

A small, readable research repository for long-tailed recognition (a few "head"
classes have many images, many "tail" classes have very few). The study makes one
point: **on long-tail data the bottleneck is the classifier, not the features.**
Each method fixes the imbalance at a different level, so the **balanced / few-shot
accuracy** rises step by step:

| # | `METHOD` | Fixes at level | Idea | Code |
|---|----------|----------------|------|------|
| 1 | `baseline` | — | ResNet + linear softmax head + cross-entropy (head-biased reference) | `src/trainers/baseline_trainer.py` |
| 2 | `balanced_softmax` | loss | Add the class-prior to the logits while training (Ren et al. 2020) | `src/trainers/losses.py` |
| 3 | `decoupling` | classifier | Train features, then re-train the head on a class-balanced sampler — cRT (Kang et al. 2020) | `src/trainers/decoupling_trainer.py` |
| 4 | `supcon` | representation | SupCon features + cRT classifier — **best** | `src/trainers/contrastive_trainer.py` |
| 5 | `meta` | (bonus) | Episodic Prototypical Networks, reported on the few-shot N-way axis | `src/trainers/meta_trainer.py` |

Each method has a one-page explanation in [`docs/`](docs/). The methods share one
encoder, one dataset module, and one metrics/visualisation suite, so the
comparison is fair and the code stays small.

> **Headline metric = `balanced_accuracy` and `few_shot_accuracy`, not plain top-1.**
> On overall top-1 (dominated by head classes) a well-trained cross-entropy baseline
> is hard to beat; the long-tail story shows up on the balanced/tail metrics.

### Backbone setups

- **Main (default):** `PRETRAINED=False`, `IMAGE_SIZE=32` — a CIFAR-stem ResNet-18
  (3×3 conv, no max-pool) trained from scratch. This is the standard CIFAR-LT
  protocol, so the numbers are comparable to the literature.
- **Optional reference table:** `PRETRAINED=True`, `IMAGE_SIZE=224` — ImageNet
  ResNet-18 on up-scaled images. Higher absolute numbers, but transfer-learning,
  so report it only as a secondary table.

> The full research background (literature, roadmap) lives in `Summary.md`.

## Repository structure

```
ltfsr-meta/
├── src/
│   ├── datasets/        # cifar_lt loaders + class groups, episodic sampler
│   ├── models/          # shared encoder, baseline, prototype, projection heads
│   ├── trainers/        # one file per method + shared engine/classifier loops
│   ├── evaluation/      # imbalanced metrics + all plots
│   └── utils/           # seeding, experiment bookkeeping
├── data/
│   ├── prepare_datasets.py    # build CIFAR-100-LT (ImageFolder layout)
│   └── validate_cifar_lt.py   # sanity-check the prepared dataset
├── docs/                # 01_baseline.md … 05_meta.md (the algorithms)
├── notebooks/
│   └── run_experiment.ipynb   # the ONLY file you run on Kaggle
├── outputs/             # per-run results (gitignored)
└── requirements.txt
```

## The dataset

`data/prepare_datasets.py` downloads CIFAR-100 via torchvision and exports a
long-tail split as an `ImageFolder`:

```
CIFAR-100-LT/
├── class_counts.json          # train images kept per class (the LT profile)
├── train/class_000 … class_099/
└── test/class_000  … class_099/
```

Head class ≈ 500 images, tail class ≈ 5 images (imbalance factor 100).

## Metrics (built for imbalance)

Accuracy alone is misleading on long-tail data, so every run reports
(`src/evaluation/metrics.py`): **Accuracy, Balanced Accuracy, Macro
Precision/Recall/F1, Weighted F1, Many/Medium/Few-shot accuracy, G-Mean, MCC**.

## Visualisations (saved automatically to the run folder)

Training curves (loss & accuracy), raw + normalised confusion matrices, class
frequency histogram, head/medium/tail group sizes, and a t-SNE of the learned
features.

---

## Run locally

```bash
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 1. Build the dataset (once)
python data/prepare_datasets.py --dataset cifar100-lt --data_dir ./data --overwrite

# 2. Check it
python data/validate_cifar_lt.py --data_dir ./data/CIFAR-100-LT

# 3. Run an experiment — open the notebook and Run All
jupyter notebook notebooks/run_experiment.ipynb
```

The notebook is the single entry point: set the config in the first cell, choose
a method, and Run All. All training logic lives in `src/` — nothing is duplicated
in the notebook.

## Run on Kaggle

1. **Add the data.** Either upload the prepared `CIFAR-100-LT/` folder as a
   Kaggle *Dataset* (it appears under `/kaggle/input/...`), or let the notebook
   build it into `/kaggle/working` (set `BUILD_DATASET = True`).
2. **Add the code.** Upload the repo as a Kaggle Dataset/Utility, or
   `!git clone` it in the first cell. The notebook auto-detects where `src/`
   lives and adds it to `sys.path`.
3. **Open `notebooks/run_experiment.ipynb`**, set `DATA_DIR` /
   `OUTPUT_DIR` / `METHOD` in the config cell, enable GPU, and **Run All**.

Set `MAX_TRAIN_SAMPLES` / `MAX_TEST_SAMPLES` (and small `EPOCHS`) for a quick
smoke test before a full run.

## Reproducibility

`src/utils/seed.py` seeds Python, NumPy and PyTorch (CPU + CUDA) and enables
deterministic cuDNN. Each run writes its `config.json`, `metrics.csv`,
`metrics.json`, the best checkpoint, and all figures into its own folder under
`outputs/`.
