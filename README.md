# LTFSR-Meta — Long-Tailed Recognition: from-scratch vs. foundation models

A small, readable research repository for long-tailed recognition (a few "head"
classes have many images, many "tail" classes have very few), studied on **two
datasets**: **CIFAR-100-LT** (IF=100) and **CUB-200-LT** (fine-grained, IF=10).

The study runs in **three tracks** and asks, for the few-image tail, *where should the
knowledge come from?* Full results & analysis: **[`REPORT.md`](REPORT.md)**.

1. **From-scratch** (Track 1) — ResNet-18 with a tail-aware fix at each level:
   `baseline` (CE) → `balanced_softmax` (loss) → `decoupling`/cRT (classifier) →
   `supcon` (representation) → `meta` (ProtoNet) → **`cmo`** (data: tail-rich CutMix +
   Balanced-Softmax; **best from-scratch**, ≈0.47 bal-acc on CIFAR).
2. **Reuse** (Track 2 / Phase 0) — no retraining: `ensemble`(+TTA), `tier_fusion`,
   `tau_norm`, CLIP `vlm_fusion`.
3. **Foundation models** (Track 3 / Phase 2–3) — adapt a **frozen** CLIP / DINOv2 by
   training only a tiny head/adapter: `clip_zeroshot`, `tip_adapter`, **`lift`**,
   **`dino_lift`**, LLM-enriched prototypes, feature diffusion/mixup, GLA, and a
   tail-aware **`fusion`**. Main contribution and strongest result (**≈0.82 CIFAR /
   0.85 CUB**); a second vision FM (**DINOv2**) helps the tail most.

Per-method explanations in [`docs/`](docs/); how-to & per-phase guides in
[`guides/`](guides/). Methods share one dataset module, metrics and visualisation
suite, so the comparison is fair and the code stays small.

> **Headline metric = `balanced_accuracy` and `few_shot_accuracy`, not plain top-1.**
> Test sets are balanced (100/class CIFAR, 10/class CUB) → `accuracy == balanced_accuracy`.

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
│   ├── run_experiment.ipynb   # run ONE method (set METHOD, Run All)
│   └── run_all_methods.ipynb  # run ALL methods + comparison plots in one pass
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
