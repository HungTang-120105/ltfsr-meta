# LTFSR-Meta — Long-Tailed Recognition: from-scratch vs. foundation models

A small, readable research repository for long-tailed recognition (a few "head"
classes have many images, many "tail" classes have very few), studied on **two
datasets**: **CIFAR-100-LT** (IF=100) and **CUB-200-LT** (fine-grained, IF=10).

The study runs in **three tracks** and asks, for the few-image tail, *where should the
knowledge come from?*

- **Full results & analysis (markdown):** **[`REPORT.md`](REPORT.md)**.
- **Formal report (LaTeX → PDF, in Vietnamese, course IT3190):** **[`Report_tex/`](Report_tex/)**
  — compile `Report_tex/main.tex` (e.g. `pdflatex main.tex` ×2, or `latexmk -pdf main.tex`) to
  produce `main.pdf`. The report covers the problem, data analysis (EDA), theory of every method,
  evaluation protocol/metrics, results, ablations and conclusions.

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
│   ├── datasets/        # cifar_lt loaders + class groups, episodic sampler, augment
│   ├── models/          # shared encoder, baseline, prototype, projection heads
│   ├── trainers/        # one file per Track-1 method + shared engine/classifier loops
│   ├── experts/         # Track-3 foundation-model experts: CLIP, Tip-Adapter, LIFT,
│   │                    #   DINOv2, LLM prompts (CuPL), feature diffusion/mixup, GLA
│   ├── evaluation/      # imbalanced metrics, ensemble/posthoc/fusion + all plots
│   └── utils/           # seeding, experiment bookkeeping
├── data/
│   ├── prepare_datasets.py    # build CIFAR-100-LT (ImageFolder layout)
│   ├── prepare_cub_lt.py      # build CUB-200-LT (fine-grained, IF=10)
│   └── validate_cifar_lt.py   # sanity-check the prepared dataset
├── docs/                # 01_baseline … 07_clip_adaptation (per-method algorithms)
├── guides/              # 00_muc_tieu … 05_new_dataset (goals, how-to-run, per-phase)
├── notebooks/
│   ├── run_pipeline.ipynb            # all 4 phases in one file (toggles RUN_PHASE1/0/2/3)
│   ├── run_experiment.ipynb          # run ONE method (set METHOD, Run All)
│   ├── run_all_methods.ipynb         # run ALL Track-1 methods + comparison plots
│   ├── phase0_reuse.ipynb            # reuse checkpoints: ensemble / fusion / τ-norm / CLIP
│   ├── phase2_clip_adapt.ipynb       # adapt frozen CLIP: Tip-Adapter / LIFT
│   ├── phase3_knowledge_sources.ipynb# research study: LLM vs DINOv2 vs diffusion + GLA + fusion
│   └── visualize_cifar_lt.ipynb      # dataset EDA figures
├── Report_tex/          # formal report (LaTeX): main.tex + chapters/ → main.pdf
├── REPORT.md            # results & analysis (markdown deliverable)
├── outputs/             # per-run results + result CSVs (cifar/, cub_200_2011/)
└── requirements.txt
```

## The datasets

`data/prepare_datasets.py` downloads CIFAR-100 via torchvision and exports a
long-tail split as an `ImageFolder`:

```
CIFAR-100-LT/
├── class_counts.json          # train images kept per class (the LT profile)
├── class_names.json           # class names in label order (for CLIP / LLM)
├── train/class_000 … class_099/
└── test/class_000  … class_099/
```

Head class ≈ 500 images, tail class ≈ 5 images (imbalance factor 100); the test set is
balanced at 100/class.

`data/prepare_cub_lt.py` builds **CUB-200-LT** in the same layout — 200 bird species
(fine-grained), head 50 / tail 5 (IF=10), balanced test 10/class. Any dataset that follows
this `ImageFolder + class_counts.json + class_names.json` layout works with no code changes
(see `guides/05_new_dataset.md`).

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

# 3a. Run ONE method quickly — open the notebook and Run All
jupyter notebook notebooks/run_experiment.ipynb

# 3b. Or run the WHOLE study (all 4 phases) in one file
jupyter notebook notebooks/run_pipeline.ipynb
```

The notebook is the single entry point: set the config in the first cell, choose
a method (or toggle the phases), and Run All. All training logic lives in `src/` —
nothing is duplicated in the notebook.

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
