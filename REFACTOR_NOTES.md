# LTFSR-Meta — Refactor Notes

> A single-file record of the repository review and transformation.
> Read top-to-bottom: **what the repo was → what was wrong → what it is now → how to run it.**

---

## Table of contents

1. [Phase 1 — Understanding the original repo](#phase-1--understanding-the-original-repo)
2. [Phase 2 — Audit (issues found)](#phase-2--audit-issues-found)
3. [What changed (decisions)](#what-changed-decisions)
4. [Final repository tree](#final-repository-tree)
5. [Architecture overview](#architecture-overview)
6. [The four methods](#the-four-methods)
7. [Files added / modified / removed](#files-added--modified--removed)
8. [Metrics & visualizations](#metrics--visualizations)
9. [How to run — local](#how-to-run--local)
10. [How to run — Kaggle](#how-to-run--kaggle)
11. [The notebook (cell-by-cell)](#the-notebook-cell-by-cell)
12. [Verification performed](#verification-performed)
13. [Known limitations / future work](#known-limitations--future-work)

---

## Phase 1 — Understanding the original repo

**Project:** Long-Tail Few-Shot Recognition (LTFSR) on CIFAR-100-LT.
`Summary.md` is a literature/strategy report describing the intended arc
(Baseline → Prototype → Meta-Learning → Contrastive). **Only the baseline
actually ran**; the meta/full configs had no trainers.

| Component | Original file | What it did |
|---|---|---|
| Dataset pipeline | `data/prepare_datasets.py` | Download CIFAR via torchvision, export ImageFolder layout, build long-tail split (exp decay, imbalance 100), write manifests + `class_counts.json` |
| Dataset validation | `data/validate_cifar_lt.py` | Check folders/manifests/counts |
| Model | `models/baseline_model.py` | `resnet18` (fc→Identity) + `Linear(512, 100)` |
| Training | `training/train_baseline.py` | YAML config, ImageFolder loaders, SGD + cosine, eval on **test** each epoch, save best by test acc |
| Validation / Test | *(same as test)* | No separate val split; `train_split: 0.8` was unused |
| Configs | `config/*.yaml` | Only `config_baseline.yaml` had a trainer |
| Entry points | 3 CLI scripts | No notebook existed despite Kaggle being the goal |

**Original workflow:** `prepare_datasets → validate → train_baseline → best_model.pt`
(stdout only; no metrics.csv, no plots, no rich metrics).

---

## Phase 2 — Audit (issues found)

| # | Issue | Why it mattered | Fix |
|---|---|---|---|
| 1 | No reproducibility seeding (`seed: 42` unused) | Runs not reproducible | `src/utils/seed.py` seeds Python/NumPy/Torch/CUDA + deterministic cuDNN |
| 2 | Only top-1 accuracy, averaged per batch | Wrong headline metric for long-tail; numerically sloppy | Full-dataset metric suite (`src/evaluation/metrics.py`) |
| 3 | No visualizations | Research repo needs curves/CM/etc. | `src/evaluation/visualize.py` (7 figures, auto-saved) |
| 4 | No experiment logging | Can't compare runs | history → `metrics.csv`; `config.json` + `metrics.json` per run |
| 5 | Test set used as validation; `train_split` ignored | Looked like leakage | Documented the standard CIFAR-LT protocol (train on LT split, report on balanced test) |
| 6 | Configs with no trainers | Promised features that didn't exist | Meta-learning **implemented** (per your request), one file per method |
| 7 | 50 MB `CIFAR-100-LT.zip` + empty `mini-ImageNet-LT/` stub | Repo bloat / dead stub | Removed both (kept prepared images locally, gitignored) |
| 8 | `requirements.txt` over-specified | Slow/fragile installs | Trimmed to only-imported deps |
| 9 | `setup.py`, root `__init__.py`, empty packages | Enterprise scaffolding | Removed; replaced with `src/` layout |
| 10 | Duplicated DataLoader rebuild, unused `BaselineModelConfig`, `# type: ignore` hack | Harder to read | Single loader path via shared `engine.py`/`classifier.py` |
| 11 | `device: "cuda"` / `num_workers: 8` defaults | Bad on Kaggle CPU | Auto-detect device; `num_workers=2` default |
| 12 | No Kaggle notebook | The #1 stated goal was unmet | `notebooks/run_experiment.ipynb` (single entry point) |

---

## What changed (decisions)

- **Implemented the full method progression** (your choice): `baseline`,
  `prototype`, `contrastive`, `meta` — each in its own trainer file with a
  matching `docs/0X_*.md` explanation, so the algorithm and the improvement are
  visible step by step.
- **Removed** the 50 MB zip and the empty mini-ImageNet stub (your choice); kept
  the prepared `data/CIFAR-100-LT/` images locally (gitignored).
- **Preserved** the original working logic: long-tail split algorithm,
  ResNet-18 + head architecture, SGD + cosine recipe, CIFAR normalization,
  ImageFolder layout. The new meta-learning code reuses the same shared encoder
  and distance helpers — an extension, not a redesign.
- **No over-engineering**: no Hydra/OmegaConf/Lightning/MLflow/W&B/registries/
  ABCs. Config is a plain dict in the notebook; logging is pandas + JSON.

---

## Final repository tree

```
ltfsr-meta/
├── README.md                      # clean academic guide (rewritten, EN)
├── Summary.md                     # research background/roadmap (unchanged)
├── REFACTOR_NOTES.md              # this file
├── requirements.txt               # only-imported deps
├── .gitignore                     # ignores outputs/* + run artifacts
│
├── src/
│   ├── datasets/
│   │   ├── cifar_lt.py            # transforms, loaders, class-count + shot groups
│   │   └── episodic.py            # N-way K-shot episode sampling
│   ├── models/
│   │   ├── backbone.py            # ONE shared ResNet-18 encoder
│   │   ├── baseline.py            # encoder + linear head        (Method 1)
│   │   ├── prototype.py           # encoder + distance head      (Method 2)
│   │   └── projection.py          # SupCon projection head       (Method 3)
│   ├── trainers/
│   │   ├── engine.py              # shared train_one_epoch / evaluate
│   │   ├── classifier.py          # shared SGD+cosine fit loop
│   │   ├── baseline_trainer.py    # Method 1
│   │   ├── prototype_trainer.py   # Method 2
│   │   ├── contrastive_trainer.py # Method 3 (SupCon + linear probe)
│   │   └── meta_trainer.py        # Method 4 (episodic ProtoNet + 100-way eval)
│   ├── evaluation/
│   │   ├── metrics.py             # imbalanced metric suite
│   │   └── visualize.py           # all plots, each saves a PNG
│   └── utils/
│       ├── seed.py                # set_seed + resolve_device
│       └── experiment.py          # run dir, config/metrics/history saving
│
├── data/
│   ├── prepare_datasets.py        # builds CIFAR-100-LT (unchanged)
│   ├── validate_cifar_lt.py       # sanity check (unchanged)
│   └── CIFAR-100-LT/              # prepared images (local only, gitignored)
│
├── docs/
│   ├── 01_baseline.md
│   ├── 02_prototype.md
│   ├── 03_contrastive.md
│   └── 04_meta_learning.md
│
├── notebooks/
│   └── run_experiment.ipynb       # the ONE file to run (20 cells)
│
└── outputs/                       # per-run results (.gitkeep tracked, rest ignored)
```

---

## Architecture overview

```
                          notebooks/run_experiment.ipynb
                      (config + paths only — no training logic)
                                       │
        ┌──────────────────────────────┼───────────────────────────────┐
        ▼                              ▼                                ▼
  src/datasets         ─────►   src/trainers (4 methods)   ─────►  src/evaluation
  cifar_lt: loaders,           baseline ─┐                          metrics.py
  shot groups (many/           prototype ┼─ share engine.py +       visualize.py
  med/few)                     contrastive┘  classifier.py
  episodic: N-way K-shot       meta (episodic, own loop)
                                       │
                                       ▼
                          src/models (ONE shared encoder)
                                       │
                                       ▼
                          src/utils: seed + experiment
                  outputs/<method>/{config.json, metrics.csv,
                  metrics.json, best_model.pt, *.png}
```

All four methods embed images with the **same ResNet-18 encoder**
(`src/models/backbone.py`), so performance differences are attributable to the
method, not the backbone.

---

## The four methods

| # | Method | Idea | Trainer | Doc |
|---|--------|------|---------|-----|
| 1 | **Baseline** | ResNet-18 + linear softmax head + cross-entropy | `baseline_trainer.py` | `docs/01_baseline.md` |
| 2 | **Prototype** | Replace linear head with learnable per-class prototypes; classify by Euclidean distance (`logit = −‖f − pᶜ‖²`) | `prototype_trainer.py` | `docs/02_prototype.md` |
| 3 | **Contrastive (SupCon)** | Stage 1: contrastive pre-train the encoder (two views/image, same-class attract). Stage 2: freeze encoder, train a linear probe | `contrastive_trainer.py` | `docs/03_contrastive.md` |
| 4 | **Meta-learning** | Episodic Prototypical Networks: N-way K-shot episodes, prototypes from support set, classify query by distance. Final eval = full 100-way via global prototypes | `meta_trainer.py` | `docs/04_meta_learning.md` |

**Why this progression matters on long-tail data:** the linear baseline's weight
norms grow with class frequency (head classes dominate, tail under-predicted).
Distance-based scoring (Methods 2 & 4) removes that bias; SupCon (Method 3)
improves the representation geometry; meta-learning explicitly rehearses
few-example learning, which is what tail classes need. Compare `g_mean`,
`balanced_accuracy`, and `few_shot_accuracy` across methods to see the effect.

---

## Files added / modified / removed

### Added
- `src/` package (12 modules + `__init__.py`s): datasets, models, trainers,
  evaluation, utils — see tree above.
- `docs/01–04_*.md` — one explanation page per method.
- `notebooks/run_experiment.ipynb` — single Kaggle entry point.
- `outputs/.gitkeep`, `REFACTOR_NOTES.md` (this file).

### Modified
- `README.md` — rewritten clean English guide.
- `requirements.txt` — removed pytorch-lightning, timm, tensorboard, scipy,
  seaborn, tqdm, pyyaml, black, isort, flake8, jupyter; kept only imported deps.
- `.gitignore` — ignore `outputs/*` (keep `.gitkeep`).

### Removed
- `setup.py`, root `__init__.py`.
- `config/config_baseline.yaml`, `config_meta_learning.yaml`, `config_full_model.yaml`.
- Old top-level `models/`, `training/`, `evaluation/`, `utils/`, `data/__init__.py`
  (logic migrated into `src/`).
- `data/CIFAR-100-LT.zip` (50 MB) and `data/mini-ImageNet-LT/` (empty stub).

---

## Metrics & visualizations

**Metrics** (`src/evaluation/metrics.py`, computed over the full test set):
Accuracy, Balanced Accuracy, Macro Precision / Recall / F1, Weighted F1,
**Many / Medium / Few-shot accuracy** (split by `class_counts`), **G-Mean**, **MCC**.

Shot groups: `many` = >100 train images, `few` = <20, else `medium`
(CIFAR-100-LT: 35 / 35 / 30 classes).

**Visualizations** (`src/evaluation/visualize.py`, each saved as PNG to the run folder):
- `curve_loss.png`, `curve_accuracy.png` (train vs val)
- `confusion_matrix.png`, `confusion_matrix_normalized.png`
- `class_frequency.png` (long-tail profile)
- `shot_distribution.png` (head/medium/tail group sizes)
- `tsne.png` (2-D t-SNE of learned features)

---

## How to run — local

```bash
python -m venv venv
venv\Scripts\activate            # Windows  (source venv/bin/activate on Linux/Mac)
pip install -r requirements.txt

# 1. Build the dataset (once)
python data/prepare_datasets.py --dataset cifar100-lt --data_dir ./data --overwrite

# 2. Validate it
python data/validate_cifar_lt.py --data_dir ./data/CIFAR-100-LT

# 3. Run an experiment — open the notebook, set METHOD, Run All
jupyter notebook notebooks/run_experiment.ipynb
```

---

## How to run — Kaggle

1. **Add the data.** Upload the prepared `CIFAR-100-LT/` folder as a Kaggle
   *Dataset* (appears under `/kaggle/input/...`), **or** set `BUILD_DATASET = True`
   in the notebook to build it into `/kaggle/working`.
2. **Add the code.** Upload the repo as a Dataset/Utility, or `!git clone` it in
   the first cell — the notebook auto-detects where `src/` lives.
3. **Open `notebooks/run_experiment.ipynb`**, set `DATA_DIR` / `OUTPUT_DIR` /
   `METHOD`, enable GPU, **Run All**.
4. For a quick smoke test first: set `MAX_TRAIN_SAMPLES` / `MAX_TEST_SAMPLES` and
   a small `EPOCHS`.

---

## The notebook (cell-by-cell)

`notebooks/run_experiment.ipynb` — 20 cells:

1. Make `src/` importable (auto-detect project root; works local + Kaggle).
2. **Configuration** — the only cell you normally edit: paths, `METHOD`,
   batch/lr/epochs/seed/pretrained/device, smoke-test limits, contrastive params
   (`PRETRAIN_EPOCHS`, `PROBE_EPOCHS`, `TEMPERATURE`), few-shot params
   (`N_WAY`, `K_SHOT`, `N_QUERY`, `EPISODES_PER_EPOCH`, `META_LR`).
3. Imports + `set_seed` + device + create run folder.
4. Optional build + load dataset; print shot-group sizes and head/tail counts.
5. Data plots (class frequency, shot distribution).
6. Train the chosen method (single dispatch over the 4 trainers).
7. Compute metrics; save `config.json`, `metrics.json`, `metrics.csv`.
8. Result plots (curves, confusion matrices, t-SNE).
9. Summary.

**To compare methods:** change `METHOD` and Run All — each writes to its own
`outputs/<method>/` folder, so nothing is overwritten.

---

## Verification performed

- **Module smoke test:** every module exercised on a tiny CPU subset — baseline,
  prototype, SupCon loss + contrastive model forward, meta (episodic + global
  100-way eval), all metrics, all 7 figures. ✅
- **Notebook glue test:** executed the actual notebook code cells on a tiny CPU
  config for `baseline`, `prototype`, and `meta` — training, full metric suite,
  JSON/CSV saving, and all figures produced for each. ✅
- (Temporary test scripts were deleted after verification.)

> Low accuracy in those tests is expected: 1 epoch, random init, ~800 random
> samples. They verify the *pipeline*, not final performance.

---

## Known limitations / future work

- **No separate validation split.** This follows the standard CIFAR-100-LT
  protocol (train on the long-tail split, report on the balanced test set). For a
  stricter setup, hold out a validation split in the data cell.
- **Contrastive pre-training** iterates the full train set per epoch; on Kaggle
  CPU this is slow — use GPU or `MAX_TRAIN_SAMPLES` for quick checks.
- **Natural extension** (noted in `Summary.md`): initialise the meta-learning
  encoder from a SupCon-pretrained encoder to combine Methods 3 and 4.
- **Not committed to git.** The working tree has staged deletions and untracked
  new files; commit when you've reviewed.
