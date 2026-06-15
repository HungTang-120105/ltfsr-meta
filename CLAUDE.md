# LTFSR-Meta — project context for Claude

Long-tail image recognition on **CIFAR-100-LT** (imbalance factor 100). The goal is a
clear, reportable comparison of long-tail methods (student research project, runs on
**Kaggle GPU**). Code must stay **simple and readable**.

## Core conventions (read first)

- **Test set is balanced** (100 img/class) → `accuracy == balanced_accuracy`.
  **Headline metrics = `balanced_accuracy` and `few_shot_accuracy`**, NOT raw accuracy.
- **Backbone** (`src/models/backbone.py` `build_encoder`):
  - `pretrained=False` (MAIN, default): **CIFAR stem** (3×3 conv, no maxpool), trained
    **from scratch** on 32×32. This is the standard CIFAR-LT protocol.
  - `pretrained=True`: keep ImageNet stem → **must** set `IMAGE_SIZE=224` (optional table).
- **Train/val/test split**: a stratified **`VAL_FRACTION=0.1`** is held out from train for
  **model selection**; each class keeps ≥1 train image (tail classes keep all 5, so val may
  contain 0 tail). **Test is never used for selection** — only the final report.
  Helpers: `split_indices_by_class`, `subset` in `src/datasets/cifar_lt.py`.
- **Multi-dataset**: any dataset in the ImageFolder LT layout (`train/class_XXX`, `test/class_XXX`,
  `class_counts.json`, `class_names.json`) works — only data-reading differs. `load_class_names`
  reads names for CLIP/LLM (falls back to CIFAR-100). `data/prepare_cub_lt.py` builds **CUB-200-LT**
  (fine-grained, IF=10, names work for CLIP). Phase 2/3 notebooks auto-derive `NUM_CLASSES`/`CLASS_NAMES`
  + take `MANY_THRESHOLD`/`FEW_THRESHOLD` (CUB: 15/6). See `guides/05_new_dataset.md`.
- **Model selection = best checkpoint by BALANCED accuracy on val** (val is long-tail-shaped,
  so raw accuracy is head-biased and would penalise tail-aware methods). See `classifier.py`.

## Methods (`METHOD` names) and files

| METHOD | level | file |
|---|---|---|
| `baseline` | — | `trainers/baseline_trainer.py` (CE) |
| `balanced_softmax` | loss | `trainers/losses.py` (logit + log prior) |
| `decoupling` | classifier | `trainers/decoupling_trainer.py` (cRT: freeze encoder, retrain head on balanced sampler) |
| `supcon` | representation | `trainers/contrastive_trainer.py` (SupCon → cRT) |
| `cmo` / `mixup` / `cutmix` | data (Phase 1) | `trainers/augment_trainer.py` + `datasets/augment.py` (CMO = tail-rich CutMix + Balanced-Softmax) |
| `meta` | bonus | `trainers/meta_trainer.py` (episodic ProtoNet; report on few-shot N-way axis, weak at 100-way by design) |

Shared: `trainers/engine.py` (train/eval loops, `eval_encoder` flag freezes BN for cRT),
`trainers/classifier.py` (`fit_classifier`), `models/baseline.py` (encoder + linear head).

## Phase 0 — reuse checkpoints, NO training (`notebooks/phase0_reuse.ipynb`)

- `evaluation/ensemble.py`: `load_classifier`, `predict_probs` (+TTA flip), `ensemble_predict`.
- `evaluation/posthoc.py`: `tier_fusion` (parametric head for head classes, NCM prototypes for
  tail), `tau_normalized_predict` (τ-norm; τ chosen on val).
- `experts/clip_expert.py`: **CLIP zero-shot expert** (vision-language fusion, the "fancy" method).
  `load_clip` + `clip_probs`; CIFAR-100 class names embedded in label order.
  Fusion = val-tuned convex blend `α·CLIP + (1−α)·vision` (α picked on val by balanced acc).
  Needs `pip install open_clip_torch` + internet on Kaggle. Set `BEST_SINGLE="cmo"`.

## Phase 2 — adapt a frozen CLIP, no backbone training (`notebooks/phase2_clip_adapt.ipynb`)

The strongest "fancy" track. Stop training from scratch; **adapt frozen CLIP** to the long
tail, learning only a few params. All run on **cached** CLIP features (`encode_clip_features`
in `clip_expert.py`, run once per split). External-knowledge (VLM) track, like CLIP fusion.

- `experts/tip_adapter.py`: **Tip-Adapter** (ECCV 2022). `build_cache` + `cache_logits` =
  retrieval-augmented k-NN-soft vote over training features, blended with zero-shot
  (`alpha`/`beta` tuned on balanced val by `tune_alpha_beta`). `TipAdapterF` +
  `train_tip_adapter_f` = trainable cache keys, fine-tuned with Balanced Softmax.
- `experts/lift.py`: **LIFT-style** (ICML 2024). `ResidualAdapter` (gate=0 init → starts at
  zero-shot, <1% params) + `CosineClassifier` (init from text features = semantic init), trained
  with `BalancedSoftmaxLoss`; `train_lift` selects best epoch by balanced val acc.
- `experts/clip_finetune.py`: **fine-tuning-depth ablation** (`linear_probe`/`last_block`/`full_ft`)
  — trains the ViT on images (no caching, HEAVY), `RUN_FT_ABLATION` toggle in phase2 (default off).
  Shows "heavy FT hurts the tail" → justifies the frozen design. NOTE: this is the ONLY place the
  CLIP backbone weights are trained; everything else freezes CLIP and adapts cached features.

## Phase 3 — which external knowledge helps the tail? (`notebooks/phase3_knowledge_sources.ipynb`)

The **research-study** track (main contribution). Research question: *for the 5-image tail, which
external knowledge helps most — language (LLM), a second vision FM (DINOv2), or generative
(diffusion) — and do they complement each other?* All on cached frozen features.

- `experts/llm_prompts.py` (A): free LLM on Kaggle generates per-class descriptions (cached to
  `outputs/class_descriptions.json`); `clip_expert.encode_text_prototypes` averages their CLIP
  embeddings into richer prototypes (CuPL-style) for zero-shot + LIFT head init.
- `experts/dino_expert.py` (D): frozen **DINOv2** features + LIFT with class-mean (NCM) init (no text encoder).
- `experts/feature_diffusion.py` (C): tiny class-conditional DDPM over feature vectors synthesises
  **tail** features (LDMLR-style); LIFT trains on real+synthetic.
- `experts/feature_mixup.py`: tail-aware feature mixup = **cmo** moved into feature space; enabled
  via `train_lift(..., mixup_alpha>0)` (soft two-label BalancedSoftmax). `balanced_softmax` is
  already the LIFT/Tip-Adapter-F training loss, so the old data+loss methods both live on here.
- `experts/gla.py` (B): **Generalized Logit Adjustment** — removes the foundation model's own
  pretraining label bias (generalises `balanced_softmax`); strength tuned on val, `0` in the grid.
- `evaluation/fusion.py`: N-expert **tail-aware** fusion (per-shot-group weights, tuned on val;
  few-group tied to medium since val lacks tail) + `complementarity_report`. Generalises tier/vlm fusion.
- `cmo` checkpoint = the **from-scratch control** every external source is measured against.
- Output: `outputs/knowledge_sources.csv` (per-shot-group table — the headline).

## Notebooks

- `run_pipeline.ipynb` — **all 4 phases in one file** with toggles (`RUN_PHASE1/0/2/3`). Each phase
  is a function (isolated scope, no var collisions); shared config on top; `USE_MULTI_GPU` splits
  feature-extraction batches across GPUs (Kaggle 2×T4). Runs Phase 1→0→2→3 in one session so cmo
  checkpoint flows to 0/3. For CUB: toggle off 1/0, set thresholds 15/6, `USE_CMO=False`.
- `run_all_methods.ipynb` — trains ALL `METHODS` back-to-back, writes `outputs/<method>/`,
  builds `comparison.csv` + comparison/overlay plots. (The standalone Phase-1 notebook.)
- `phase0_reuse.ipynb` — reuse trained checkpoints (ensemble/TTA/tier_fusion/τ-norm/CLIP fusion).
- `phase2_clip_adapt.ipynb` — adapt frozen CLIP (Tip-Adapter / Tip-Adapter-F / LIFT). VLM track.
- `phase3_knowledge_sources.ipynb` — research study: LLM vs DINOv2 vs diffusion + GLA + fusion.
- `run_experiment.ipynb` — run a single method (quick look).
- Smoke-test first: `MAX_TRAIN_SAMPLES=2000, EPOCHS=5`, then full `EPOCHS=200`.

## Gotchas / decisions already made

- **PyTorch 2.6**: `torch.load` defaults to `weights_only=True`. All 4 load sites use
  `weights_only=False`, and checkpoints store scalars as python `float`. Do not revert.
- sklearn warning `y_pred contains classes not in y_true` is **harmless** (LT val lacks some classes).
- SupCon stage-1 pretrains only on the **train split** (no val leak) — `pretrain_encoder(train_dataset,...)`.
- `cmo` = Balanced-Softmax + CMO (a single combined method, not "CMO on every method").
- Present CLIP fusion as a **separate "external knowledge" table**, not in the from-scratch leaderboard.

## Expected ballpark (CIFAR-100-LT IF=100, from scratch)

baseline ~0.40 bal-acc; balanced_softmax ~0.44; cmo best ~0.47 (tail ~0.30); ensemble/fusion higher.
CLIP zero-shot ~0.65 (uniform) → VLM fusion ~0.66–0.70. (Numbers shift with the val-selection rerun.)
Phase 2 (adapt frozen CLIP): tip_adapter ~0.68–0.72, tip_adapter_f ~0.71–0.75, **lift ~0.76–0.83**.

## More detail

`guides/00_muc_tieu_va_thong_diep.md` (project goals + message + expected outputs per phase),
`guides/01_how_to_run.md` (run guide), `guides/02_vlm_fusion.md` (CLIP fusion),
`guides/03_clip_adaptation.md` (Tip-Adapter + LIFT),
`guides/04_knowledge_sources.md` (Phase 3 research study: LLM/DINOv2/diffusion + GLA + fusion),
`guides/05_new_dataset.md` (multi-dataset: ImageFolder LT layout + `class_names.json`; CUB-200-LT),
`REFACTOR_NOTES.md` (history),
`docs/01–07_*.md` (per-method explanations: 06 = cmo, 07 = CLIP fusion/Tip-Adapter/LIFT).

## Status

Phase 0 + Phase 1 + Phase 2 + Phase 3 + val-split + balanced-selection implemented. Phase 0/1
smoke-tested; Phase 2 real numbers obtained (lift ~0.72 with ViT-B/32; Tip-Adapter cache fixed to
class-balanced). Phase 3 (LLM prototypes / DINOv2 / feature-diffusion / GLA / tail-aware fusion)
code-complete and numerically smoke-tested on synthetic features —
**run `phase3_knowledge_sources.ipynb` on Kaggle (GPU + internet) for the real study**.
Not committed to git yet. Latest results: `outputs/comparison.csv`, `outputs/comparison_vlm.csv`,
`outputs/knowledge_sources.csv`.
