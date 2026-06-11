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

## Notebooks

- `run_all_methods.ipynb` — trains ALL `METHODS` back-to-back, writes `outputs/<method>/`,
  builds `comparison.csv` + comparison/overlay plots. **Primary entry point.**
- `phase0_reuse.ipynb` — reuse trained checkpoints (ensemble/TTA/tier_fusion/τ-norm/CLIP fusion).
- `phase2_clip_adapt.ipynb` — adapt frozen CLIP (Tip-Adapter / Tip-Adapter-F / LIFT). Highest VLM track.
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

`guides/01_how_to_run.md` (run guide), `guides/02_vlm_fusion.md` (CLIP fusion),
`guides/03_clip_adaptation.md` (Tip-Adapter + LIFT), `REFACTOR_NOTES.md` (history),
`docs/01–05_*.md` (per-method explanations).

## Status

Phase 0 + Phase 1 + Phase 2 + val-split + balanced-selection implemented. Phase 0/1 smoke-tested;
Phase 2 (Tip-Adapter + LIFT) code-complete and numerically smoke-tested on synthetic features —
**run `phase2_clip_adapt.ipynb` on Kaggle to get the real numbers**.
Not committed to git yet. Latest results live in `outputs/comparison.csv`.
