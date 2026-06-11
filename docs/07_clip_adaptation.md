# Method 7 (external knowledge) — CLIP fusion, Tip-Adapter & LIFT

**Code:** `src/experts/clip_expert.py`, `src/experts/tip_adapter.py`, `src/experts/lift.py`
**Run guides:** `guides/02_vlm_fusion.md` (fusion), `guides/03_clip_adaptation.md` (Tip-Adapter + LIFT)

## The idea

Methods 1–6 train a small network **from scratch**, so the 5-image tail classes can only ever
be learned from those 5 images. Method 7 steps outside that budget: it borrows a **frozen
foundation model — CLIP** — which already understands the *meaning of class names* and has
never seen our long-tailed data, so it carries **no head/tail bias**. Reported in a **separate
"external-knowledge (VLM)" table**, since it uses knowledge our from-scratch models do not.

Four steps, each adding very little capacity but more tail accuracy than the last:

```
clip_only      : zero-shot — match each image to the text "a photo of a {class}"   (no training)
vlm_fusion     : α·CLIP + (1-α)·vision model, α picked on val                       (no training)
tip_adapter    : retrieval — vote by similarity to a CACHE of training features      (no training)
tip_adapter_f  : same cache, keys fine-tuned a few epochs with Balanced Softmax      (light)
lift           : tiny residual adapter + cosine head (text-init), logit-adjusted loss (<1% params)
```

## Why each helps long-tail data

- **clip_only / vlm_fusion** — CLIP recognises a class from its *name*, so a class with 5
  images is no harder than one with 500. Fusing it with the vision model (which is strong on
  the head) closes the tail gap with a single val-tuned weight `α`.
- **Tip-Adapter** — keeps a *cache* (keys = training image features, values = labels) and
  classifies a test image by soft nearest-neighbour vote, blended with zero-shot. Training-free;
  `α`/`β` chosen on the *balanced* val so the head-heavy cache does not swamp the tail.
- **LIFT** (the lesson: *heavy fine-tuning hurts the tail*) — freeze CLIP, add a small
  residual adapter whose gate starts at **0** (so the model *begins exactly at zero-shot*),
  and a cosine head **initialised from the text features** (so even the 5-image tail starts
  from a meaningful prototype). Train only these with the same logit-adjusted loss as Method 2.

All of this runs on **cached** CLIP features — the backbone is run once and never trained — so
it is cheap enough for Kaggle while reaching the project's highest tail accuracy.

## How we report it

Everything is selected on the **validation split** (balanced accuracy) exactly like the
from-scratch methods, but written to a **separate** table (`comparison_vlm.csv`) so the
external-knowledge results are never mixed into the from-scratch leaderboard. Expected order:
`clip_only` < `vlm_fusion` ≲ `tip_adapter` < `tip_adapter_f` < **`lift`**.
