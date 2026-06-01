# Method 3 — Supervised Contrastive Learning (SupCon)

**Code:** `src/models/projection.py`, `src/trainers/contrastive_trainer.py`

## The idea

Better classification starts with a better feature space. SupCon trains the
encoder in **two stages**:

**Stage 1 — contrastive pre-training.** For each image we create two augmented
views and project the encoder features into a small normalised embedding space
(via a 2-layer MLP projection head). The SupCon loss pulls embeddings of
**all same-class images in the batch together** and pushes different classes
apart:

```
two views per image → encoder → projection head → L2-normalised embedding
loss: same-class pairs attract, different-class pairs repel
```

**Stage 2 — linear probe.** The projection head is thrown away, the encoder is
**frozen**, and a single linear layer is trained on top to do the actual 100-way
classification. We reuse `fit_classifier` for this stage — freezing the encoder
is just `requires_grad = False`.

## Why it helps on long-tail data

The contrastive objective shapes the geometry of the feature space using *all*
pairwise relationships in a batch, which gives a cleaner, more transferable
representation than cross-entropy alone. With well-separated clusters, even
tail classes — which contribute few but still informative positive pairs — end
up with a more faithful region in feature space, so the downstream classifier
generalises better to rare classes.

## Knobs that matter

- `temperature` (default `0.07`): lower = harder contrast.
- `pretrain_epochs` vs `probe_epochs`: most of the work is in Stage 1.

## What to look at after running

- `pretrain_metrics.csv`: the SupCon loss should fall steadily.
- Compare `macro_f1` / `g_mean` to Methods 1–2 — representation quality shows up
  most on the balanced (macro) metrics.
