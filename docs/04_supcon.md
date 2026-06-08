# Method 4 — Supervised Contrastive + cRT (representation-level fix, best)

**Code:** `src/trainers/contrastive_trainer.py` (+ `rebalance_classifier` from `decoupling_trainer.py`)

## The idea

Methods 2–3 fix the loss and the classifier; Method 4 also improves the **feature
space itself**, then reuses the cRT classifier from Method 3. Two stages:

**Stage 1 — SupCon pre-training.** For each image we make two augmented views and
project the encoder features into a small L2-normalised space (a 2-layer MLP head).
The Supervised Contrastive loss (Khosla et al., 2020) pulls **all same-class
embeddings in the batch together** and pushes different classes apart:

```
two views/image → encoder → projection head → embedding
loss: same-class pairs attract, different-class pairs repel
```

**Stage 2 — cRT (not a plain linear probe).** Drop the projection head, freeze the
encoder, and train a fresh head with the **class-balanced sampler** — the *same*
`rebalance_classifier` used by Method 3.

> Why this matters: the original code froze the encoder and trained a normal
> (instance-balanced) linear probe, which **inherits the head bias** and loses to a
> fine-tuned baseline. Pairing the strong SupCon features with a *balanced*
> classifier (cRT) is what lets the representation actually help the tail.

## Why it helps on long-tail data

SupCon shapes the geometry using all pairwise relationships in a batch, giving
tighter, better-separated clusters than cross-entropy alone — a more faithful
region in space even for rare classes. A balanced classifier on top then reads out
those clusters without frequency bias. This combination is typically the **best**
balanced/few-shot accuracy in the study.

## Knobs that matter

- `pretrain_epochs` (default 200) and `pretrain_lr` (default 0.5): SupCon needs a
  long pre-training stage and a fairly large LR/batch to work well.
- `temperature` (default 0.07): lower = harder contrast.

## What to look at after running

- `pretrain_metrics.csv`: the SupCon loss should fall steadily.
- Compare `balanced_accuracy` / `macro_f1` / `few_shot_accuracy` to Methods 1–3 —
  this method should top the comparison table.
