# Method 3 — Decoupling / cRT (classifier-level fix)

**Code:** `src/trainers/decoupling_trainer.py` (+ `make_balanced_loader` in `src/datasets/cifar_lt.py`)

## The idea

Kang et al. (ICLR 2020, *Decoupling Representation and Classifier for Long-Tailed
Recognition*) showed that a model trained on long-tail data already learns **good
features** — the part that is biased toward head classes is the **linear
classifier**. So we separate the two and fix only the broken part:

```
Stage 1 (representation): train the whole network normally with cross-entropy
                          on the natural long-tail distribution.
Stage 2 (cRT = classifier Re-Training):
                          freeze the encoder, throw away the biased head, and
                          train a FRESH linear head for a few epochs using a
                          CLASS-BALANCED sampler. Features never move.
```

The class-balanced sampler (`make_balanced_loader`) draws every class with equal
probability, so the new head sees head and tail classes equally often and learns
an unbiased decision boundary.

## Why it helps on long-tail data

This directly demonstrates the report's thesis: *the bottleneck is the classifier,
not the features.* Re-balancing only the head — a cheap 10-epoch stage on frozen
features — typically gives the **largest single jump in tail accuracy** in the
whole progression, without retraining the expensive backbone.

This is the principled version of the old "prototype head" idea: a nearest-class-
mean / balanced linear classifier on a fixed encoder, done in the way the
literature shows actually works.

## What to look at after running

- `metrics.csv` has a `stage` column (`representation` then `crt`). Watch
  `balanced_accuracy` jump at the stage boundary.
- Compare `few_shot_accuracy` to Methods 1–2 — it should rise again.
