# Method 2 — Prototype Classifier (distance head)

**Code:** `src/models/prototype.py`, `src/trainers/prototype_trainer.py`

## The idea

Keep the same ResNet-18 encoder, but **replace the linear head with one learnable
prototype vector per class**. An image is classified by how close its feature is
to each prototype, using (negative) squared Euclidean distance as the logit:

```
logitᶜ = − ‖ f − prototypeᶜ ‖²
```

Cross-entropy over these logits pulls each image's feature towards its own class
prototype and away from the others. Everything else (optimizer, schedule,
training loop) is identical to the baseline — only the head changes. This is why
`baseline_trainer` and `prototype_trainer` share `fit_classifier`.

## Why it helps on long-tail data

Distances do **not** depend on the magnitude of a weight vector, so the
weight-norm bias that hurts the linear baseline disappears. Every class is
represented by a point in feature space and competes on equal footing,
regardless of how many training images it had. In practice this lifts
`few_shot_accuracy` and `g_mean` relative to the baseline.

## What to look at after running

- Compare `g_mean` and `few_shot_accuracy` against Method 1 — distance-based
  scoring should narrow the head-vs-tail gap.
- `tsne.png`: classes should form tighter, better-separated clusters.

This method is also the conceptual bridge to meta-learning (Method 4), which
uses the *same* prototype/distance idea but builds the prototypes on the fly from
a tiny support set instead of learning them as parameters.
