# Method 2 — Balanced Softmax (loss-level fix)

**Code:** `src/trainers/losses.py`, `src/trainers/baseline_trainer.py`

## The idea

Keep the **exact same** model and training loop as the baseline (ResNet + linear
head + SGD). Change only the loss. A plain softmax implicitly assumes every class
is equally likely, but our training set is long-tailed, so the model drifts toward
the frequent (head) classes.

Balanced Softmax (Ren et al., NeurIPS 2020) corrects this by adding the **log of
each class's training frequency to the logits during training**:

```
loss = CrossEntropy( logits + log(class_prior),  target )      # training only
```

At test time we use the raw logits (the test set is balanced, so the prior is
uniform) — **the evaluation code does not change at all**. This is why Method 2
reuses `train_baseline`: we just pass `criterion=BalancedSoftmaxLoss(counts)`.

## Why it helps on long-tail data

Adding `log(prior)` to a head class's logit makes the model work *harder* to
predict it during training (it already has a built-in advantage), and conversely
gives tail classes a relative boost. The net effect is a decision boundary that is
no longer biased by class frequency — the simplest principled fix for imbalance,
and competitive with much heavier methods (LDAM-DRW, etc.) in just two lines.

## What to look at after running

- Compare `balanced_accuracy` and `few_shot_accuracy` against Method 1 — both
  should rise clearly, while plain `accuracy` may move only a little.
- `confusion_matrix_normalized.png`: tail-class rows should start filling in.
