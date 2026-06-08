# Method 1 — Baseline (ResNet-18 + Cross-Entropy)

**Code:** `src/models/baseline.py`, `src/trainers/baseline_trainer.py`

## The idea

The simplest image classifier: a ResNet-18 encoder produces a 512-d feature
vector, a single linear layer turns it into 100 class scores (logits), and
softmax + cross-entropy trains everything end to end.

```
image → ResNet-18 → 512-d feature → Linear(512, 100) → softmax → class
```

## Why it struggles on long-tail data

For a linear softmax head the score of class *c* is `wᶜ · f` (a dot product).
During training on imbalanced data the **weight norm `‖wᶜ‖` grows roughly with
the number of training images** in class *c*. Head classes therefore get larger
scores and wider decision regions, while tail classes are squeezed into narrow
regions and are systematically under-predicted.

This is exactly why the headline **accuracy looks fine but `few_shot_accuracy`
and `g_mean` are low** — the model is right on frequent classes and wrong on rare
ones. The baseline is the reference point that the next methods try to beat on the
*tail* metrics (balanced / few-shot accuracy), **not** on overall top-1 — on plain
top-1 a well-trained cross-entropy baseline is already very hard to beat, which is
why the comparison is framed around the long-tail metrics.

## What to look at after running

- `metrics.json`: compare `many_shot_accuracy` vs `few_shot_accuracy` — the gap
  is the long-tail problem in a single number.
- `confusion_matrix_normalized.png`: tail-class rows are mostly empty.
