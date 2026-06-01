# Method 4 — Meta-Learning (episodic Prototypical Networks)

**Code:** `src/datasets/episodic.py`, `src/trainers/meta_trainer.py`

## The idea

Instead of training on the whole dataset at once, we train on thousands of small
**episodes**, each a self-contained few-shot task:

```
Episode = pick N classes (N-way)
          → K labelled support images per class (K-shot)
          → Q query images per class to classify
```

Within an episode:
1. Encode the support images and average them per class → **prototypes**
   (the mean feature, computed on the fly — `compute_prototypes`).
2. Encode the query images and classify each by nearest prototype
   (same Euclidean-distance rule as Method 2).
3. Cross-entropy on the query predictions trains the **encoder** so that *a few
   examples are enough to form a good prototype*.

This is "learning to learn from few examples", which is precisely the skill the
tail of a long-tail dataset demands.

## Long-tail handling

`sample_episode` lets tail classes join episodes with a smaller support set
(rather than dropping classes that have fewer than K images), so the rare
classes are not silently excluded from training.

## Fair final evaluation

Episode accuracy is only a *training* signal. For a comparison with the other
methods we run a full **100-way** test (`evaluate_meta`): prototypes are computed
from the **entire training set**, and every test image is assigned to its nearest
global prototype. The reported `metrics.json` is therefore directly comparable to
Methods 1–3.

## What to look at after running

- `metrics.csv`: the episode accuracy curve (`train_accuracy` / `val_accuracy`)
  should climb as the encoder learns to form prototypes.
- Compare `g_mean`, `balanced_accuracy`, and `few_shot_accuracy` against the
  earlier methods — meta-learning is expected to be strongest on the tail.

## Relationship to the other methods

- Method 2 learns **fixed** prototypes as parameters.
- Method 4 learns an **encoder** that produces good prototypes from any few
  examples — a strictly more flexible version of the same distance idea.
- SupCon (Method 3) and meta-learning are complementary: a SupCon-pretrained
  encoder is a strong initialisation for episodic training (a natural extension
  left as future work in `Summary.md`).
