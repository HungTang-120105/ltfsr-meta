# Method 5 (bonus) — Meta-Learning (episodic Prototypical Networks)

**Code:** `src/datasets/episodic.py`, `src/trainers/meta_trainer.py`

## The idea

Instead of training on the whole dataset at once, train on thousands of small
**episodes**, each a self-contained few-shot task:

```
Episode = pick N classes (N-way)
          → K labelled support images per class (K-shot)
          → Q query images per class to classify
```

Within an episode: encode the support images and average them per class →
**prototypes**; classify each query image by nearest prototype (Euclidean
distance); cross-entropy on the query predictions trains the **encoder** so that a
few examples are enough to form a good prototype. This is "learning to learn from
few examples".

## How to report it (important)

Meta-learning is included as a **complementary experiment, not a 4th competitor on
overall accuracy.** Episodic ProtoNet is optimised for *N-way* discrimination, so
its natural, honest metric is the **few-shot episodic accuracy** (the
`val_accuracy` column in `metrics.csv`), where it is strong.

Forcing it onto the full **100-way** top-1 axis — as `evaluate_meta` also reports
for completeness — is known to underperform an end-to-end cross-entropy classifier,
so do **not** read its 100-way number as "meta is worse than baseline". Report it on
the axis it was designed for, alongside a sentence explaining the regime difference.

## Long-tail handling

`sample_episode` lets tail classes join episodes with a smaller support set (rather
than dropping classes with fewer than K images), so rare classes are not excluded.

## Relationship to the other methods

- Methods 3–4 build a balanced classifier on a fixed encoder.
- Meta-learning instead learns an encoder that produces good prototypes from any
  few examples — the few-shot view of the same distance idea.
- Natural extension: initialise the episodic encoder from the SupCon-pretrained
  encoder (Method 4) to combine representation quality with few-shot adaptation.
