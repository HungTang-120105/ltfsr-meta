"""Fuse several knowledge experts and test whether they complement each other.

This generalises the project's earlier two-expert blends (``vlm_fusion``,
``tier_fusion``) to **N experts** and is the tool that answers the research
question's second half — *do the external-knowledge sources complement each other?*

Each expert is summarised by its per-sample class **probabilities** on the same
val/test samples (aligned order). Two fusion modes:

* ``tune_weights`` — one convex weight per expert, picked on balanced val accuracy.
* ``tune_group_weights`` — **tail-aware**: a separate weight vector per shot-group
  (many/medium/few), so the tail can lean on whichever expert is strongest there.
  Generalises ``tier_fusion``'s per-class blend. The long-tail val split usually has
  **no few-shot samples**, so the few-group weights are tied to the medium-group
  solution (documented, not silently guessed).

``complementarity_report`` then prints, per shot-group, every single expert vs the
fusion — the headline table for the presentation.
"""

from __future__ import annotations

import numpy as np

from src.evaluation.metrics import balanced_accuracy, compute_metrics


def to_probs(scores: np.ndarray) -> np.ndarray:
    """Accept logits or probabilities; return row-normalized probabilities."""
    scores = np.asarray(scores, dtype=np.float64)
    if np.all(scores >= 0) and np.allclose(scores.sum(axis=1), 1.0, atol=1e-3):
        return scores                                  # already probabilities
    z = scores - scores.max(axis=1, keepdims=True)     # softmax (stable)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _simplex_grid(n: int, step: float = 0.25):
    """All convex weight vectors over ``n`` experts on a ``step`` grid (sum to 1)."""
    levels = int(round(1.0 / step))

    def rec(remaining, slots):
        if slots == 1:
            yield (remaining,)
            return
        for k in range(remaining + 1):
            for rest in rec(remaining - k, slots - 1):
                yield (k, *rest)

    return [tuple(v / levels for v in combo) for combo in rec(levels, n)]


def fuse(probs_list: list[np.ndarray], weights) -> np.ndarray:
    """Weighted average of expert probabilities -> fused probabilities ``(N, C)``."""
    return sum(w * p for w, p in zip(weights, probs_list))


def tune_weights(val_probs: list[np.ndarray], y_val: np.ndarray,
                 step: float = 0.25) -> tuple[tuple, float]:
    """Best single convex weight vector over experts by balanced val accuracy."""
    best = (None, -1.0)
    for weights in _simplex_grid(len(val_probs), step):
        score = balanced_accuracy(y_val, fuse(val_probs, weights).argmax(1))
        if score > best[1]:
            best = (weights, float(score))
    return best


def fuse_group(probs_list: list[np.ndarray], group_weights: dict,
               class_group: np.ndarray) -> np.ndarray:
    """Per-shot-group weighting: class column ``c`` uses ``group_weights[group(c)]``.

    ``group_weights`` maps group name -> weight tuple over experts; ``class_group`` is
    a ``(C,)`` array of each class's group name. Like ``tier_fusion``, the weighting is
    on class *columns*, so it needs no knowledge of the test labels.
    """
    num_classes = probs_list[0].shape[1]
    col_w = np.zeros((len(probs_list), num_classes))
    for c in range(num_classes):
        col_w[:, c] = group_weights[class_group[c]]
    return sum(col_w[e][None, :] * probs_list[e] for e in range(len(probs_list)))


def tune_group_weights(val_probs: list[np.ndarray], y_val: np.ndarray,
                       shot_groups: dict, num_classes: int, step: float = 0.25):
    """Tune a weight vector per shot-group on balanced val accuracy (coordinate-style).

    Groups absent from the val split (typically ``few``) inherit the ``medium`` weights.
    Returns ``(group_weights dict, val_balanced_acc)``.
    """
    class_group = np.empty(num_classes, dtype=object)
    for group, ids in shot_groups.items():
        for c in ids:
            class_group[c] = group
    present = set(np.unique(y_val).tolist())
    grid = _simplex_grid(len(val_probs), step)

    group_weights = {}
    for group in ("many", "medium", "few"):
        ids = [c for c in shot_groups.get(group, []) if c in present]
        if not ids:                                    # e.g. tail absent from val
            continue
        # Optimise this group's weights while keeping a uniform blend elsewhere.
        uniform = tuple([1.0 / len(val_probs)] * len(val_probs))
        best = (uniform, -1.0)
        for w in grid:
            trial = {g: (w if g == group else uniform) for g in ("many", "medium", "few")}
            fused = fuse_group(val_probs, trial, class_group)
            # score only on this group's classes (where this weight matters)
            mask = np.isin(y_val, ids)
            score = balanced_accuracy(y_val[mask], fused[mask].argmax(1))
            if score > best[1]:
                best = (w, float(score))
        group_weights[group] = best[0]

    if "medium" in group_weights:                      # tie absent few to medium
        group_weights.setdefault("few", group_weights["medium"])
    for group in ("many", "medium", "few"):            # any still-missing -> uniform
        group_weights.setdefault(group, tuple([1.0 / len(val_probs)] * len(val_probs)))

    fused = fuse_group(val_probs, group_weights, class_group)
    return group_weights, float(balanced_accuracy(y_val, fused.argmax(1)))


def complementarity_report(expert_probs: dict, y_true: np.ndarray, num_classes: int,
                           shot_groups: dict, fused_probs: np.ndarray | None = None) -> dict:
    """Per-shot-group balanced accuracy for each expert (and the fusion).

    Returns ``{name: {balanced_accuracy, many/medium/few_shot_accuracy}}`` — the table
    that shows *which* knowledge source wins *where*, and whether the fusion beats them.
    """
    report = {}
    items = dict(expert_probs)
    if fused_probs is not None:
        items["fusion"] = fused_probs
    for name, probs in items.items():
        m = compute_metrics(y_true, np.asarray(probs).argmax(1), num_classes, shot_groups)
        report[name] = {k: m[k] for k in ("balanced_accuracy", "many_shot_accuracy",
                                          "medium_shot_accuracy", "few_shot_accuracy")}
    return report


# ---------------------------------------------------------------------------
# Greedy ensemble selection (added below; does not touch the functions above).
# ---------------------------------------------------------------------------
def greedy_group_weights(val_probs: list[np.ndarray], y_val: np.ndarray, shot_groups: dict,
                         num_classes: int, max_iters: int = 25) -> tuple[dict, float]:
    """Per-shot-group **greedy ensemble selection** (Caruana et al., ICML 2004).

    For each group, start from the single best expert and **only add an expert (with
    replacement) if it improves that group's balanced val accuracy** — so the result is
    ``>= best single expert`` on val *by construction* (monotonic). This is the safe
    alternative to ``tune_group_weights`` when one expert dominates: the weighted grid can
    overfit a small val group and regress on test (e.g. the medium group), whereas greedy
    selection never drops below the best single on the val it selects on. "Greedy soup"
    (Wortsman et al., 2022) is the modern incarnation of the same add-if-it-helps rule.

    Returns ``(group_weights dict, val_balanced_acc)`` — same shape as
    ``tune_group_weights``, so it plugs straight into :func:`fuse_group`. As there, the
    few group (usually absent from a long-tail val) inherits the medium weights.
    """
    n = len(val_probs)
    uniform = tuple([1.0 / n] * n)
    eye = np.eye(n)
    class_group = np.empty(num_classes, dtype=object)
    for group, ids in shot_groups.items():
        for c in ids:
            class_group[c] = group
    present = set(np.unique(y_val).tolist())

    def score(group, weight, ids):
        # Same scoring convention as tune_group_weights: this group's weight on its
        # columns, uniform elsewhere, balanced accuracy on this group's val classes.
        trial = {g: (weight if g == group else uniform) for g in ("many", "medium", "few")}
        fused = fuse_group(val_probs, trial, class_group)
        mask = np.isin(y_val, ids)
        return balanced_accuracy(y_val[mask], fused[mask].argmax(1))

    group_weights = {}
    for group in ("many", "medium", "few"):
        ids = [c for c in shot_groups.get(group, []) if c in present]
        if not ids:
            continue
        singles = [score(group, tuple(eye[e].tolist()), ids) for e in range(n)]
        counts = eye[int(np.argmax(singles))].copy()      # start from the best single expert
        current = max(singles)
        for _ in range(max_iters):
            best_gain, pick = current, None
            for e in range(n):
                cand = counts.copy()
                cand[e] += 1.0
                s = score(group, tuple((cand / cand.sum()).tolist()), ids)
                if s > best_gain:
                    best_gain, pick = s, e
            if pick is None:                               # nothing improves -> stop (monotonic)
                break
            counts[pick] += 1.0
            current = best_gain
        group_weights[group] = tuple((counts / counts.sum()).tolist())

    if "medium" in group_weights:
        group_weights.setdefault("few", group_weights["medium"])
    for group in ("many", "medium", "few"):
        group_weights.setdefault(group, uniform)

    fused = fuse_group(val_probs, group_weights, class_group)
    return group_weights, float(balanced_accuracy(y_val, fused.argmax(1)))
