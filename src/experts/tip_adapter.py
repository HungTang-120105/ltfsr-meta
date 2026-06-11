"""Tip-Adapter: a training-free (and a lightly fine-tuned) CLIP cache model.

Zhang et al., *Tip-Adapter: Training-free Adaption of CLIP* (ECCV 2022). The idea
is **retrieval-augmented** classification on top of a frozen CLIP:

* Build a *cache* from the training set — keys are CLIP image features, values are
  the one-hot labels. A test image is classified by how similar it is to the
  cached training features (a soft nearest-neighbour vote), blended with CLIP's
  own zero-shot text prediction.
* ``tip_adapter_logits`` is **training-free**: nothing is learned, ``alpha`` (cache
  weight) and ``beta`` (sharpness) are just picked on the validation split.
* ``TipAdapterF`` makes the cache keys trainable and fine-tunes them for a few
  epochs — a tiny, cheap step that usually adds a couple of points.

Everything runs on the cached features from ``encode_clip_features`` (the frozen
backbone is never re-run), so it is fast enough for Kaggle.

For long-tail data the cache is naturally head-heavy (head classes contribute more
keys), so we (a) select ``alpha``/``beta`` on the *balanced* validation accuracy
and (b) train the fine-tuned variant with Balanced Softmax — both keep the tail in
view. Returns engine-style ``{y_true, y_pred}`` dicts for ``compute_metrics``.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.trainers.losses import BalancedSoftmaxLoss


def build_cache(train_features: torch.Tensor, train_labels: torch.Tensor,
                num_classes: int, balanced: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """Cache keys (training features) and values (labels).

    ``balanced=True`` (the long-tail default) divides each class column by its key
    count, so the cache vote is the **mean** affinity to a class, not the **sum**.
    Without it a head class with 500 keys out-votes a tail class with 5 purely by
    count, collapsing every prediction onto the head. Set False for the original
    (balanced few-shot) Tip-Adapter behaviour.
    """
    keys = train_features
    values = F.one_hot(train_labels.long(), num_classes).float()
    if balanced:
        values = values / values.sum(dim=0).clamp_min(1.0)   # column c -> 1 / count(c)
    return keys, values


def cache_logits(query: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
                 beta: float) -> torch.Tensor:
    """Soft nearest-neighbour vote over the cache.

    ``query`` and ``keys`` are unit-norm, so ``query @ keys.T`` is cosine
    similarity in ``[-1, 1]``. ``exp(-beta * (1 - sim))`` turns it into a sharp,
    positive affinity (``beta`` controls sharpness); multiplying by ``values``
    aggregates each neighbour's affinity into its class (a per-class mean when the
    cache was built balanced, a sum otherwise).
    """
    affinity = query @ keys.t()
    return ((-beta * (1.0 - affinity)).exp()) @ values


def tip_adapter_logits(query: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
                       clip_logits: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    """Final logits = zero-shot CLIP + ``alpha`` * cache vote."""
    return clip_logits + alpha * cache_logits(query, keys, values, beta)


def tune_alpha_beta(val_query: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
                    val_clip_logits: torch.Tensor, val_labels: np.ndarray,
                    alphas=None, betas=None) -> tuple[float, float, float]:
    """Grid-search ``alpha``/``beta`` on the validation split (balanced accuracy).

    Returns ``(best_alpha, best_beta, best_val_balanced_acc)``.
    """
    from src.evaluation.metrics import balanced_accuracy

    # Balanced cache logits are per-class means in [0, 1]; CLIP logits live on the
    # ~100 logit-scale, so alpha must reach into the tens to matter — search wide.
    alphas = np.array([1, 2, 5, 10, 20, 35, 50, 75, 100], dtype=float) if alphas is None else alphas
    betas = np.array([1, 2, 3, 5, 7], dtype=float) if betas is None else betas
    best = (1.0, 5.0, -1.0)
    for beta in betas:
        cache = cache_logits(val_query, keys, values, beta)
        for alpha in alphas:
            pred = (val_clip_logits + alpha * cache).argmax(dim=1).cpu().numpy()
            score = balanced_accuracy(val_labels, pred)
            if score > best[2]:
                best = (float(alpha), float(beta), float(score))
    return best


class TipAdapterF(nn.Module):
    """Fine-tuned Tip-Adapter: the cache keys become a trainable linear layer.

    ``adapter.weight`` is initialised to the cache keys, so at step 0 the model is
    exactly the training-free Tip-Adapter; a few epochs of training then adjust the
    keys to better separate the classes.
    """

    def __init__(self, keys: torch.Tensor, values: torch.Tensor, beta: float) -> None:
        super().__init__()
        num_keys, dim = keys.shape
        self.adapter = nn.Linear(dim, num_keys, bias=False)
        self.adapter.weight.data = keys.clone()
        self.register_buffer("values", values)
        self.beta = beta

    def forward(self, query: torch.Tensor, clip_logits: torch.Tensor, alpha: float) -> torch.Tensor:
        affinity = self.adapter(query)
        cache = ((-self.beta * (1.0 - affinity)).exp()) @ self.values
        return clip_logits + alpha * cache


def train_tip_adapter_f(model: TipAdapterF, train_query: torch.Tensor,
                        train_clip_logits: torch.Tensor, train_labels: torch.Tensor,
                        alpha: float, class_counts: list[int], device,
                        epochs: int = 20, lr: float = 1e-3, batch_size: int = 256) -> None:
    """Fine-tune the cache keys with Balanced Softmax (long-tail-aware) in place."""
    model.to(device).train()
    criterion = BalancedSoftmaxLoss(class_counts).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    dataset = torch.utils.data.TensorDataset(train_query, train_clip_logits, train_labels.long())
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for _ in range(epochs):
        for query, clip_logits, labels in loader:
            logits = model(query.to(device), clip_logits.to(device), alpha)
            loss = criterion(logits, labels.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()


@torch.no_grad()
def predict(query: torch.Tensor, labels: torch.Tensor, keys: torch.Tensor,
            values: torch.Tensor, clip_logits: torch.Tensor, alpha: float, beta: float,
            model: TipAdapterF | None = None, device=None) -> dict:
    """Predict with either the training-free cache (``model=None``) or ``TipAdapterF``."""
    if model is None:
        logits = tip_adapter_logits(query, keys, values, clip_logits, alpha, beta)
    else:
        model.eval()
        logits = model(query.to(device), clip_logits.to(device), alpha).cpu()
    return {"y_true": labels.cpu().numpy(), "y_pred": logits.argmax(dim=1).cpu().numpy()}
