"""Reuse already-trained checkpoints: ensembling and test-time augmentation.

Nothing here trains a model. Each helper loads finished checkpoints from
``outputs/<method>/`` and combines their predictions, so it is cheap and runs at
inference only. Ensembling averages class probabilities across models (their
errors are different, so the average is steadier); TTA averages a few views of
each test image.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.models.baseline import BaselineClassifier


def load_classifier(run_dir: Path, num_classes: int, device: torch.device,
                    pretrained: bool = False, checkpoint_name: str = "best_model.pt") -> BaselineClassifier:
    """Rebuild a BaselineClassifier and load a saved checkpoint.

    ``pretrained`` must match how the checkpoint was trained (False = CIFAR stem),
    otherwise the architecture will not match the saved weights.
    """
    model = BaselineClassifier(num_classes=num_classes, pretrained=pretrained).to(device)
    state = torch.load(Path(run_dir) / checkpoint_name, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def predict_probs(model, loader, device, tta: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Return (probabilities ``(N, C)``, true labels ``(N,)``) for one model.

    With ``tta=True`` the prediction is averaged with the horizontally-flipped
    image — a cheap, reliable form of test-time augmentation for natural images.
    """
    model.eval()
    all_probs, all_true = [], []
    for images, targets in loader:
        images = images.to(device)
        probs = model(images).softmax(dim=1)
        if tta:
            probs = (probs + model(torch.flip(images, dims=[3])).softmax(dim=1)) / 2
        all_probs.append(probs.cpu().numpy())
        all_true.append(targets.numpy())
    return np.concatenate(all_probs), np.concatenate(all_true)


def ensemble_predict(models: list, loader, device, tta: bool = False,
                     weights: list[float] | None = None) -> dict:
    """Average class probabilities across models; return an engine-style result.

    ``weights`` optionally weights each model (defaults to equal). The returned
    dict has ``y_true`` / ``y_pred`` so it plugs straight into ``compute_metrics``.
    """
    summed, y_true = None, None
    for index, model in enumerate(models):
        probs, y_true = predict_probs(model, loader, device, tta=tta)
        weight = 1.0 if weights is None else weights[index]
        summed = weight * probs if summed is None else summed + weight * probs
    return {"y_true": y_true, "y_pred": summed.argmax(axis=1)}
