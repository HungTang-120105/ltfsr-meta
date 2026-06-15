"""Module D — DINOv2 as a second, complementary vision foundation model.

CLIP is trained with *language* supervision, so its features are organised around
nameable semantics. DINOv2 (Oquab et al., 2023) is trained **self-supervised on
images only**, so it captures fine-grained visual structure CLIP can miss. The
research question asks whether this *different* kind of external knowledge helps the
tail — and whether it **complements** CLIP. We use it exactly like the CLIP track:
freeze the backbone, cache features once, train a tiny head.

DINOv2 has **no text encoder**, so there is no zero-shot path: we initialise the
LIFT cosine head from **class-mean features (NCM)** instead of text prototypes
(``class_mean_prototypes`` below), then reuse ``src.experts.lift.train_lift`` to
fine-tune the adapter + head with the same logit-adjusted loss.

Needs ``torch.hub`` access to ``facebookresearch/dinov2`` (internet on Kaggle the
first time). Feature extraction is a few minutes; training is on cached features.
"""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

# DINOv2 uses ImageNet normalisation and a 14-pixel patch (224 = 16 patches).
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def load_dino(device, model_name: str = "dinov2_vits14") -> dict:
    """Load a frozen DINOv2 backbone and its preprocessing transform."""
    # trust_repo=True avoids the interactive "trust this repo?" prompt, which would
    # otherwise hang Kaggle's non-interactive kernel on the first download.
    model = torch.hub.load("facebookresearch/dinov2", model_name, trust_repo=True)
    model = model.to(device).eval()
    preprocess = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])
    return {"model": model, "preprocess": preprocess}


@torch.no_grad()
def encode_dino_features(dataset, dino_bundle: dict, device, batch_size: int = 128,
                         num_workers: int = 2, multi_gpu: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """L2-normalized DINOv2 CLS features for ``dataset`` -> (features ``(N, D)``, labels).

    Same dataset-copy trick as the CLIP expert: swap in DINOv2's preprocessing so the
    caller's dataset is untouched. Returned on CPU for caching. ``multi_gpu=True`` splits
    each batch across all available GPUs (forward-only; DINOv2's ``forward`` already
    returns the CLS feature, so DataParallel wraps it directly).
    """
    model = dino_bundle["model"]
    if multi_gpu and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    dino_dataset = copy.copy(dataset)
    dino_dataset.transform = dino_bundle["preprocess"]
    loader = DataLoader(dino_dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=torch.cuda.is_available())

    all_features, all_labels = [], []
    for images, targets in loader:
        features = F.normalize(model(images.to(device)), dim=-1)
        all_features.append(features.float().cpu())
        all_labels.append(targets)
    return torch.cat(all_features), torch.cat(all_labels)


def class_mean_prototypes(features: torch.Tensor, labels: torch.Tensor,
                          num_classes: int) -> torch.Tensor:
    """L2-normalized class-mean features, shape ``(C, D)`` (NCM head initialisation).

    Used in place of CLIP's text prototypes to seed LIFT's cosine classifier for the
    text-free DINOv2 expert. Empty classes (none in train) fall back to a zero vector.
    """
    dim = features.shape[1]
    prototypes = torch.zeros(num_classes, dim)
    for c in range(num_classes):
        mask = labels == c
        if mask.any():
            prototypes[c] = F.normalize(features[mask].mean(dim=0), dim=0)
    return prototypes
