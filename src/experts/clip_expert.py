"""CLIP zero-shot expert for the vision-language fusion.

A frozen CLIP model classifies CIFAR-100 images from the **names** of the classes
(zero-shot): it never sees our long-tailed training data, so it has no head/tail
bias. We use its per-class probabilities as a second expert and fuse them with the
trained vision model (see ``notebooks/phase0_reuse.ipynb``).

Requires ``open_clip_torch`` (``pip install open_clip_torch``); the import is lazy
so the rest of the repo works without it.

CIFAR-100 fine-label names are listed in the canonical torchvision label order, so
``CIFAR100_CLASSES[i]`` is the name of ImageFolder class ``class_{i:03d}`` (which
``data/prepare_datasets.py`` writes as ``class_{label:03d}``).
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

CIFAR100_CLASSES = [
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle",
    "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel",
    "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock",
    "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster",
    "house", "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion",
    "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain", "mouse",
    "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear",
    "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine", "possum",
    "rabbit", "raccoon", "ray", "road", "rocket", "rose", "sea", "seal", "shark",
    "shrew", "skunk", "skyscraper", "snail", "snake", "spider", "squirrel",
    "streetcar", "sunflower", "sweet_pepper", "table", "tank", "telephone",
    "television", "tiger", "tractor", "train", "trout", "tulip", "turtle",
    "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm",
]


def load_clip(class_names: list[str], device, model_name: str = "ViT-B-32",
              pretrained: str = "openai", prompt: str = "a photo of a {}") -> dict:
    """Load a frozen CLIP model and pre-encode the class-name text prompts."""
    try:
        import open_clip
    except ImportError as error:
        raise ImportError("CLIP expert needs open_clip: pip install open_clip_torch") from error

    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(model_name)

    prompts = [prompt.format(name.replace("_", " ")) for name in class_names]
    with torch.no_grad():
        text_features = F.normalize(model.encode_text(tokenizer(prompts).to(device)), dim=-1)
    return {
        "model": model,
        "preprocess": preprocess,
        "text_features": text_features,
        # exp() of the trained temperature; image @ text similarities are scaled by
        # this before softmax, so reuse it to keep our logits on CLIP's own scale.
        "logit_scale": float(model.logit_scale.exp().item()),
    }


@torch.no_grad()
def encode_clip_features(dataset, clip_bundle: dict, device, batch_size: int = 128,
                         num_workers: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    """L2-normalized CLIP image features for ``dataset`` -> (features ``(N, D)``, labels).

    Returned as CPU tensors so they can be cached once and reused by every adapter
    (Tip-Adapter, LIFT) without re-running the frozen backbone. Same dataset-copy
    trick as :func:`clip_probs` so CLIP's own preprocessing is used.
    """
    model = clip_bundle["model"]
    clip_dataset = copy.copy(dataset)
    clip_dataset.transform = clip_bundle["preprocess"]
    loader = DataLoader(clip_dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=torch.cuda.is_available())

    all_features, all_labels = [], []
    for images, targets in loader:
        features = F.normalize(model.encode_image(images.to(device)), dim=-1)
        all_features.append(features.float().cpu())
        all_labels.append(targets)
    return torch.cat(all_features), torch.cat(all_labels)


def clip_zero_shot_logits(features: torch.Tensor, clip_bundle: dict) -> torch.Tensor:
    """Zero-shot logits = ``logit_scale * features @ text_features^T`` (no softmax).

    ``features`` and ``text_features`` are unit-norm, so this is scaled cosine
    similarity — the same quantity :func:`clip_probs` feeds to softmax.
    """
    text_features = clip_bundle["text_features"].to(features.device).float()
    return clip_bundle["logit_scale"] * features @ text_features.t()


@torch.no_grad()
def clip_probs(dataset, clip_bundle: dict, device, batch_size: int = 128,
               num_workers: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Zero-shot class probabilities for ``dataset`` -> (probs ``(N, C)``, labels).

    A shallow copy of the dataset is taken so its transform can be swapped for
    CLIP's own preprocessing without disturbing the caller's dataset.
    """
    model, text_features = clip_bundle["model"], clip_bundle["text_features"]
    clip_dataset = copy.copy(dataset)
    clip_dataset.transform = clip_bundle["preprocess"]
    loader = DataLoader(clip_dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=torch.cuda.is_available())

    logit_scale = model.logit_scale.exp()
    all_probs, all_labels = [], []
    for images, targets in loader:
        image_features = F.normalize(model.encode_image(images.to(device)), dim=-1)
        logits = logit_scale * image_features @ text_features.t()
        all_probs.append(logits.softmax(dim=1).cpu().numpy())
        all_labels.append(targets.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)
