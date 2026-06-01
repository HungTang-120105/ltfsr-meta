"""Reproducibility helpers: seed every random source used in the project."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch (CPU + CUDA) for reproducible runs.

    Args:
        seed: The integer seed shared by all random number generators.
        deterministic: If True, force cuDNN into deterministic mode. This makes
            results repeatable at a small speed cost, which is the right
            trade-off for a research project.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(device_name: str | None = None) -> torch.device:
    """Return a torch.device, falling back to CPU when CUDA is unavailable.

    Args:
        device_name: "cuda", "cpu", or None to auto-detect. Asking for CUDA on a
            CPU-only machine (common on Kaggle) prints a notice and uses the CPU.
    """
    if device_name is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU.")
        return torch.device("cpu")
    return device
