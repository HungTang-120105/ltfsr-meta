"""Module C — generative knowledge: a diffusion model that synthesises tail features.

The tail fails because it has too few examples. Instead of generating pixels (slow,
and a from-scratch image diffusion needs lots of data), we generate in the **frozen
feature space** — LDMLR (Han et al., 2024) showed feature-level augmentation beats
image-level for long-tail and is far cheaper. A small class-conditional DDPM is
trained on the cached CLIP/DINOv2 features, then samples extra **tail** features to
balance the training set; LIFT is then trained on real + synthetic features.

Compact and Kaggle-cheap: the denoiser is a 3-layer MLP over ``D``-dim vectors, a
few hundred steps of training (seconds–minutes), no pixels involved.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class FeatureDenoiser(nn.Module):
    """Predict the noise added to a feature, conditioned on timestep and class."""

    def __init__(self, dim: int, num_classes: int, num_steps: int,
                 hidden: int = 512, emb: int = 128) -> None:
        super().__init__()
        self.t_emb = nn.Embedding(num_steps, emb)
        self.y_emb = nn.Embedding(num_classes, emb)
        self.net = nn.Sequential(
            nn.Linear(dim + 2 * emb, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, self.t_emb(t), self.y_emb(y)], dim=-1))


class FeatureDiffusion:
    """A tiny class-conditional DDPM over feature vectors."""

    def __init__(self, dim: int, num_classes: int, device, num_steps: int = 100) -> None:
        self.device = device
        self.num_steps = num_steps
        self.model = FeatureDenoiser(dim, num_classes, num_steps).to(device)
        betas = torch.linspace(1e-4, 0.02, num_steps, device=device)
        self.alpha_cumprod = torch.cumprod(1.0 - betas, dim=0)   # (T,)
        self.betas = betas

    def train(self, features: torch.Tensor, labels: torch.Tensor,
              epochs: int = 200, lr: float = 1e-3, batch_size: int = 256) -> None:
        """Standard DDPM noise-prediction training on (features, labels)."""
        self.model.train()
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr)
        ds = torch.utils.data.TensorDataset(features, labels.long())
        loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
        for _ in range(epochs):
            for x0, y in loader:
                x0, y = x0.to(self.device), y.to(self.device)
                t = torch.randint(0, self.num_steps, (x0.size(0),), device=self.device)
                noise = torch.randn_like(x0)
                acp = self.alpha_cumprod[t].unsqueeze(1)
                x_t = acp.sqrt() * x0 + (1 - acp).sqrt() * noise
                loss = F.mse_loss(self.model(x_t, t, y), noise)
                opt.zero_grad(); loss.backward(); opt.step()
        self.model.eval()

    @torch.no_grad()
    def sample(self, labels: torch.Tensor) -> torch.Tensor:
        """Reverse-diffuse Gaussian noise into class-conditional features (L2-normalized)."""
        labels = labels.to(self.device)
        x = torch.randn(labels.size(0), self.model.net[-1].out_features, device=self.device)
        for step in reversed(range(self.num_steps)):
            t = torch.full((labels.size(0),), step, device=self.device, dtype=torch.long)
            acp = self.alpha_cumprod[step]
            beta = self.betas[step]
            alpha = 1.0 - beta
            eps = self.model(x, t, labels)
            mean = (x - beta / (1 - acp).sqrt() * eps) / alpha.sqrt()
            x = mean + (beta.sqrt() * torch.randn_like(x) if step > 0 else 0.0)
        return F.normalize(x, dim=-1).cpu()


def augment_tail(features: torch.Tensor, labels: torch.Tensor, diffusion: FeatureDiffusion,
                 num_classes: int, target_count: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Top up under-represented classes with synthetic features up to ``target_count``.

    ``target_count`` defaults to the median class count — enough to relieve the tail
    without drowning the real data. Returns the concatenated (real + synthetic) set.
    """
    counts = np.bincount(labels.numpy(), minlength=num_classes)
    target = int(np.median(counts[counts > 0])) if target_count is None else target_count

    need_labels = []
    for c in range(num_classes):
        deficit = target - int(counts[c])
        if counts[c] > 0 and deficit > 0:
            need_labels += [c] * deficit
    if not need_labels:
        return features, labels

    synth = diffusion.sample(torch.tensor(need_labels))
    return (torch.cat([features, synth]),
            torch.cat([labels, torch.tensor(need_labels, dtype=labels.dtype)]))
