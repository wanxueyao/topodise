"""Feature-space diffusion augmentation for TopoDiSE Stage 2.

Lightweight conditional MLP that learns to denoise the encoder feature `z`.
Used to synthesise additional `z_aug` samples in feature space (no image
decoding required), expanding minority-class coverage on a small downstream
dataset.

Design choices (see files/topodise_enhancement/topodise_high_accuracy_strategy_20260522.md
section 3 P0-A for the rationale):
- predicts `z0` (clean feature) rather than `eps` (noise): for low-dim features
  on small datasets, x0-prediction is more stable than eps-prediction.
- conditioned on a sinusoidal time embedding and (optionally) class label.
- diffusion schedule is **shared** with the image-space schedule on purpose;
  by sampling `t` only in `[1, T_aug]` (with T_aug << T) we restrict noise to
  a low-distortion regime so `z_aug` stays close to the semantic manifold.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def q_sample_feature(z0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor,
                     alpha_bar: torch.Tensor) -> torch.Tensor:
    """Forward-diffuse a feature vector `z0 : [B, D]` to step `t`."""
    ab = alpha_bar[t].view(-1, 1)
    return ab.sqrt() * z0 + (1.0 - ab).sqrt() * noise


def _sinusoidal_t_emb(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Standard sinusoidal time embedding; t: [B] long."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0)
        * torch.arange(0, half, device=t.device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
    emb = torch.cat([args.sin(), args.cos()], dim=-1)
    if emb.size(-1) < dim:  # pad if odd dim
        emb = F.pad(emb, (0, dim - emb.size(-1)))
    return emb


class FeatureDiffusionHead(nn.Module):
    """MLP denoiser g_phi(z_t, t [, y]) -> z0_hat.

    Args:
        z_dim:       feature dim of the encoder output.
        hidden:      MLP hidden width.
        layers:      number of MLP layers (>=2).
        t_emb_dim:   sinusoidal time embedding dim.
        n_classes:   if > 0, concat a one-hot class embedding to the input.
    """

    def __init__(self, z_dim: int = 512, hidden: int = 256, layers: int = 3,
                 t_emb_dim: int = 128, n_classes: int = 0):
        super().__init__()
        assert layers >= 2, "FeatureDiffusionHead requires layers>=2"
        self.z_dim = z_dim
        self.t_emb_dim = t_emb_dim
        self.n_classes = n_classes

        in_dim = z_dim + t_emb_dim + (n_classes if n_classes > 0 else 0)
        modules = []
        d = in_dim
        for _ in range(layers - 1):
            modules += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        modules += [nn.Linear(d, z_dim)]
        self.net = nn.Sequential(*modules)

    def forward(self, z_t: torch.Tensor, t: torch.Tensor,
                y: Optional[torch.Tensor] = None) -> torch.Tensor:
        t_emb = _sinusoidal_t_emb(t, self.t_emb_dim)
        parts = [z_t, t_emb]
        if self.n_classes > 0:
            if y is None:
                y_oh = torch.zeros(z_t.size(0), self.n_classes,
                                   device=z_t.device, dtype=z_t.dtype)
            else:
                y_clamped = y.clamp(min=0).long()
                y_oh = F.one_hot(y_clamped, num_classes=self.n_classes).to(z_t.dtype)
                # zero-out class info where label is missing (y < 0)
                valid = (y >= 0).float().unsqueeze(-1)
                y_oh = y_oh * valid
            parts.append(y_oh)
        return self.net(torch.cat(parts, dim=-1))
