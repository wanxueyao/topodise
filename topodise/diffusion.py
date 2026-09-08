"""Diffusion schedule + q_sample + DDIM sampler."""
from dataclasses import dataclass
from typing import Optional
import math

import torch


@dataclass
class DiffusionSchedule:
    T: int = 1000
    schedule: str = "cosine"
    s: float = 0.008

    def __post_init__(self):
        steps = self.T
        if self.schedule == "cosine":
            t = torch.linspace(0, steps, steps + 1) / steps
            alpha_bar = torch.cos(((t + self.s) / (1 + self.s)) * math.pi * 0.5) ** 2
            alpha_bar = alpha_bar / alpha_bar[0]
            betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
            betas = betas.clamp(1e-5, 0.999)
        else:
            betas = torch.linspace(1e-4, 0.02, steps)
        alphas = 1.0 - betas
        self.alpha_bar = torch.cumprod(alphas, dim=0)  # [T]
        self.betas = betas
        self.alphas = alphas

    def to(self, device):
        self.alpha_bar = self.alpha_bar.to(device)
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        return self


def q_sample(x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor, alpha_bar: torch.Tensor) -> torch.Tensor:
    ab = alpha_bar[t].view(-1, 1, 1, 1)
    return ab.sqrt() * x0 + (1 - ab).sqrt() * noise


@torch.no_grad()
def ddim_sample(
    unet,
    schedule: DiffusionSchedule,
    z_sem: torch.Tensor,
    shape: tuple,
    n_steps: int = 50,
    eta: float = 0.0,
    x_T: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    device = z_sem.device
    schedule = schedule.to(device)
    B = z_sem.size(0)
    x = x_T if x_T is not None else torch.randn(B, *shape, device=device)
    ts = torch.linspace(schedule.T - 1, 0, n_steps + 1).long().to(device)
    for i in range(n_steps):
        t = ts[i].expand(B)
        t_prev = ts[i + 1]
        ab_t = schedule.alpha_bar[t].view(-1, 1, 1, 1)
        ab_p = schedule.alpha_bar[t_prev].view(-1, 1, 1, 1)
        eps = unet(x, t, z_sem)
        x0_hat = (x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt().clamp(min=1e-8)
        x = ab_p.sqrt() * x0_hat + (1 - ab_p).sqrt() * eps
    return x
