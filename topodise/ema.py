"""Lightweight EMA for nn.Module parameters."""
from copy import deepcopy
import torch
import torch.nn as nn


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        for ema_p, p in zip(self.shadow.parameters(), model.parameters()):
            ema_p.mul_(d).add_(p.detach(), alpha=1 - d)
        for ema_b, b in zip(self.shadow.buffers(), model.buffers()):
            ema_b.copy_(b)

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, sd):
        self.shadow.load_state_dict(sd)
