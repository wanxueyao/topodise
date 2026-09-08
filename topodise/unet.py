"""Conditional denoising U-Net with AdaGN and classifier-free guidance."""
from dataclasses import dataclass
from typing import Tuple, Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class UNetConfig:
    in_channels: int = 1
    out_channels: int = 1
    model_channels: int = 64           # smoke 可设 32
    channel_mult: Tuple[int, ...] = (1, 2, 2, 4)
    z_dim: int = 512
    time_dim: int = 256
    gn_groups: int = 8
    image_size: int = 256


def _gn(c, g):
    return nn.GroupNorm(min(g, c), c)


def _sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(0, half, device=t.device, dtype=torch.float32) / max(half - 1, 1)
    )
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
    return torch.cat([args.sin(), args.cos()], dim=-1)


class _ResBlockAdaGN(nn.Module):
    """ResBlock with AdaGN modulation from (t_emb + z_emb)."""

    def __init__(self, in_ch, out_ch, cond_dim, gn_groups):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.gn1 = _gn(out_ch, gn_groups)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.gn2 = _gn(out_ch, gn_groups)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.cond = nn.Linear(cond_dim, 2 * out_ch)

    def forward(self, x, cond):
        h = self.gn1(self.conv1(x))
        sb = self.cond(cond)
        s, b = sb.chunk(2, dim=-1)
        h = h * (1 + s.unsqueeze(-1).unsqueeze(-1)) + b.unsqueeze(-1).unsqueeze(-1)
        h = F.silu(h)
        h = self.gn2(self.conv2(h))
        h = F.silu(h)
        return h + self.skip(x)


class ConditionalUNet(nn.Module):
    def __init__(self, cfg: Optional[UNetConfig] = None):
        super().__init__()
        self.cfg = cfg or UNetConfig()
        c = self.cfg
        self.time_mlp = nn.Sequential(
            nn.Linear(c.time_dim, c.time_dim),
            nn.SiLU(),
            nn.Linear(c.time_dim, c.time_dim),
        )
        self.z_proj = nn.Sequential(
            nn.Linear(c.z_dim, c.time_dim),
            nn.SiLU(),
            nn.Linear(c.time_dim, c.time_dim),
        )
        cond_dim = c.time_dim

        ch = c.model_channels
        mults = c.channel_mult
        self.in_conv = nn.Conv2d(c.in_channels, ch, 3, padding=1)

        # encoder
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        in_ch = ch
        skip_chs = [ch]
        for i, m in enumerate(mults):
            out_ch = ch * m
            self.down_blocks.append(_ResBlockAdaGN(in_ch, out_ch, cond_dim, c.gn_groups))
            skip_chs.append(out_ch)
            if i < len(mults) - 1:
                self.down_samples.append(nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1))
            else:
                self.down_samples.append(nn.Identity())
            in_ch = out_ch

        # mid
        self.mid_block = _ResBlockAdaGN(in_ch, in_ch, cond_dim, c.gn_groups)

        # decoder
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        for i, m in enumerate(reversed(mults)):
            skip_ch = skip_chs[-(i + 1)]
            out_ch = ch * m
            self.up_blocks.append(_ResBlockAdaGN(in_ch + skip_ch, out_ch, cond_dim, c.gn_groups))
            if i < len(mults) - 1:
                self.up_samples.append(
                    nn.ConvTranspose2d(out_ch, out_ch, 4, stride=2, padding=1)
                )
            else:
                self.up_samples.append(nn.Identity())
            in_ch = out_ch

        self.out_norm = _gn(in_ch, c.gn_groups)
        self.out_conv = nn.Conv2d(in_ch, c.out_channels, 3, padding=1)

    def forward(self, x_t, t, z):
        t_emb = self.time_mlp(_sinusoidal_time_embedding(t, self.cfg.time_dim))
        z_emb = self.z_proj(z)
        cond = t_emb + z_emb

        h = self.in_conv(x_t)
        skips = [h]
        for blk, down in zip(self.down_blocks, self.down_samples):
            h = blk(h, cond)
            skips.append(h)
            h = down(h)
        h = self.mid_block(h, cond)
        for blk, up in zip(self.up_blocks, self.up_samples):
            skip = skips.pop()
            # align spatial dims if needed
            if skip.shape[-2:] != h.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = torch.cat([h, skip], dim=1)
            h = blk(h, cond)
            h = up(h)
        h = F.silu(self.out_norm(h))
        return self.out_conv(h)
