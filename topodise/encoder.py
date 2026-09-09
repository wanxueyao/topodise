"""Dual-branch topology-aware semantic encoder (E_sem)."""
from dataclasses import dataclass
from typing import List, Tuple, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class EncoderConfig:
    in_channels: int = 1
    base_channels: int = 64            # smoke 可设 32
    channel_mult: Tuple[int, ...] = (1, 2, 4, 8)  # 共 4 个 stage
    blocks_per_stage: Tuple[int, ...] = (2, 2, 2, 2)
    z_dim: int = 512
    fusion: Literal["cross_attn", "concat_pool", "late_concat"] = "cross_attn"
    n_heads: int = 4
    n_fusion_layers: int = 2
    gn_groups: int = 8
    image_size: int = 224


def _gn(c, groups):
    return nn.GroupNorm(min(groups, c), c)


class _ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, gn_groups):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.gn1 = _gn(out_ch, gn_groups)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.gn2 = _gn(out_ch, gn_groups)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        h = F.silu(self.gn1(self.conv1(x)))
        h = F.silu(self.gn2(self.conv2(h)))
        return h + self.skip(x)


class _Branch(nn.Module):
    """Single-stream ResNet-like backbone returning per-stage features and CLS."""

    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        ch = cfg.base_channels
        self.stem = nn.Sequential(
            nn.Conv2d(cfg.in_channels, ch, 7, stride=2, padding=3),
            _gn(ch, cfg.gn_groups),
            nn.SiLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        stages = []
        in_ch = ch
        for i, (mult, n_blocks) in enumerate(zip(cfg.channel_mult, cfg.blocks_per_stage)):
            out_ch = ch * mult
            blocks = []
            for j in range(n_blocks):
                blocks.append(_ResBlock(in_ch if j == 0 else out_ch, out_ch, cfg.gn_groups))
            # downsample at end of stage (except last)
            if i < len(cfg.channel_mult) - 1:
                blocks.append(nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1))
            stages.append(nn.Sequential(*blocks))
            in_ch = out_ch
        self.stages = nn.ModuleList(stages)
        self.final_ch = in_ch
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        h = self.stem(x)
        feats: List[torch.Tensor] = []
        for s in self.stages:
            h = s(h)
            feats.append(h)
        cls = self.pool(h).flatten(1)  # [B, final_ch]
        return cls, feats


class _CrossAttnBlock(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.ln_q = nn.LayerNorm(dim)
        self.ln_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.ln_f = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))

    def forward(self, q, kv):
        h = self.ln_q(q)
        out, _ = self.attn(h, self.ln_kv(kv), self.ln_kv(kv), need_weights=False)
        q = q + out
        return q + self.ffn(self.ln_f(q))


class TopoDiSEEncoder(nn.Module):
    """E_sem: dual-branch (central / whorl) + fusion -> z_sem."""

    def __init__(self, cfg: Optional[EncoderConfig] = None):
        super().__init__()
        self.cfg = cfg or EncoderConfig()
        self.branch_c = _Branch(self.cfg)
        self.branch_w = _Branch(self.cfg)
        dim = self.branch_c.final_ch
        self.fusion_kind = self.cfg.fusion

        if self.cfg.fusion == "cross_attn":
            self.fuse_c = nn.ModuleList(
                [_CrossAttnBlock(dim, self.cfg.n_heads) for _ in range(self.cfg.n_fusion_layers)]
            )
            self.fuse_w = nn.ModuleList(
                [_CrossAttnBlock(dim, self.cfg.n_heads) for _ in range(self.cfg.n_fusion_layers)]
            )
            self.proj = nn.Sequential(
                nn.LayerNorm(dim * 2),
                nn.Linear(dim * 2, self.cfg.z_dim),
            )
        elif self.cfg.fusion == "concat_pool":
            self.proj = nn.Sequential(nn.LayerNorm(dim * 2), nn.Linear(dim * 2, self.cfg.z_dim))
        elif self.cfg.fusion == "late_concat":
            self.proj = nn.Sequential(nn.LayerNorm(dim * 2), nn.Linear(dim * 2, self.cfg.z_dim))
        else:
            raise ValueError(f"unknown fusion: {self.cfg.fusion}")

    def forward(
        self,
        x_central: torch.Tensor,
        x_whorl: torch.Tensor,
        return_cls: bool = False,
    ):
        cls_c0, feats_c = self.branch_c(x_central)
        cls_w0, feats_w = self.branch_w(x_whorl)

        if self.fusion_kind == "cross_attn":
            # token sequences from final feature map
            Fc = feats_c[-1].flatten(2).transpose(1, 2)  # [B, HW, C]
            Fw = feats_w[-1].flatten(2).transpose(1, 2)
            tc, tw = Fc, Fw
            for blk_c, blk_w in zip(self.fuse_c, self.fuse_w):
                new_tc = blk_c(tc, tw)
                new_tw = blk_w(tw, tc)
                tc, tw = new_tc, new_tw
            cls_c = tc.mean(dim=1)
            cls_w = tw.mean(dim=1)
        else:
            cls_c, cls_w = cls_c0, cls_w0

        z = self.proj(torch.cat([cls_c, cls_w], dim=-1))
        if return_cls:
            return z, (cls_c, cls_w), (feats_c, feats_w)
        return z, (feats_c, feats_w)
