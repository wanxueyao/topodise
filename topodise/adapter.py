"""Residual adapters with zero-init upsample for RACF."""
from typing import Dict
import torch
import torch.nn as nn

from .encoder import TopoDiSEEncoder


class ResidualAdapter(nn.Module):
    """Bottleneck adapter for 4D conv features. Zero-init keeps identity at start."""

    def __init__(self, dim: int, r: int = 64):
        super().__init__()
        self.down = nn.Conv2d(dim, r, 1, bias=False)
        self.act = nn.GELU()
        self.up = nn.Conv2d(r, dim, 1, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=5 ** 0.5)
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return x + self.up(self.act(self.down(x)))


class _StageWithAdapter(nn.Module):
    def __init__(self, stage: nn.Module, adapter: ResidualAdapter):
        super().__init__()
        self.stage = stage
        self.adapter = adapter

    def forward(self, x):
        return self.adapter(self.stage(x))


def attach_adapters(encoder: TopoDiSEEncoder, r: int = 64) -> Dict[str, ResidualAdapter]:
    """Wrap each stage in both branches with a residual adapter.

    Returns dict of adapter modules for downstream parameter filtering.
    """
    adapters: Dict[str, ResidualAdapter] = {}
    for branch_name in ("branch_c", "branch_w"):
        branch = getattr(encoder, branch_name)
        new_stages = nn.ModuleList()
        for i, stage in enumerate(branch.stages):
            # 推断 stage 输出通道：最后一个含权重的卷积
            out_ch = None
            for m in reversed(list(stage.modules())):
                if isinstance(m, nn.Conv2d):
                    out_ch = m.out_channels
                    break
            assert out_ch is not None, f"cannot infer out_ch for {branch_name}.stages[{i}]"
            adapter = ResidualAdapter(out_ch, r=r)
            key = f"{branch_name}.adapter_{i}"
            adapters[key] = adapter
            new_stages.append(_StageWithAdapter(stage, adapter))
        branch.stages = new_stages
    return adapters
