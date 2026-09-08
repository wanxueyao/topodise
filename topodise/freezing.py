"""Stage-wise freezing policies (surgical fine-tuning)."""
from typing import Literal
import torch.nn as nn

from .encoder import TopoDiSEEncoder


FreezePolicy = Literal["none", "rscp_high", "racf_all"]


def apply_freeze_policy(
    encoder: TopoDiSEEncoder,
    unet: nn.Module = None,
    topo_head: nn.Module = None,
    policy: FreezePolicy = "none",
) -> dict:
    """Apply requires_grad flags. Returns summary dict for logging."""
    trainable = {"encoder": 0, "unet": 0, "topo_head": 0}
    total = {"encoder": 0, "unet": 0, "topo_head": 0}

    def _stat(module, key):
        for p in module.parameters():
            total[key] += p.numel()
            if p.requires_grad:
                trainable[key] += p.numel()

    if policy == "none":
        # 全部可训练（阶段 0）
        for m, k in [(encoder, "encoder"), (unet, "unet"), (topo_head, "topo_head")]:
            if m is None:
                continue
            for p in m.parameters():
                p.requires_grad_(True)
            _stat(m, k)

    elif policy == "rscp_high":
        # 阶段 1：冻结 stage1/2，释放 stage3/4 + cross-attn + GN affine
        for p in encoder.parameters():
            p.requires_grad_(False)
        # 释放后半 stage
        for branch_name in ("branch_c", "branch_w"):
            branch = getattr(encoder, branch_name)
            n = len(branch.stages)
            for i in range(n // 2, n):
                for p in branch.stages[i].parameters():
                    p.requires_grad_(True)
        # cross-attn / proj 全部释放
        if hasattr(encoder, "fuse_c"):
            for p in encoder.fuse_c.parameters():
                p.requires_grad_(True)
            for p in encoder.fuse_w.parameters():
                p.requires_grad_(True)
        for p in encoder.proj.parameters():
            p.requires_grad_(True)
        # 早期 GN 仿射放开（feature norm refit）
        for branch_name in ("branch_c", "branch_w"):
            branch = getattr(encoder, branch_name)
            for i in range(min(len(branch.stages) // 2, len(branch.stages))):
                for m in branch.stages[i].modules():
                    if isinstance(m, nn.GroupNorm):
                        for p in m.parameters():
                            p.requires_grad_(True)
        # ε_θ, ψ 全部冻结
        if unet is not None:
            for p in unet.parameters():
                p.requires_grad_(False)
        if topo_head is not None:
            for p in topo_head.parameters():
                p.requires_grad_(False)
        _stat(encoder, "encoder")
        if unet is not None:
            _stat(unet, "unet")
        if topo_head is not None:
            _stat(topo_head, "topo_head")

    elif policy == "racf_all":
        # 阶段 2：encoder/unet/ψ 全冻结；适配器与分类头另外 requires_grad=True
        for m, k in [(encoder, "encoder"), (unet, "unet"), (topo_head, "topo_head")]:
            if m is None:
                continue
            for p in m.parameters():
                p.requires_grad_(False)
            _stat(m, k)

    else:
        raise ValueError(f"unknown policy: {policy}")

    return {
        "policy": policy,
        "trainable_params": trainable,
        "total_params": total,
    }
