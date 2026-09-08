"""TopoDiSE Stage0/Stage1 的可复现 seed 开关。"""
from __future__ import annotations

from typing import Any, Dict

import torch


REFERENCE_RECIPE = "c8bf6b548656"
FIXED_TORCH_SEED = 42


def configure_reproducibility(enabled: bool) -> Dict[str, Any]:
    """仅在开启时执行 ``torch.manual_seed(42)``。

    此开关不控制 RNG checkpoint，也不修改任何 PyTorch/CUDA 后端选项。
    """
    if not enabled:
        return {
            "enabled": False,
            "strategy": "environment_default",
            "recipe": None,
            "seed": None,
            "note": "No seed or backend option was changed.",
        }

    torch.manual_seed(FIXED_TORCH_SEED)
    return {
        "enabled": True,
        "strategy": "fixed_torch_manual_seed",
        "recipe": REFERENCE_RECIPE,
        "seed": FIXED_TORCH_SEED,
        "note": "Only torch.manual_seed(42) was executed.",
    }


def capture_rng_state() -> Dict[str, Any]:
    """捕获断点续训所需的 PyTorch 随机数状态。"""
    state: Dict[str, Any] = {"torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Dict[str, Any]) -> None:
    """恢复 :func:`capture_rng_state` 保存的随机数状态。"""
    if not state:
        return
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
