"""TopoDiSE Stage0/Stage1 reproducibility modes."""
from __future__ import annotations

import os
import random
from typing import Any, Dict

import numpy as np
import torch


REFERENCE_RECIPE = "c8bf6b548656"
FIXED_TORCH_SEED = 42
REPRODUCIBILITY_ENV = "TOPODISE_REPRODUCIBILITY"
TRAINING_MODES = ("strict", "portable")
ALL_MODES = (*TRAINING_MODES, "checkpoint")


def normalize_reproducibility(value: Any) -> str:
    """Normalize descriptive modes and legacy bool/0/1 spellings."""
    if isinstance(value, bool):
        return "strict" if value else "portable"
    normalized = str(value).strip().lower()
    aliases = {
        "1": "strict", "true": "strict", "on": "strict", "yes": "strict",
        "2": "portable", "0": "portable", "false": "portable",
        "off": "portable", "no": "portable", "": "portable",
        "3": "checkpoint", "reference": "checkpoint", "c8": "checkpoint",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in ALL_MODES:
        raise ValueError(
            f"reproducibility must be one of {ALL_MODES}, got {value!r}"
        )
    return normalized


def resolve_reproducibility(value: Any) -> str:
    """Resolve a config value, allowing an explicit environment override."""
    return normalize_reproducibility(os.environ.get(REPRODUCIBILITY_ENV, value))


def configure_reproducibility(
    value: Any,
    seed: int = FIXED_TORCH_SEED,
) -> Dict[str, Any]:
    """Configure a strict or portable from-scratch training process.

    ``strict`` fixes every relevant RNG and deterministic backend setting.
    ``portable`` preserves the ordinary device-adaptive training path and does
    not force any RNG or backend option. The ``checkpoint`` mode is handled by
    the batch controller and must never enter a Stage0/Stage1 training function.
    """
    mode = resolve_reproducibility(value)
    if mode == "checkpoint":
        raise RuntimeError(
            "checkpoint mode skips Stage0/Stage1; run it through the combined "
            "TopoDiSE batch entry point with --checkpoint-dir"
        )
    deterministic = mode == "strict"

    seed = int(seed)
    if not deterministic:
        return {
            "enabled": False,
            "mode": mode,
            "strategy": "environment_default_portable_kernels",
            "recipe": REFERENCE_RECIPE,
            "seed": None,
            "deterministic": False,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "note": "No RNG or backend option was changed.",
        }

    # Must be present before the first CUDA operation. Launch scripts also set
    # it before Python starts so subprocess entry points are covered.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Match the numeric path of the recorded c8 run: PyTorch 2.9 keeps
    # matmul TF32 disabled but enables cuDNN TF32 by default. cuDNN's TF32
    # convolution kernels are deterministic under the policy above, so this
    # preserves strict rerun semantics without drifting from the reference
    # training trajectory solely because strict mode was enabled.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.use_deterministic_algorithms(True)
    return {
        "enabled": True,
        "mode": mode,
        "strategy": "all_rngs_deterministic_kernels",
        "recipe": REFERENCE_RECIPE,
        "seed": seed,
        "deterministic": deterministic,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "note": (
            "Python, NumPy, PyTorch CPU and CUDA RNGs were seeded; strict "
            "deterministic kernels are enabled."
        ),
    }


def seed_data_loader_worker(_worker_id: int) -> None:
    """Seed Python and NumPy from PyTorch's deterministic worker seed."""
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def capture_rng_state() -> Dict[str, Any]:
    """Capture RNG state required for exact checkpoint continuation."""
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Dict[str, Any]) -> None:
    """Restore :func:`capture_rng_state` output."""
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
