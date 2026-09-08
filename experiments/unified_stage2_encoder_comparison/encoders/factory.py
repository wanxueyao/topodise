from __future__ import annotations

from pathlib import Path

from ..utils import resolve_codes_path
from .base import FrozenRegionEncoder


SUPPORTED_ENCODERS = (
    "topodise_stage1",
    "topodise_native_zsem",
    "simclr_resnet18",
    "dinov2_vits14",
)


def encoder_spec(cfg: dict, name: str) -> dict:
    """在不修改锁定科学配置的前提下解析扩展编码器。"""
    source_name = "topodise_stage1" if name == "topodise_native_zsem" else name
    try:
        return cfg["encoders"][source_name]
    except KeyError as error:
        raise KeyError(f"encoder is not configured: {name}") from error


def _checkpoint(spec: dict) -> Path:
    value = str(spec.get("checkpoint", "")).strip()
    if not value:
        raise ValueError("encoder checkpoint path is empty in the experiment config")
    return resolve_codes_path(value)


def build_frozen_encoder(name: str, spec: dict, device_request: str = "auto") -> FrozenRegionEncoder:
    if name == "topodise_stage1":
        from ..pretrain_topodise import ensure_topodise_checkpoint
        from .topodise import TopoDiSEStage1Frozen

        checkpoint = ensure_topodise_checkpoint(_checkpoint(spec), spec, device_request)
        model = TopoDiSEStage1Frozen(checkpoint, bool(spec.get("use_ema", True)))
    elif name == "topodise_native_zsem":
        from .topodise import TopoDiSEStage1NativeZSemFrozen

        checkpoint = _checkpoint(spec)
        if not checkpoint.is_file():
            raise FileNotFoundError(
                "topodise_native_zsem is a reuse-only add-on experiment and will not retrain "
                f"Stage0/1. Copy the completed Stage1 checkpoint to: {checkpoint}"
            )
        model = TopoDiSEStage1NativeZSemFrozen(
            checkpoint, bool(spec.get("use_ema", True))
        )
    elif name == "simclr_resnet18":
        from ..pretrain_simclr import ensure_simclr_checkpoint
        from .simclr import SimCLRResNet18Frozen

        checkpoint = ensure_simclr_checkpoint(_checkpoint(spec), spec, device_request)
        model = SimCLRResNet18Frozen(checkpoint)
    elif name == "dinov2_vits14":
        from .dinov2 import DINOv2ViTS14Frozen

        model = DINOv2ViTS14Frozen(
            _checkpoint(spec), str(spec.get("model_name", "vit_small_patch14_dinov2"))
        )
    else:
        raise KeyError(f"unknown encoder: {name}")
    model.assert_frozen()
    return model
