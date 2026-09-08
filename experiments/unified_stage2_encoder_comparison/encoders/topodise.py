from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from topodise.data.transforms import build_eval_transform
from topodise.encoder import EncoderConfig, TopoDiSEEncoder

from ..utils import sha256_file
from .base import FrozenRegionEncoder


class TopoDiSEStage1Frozen(FrozenRegionEncoder):
    encoder_name = "topodise_stage1"

    def __init__(self, checkpoint: Path, use_ema: bool = True):
        super().__init__()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"TopoDiSE Stage1 checkpoint not found: {checkpoint}")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        parent = payload.get("parent_config") or payload.get("config")
        if not isinstance(parent, dict):
            raise KeyError("TopoDiSE checkpoint has no parent_config/config")
        cfg = EncoderConfig(
            in_channels=1,
            base_channels=int(parent["encoder_base_channels"]),
            channel_mult=tuple(parent["encoder_channel_mult"]),
            blocks_per_stage=tuple(parent["encoder_blocks_per_stage"]),
            z_dim=int(parent["z_dim"]),
            image_size=int(parent.get("image_size", 256)),
        )
        self.backbone = TopoDiSEEncoder(cfg)
        state = payload.get("ema_encoder") if use_ema else None
        state = state or payload.get("encoder")
        if state is None:
            raise KeyError("TopoDiSE checkpoint has neither ema_encoder nor encoder state")
        missing, unexpected = self.backbone.load_state_dict(state, strict=False)
        self.load_missing_keys = tuple(missing)
        self.load_unexpected_keys = tuple(unexpected)
        branch_missing = [name for name in missing if name.startswith(("branch_c.", "branch_w."))]
        branch_unexpected = [name for name in unexpected if name.startswith(("branch_c.", "branch_w."))]
        if branch_missing or branch_unexpected:
            raise RuntimeError(
                f"TopoDiSE branch mismatch: missing={branch_missing}, unexpected={branch_unexpected}"
            )
        self.raw_dim = int(self.backbone.branch_c.final_ch)
        if self.raw_dim != int(self.backbone.branch_w.final_ch):
            raise RuntimeError("TopoDiSE central/whorl branch dimensions differ")
        self.transform = build_eval_transform(int(parent.get("image_size", 256)))
        self.checkpoint_path = checkpoint.resolve()
        self.checkpoint_sha256 = sha256_file(checkpoint)
        self.state_source = "ema_encoder" if use_ema and payload.get("ema_encoder") is not None else "encoder"
        self.parent_checkpoint_sha256 = payload.get("parent_checkpoint_sha256")
        self.preprocessing_name = f"topodise_gray_minus1_1_center_{parent.get('image_size', 256)}"
        self.freeze()

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        return self.transform(image)

    def encode_region(self, images: torch.Tensor, region: str) -> torch.Tensor:
        if region == "central":
            cls, _ = self.backbone.branch_c(images)
        elif region == "whorl":
            cls, _ = self.backbone.branch_w(images)
        else:
            raise ValueError(f"unknown region: {region}")
        return cls

    def metadata(self) -> dict:
        return {
            **super().metadata(),
            "state_source": self.state_source,
            "stage0_parent_checkpoint_sha256": self.parent_checkpoint_sha256,
            "feature_path": "branch_c/branch_w pooled CLS only; fusion and proj bypassed",
        }


class TopoDiSEStage1NativeZSemFrozen(TopoDiSEStage1Frozen):
    """论文定义的原生成对融合后冻结 Stage1 表征。"""

    encoder_name = "topodise_native_zsem"
    feature_mode = "native_pair"

    def __init__(self, checkpoint: Path, use_ema: bool = True):
        super().__init__(checkpoint, use_ema)
        native_prefixes = ("fuse_c.", "fuse_w.", "proj.")
        native_missing = [
            name for name in self.load_missing_keys if name.startswith(native_prefixes)
        ]
        native_unexpected = [
            name for name in self.load_unexpected_keys if name.startswith(native_prefixes)
        ]
        if native_missing or native_unexpected:
            raise RuntimeError(
                "native TopoDiSE fusion/projection checkpoint mismatch: "
                f"missing={native_missing}, unexpected={native_unexpected}"
            )
        self.raw_dim = int(self.backbone.cfg.z_dim)

    def encode_pair(
        self, central_images: torch.Tensor, whorl_images: torch.Tensor
    ) -> torch.Tensor:
        z_sem, _ = self.backbone(central_images, whorl_images)
        return z_sem

    def metadata(self) -> dict:
        metadata = super().metadata()
        metadata.update({
            "encoder": self.encoder_name,
            "raw_dim": self.raw_dim,
            "feature_mode": self.feature_mode,
            "feature_path": (
                "native TopoDiSE z_sem after frozen branch_c/branch_w, "
                "bidirectional cross-attention fusion, and proj"
            ),
            "native_fusion_frozen": True,
            "native_projection_frozen": True,
        })
        return metadata
