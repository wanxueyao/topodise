from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import torch
from PIL import Image
from torch import nn


class FrozenRegionEncoder(nn.Module, ABC):
    """为区域特征或原生成对特征提供冻结特征源。"""

    encoder_name: str
    raw_dim: int
    feature_mode: str = "region"
    checkpoint_path: Path
    checkpoint_sha256: str
    preprocessing_name: str

    @abstractmethod
    def preprocess(self, image: Image.Image) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def encode_region(self, images: torch.Tensor, region: str) -> torch.Tensor:
        raise NotImplementedError

    def encode_pair(
        self, central_images: torch.Tensor, whorl_images: torch.Tensor
    ) -> torch.Tensor:
        raise RuntimeError(f"{self.encoder_name} does not expose native pair features")

    def freeze(self) -> "FrozenRegionEncoder":
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        return self

    def train(self, mode: bool = True):
        # 即使下游分类头处于训练模式，冻结的编码器也必须保持评估模式。
        return super().train(False)

    def assert_frozen(self) -> None:
        trainable = [name for name, p in self.named_parameters() if p.requires_grad]
        if trainable:
            raise AssertionError(f"frozen encoder has trainable parameters: {trainable[:10]}")
        if self.training:
            raise AssertionError("frozen encoder unexpectedly entered train mode")

    def metadata(self) -> dict:
        return {
            "encoder": self.encoder_name,
            "raw_dim": self.raw_dim,
            "feature_mode": self.feature_mode,
            "checkpoint": str(self.checkpoint_path),
            "checkpoint_sha256": self.checkpoint_sha256,
            "preprocessing": self.preprocessing_name,
            "all_parameters_frozen": True,
        }
