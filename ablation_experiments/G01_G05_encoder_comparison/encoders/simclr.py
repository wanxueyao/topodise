from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from core.basic_models import get_backbone

from ..utils import sha256_file
from .base import FrozenRegionEncoder


class SimCLRResNet18Frozen(FrozenRegionEncoder):
    encoder_name = "simclr_resnet18"
    raw_dim = 512

    def __init__(self, checkpoint: Path):
        super().__init__()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"internal error: ensured SimCLR cache is absent: {checkpoint}")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = payload.get("backbone_state", payload) if isinstance(payload, dict) else payload
        if not isinstance(state, dict):
            raise TypeError("unsupported SimCLR checkpoint payload")
        meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
        self.pretrain_meta = meta
        backbone = str(meta.get("backbone", "resnet18"))
        if backbone != "resnet18":
            raise ValueError(f"Core protocol requires SimCLR ResNet-18, got {backbone!r}")
        self.backbone = get_backbone("resnet18", pretrained=False)
        self.backbone.load_state_dict(state, strict=True)
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.checkpoint_path = checkpoint.resolve()
        self.checkpoint_sha256 = sha256_file(checkpoint)
        self.preprocessing_name = "rgb_imagenet_resize256_centercrop224"
        self.freeze()

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        return self.transform(image.convert("RGB"))

    def encode_region(self, images: torch.Tensor, region: str) -> torch.Tensor:
        if region not in ("central", "whorl"):
            raise ValueError(f"unknown region: {region}")
        return self.backbone(images)

    def metadata(self) -> dict:
        return {**super().metadata(), "simclr_pretraining": self.pretrain_meta}
