from __future__ import annotations

from pathlib import Path

import timm
import torch
from PIL import Image
from torchvision import transforms

from ..utils import sha256_file
from .base import FrozenRegionEncoder


def _unwrap_state(payload):
    if not isinstance(payload, dict):
        return payload
    for key in ("model", "state_dict", "teacher", "student"):
        if key in payload and isinstance(payload[key], dict):
            payload = payload[key]
            break
    cleaned = {}
    for key, value in payload.items():
        new_key = key
        for prefix in ("module.", "backbone."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        cleaned[new_key] = value
    return cleaned


class DINOv2ViTS14Frozen(FrozenRegionEncoder):
    encoder_name = "dinov2_vits14"

    def __init__(self, checkpoint: Path, model_name: str = "vit_small_patch14_dinov2"):
        super().__init__()
        if not checkpoint.is_file():
            raise FileNotFoundError(
                "DINOv2-S/14 local checkpoint not found. Automatic downloads are intentionally "
                f"disabled for reproducibility: {checkpoint}"
            )
        if model_name not in timm.list_models():
            raise ValueError(f"timm {timm.__version__} does not provide {model_name!r}")
        self.backbone = timm.create_model(
            model_name, pretrained=False, num_classes=0, dynamic_img_size=True
        )
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = _unwrap_state(payload)
        missing, unexpected = self.backbone.load_state_dict(state, strict=False)
        allowed_missing = {"head.weight", "head.bias"}
        significant_missing = [x for x in missing if x not in allowed_missing]
        # Meta 官方 DINOv2 检查点包含仅训练时使用的遮罩标记，
        # 而 timm 的推理骨干网络有意省略了该参数。
        allowed_unexpected = {"mask_token"}
        significant_unexpected = [
            x for x in unexpected
            if not x.startswith("head.") and x not in allowed_unexpected
        ]
        if significant_missing or significant_unexpected:
            raise RuntimeError(
                "DINOv2 checkpoint/model mismatch: "
                f"missing={significant_missing[:20]}, unexpected={significant_unexpected[:20]}"
            )
        self.raw_dim = int(self.backbone.num_features)
        self.model_name = model_name
        if self.raw_dim != 384:
            raise ValueError(f"DINOv2-S/14 must produce 384-D features, got {self.raw_dim}")
        self.transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.checkpoint_path = checkpoint.resolve()
        self.checkpoint_sha256 = sha256_file(checkpoint)
        self.preprocessing_name = "dinov2_rgb_bicubic256_centercrop224_imagenet_norm"
        self.freeze()

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        return self.transform(image.convert("RGB"))

    def encode_region(self, images: torch.Tensor, region: str) -> torch.Tensor:
        if region not in ("central", "whorl"):
            raise ValueError(f"unknown region: {region}")
        output = self.backbone.forward_features(images)
        if isinstance(output, dict):
            for key in ("x_norm_clstoken", "x_prenorm"):
                if key in output:
                    value = output[key]
                    return value[:, 0] if value.ndim == 3 else value
            raise KeyError(f"DINOv2 forward_features keys not understood: {sorted(output)}")
        return output[:, 0] if output.ndim == 3 else output

    def metadata(self) -> dict:
        return {**super().metadata(), "timm_model_name": self.model_name, "weights_local_only": True}
