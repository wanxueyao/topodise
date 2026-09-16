"""Online SimCLR/DINOv2 bridge into the unchanged full TopoDiSE MIL trainer."""
from pathlib import Path
import torch
from torch import nn


class VectorAdapter(nn.Module):
    def __init__(self, dim, rank):
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return x + self.up(torch.nn.functional.gelu(self.down(x)))


class Stage2EncoderBridge(nn.Module):
    def __init__(self, frozen, common_dim=512):
        super().__init__()
        self.frozen = frozen
        self.raw_dim, self.common_dim = frozen.raw_dim, common_dim
        self.pair_projection = nn.Sequential(nn.LayerNorm(2 * self.raw_dim),
                                             nn.Linear(2 * self.raw_dim, common_dim))
        self.central_adapter = nn.Identity()
        self.whorl_adapter = nn.Identity()
        self.register_buffer("rgb_mean", torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer("rgb_std", torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))

    def attach_stage2_adapters(self, rank):
        # The full trainer freezes first, then activates only these new modules.
        self.central_adapter = VectorAdapter(self.raw_dim, rank)
        self.whorl_adapter = VectorAdapter(self.raw_dim, rank)
        return {"central_adapter": self.central_adapter, "whorl_adapter": self.whorl_adapter,
                "pair_projection": self.pair_projection}

    def train(self, mode=True):
        super().train(mode)
        self.frozen.eval()
        return self

    def _region(self, gray, region):
        # Stage2's existing geometry/augmentations and morphology remain shared.
        # Its tensors are grayscale [-1,1]; restore [0,1], repeat RGB, normalize.
        rgb = ((gray + 1) * .5).expand(-1, 3, -1, -1)
        rgb = (rgb - self.rgb_mean) / self.rgb_std
        with torch.no_grad():
            return self.frozen.encode_region(rgb, region)

    def forward(self, central, whorl, return_cls=False):
        c = self.central_adapter(self._region(central, "central"))
        w = self.whorl_adapter(self._region(whorl, "whorl"))
        z = self.pair_projection(torch.cat([c, w], dim=1))
        return z, {"cls_c": c, "cls_w": w}


def build_stage2_encoder(cfg, device):
    path = Path(cfg.encoder_checkpoint)
    if cfg.encoder_backend == "simclr_resnet18":
        from .simclr import SimCLRResNet18Frozen
        frozen = SimCLRResNet18Frozen(path)
    elif cfg.encoder_backend == "dinov2_vits14":
        from .dinov2 import DINOv2ViTS14Frozen
        frozen = DINOv2ViTS14Frozen(path, cfg.encoder_model_name)
    else:
        raise ValueError(f"Unknown Stage2 encoder backend: {cfg.encoder_backend}")
    encoder = Stage2EncoderBridge(frozen, cfg.encoder_common_dim).to(device)
    pc = {"z_dim": cfg.encoder_common_dim, "T": 1000, "schedule": "cosine",
          "encoder_backend": cfg.encoder_backend, "checkpoint_sha256": frozen.checkpoint_sha256}
    return encoder, nn.Identity().to(device), nn.Identity().to(device), pc
