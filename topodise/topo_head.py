"""Auxiliary head ψ for stage-dependent two-channel structural supervision."""
import torch.nn as nn
import torch.nn.functional as F


class TopoHead(nn.Module):
    """Decode z_sem into two structural logit maps at the target resolution.

    In the formal pipeline, Stage0 supervises GT nerve presence and high pooled
    mask density; Stage1 uses low/high-threshold HGrad proxy targets with this
    head's parameters frozen. The channels are not skeleton/branch-point labels.
    """

    def __init__(self, z_dim: int = 512, base_ch: int = 128, target_size: int = 32):
        super().__init__()
        self.target_size = target_size
        self.proj = nn.Linear(z_dim, 4 * 4 * base_ch)
        self.base_ch = base_ch
        ups = []
        in_ch = base_ch
        # 4 -> 8 -> 16 -> 32
        for _ in range(3):
            out_ch = max(in_ch // 2, 32)
            ups.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(in_ch, out_ch, 3, padding=1),
                    nn.GroupNorm(min(8, out_ch), out_ch),
                    nn.SiLU(),
                )
            )
            in_ch = out_ch
        self.ups = nn.Sequential(*ups)
        self.head = nn.Conv2d(in_ch, 2, 1)

    def forward(self, z):
        h = self.proj(z).view(z.size(0), self.base_ch, 4, 4)
        h = self.ups(h)
        if h.shape[-1] != self.target_size:
            h = F.interpolate(h, size=self.target_size, mode="nearest")
        return self.head(h)  # [B, 2, 32, 32]
