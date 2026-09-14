"""Shared Stage0/1 dual-threshold HGrad; delegate other targets to historical code."""
import torch
import torch.nn.functional as F

from topodise.topo_targets import _binarize, compute_topo_target as historical_target


@torch.no_grad()
def compute_topo_target(x, mode="hgrad", out_size=32,
                        sigmas=(1.0, 2.0, 3.0), beta=0.5, c=0.5, mask=None):
    if mode != "hgrad_dual":
        return historical_target(x, mode, out_size, sigmas, beta, c, mask)
    if x.ndim != 4 or x.shape[1] != 1 or out_size < 2:
        raise ValueError("hgrad_dual requires [B,1,H,W] input and out_size >= 2")
    # Preserve the legacy horizontal difference, padding and pooling. Only
    # thresholding changes: per-image mean / mean+std, as in Frangi.
    gradient = F.pad((x[..., 1:] - x[..., :-1]).abs(), (1, 0, 0, 0))
    return torch.cat([
        _binarize(gradient, out_size, k=0.0),
        _binarize(gradient, out_size, k=1.0),
    ], dim=1)
