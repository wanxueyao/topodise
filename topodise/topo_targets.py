"""Topology supervision targets for the Stage0/1 topo head.

The legacy ``hgrad`` mode is a single-direction horizontal-gradient placeholder
(kept for exact numeric parity with old checkpoints). ``sobel`` and ``frangi``
provide real ridge / vesselness topology targets for the corneal nerve fibers,
computed on-the-fly with no external dependencies (pure torch, runs on GPU).

All modes return a ``[B, 2, out_size, out_size]`` binary target in {0, 1} so the
two-channel ``TopoHead`` (skeleton + branch logits) can be supervised directly.
"""
from typing import Sequence

import torch
import torch.nn.functional as F


def _gaussian_kernel1d(sigma: float, device, dtype):
    radius = max(1, int(3.0 * sigma + 0.5))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-(x ** 2) / (2.0 * sigma * sigma))
    k = k / k.sum()
    return k, radius


def _gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable depthwise Gaussian blur (reflect padding)."""
    k, r = _gaussian_kernel1d(sigma, x.device, x.dtype)
    C = x.size(1)
    kx = k.view(1, 1, 1, -1).expand(C, 1, 1, -1)
    ky = k.view(1, 1, -1, 1).expand(C, 1, -1, 1)
    x = F.pad(x, (r, r, 0, 0), mode="reflect")
    x = F.conv2d(x, kx, groups=C)
    x = F.pad(x, (0, 0, r, r), mode="reflect")
    x = F.conv2d(x, ky, groups=C)
    return x


def _sobel(x: torch.Tensor):
    kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
                      device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    ky = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
                      device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    C = x.size(1)
    kx = kx.expand(C, 1, 3, 3)
    ky = ky.expand(C, 1, 3, 3)
    xp = F.pad(x, (1, 1, 1, 1), mode="reflect")
    gx = F.conv2d(xp, kx, groups=C)
    gy = F.conv2d(xp, ky, groups=C)
    return gx, gy


def _binarize(m: torch.Tensor, out_size: int, k: float = 0.0) -> torch.Tensor:
    """Pool to out_size and threshold per-sample at mean + k*std -> {0,1}."""
    t = F.adaptive_avg_pool2d(m, out_size)
    flat = t.flatten(1)
    thr = flat.mean(dim=1, keepdim=True)
    if k != 0.0:
        thr = thr + k * flat.std(dim=1, keepdim=True)
    thr = thr.view(-1, 1, 1, 1)
    return (t > thr).float()


def _frangi_vesselness(x: torch.Tensor, sigmas: Sequence[float],
                       beta: float = 0.5, c: float = 0.5) -> torch.Tensor:
    """2D Frangi vesselness for bright ridges on dark background ([B,1,H,W])."""
    best = torch.zeros_like(x)
    for sigma in sigmas:
        xs = _gaussian_blur(x, float(sigma))
        gx, gy = _sobel(xs)
        gxx, _ = _sobel(gx)
        gxy, gyy = _sobel(gy)
        s2 = float(sigma) * float(sigma)
        Dxx = gxx * s2
        Dyy = gyy * s2
        Dxy = gxy * s2
        tmp = torch.sqrt(((Dxx - Dyy) * 0.5) ** 2 + Dxy ** 2 + 1e-12)
        mu = (Dxx + Dyy) * 0.5
        lam_a = mu + tmp
        lam_b = mu - tmp
        # order by absolute magnitude: |a1| <= |a2|
        swap = lam_a.abs() <= lam_b.abs()
        a1 = torch.where(swap, lam_a, lam_b)
        a2 = torch.where(swap, lam_b, lam_a)
        Rb = a1 / (a2 + 1e-12)
        S = torch.sqrt(a1 ** 2 + a2 ** 2 + 1e-12)
        V = torch.exp(-(Rb ** 2) / (2.0 * beta * beta)) * (1.0 - torch.exp(-(S ** 2) / (2.0 * c * c)))
        V = torch.where(a2 < 0, V, torch.zeros_like(V))  # bright ridges only
        best = torch.maximum(best, V)
    return best


def compute_topo_target(x: torch.Tensor, mode: str = "hgrad", out_size: int = 32,
                        sigmas: Sequence[float] = (1.0, 2.0, 3.0),
                        beta: float = 0.5, c: float = 0.5,
                        mask: torch.Tensor = None) -> torch.Tensor:
    """Build a ``[B, 2, out_size, out_size]`` topology target in {0, 1}.

    Channel 0 ~ skeleton / ridge map, channel 1 ~ branch / ridge-core cue.

    Modes
    -----
    - ``hgrad``  : legacy single-direction horizontal-gradient placeholder
                   (both channels identical; exact numeric parity).
    - ``sobel``  : ch0 = gradient-magnitude ridge, ch1 = coarser-scale ridge.
    - ``frangi`` : ch0 = multi-scale Frangi vesselness, ch1 = sparse ridge cores.
    - ``gt``     : ground-truth nerve-fiber segmentation ``mask`` ([B,1,H,W] in
                   {0,1}). ch0 = density>0 region, ch1 = denser ridge cores.
    """
    if mode == "gt":
        if mask is None:
            raise ValueError("topo_target='gt' requires a `mask` tensor")
        m = mask.float()
        if m.dim() == 3:
            m = m.unsqueeze(1)
        dens = F.adaptive_avg_pool2d(m, out_size)   # nerve density per cell
        ch0 = (dens > 0).float()                    # any nerve present
        flat = dens.flatten(1)
        thr = (flat.mean(dim=1, keepdim=True) + flat.std(dim=1, keepdim=True)).view(-1, 1, 1, 1)
        ch1 = (dens > thr).float()                  # denser ridge cores
        return torch.cat([ch0, ch1], dim=1)

    if mode == "hgrad_dual":
        gradient = F.pad((x[..., 1:] - x[..., :-1]).abs(), (1, 0, 0, 0))
        return torch.cat([_binarize(gradient, out_size, k=0.0),
                          _binarize(gradient, out_size, k=1.0)], dim=1)

    if mode == "hgrad":
        dx = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs()
        grad_map = F.pad(dx, (1, 0, 0, 0))
        t = F.adaptive_avg_pool2d(grad_map, out_size)
        t = (t > t.mean()).float()
        return torch.cat([t, t], dim=1)

    if mode == "sobel":
        gx, gy = _sobel(x)
        mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-12)
        ch0 = _binarize(mag, out_size, k=0.0)
        ch1 = _binarize(_gaussian_blur(mag, 2.0), out_size, k=0.5)
        return torch.cat([ch0, ch1], dim=1)

    if mode == "frangi":
        V = _frangi_vesselness(x, sigmas, beta=beta, c=c)
        ch0 = _binarize(V, out_size, k=0.0)
        ch1 = _binarize(V, out_size, k=1.0)   # sparser ridge cores / branches
        return torch.cat([ch0, ch1], dim=1)

    raise ValueError(f"unknown topo_target mode: {mode}")
