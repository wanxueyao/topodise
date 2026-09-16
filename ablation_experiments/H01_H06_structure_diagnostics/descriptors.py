"""Pinned Zhang-Suen thinning and branch-pixel (not node) descriptors."""
from typing import Tuple
import numpy as np
import torch

def zhang_suen_skeleton(mask: np.ndarray, max_iter: int = 256) -> np.ndarray:
    """Pure NumPy Zhang-Suen thinning fallback for binary 2D masks."""
    img = (mask > 0).astype(np.uint8).copy()
    if img.ndim != 2:
        raise ValueError("skeleton mask must be 2D")
    rows, cols = img.shape
    if rows < 3 or cols < 3:
        return img.astype(bool)

    def neighbors(a: np.ndarray) -> Tuple[np.ndarray, ...]:
        p2 = a[:-2, 1:-1]
        p3 = a[:-2, 2:]
        p4 = a[1:-1, 2:]
        p5 = a[2:, 2:]
        p6 = a[2:, 1:-1]
        p7 = a[2:, :-2]
        p8 = a[1:-1, :-2]
        p9 = a[:-2, :-2]
        return p2, p3, p4, p5, p6, p7, p8, p9

    for _ in range(max_iter):
        changed = False
        for step in (0, 1):
            p2, p3, p4, p5, p6, p7, p8, p9 = neighbors(img)
            center = img[1:-1, 1:-1]
            nsum = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            seq = (p2, p3, p4, p5, p6, p7, p8, p9, p2)
            transitions = sum(((seq[i] == 0) & (seq[i + 1] == 1)).astype(np.uint8) for i in range(8))
            if step == 0:
                cond = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                cond = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            remove = (center == 1) & (nsum >= 2) & (nsum <= 6) & (transitions == 1) & cond
            if remove.any():
                img[1:-1, 1:-1][remove] = 0
                changed = True
        if not changed:
            break
    return img.astype(bool)


def branch_and_endpoint_counts(skel: np.ndarray) -> Tuple[int, int]:
    s = (skel > 0).astype(np.uint8)
    if s.sum() == 0:
        return 0, 0
    padded = np.pad(s, 1, mode="constant")
    neigh = np.zeros_like(s, dtype=np.uint8)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            neigh += padded[1 + dy:1 + dy + s.shape[0], 1 + dx:1 + dx + s.shape[1]]
    endpoints = int(((s == 1) & (neigh == 1)).sum())
    branches = int(((s == 1) & (neigh >= 3)).sum())
    return branches, endpoints


def compute_mask_descriptors(mask_tensor: torch.Tensor) -> Tuple[float, float, float]:
    mask_np = mask_tensor.squeeze(0).cpu().numpy() > 0.5
    skel = zhang_suen_skeleton(mask_np)
    area = float(mask_np.size)
    length_density = float(skel.sum()) / max(area, 1.0)
    branches, endpoints = branch_and_endpoint_counts(skel)
    branch_density = float(branches) / max(float(skel.sum()), 1.0)
    endpoint_density = float(endpoints) / max(float(skel.sum()), 1.0)
    return length_density, branch_density, endpoint_density
