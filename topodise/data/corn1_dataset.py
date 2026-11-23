"""CORN-1 dataset for Stage0 representation pretraining.

Returns (xc, xw, x_full) where (xc, xw) is a pseudo dual-view augmentation pair
constructed from the same image (since CORN-1 has no central/whorl pairing).

If the standard CORN-1 layout (``images/<split>_orig`` + ``labels/<split>_label``)
is detected, only the real images are used for training (labels are NOT loaded as
images), and the matching nerve-fiber segmentation mask is returned as ``x_mask``
for use as a ground-truth topology target.
"""
import os
from typing import Optional, Callable
from PIL import Image
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset


_IMG_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")


def _derive_label_path(img_path: str) -> Optional[str]:
    """Map ``.../images/<split>_orig/<f>`` -> ``.../labels/<split>_label/<f>``."""
    parts = img_path.split(os.sep)
    try:
        i = len(parts) - 1 - parts[::-1].index("images")
    except ValueError:
        return None
    parts = list(parts)
    parts[i] = "labels"
    if i + 1 < len(parts):
        parts[i + 1] = parts[i + 1].replace("_orig", "_label")
    cand = os.sep.join(parts)
    return cand if os.path.exists(cand) else None


class CORN1Dataset(Dataset):
    def __init__(self, root: str, transform: Optional[Callable] = None,
                 transform_paired: Optional[Callable] = None,
                 recursive: bool = True,
                 load_masks: bool = False,
                 image_size: Optional[int] = None):
        """
        Args:
            root: directory containing image files (or a CORN-1 root with
                ``images/`` and ``labels/`` subdirs).
            transform: applied for x_full (and used for both branches if paired is None)
            transform_paired: independent transform applied a second time for whorl-view
            load_masks: if True and the CORN-1 images/labels layout is detected,
                return the nerve-fiber segmentation mask as ``x_mask``.
            image_size: target size for the deterministic mask resize (defaults to
                512 if unknown; the topo target pools to 32x32 anyway).
        """
        self.root = root
        self.transform = transform
        self.transform_paired = transform_paired or transform
        self.load_masks = load_masks
        self.mask_size = int(image_size) if image_size else 512

        # Prefer the canonical CORN-1 layout: only read from images/ so that the
        # binary label masks are never loaded as training images.
        images_dir = os.path.join(root, "images")
        scan_root = images_dir if os.path.isdir(images_dir) else root

        paths = []
        if recursive:
            for dp, _, fs in os.walk(scan_root):
                for f in fs:
                    if f.lower().endswith(_IMG_EXTS):
                        paths.append(os.path.join(dp, f))
        else:
            for f in os.listdir(scan_root):
                if f.lower().endswith(_IMG_EXTS):
                    paths.append(os.path.join(scan_root, f))
        paths.sort()
        if not paths:
            raise FileNotFoundError(f"no images found under {scan_root}")
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def _load_mask(self, img_path):
        lbl_path = _derive_label_path(img_path)
        if lbl_path is None:
            return None
        m = Image.open(lbl_path).convert("L")
        m = TF.resize(m, [self.mask_size, self.mask_size],
                      interpolation=TF.InterpolationMode.NEAREST)
        m = TF.to_tensor(m)               # [1,H,W] in [0,1]
        return (m > 0.5).float()

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("L")
        x_full = self.transform(img) if self.transform else None
        xc = self.transform(img) if self.transform else None
        xw = self.transform_paired(img) if self.transform_paired else xc
        out = {"x_full": x_full, "x_central": xc, "x_whorl": xw}
        if self.load_masks:
            m = self._load_mask(path)
            if m is not None:
                out["x_mask"] = m
        return out
