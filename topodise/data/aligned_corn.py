"""Align the GT mask with the independently augmented clean reference x_full.

The encoder views x_central/x_whorl are separate augmentation draws and need not
share the reference geometry. Mask replay preserves the subsequent image RNG.
"""
from copy import deepcopy

import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

from topodise.data.corn1_dataset import CORN1Dataset, _derive_label_path
from topodise.data.transforms import _GrayNorm


class AlignedCORN1Dataset(CORN1Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_transform = None
        if not self.load_masks:
            return
        missing = [p for p in self.paths if _derive_label_path(p) is None]
        if missing:
            raise FileNotFoundError(f"Missing {len(missing)} CORN-1 masks; first: {missing[0]}")
        if not isinstance(self.transform, T.Compose):
            raise TypeError("GT training requires the standard composed pretrain transform")
        # Fail closed if the upstream pipeline gains an unrecognized stochastic
        # transform. The current transforms all draw from the CPU torch RNG.
        allowed = (T.Resize, T.RandomCrop, T.RandomHorizontalFlip, T.RandomAffine, _GrayNorm)
        if any(not isinstance(t, allowed) for t in self.transform.transforms):
            raise TypeError("Update paired mask transform for the changed image pipeline")
        self.mask_transform = deepcopy(self.transform)
        for transform in self.mask_transform.transforms:
            if isinstance(transform, (T.Resize, T.RandomAffine)):
                transform.interpolation = InterpolationMode.NEAREST

    def __getitem__(self, idx):
        if not self.load_masks:
            return super().__getitem__(idx)
        path = self.paths[idx]
        with Image.open(path) as source:
            img = source.convert("L")
        # Save the reference draw so only its geometry is replayed for the mask.
        before = torch.get_rng_state()
        x_full = self.transform(img)
        # Independent encoder views; neither is forced to align with x_full.
        xc = self.transform(img)
        xw = self.transform_paired(img)
        after = torch.get_rng_state()
        with Image.open(_derive_label_path(path)) as source:
            mask = source.convert("L")
        if mask.size != img.size:
            raise ValueError(f"Image/mask dimensions differ: {path}")
        try:
            torch.set_rng_state(before)
            mask = (self.mask_transform(mask) > 0).float()
        finally:
            # GT must not change the subsequent image draws compared to Frangi
            # or HGrad, including when dataset workers perform augmentation.
            torch.set_rng_state(after)
        return {"x_full": x_full, "x_central": xc, "x_whorl": xw, "x_mask": mask}
