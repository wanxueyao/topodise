"""
prepare_dataset.py  –  Convert CORN-1 TIF images to nnUNet v2 raw format.

Directory layout produced:
    codes/unet/nnunet_data/nnUNet_raw/Dataset001_Cornea/
        imagesTr/   {case}_0000.nii.gz   (grayscale, float32)
        labelsTr/   {case}.nii.gz        (int, 0=bg / 1=cell_boundary)
        imagesTs/   {case}_0000.nii.gz
        dataset.json

nnUNet uses its own internal train/val split via cross-validation, so the
original train+val images are merged into imagesTr/.  The original test
split is kept in imagesTs/ for final evaluation.

Usage:
    python prepare_dataset.py
    python prepare_dataset.py --include_val_in_train  (default: True)
"""

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from unet.config import (
    CHANNEL_NAMES, DATASET_ID, DATASET_NAME,
    IMAGE_TEST, IMAGE_TRAIN, IMAGE_VAL,
    LABEL_TRAIN, LABEL_VAL,
    LABELS, NNUNET_RAW, PIXEL_SPACING,
)


def tif_to_nifti_image(tif_path: Path) -> nib.Nifti1Image:
    """Load a grayscale TIF and return a 2-D NIfTI image.

    nnUNet v2 expects shape  (1, H, W)  for 2-D single-channel images.
    Spacing is set to [999.0, sy, sx]; the large first-axis spacing is a
    conventional hint so nnUNet recognises this as a 2-D dataset.
    """
    arr = np.array(Image.open(tif_path)).astype(np.float32)  # (H, W)
    arr = arr[np.newaxis, ...]                                 # (1, H, W)

    # NIfTI affine: diagonal with spacings [sz, sy, sx]
    sz = 999.0
    sy, sx = float(PIXEL_SPACING[0]), float(PIXEL_SPACING[1])
    affine = np.diag([sz, sy, sx, 1.0])
    return nib.Nifti1Image(arr, affine=affine)


def tif_to_nifti_label(tif_path: Path) -> nib.Nifti1Image:
    """Load a binary TIF label (0 / 255) and return a 2-D NIfTI label image.

    Pixel value 255 → 1  (cell boundary).
    Shape: (1, H, W), dtype int16.
    """
    arr = np.array(Image.open(tif_path)).astype(np.uint8)     # (H, W)
    arr = (arr > 128).astype(np.int16)                        # binarise → 0/1
    arr = arr[np.newaxis, ...]                                 # (1, H, W)

    sz = 999.0
    sy, sx = float(PIXEL_SPACING[0]), float(PIXEL_SPACING[1])
    affine = np.diag([sz, sy, sx, 1.0])
    return nib.Nifti1Image(arr, affine=affine)


def copy_split(
    image_dir: Path,
    label_dir: Path | None,
    out_image_dir: Path,
    out_label_dir: Path | None,
    desc: str,
) -> list[str]:
    """Convert one split (train/val/test) and return list of case names."""
    out_image_dir.mkdir(parents=True, exist_ok=True)
    if out_label_dir is not None:
        out_label_dir.mkdir(parents=True, exist_ok=True)

    tif_files = sorted(image_dir.glob("*.tif"))
    case_names = []

    for tif_path in tqdm(tif_files, desc=desc):
        stem = tif_path.stem          # e.g. "batch_1_1"
        case_names.append(stem)

        # image
        out_img = out_image_dir / f"{stem}_0000.nii.gz"
        if not out_img.exists():
            nib.save(tif_to_nifti_image(tif_path), str(out_img))

        # label (only for training splits)
        if label_dir is not None and out_label_dir is not None:
            label_path = label_dir / tif_path.name
            if not label_path.exists():
                raise FileNotFoundError(
                    f"Label not found for image {tif_path.name}: expected {label_path}"
                )
            out_lbl = out_label_dir / f"{stem}.nii.gz"
            if not out_lbl.exists():
                nib.save(tif_to_nifti_label(label_path), str(out_lbl))

    return case_names


def build_dataset_json(
    dataset_dir: Path,
    training_cases: list[str],
    test_cases: list[str],
) -> None:
    """Write dataset.json in nnUNet v2 format."""
    dataset_json = {
        "channel_names": CHANNEL_NAMES,
        "labels": LABELS,
        "numTraining": len(training_cases),
        "file_ending": ".nii.gz",
        "name": DATASET_NAME,
        "description": "Corneal endothelial cell segmentation (CORN-1 dataset)",
        "reference": "",
        "licence": "",
        "release": "1.0",
        "overwrite_image_reader_writer": "NibabelIOWithReorient",
    }
    out_path = dataset_dir / "dataset.json"
    with open(out_path, "w") as f:
        json.dump(dataset_json, f, indent=4)
    print(f"[OK] dataset.json  →  {out_path}")


def prepare(include_val_in_train: bool = True) -> None:
    dataset_dir = NNUNET_RAW / DATASET_NAME
    images_tr   = dataset_dir / "imagesTr"
    labels_tr   = dataset_dir / "labelsTr"
    images_ts   = dataset_dir / "imagesTs"

    print(f"Output directory: {dataset_dir}")

    # --- training split ---
    train_cases = copy_split(IMAGE_TRAIN, LABEL_TRAIN, images_tr, labels_tr,
                             desc="Train images")

    # --- validation split (merge into train, nnUNet manages splits itself) ---
    if include_val_in_train:
        val_cases = copy_split(IMAGE_VAL, LABEL_VAL, images_tr, labels_tr,
                               desc="Val   images (merged into train)")
        training_cases = train_cases + val_cases
    else:
        training_cases = train_cases

    # --- test split (images only, no labels) ---
    test_cases = copy_split(IMAGE_TEST, None, images_ts, None,
                            desc="Test  images")

    build_dataset_json(dataset_dir, training_cases, test_cases)

    print("\n[Summary]")
    print(f"  Training cases : {len(training_cases)}")
    print(f"  Test cases     : {len(test_cases)}")
    print(f"  Dataset dir    : {dataset_dir}")
    print(
        "\nNext step:\n"
        f"  python train.py\n"
        f"  (or run manually: nnUNetv2_plan_and_preprocess -d {DATASET_ID} -c 2d)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare CORN-1 dataset for nnUNet v2")
    parser.add_argument(
        "--no_val_merge",
        action="store_true",
        help="Do NOT merge validation images into the training set "
             "(by default val is merged; nnUNet does its own cross-validation)",
    )
    args = parser.parse_args()
    prepare(include_val_in_train=not args.no_val_merge)
