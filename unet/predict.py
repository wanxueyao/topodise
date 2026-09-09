"""
predict.py  –  Run inference with a trained nnUNet v2 corneal segmentation model.

Accepts raw TIF images (or a folder of TIFs) and writes binary segmentation
masks as PNG / TIF files.  Can also be imported as a Python module.

Command-line usage
------------------
  # Predict on a folder of TIF images:
  python predict.py \\
      --input  /path/to/images/ \\
      --output /path/to/masks/  \\
      --model  codes/unet/nnunet_data/nnUNet_results/Dataset001_Cornea

  # Predict on a single image:
  python predict.py \\
      --input  image.tif \\
      --output output_mask.png \\
      --model  codes/unet/nnunet_data/nnUNet_results/Dataset001_Cornea

  # Predict using a specific fold (default: fold_all or best fold available):
  python predict.py --input … --output … --model … --fold 0

Python API
----------
  from predict import load_predictor, predict_single, predict_folder

  predictor = load_predictor(model_dir, fold="all")

  mask = predict_single("image.tif", predictor)      # returns np.ndarray (H, W) bool
  predict_folder("input/", "output/", predictor)
"""

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from unet.config import (
    DATASET_NAME, NNUNET_CONFIG,
    NNUNET_PREPROCESSED, NNUNET_RAW, NNUNET_RESULTS,
    PIXEL_SPACING,
)


# --------------------------------------------------------------------------- #
# Environment helpers
# --------------------------------------------------------------------------- #

def _set_nnunet_env() -> None:
    """Set nnUNet path environment variables to the local data directory."""
    os.environ["nnUNet_raw"]          = str(NNUNET_RAW)
    os.environ["nnUNet_preprocessed"] = str(NNUNET_PREPROCESSED)
    os.environ["nnUNet_results"]      = str(NNUNET_RESULTS)


def _resolve_model_dir(model_dir: str | Path | None) -> Path:
    """Return the resolved model directory, defaulting to the project results."""
    if model_dir is None:
        model_dir = NNUNET_RESULTS / DATASET_NAME
    model_dir = Path(model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(
            f"Model directory not found: {model_dir}\n"
            "Train the model first with:  python train.py"
        )
    return model_dir


def _find_trained_model_dir(model_dir: Path) -> Path:
    """Return the trainer/config directory that contains the fold checkpoints."""
    config_dirs = [d for d in model_dir.iterdir() if d.is_dir() and NNUNET_CONFIG in d.name]
    if not config_dirs:
        raise FileNotFoundError(
            f"No trained model with config '{NNUNET_CONFIG}' found in {model_dir}.\n"
            "Train the model first with:  python train.py"
        )
    return config_dirs[0]


def get_available_folds(model_dir: str | Path | None = None) -> list[int]:
    """Return all completed fold indices that can be used for nnU-Net inference."""
    resolved_model_dir = _resolve_model_dir(model_dir)
    trained_model_dir = _find_trained_model_dir(resolved_model_dir)
    folds = sorted(
        int(d.name.split("_")[1])
        for d in trained_model_dir.iterdir()
        if d.is_dir() and d.name.startswith("fold_") and (d / "checkpoint_final.pth").exists()
    )
    if not folds:
        raise FileNotFoundError(
            f"No completed fold checkpoints in {trained_model_dir}.\n"
            "Make sure training finished (checkpoint_final.pth must exist)."
        )
    return folds


# --------------------------------------------------------------------------- #
# TIF → temporary NIfTI helpers (nnUNet reads .nii.gz internally)
# --------------------------------------------------------------------------- #

def _tif_to_nifti(tif_path: Path, out_path: Path) -> None:
    """Convert a single TIF image to a 2-D NIfTI file for nnUNet."""
    arr = np.array(Image.open(tif_path)).astype(np.float32)
    arr = arr[np.newaxis, ...]   # (1, H, W)
    sy, sx = float(PIXEL_SPACING[0]), float(PIXEL_SPACING[1])
    affine = np.diag([999.0, sy, sx, 1.0])
    nib.save(nib.Nifti1Image(arr, affine=affine), str(out_path))


def _nifti_to_mask(nii_path: Path) -> np.ndarray:
    """Load a NIfTI segmentation file produced by nnUNet and return a 2-D bool array."""
    arr = nib.load(str(nii_path)).get_fdata()   # shape (1, H, W) or (H, W)
    if arr.ndim == 3:
        arr = arr[0]
    return (arr > 0).astype(np.uint8)


def _find_predicted_nifti(out_dir: Path, expected_stem: str, wait_seconds: float = 20.0) -> Path:
    expected = out_dir / f"{expected_stem}.nii.gz"
    deadline = time.time() + wait_seconds

    while time.time() < deadline:
        if expected.exists():
            return expected

        candidates = sorted(out_dir.glob("*.nii.gz"))
        if len(candidates) == 1:
            return candidates[0]

        if len(candidates) > 1:
            exact_name = [p for p in candidates if expected_stem in p.name]
            if len(exact_name) == 1:
                return exact_name[0]

        time.sleep(0.2)

    candidates = sorted(out_dir.glob("*.nii.gz"))
    raise FileNotFoundError(
        f"Prediction output not found in {out_dir}. "
        f"Expected {expected.name}, candidates={[p.name for p in candidates]}"
    )


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def load_predictor(
    model_dir: str | Path | None = None,
    fold: str | int = "all",
    use_folds: list[int] | None = None,
):
    """
    Load a trained nnUNet v2 predictor.

    Parameters
    ----------
    model_dir : path to the nnUNet results folder for this dataset.
                Defaults to codes/unet/nnunet_data/nnUNet_results/Dataset001_Cornea
    fold      : which fold checkpoint to load ('all' loads every completed fold
                and lets nnU-Net ensemble them internally).
    use_folds : explicit list of fold indices to load (overrides `fold`).

    Returns
    -------
    nnUNetPredictor instance (ready for inference)
    """
    _set_nnunet_env()

    # Lazy import so the module can be imported even when nnunetv2 is absent
    # (it will only fail when load_predictor() is actually called).
    try:
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    except ImportError:
        sys.exit(
            "[ERROR] nnunetv2 is not installed.\n"
            "        Install it with:  pip install nnunetv2"
        )

    model_dir = _resolve_model_dir(model_dir)

    trained_model_dir = _find_trained_model_dir(model_dir)

    # Determine folds to load
    if use_folds is not None:
        folds = sorted(set(int(f) for f in use_folds))
    elif fold == "all":
        folds = get_available_folds(model_dir)
    else:
        folds = [int(fold)]

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=_get_torch_device(),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )
    predictor.initialize_from_trained_model_folder(
        str(trained_model_dir),
        use_folds=folds,
        checkpoint_name="checkpoint_final.pth",
    )
    predictor.loaded_folds = tuple(folds)
    predictor.model_dir = trained_model_dir
    return predictor


def predict_single(
    image_path: str | Path,
    predictor=None,
    model_dir: str | Path | None = None,
    fold: str | int = "all",
) -> np.ndarray:
    """
    Segment a single TIF image.

    Parameters
    ----------
    image_path : path to input .tif image
    predictor  : a loaded nnUNetPredictor (optional; created from model_dir if None)
    model_dir  : model directory (used only when predictor is None)
    fold       : fold to use (used only when predictor is None)

    Returns
    -------
    mask : np.ndarray of shape (H, W), dtype uint8, values 0/1
    """
    if predictor is None:
        predictor = load_predictor(model_dir, fold)

    image_path = Path(image_path)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        # Write input NIfTI
        in_nii = tmp / f"{image_path.stem}_0000.nii.gz"
        _tif_to_nifti(image_path, in_nii)
        # Run prediction
        out_dir = tmp / "out"
        out_dir.mkdir()
        predictor.predict_from_files(
            [[str(in_nii)]],
            [str(out_dir / f"{image_path.stem}.nii.gz")],
            save_probabilities=False,
            overwrite=True,
            num_processes_preprocessing=1,
            num_processes_segmentation_export=1,
        )
        out_pred = _find_predicted_nifti(out_dir, image_path.stem)
        mask = _nifti_to_mask(out_pred)
    return mask


def predict_folder(
    input_dir: str | Path,
    output_dir: str | Path,
    predictor=None,
    model_dir: str | Path | None = None,
    fold: str | int = "all",
    save_as: str = "png",
) -> None:
    """
    Segment all TIF images in input_dir and save masks to output_dir.

    Parameters
    ----------
    input_dir  : directory containing *.tif images
    output_dir : directory where segmentation masks will be saved
    predictor  : loaded nnUNetPredictor (created if None)
    model_dir  : used only when predictor is None
    fold       : fold to use (used only when predictor is None)
    save_as    : output format 'png' or 'tif'
    """
    if predictor is None:
        predictor = load_predictor(model_dir, fold)

    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tif_files = sorted(input_dir.glob("*.tif"))
    if not tif_files:
        print(f"[WARN] No .tif files found in {input_dir}")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        in_dir  = tmp / "in"
        out_dir = tmp / "out"
        in_dir.mkdir()
        out_dir.mkdir()

        # Convert all TIFs to NIfTI
        list_of_lists = []
        out_paths = []
        for tif_path in tif_files:
            nii_path = in_dir / f"{tif_path.stem}_0000.nii.gz"
            _tif_to_nifti(tif_path, nii_path)
            list_of_lists.append([str(nii_path)])
            out_paths.append(str(out_dir / f"{tif_path.stem}.nii.gz"))

        # Batch prediction
        predictor.predict_from_files(
            list_of_lists,
            out_paths,
            save_probabilities=False,
            overwrite=True,
            num_processes_preprocessing=4,
            num_processes_segmentation_export=4,
        )

        # Convert NIfTI masks back to images
        for tif_path, out_nii in zip(tif_files, out_paths):
            mask = _nifti_to_mask(Path(out_nii))
            out_name = tif_path.stem + ("." + save_as)
            # Save mask: 0 → black, 1 → white (255)
            Image.fromarray((mask * 255).astype(np.uint8)).save(
                output_dir / out_name
            )

    print(f"[Done] {len(tif_files)} masks saved to {output_dir}")


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #



def _get_torch_device():
    try:
        import torch
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    except ImportError:
        return "cpu"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Corneal segmentation inference using a trained nnUNet v2 model"
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Input TIF image file or directory containing TIF images",
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="Output mask file (PNG/TIF) or output directory",
    )
    parser.add_argument(
        "--model", "-m", default=None,
        help=(
            "Path to the nnUNet results directory for this dataset "
            f"(default: {NNUNET_RESULTS / DATASET_NAME})"
        ),
    )
    parser.add_argument(
        "--fold", default="all",
        help="Which fold to use for inference (integer 0-4 or 'all') (default: all)",
    )
    parser.add_argument(
        "--format", default="png", choices=["png", "tif"],
        help="Output mask format when --input is a directory (default: png)",
    )
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output)

    predictor = load_predictor(args.model, args.fold)

    if input_path.is_dir():
        predict_folder(input_path, output_path, predictor, save_as=args.format)
    elif input_path.is_file():
        mask = predict_single(input_path, predictor)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray((mask * 255).astype(np.uint8)).save(str(output_path))
        print(f"[Done] Mask saved to {output_path}")
    else:
        sys.exit(f"[ERROR] Input not found: {input_path}")


if __name__ == "__main__":
    main()
