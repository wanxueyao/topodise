"""
nnUNet configuration for corneal image segmentation (CORN-1 dataset).
All paths are relative to this file so the project is portable.
"""
from pathlib import Path

# --------------------------------------------------------------------------- #
# Project paths
# --------------------------------------------------------------------------- #
UNET_DIR = Path(__file__).parent.resolve()          # codes/unet/
PROJECT_ROOT = UNET_DIR.parent                      # PD/

DATASET_ROOT = PROJECT_ROOT / "datasets" / "CORN-1"
IMAGE_TRAIN  = DATASET_ROOT / "images" / "train_orig"
IMAGE_VAL    = DATASET_ROOT / "images" / "val_orig"
IMAGE_TEST   = DATASET_ROOT / "images" / "test_orig"
LABEL_TRAIN  = DATASET_ROOT / "labels" / "train_label"
LABEL_VAL    = DATASET_ROOT / "labels" / "val_label"
LABEL_TEST   = DATASET_ROOT / "labels" / "test_label"

# --------------------------------------------------------------------------- #
# nnUNet data directories (self-contained under codes/unet/)
# --------------------------------------------------------------------------- #
NNUNET_BASE         = UNET_DIR / "nnunet_data"
NNUNET_RAW          = NNUNET_BASE / "nnUNet_raw"
NNUNET_PREPROCESSED = NNUNET_BASE / "nnUNet_preprocessed"
NNUNET_RESULTS      = NNUNET_BASE / "nnUNet_results"

# --------------------------------------------------------------------------- #
# Dataset specification
# --------------------------------------------------------------------------- #
DATASET_ID   = 1                     # nnUNet dataset ID
DATASET_NAME = "Dataset001_Cornea"   # must match pattern DatasetXXX_Name

# Image modality
CHANNEL_NAMES = {0: "microscopy"}    # single-channel grayscale

# Segmentation labels: 0 = background, 1 = cell boundary
LABELS = {
    "background": 0,
    "cell_boundary": 1,
}
NUM_CLASSES = 2   # including background

# --------------------------------------------------------------------------- #
# Training configuration
# --------------------------------------------------------------------------- #
NNUNET_CONFIG   = "2d"    # 2D UNet (suitable for 384×384 natural 2D images)
FOLD            = "all"   # train on all folds (nnUNet 5-fold CV);
                          # use integer 0-4 to train a single fold

# --------------------------------------------------------------------------- #
# Image properties (used during dataset preparation)
# --------------------------------------------------------------------------- #
IMAGE_SIZE = (384, 384)   # H × W
PIXEL_SPACING = (1.0, 1.0)  # mm/pixel (arbitrary for microscopy; used by nnUNet)
