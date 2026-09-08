"""
evaluate.py  –  Compute segmentation metrics on the CORN-1 test set.

Metrics computed per image and averaged:
  - Dice coefficient (F1)
  - IoU  (Jaccard index)
  - Pixel accuracy

Optionally uses the built-in nnUNet evaluation pipeline, or a standalone
NumPy implementation (no nnUNet required for evaluation).

Usage
-----
  # Evaluate predictions stored in a folder of PNG/TIF masks:
  python evaluate.py \\
      --pred  /path/to/predicted_masks/ \\
      --gt    codes/datasets/CORN-1/labels/test_label/

  # Evaluate and save a CSV report:
  python evaluate.py \\
      --pred  /path/to/predicted_masks/ \\
      --gt    codes/datasets/CORN-1/labels/test_label/ \\
      --save  results/evaluation.csv
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from config import LABEL_TEST


# --------------------------------------------------------------------------- #
# Metric functions
# --------------------------------------------------------------------------- #

def binary_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """Dice coefficient for binary arrays (0/1)."""
    intersection = (pred & gt).sum()
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0  # both empty → perfect agreement
    return float(2 * intersection / denom)


def binary_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """Intersection-over-Union for binary arrays (0/1)."""
    intersection = (pred & gt).sum()
    union = (pred | gt).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)


def pixel_accuracy(pred: np.ndarray, gt: np.ndarray) -> float:
    """Fraction of correctly classified pixels."""
    return float((pred == gt).mean())


def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Return a dict with dice, iou, and pixel_acc."""
    pred_bool = pred.astype(bool)
    gt_bool   = gt.astype(bool)
    return {
        "dice":      binary_dice(pred_bool, gt_bool),
        "iou":       binary_iou(pred_bool, gt_bool),
        "pixel_acc": pixel_accuracy(pred.astype(np.uint8), gt.astype(np.uint8)),
    }


# --------------------------------------------------------------------------- #
# Image loading helpers
# --------------------------------------------------------------------------- #

def load_mask(path: Path) -> np.ndarray:
    """Load a PNG or TIF mask and return a binary 2-D array (H, W), dtype bool."""
    arr = np.array(Image.open(path).convert("L"))
    return (arr > 128).astype(bool)


def load_gt(path: Path) -> np.ndarray:
    """Load a ground-truth label TIF (0 / 255) and binarise it."""
    arr = np.array(Image.open(path))
    return (arr > 128).astype(bool)


# --------------------------------------------------------------------------- #
# Main evaluation routine
# --------------------------------------------------------------------------- #

def evaluate(
    pred_dir: str | Path,
    gt_dir: str | Path,
    save_csv: str | Path | None = None,
) -> dict:
    """
    Compare predicted masks against ground-truth labels.

    Parameters
    ----------
    pred_dir  : directory containing predicted binary masks (PNG or TIF)
    gt_dir    : directory containing ground-truth TIF labels (0/255)
    save_csv  : optional path to write per-image metrics as CSV

    Returns
    -------
    summary : dict with keys 'mean_dice', 'mean_iou', 'mean_pixel_acc',
              'std_dice', 'std_iou', 'n_images'
    """
    pred_dir = Path(pred_dir)
    gt_dir   = Path(gt_dir)

    pred_files = sorted(pred_dir.glob("*.png")) + sorted(pred_dir.glob("*.tif"))
    if not pred_files:
        sys.exit(f"[ERROR] No PNG/TIF files found in prediction directory: {pred_dir}")

    rows  = []
    dices = []
    ious  = []
    accs  = []

    for pred_path in pred_files:
        stem = pred_path.stem
        # find matching GT (may have .tif extension)
        gt_path = gt_dir / f"{stem}.tif"
        if not gt_path.exists():
            print(f"[WARN] GT not found for {stem}, skipping.")
            continue

        pred = load_mask(pred_path)
        gt   = load_gt(gt_path)

        if pred.shape != gt.shape:
            print(f"[WARN] Shape mismatch for {stem}: pred={pred.shape} gt={gt.shape}")
            continue

        m = compute_metrics(pred, gt)
        dices.append(m["dice"])
        ious.append(m["iou"])
        accs.append(m["pixel_acc"])
        rows.append({"name": stem, **m})

    if not dices:
        sys.exit("[ERROR] No matched image pairs found. Check paths.")

    summary = {
        "mean_dice":      float(np.mean(dices)),
        "std_dice":       float(np.std(dices)),
        "mean_iou":       float(np.mean(ious)),
        "std_iou":        float(np.std(ious)),
        "mean_pixel_acc": float(np.mean(accs)),
        "n_images":       len(dices),
    }

    # Console output
    print(f"\n{'='*50}")
    print(f" Evaluation Results  ({summary['n_images']} images)")
    print(f"{'='*50}")
    print(f"  Dice  : {summary['mean_dice']:.4f}  ± {summary['std_dice']:.4f}")
    print(f"  IoU   : {summary['mean_iou']:.4f}  ± {summary['std_iou']:.4f}")
    print(f"  Acc   : {summary['mean_pixel_acc']:.4f}")
    print(f"{'='*50}\n")

    # CSV output
    if save_csv is not None:
        save_csv = Path(save_csv)
        save_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(save_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["name", "dice", "iou", "pixel_acc"])
            writer.writeheader()
            writer.writerows(rows)
            writer.writerow({
                "name": "MEAN",
                "dice": summary["mean_dice"],
                "iou":  summary["mean_iou"],
                "pixel_acc": summary["mean_pixel_acc"],
            })
        print(f"[Saved] Per-image results → {save_csv}")

    return summary


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate corneal segmentation predictions against CORN-1 labels"
    )
    parser.add_argument(
        "--pred", "-p", required=True,
        help="Directory containing predicted segmentation masks (PNG or TIF)",
    )
    parser.add_argument(
        "--gt", "-g", default=str(LABEL_TEST),
        help=f"Directory containing ground-truth TIF labels (default: {LABEL_TEST})",
    )
    parser.add_argument(
        "--save", "-s", default=None,
        help="Optional path to save per-image metrics as a CSV file",
    )
    args = parser.parse_args()
    evaluate(args.pred, args.gt, save_csv=args.save)


if __name__ == "__main__":
    main()
