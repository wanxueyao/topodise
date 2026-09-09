"""
train.py  –  Preprocess and train an nnUNet v2 model on CORN-1.

Steps performed (in order):
  1. Set nnUNet environment variables to the local data directory.
  2. Run fingerprint extraction + experiment planning + preprocessing.
  3. Run nnUNet 2-D training for the requested fold(s).

The trained model is saved to:
    codes/unet/nnunet_data/nnUNet_results/Dataset001_Cornea/
        nnUNetTrainer__nnUNetPlans__2d/
            fold_0/ ... fold_4/   (when fold="all")
                checkpoint_best.pth
                checkpoint_final.pth
                ...

Usage:
    # Train all folds (5-fold CV, recommended):
    python train.py

    # Train a single fold (faster):
    python train.py --fold 0

    # Skip preprocessing if it was already done:
    python train.py --skip_preprocess
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


FAST_EPOCH_TRAINERS = {
    5: "nnUNetTrainer_5epochs",
    10: "nnUNetTrainer_10epochs",
    20: "nnUNetTrainer_20epochs",
    50: "nnUNetTrainer_50epochs",
    100: "nnUNetTrainer_100epochs",
    250: "nnUNetTrainer_250epochs",
    500: "nnUNetTrainer_500epochs",
    750: "nnUNetTrainer_750epochs",
    2000: "nnUNetTrainer_2000epochs",
    4000: "nnUNetTrainer_4000epochs",
    8000: "nnUNetTrainer_8000epochs",
}

DGX_SPARK_DEFAULT_FAST_EPOCHS = 250
DGX_SPARK_DEFAULT_CPU_THREADS = 8

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from unet.config import (
    DATASET_ID, DATASET_NAME, FOLD,
    NNUNET_PREPROCESSED, NNUNET_RAW, NNUNET_RESULTS,
    NNUNET_CONFIG,
)


def set_nnunet_env() -> dict:
    """Return an updated environment dict with the nnUNet path variables set."""
    env = os.environ.copy()
    python_bin_dir = str(Path(sys.executable).resolve().parent)
    env["PATH"] = os.pathsep.join([python_bin_dir, env.get("PATH", "")])
    env["nnUNet_raw"]          = str(NNUNET_RAW)
    env["nnUNet_preprocessed"] = str(NNUNET_PREPROCESSED)
    env["nnUNet_results"]      = str(NNUNET_RESULTS)
    return env


def configure_runtime_env(env: dict, cpu_threads: int | None = None) -> dict:
    """Apply conservative runtime defaults that work well on a single-GPU DGX Spark."""
    tuned_env = env.copy()
    thread_count = cpu_threads or min(DGX_SPARK_DEFAULT_CPU_THREADS, os.cpu_count() or 1)
    thread_count = max(1, thread_count)

    for variable_name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        tuned_env.setdefault(variable_name, str(thread_count))

    tuned_env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return tuned_env


def run(cmd: list[str], env: dict, dry_run: bool = False) -> None:
    """Run a command, streaming output to stdout/stderr."""
    print(f"\n[CMD] {' '.join(cmd)}\n{'='*60}")
    if dry_run:
        print("[DRY-RUN] Command not executed.")
        return
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        sys.exit(f"Command failed with exit code {result.returncode}")


def check_dataset_prepared() -> bool:
    dataset_dir = NNUNET_RAW / DATASET_NAME
    return (dataset_dir / "dataset.json").exists() and \
           (dataset_dir / "imagesTr").exists() and \
           (dataset_dir / "labelsTr").exists()


def check_preprocessed_dataset_available() -> bool:
    dataset_dir = NNUNET_PREPROCESSED / DATASET_NAME
    return dataset_dir.exists() and any(dataset_dir.iterdir())


def preprocess(env: dict, verify_integrity: bool = True, dry_run: bool = False) -> None:
    """Run nnUNetv2_plan_and_preprocess for the 2-D configuration."""
    cmd = [
        "nnUNetv2_plan_and_preprocess",
        "-d", str(DATASET_ID),
        "-c", NNUNET_CONFIG,
        "--verify_dataset_integrity",
    ]
    if not verify_integrity:
        cmd.remove("--verify_dataset_integrity")
    run(cmd, env, dry_run=dry_run)


def train_fold(
    fold: str | int,
    env: dict,
    device: str = "cuda",
    num_gpus: int = 1,
    continue_training: bool = False,
    trainer: str = "nnUNetTrainer",
    save_npz: bool = True,
    disable_checkpointing: bool = False,
    dry_run: bool = False,
) -> None:
    """Run nnUNetv2_train for a single fold or 'all'."""
    cmd = [
        "nnUNetv2_train",
        str(DATASET_ID),
        NNUNET_CONFIG,
        str(fold),
        "-tr", trainer,
        "-device", device,
    ]
    if save_npz:
        # Save softmax predictions for later ensemble/analysis. Disable for faster I/O.
        cmd.append("--npz")
    if num_gpus > 1:
        cmd += ["-num_gpus", str(num_gpus)]
    if continue_training:
        cmd.append("--c")
    if disable_checkpointing:
        cmd.append("--disable_checkpointing")
    run(cmd, env, dry_run=dry_run)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train nnUNet v2 on the CORN-1 corneal segmentation dataset"
    )
    parser.add_argument(
        "--profile",
        default="baseline",
        choices=["dgx-spark", "baseline"],
        help=(
            "Runtime profile presets. 'dgx-spark' reuses existing preprocessing, "
            "defaults to 250 epochs, disables npz export, and skips integrity verification. "
            "Use 'baseline' to keep the original behavior."
        ),
    )
    parser.add_argument(
        "--fold",
        default=str(FOLD),
        help=(
            "Which fold(s) to train.  "
            "Use an integer 0-4 for a single fold or 'all' for all 5 folds "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--skip_preprocess",
        action="store_true",
        help="Skip fingerprint extraction / preprocessing if already done.",
    )
    parser.add_argument(
        "--no_verify",
        action="store_true",
        help="Skip dataset integrity verification during preprocessing.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu", "mps"],
        help="Device for nnUNet training (default: cuda).",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to use for training (default: 1).",
    )
    parser.add_argument(
        "--cuda_devices",
        default=None,
        help=(
            "Set CUDA_VISIBLE_DEVICES, e.g. '0' or '0,1'. "
            "Overrides any existing environment variable."
        ),
    )
    parser.add_argument(
        "--continue_training",
        action="store_true",
        help="Continue training from the latest checkpoint (passes --c to nnUNetv2_train).",
    )
    parser.add_argument(
        "--trainer",
        default=None,
        help=(
            "nnUNet trainer class name. "
            "Examples: nnUNetTrainer_250epochs, nnUNetTrainer_500epochs"
        ),
    )
    parser.add_argument(
        "--fast_epochs",
        type=int,
        choices=sorted(FAST_EPOCH_TRAINERS.keys()),
        help=(
            "Use a built-in shorter/longer training-length trainer. "
            "Overrides --trainer."
        ),
    )
    parser.add_argument(
        "--cpu_threads",
        type=int,
        default=None,
        help="Cap CPU math threads used by preprocessing/training helpers.",
    )
    npz_group = parser.add_mutually_exclusive_group()
    npz_group.add_argument(
        "--no_npz",
        dest="save_npz",
        action="store_false",
        help="Disable --npz export during validation to reduce I/O and speed up training.",
    )
    npz_group.add_argument(
        "--save_npz",
        dest="save_npz",
        action="store_true",
        help="Force --npz export even under the dgx-spark profile.",
    )
    parser.set_defaults(save_npz=None)
    parser.add_argument(
        "--disable_checkpointing",
        action="store_true",
        help="Disable checkpoint saving for speed benchmarking/debug (no resume/final checkpoints).",
    )
    parser.add_argument(
        "--dry_run", "--dry-run",
        action="store_true",
        help="Print the effective nnUNet commands without executing them.",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # Validate environment
    # ------------------------------------------------------------------ #
    if not args.dry_run and not check_dataset_prepared():
        sys.exit(
            "[ERROR] Dataset not prepared.\n"
            "        Run  python prepare_dataset.py  first."
        )

    env = set_nnunet_env()
    env = configure_runtime_env(env, cpu_threads=args.cpu_threads)

    # Apply CUDA_VISIBLE_DEVICES if --cuda_devices was specified
    if args.cuda_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
        print(f"[INFO] CUDA_VISIBLE_DEVICES={args.cuda_devices}")

    trainer_name = args.trainer or "nnUNetTrainer"
    skip_preprocess = args.skip_preprocess
    verify_integrity = not args.no_verify
    save_npz = True if args.save_npz is None else args.save_npz

    if args.profile == "dgx-spark":
        env["nnUNet_compile"] = "False"
        if args.fast_epochs is None and args.trainer is None:
            args.fast_epochs = DGX_SPARK_DEFAULT_FAST_EPOCHS
        if args.save_npz is None:
            save_npz = False
        verify_integrity = False
        if check_preprocessed_dataset_available() and not args.skip_preprocess:
            skip_preprocess = True
            print("[INFO] Reusing existing preprocessing outputs for dgx-spark profile.")

    if args.fast_epochs is not None:
        trainer_name = FAST_EPOCH_TRAINERS[args.fast_epochs]

    print(f"[INFO] Using trainer: {trainer_name}")
    print(f"[INFO] save_npz={save_npz}")
    print(f"[INFO] verify_integrity={verify_integrity}")
    print(f"[INFO] cpu_threads={env.get('OMP_NUM_THREADS')}")

    # ------------------------------------------------------------------ #
    # Step 1: preprocessing
    # ------------------------------------------------------------------ #
    if not skip_preprocess:
        print("[Step 1/2] Preprocessing …")
        preprocess(env, verify_integrity=verify_integrity, dry_run=args.dry_run)
    else:
        print("[Step 1/2] Skipping preprocessing.")

    # ------------------------------------------------------------------ #
    # Step 2: training
    # ------------------------------------------------------------------ #
    fold_arg = args.fold
    if fold_arg == "all":
        folds_to_train = list(range(5))
        print("[Step 2/2] Training all 5 folds …")
    else:
        folds_to_train = [int(fold_arg)]
        print(f"[Step 2/2] Training fold {fold_arg} …")

    for fold in folds_to_train:
        train_fold(
            fold,
            env,
            device=args.device,
            num_gpus=args.num_gpus,
            continue_training=args.continue_training,
            trainer=trainer_name,
            save_npz=save_npz,
            disable_checkpointing=args.disable_checkpointing,
            dry_run=args.dry_run,
        )

    if args.dry_run:
        print("\n[Done] Command preview complete; no training was executed.")
        return

    print(
        "\n[Done] Training complete.\n"
        f"  Models saved to: {NNUNET_RESULTS / DATASET_NAME}\n"
        "\nNext step:\n"
        "  python predict.py --input <input_dir> --output <output_dir>"
    )


if __name__ == "__main__":
    main()
