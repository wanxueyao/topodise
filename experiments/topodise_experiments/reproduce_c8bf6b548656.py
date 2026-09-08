"""以可迁移方式运行论文的 c8bf6b548656 Stage0 -> Stage1 配方。

该入口显式写出 ``reproduction/_experiment_def.json`` 中的每个配置值，
避免依赖可能变化的默认值。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict


CODES_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if CODES_ROOT not in sys.path:
    sys.path.insert(0, CODES_ROOT)

from experiments.topodise_experiments.stage0_corn1_pretrain.config import Stage0Config  # noqa: E402
from experiments.topodise_experiments.stage0_corn1_pretrain.main import (  # noqa: E402
    run_single_experiment as run_stage0,
)
from experiments.topodise_experiments.stage1_rscp_finetune.config import Stage1Config  # noqa: E402
from experiments.topodise_experiments.stage1_rscp_finetune.main import (  # noqa: E402
    run_single_experiment as run_stage1,
)
from topodise.reproducibility import REFERENCE_RECIPE  # noqa: E402


REFERENCE = {
    "stage0_best_loss": 0.11062798957201711,
    "stage0_best_epoch": 100,
    "stage1_best_loss": 0.02590857886455276,
    "stage1_best_epoch": 77,
    "reference_stage1_sha256": "45146fce10112b0abbc6c782d55099cb18593ec413019f4f52cd8d469725bf03",
    "corn1_images_sha256": "7ae1b3420cfce70bab08baf9eb43ce20b284b1a826820d605a509d7fb2d58623",
    "pd_json_sha256": "d39a0f3c613d17612f94ce3c9651a2a80e47e5c10855873e6fc5966327dda9f0",
    "stage1_pair_images_sha256": "c275f2805366c79f352c4f7b46fc3abebd6c51dff2df21697962f6045fce7b4a",
    "recorded_environment": {
        "python": "3.10.19",
        "torch": "2.9.1+cu128",
        "cuda_device": "NVIDIA RTX A6000",
        "cudnn": 91002,
    },
}


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _corn1_images_sha256(data_root: str) -> str:
    root = os.path.join(data_root, "CORN-1")
    scan_root = os.path.join(root, "images")
    extensions = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")
    paths = []
    for directory, _, filenames in os.walk(scan_root):
        paths.extend(
            os.path.join(directory, name)
            for name in filenames
            if name.lower().endswith(extensions)
        )
    digest = hashlib.sha256()
    for path in sorted(paths):
        relative = os.path.relpath(path, root).replace(os.sep, "/")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\n")
    return digest.hexdigest()


def _stage1_pair_images_sha256(data_root: str, pd_json: str) -> str:
    with open(pd_json, "r", encoding="utf-8") as handle:
        patients = json.load(handle)
    paths = []
    for patient in patients:
        if patient.get("group") not in ("control", "control_cataract"):
            continue
        image_paths = patient.get("image_paths", {})
        for central, whorl in zip(image_paths.get("central", []), image_paths.get("whorl", [])):
            paths.extend((central, whorl))
    digest = hashlib.sha256()
    for relative in paths:
        digest.update(relative.replace(os.sep, "/").encode("utf-8"))
        digest.update(b"\0")
        with open(os.path.join(data_root, relative), "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\n")
    return digest.hexdigest()


def _preflight_data(data_root: str) -> Dict[str, str]:
    pd_json = os.path.join(data_root, "pd_dataset_en", "patients_dataset_en.json")
    observed = {
        "corn1_images_sha256": _corn1_images_sha256(data_root),
        "pd_json_sha256": _sha256(pd_json),
        "stage1_pair_images_sha256": _stage1_pair_images_sha256(data_root, pd_json),
    }
    mismatches = {
        key: {"expected": REFERENCE[key], "observed": value}
        for key, value in observed.items()
        if value != REFERENCE[key]
    }
    if mismatches:
        raise RuntimeError(
            "dataset does not match the c8bf6b548656 reference: "
            + json.dumps(mismatches, indent=2)
        )
    return observed


def _assert_fresh(run_dir: str, resume: bool) -> None:
    if resume:
        return
    occupied = [
        os.path.join(run_dir, "ckpt", "last.pt"),
        os.path.join(run_dir, "ckpt", "ema_best.pt"),
    ]
    if any(os.path.exists(path) for path in occupied):
        raise FileExistsError(
            f"fresh reproduction refused because checkpoints already exist in {run_dir}; "
            "choose another --output-dir or pass --resume"
        )


def _stage0_config(args: argparse.Namespace) -> Stage0Config:
    return Stage0Config(
        corn1_root=os.path.join(args.data_root, "CORN-1"),
        image_size=224,
        num_workers=args.num_workers,
        persistent_workers=False,
        prefetch_factor=2,
        pin_memory=None,
        non_blocking_transfers=False,
        expected_samples=1516,
        encoder_base_channels=64,
        encoder_channel_mult=(1, 2, 4, 8),
        encoder_blocks_per_stage=(2, 2, 2, 2),
        z_dim=512,
        unet_model_channels=64,
        unet_channel_mult=(1, 2, 2, 4),
        T=1000,
        schedule="cosine",
        epochs=200,
        batch_size=16,
        lr=1e-4,
        weight_decay=1e-2,
        grad_clip=1.0,
        ema_decay=0.9999,
        z_dropout=0.1,
        lambda_simple=1.0,
        lambda_topo=0.1,
        topo_target="hgrad",
        use_view_contrastive=False,
        output_dir=args.output_dir,
        run_name=f"{REFERENCE_RECIPE}_stage0",
        resume=args.resume,
        device=args.device,
        reproducibility=not args.no_reproducibility,
        amp=False,
        channels_last=False,
        fused_adamw=False,
    )


def _stage1_config(args: argparse.Namespace, parent_ckpt: str) -> Stage1Config:
    return Stage1Config(
        parent_ckpt=parent_ckpt,
        pd_json=os.path.join(args.data_root, "pd_dataset_en", "patients_dataset_en.json"),
        base_path=args.data_root,
        patient_groups=("control", "control_cataract"),
        allow_dm_disease=False,
        image_size=224,
        num_workers=args.num_workers,
        persistent_workers=False,
        prefetch_factor=2,
        pin_memory=None,
        non_blocking_transfers=False,
        expected_region_pairs=182,
        epochs=80,
        batch_size=16,
        lr=2e-5,
        weight_decay=1e-2,
        grad_clip=1.0,
        ema_decay=0.9999,
        lambda_simple=1.0,
        lambda_topo=0.05,
        lambda_region=0.5,
        tau_region=0.2,
        topo_target="hgrad",
        train_denoiser=False,
        output_dir=args.output_dir,
        run_name=f"{REFERENCE_RECIPE}_stage1",
        device=args.device,
        reproducibility=not args.no_reproducibility,
        parent_encoder_weights="raw",
        amp=False,
        channels_last=False,
        fused_adamw=False,
        resume=args.resume,
    )


def _write_summary(output_dir: str, summary: Dict[str, Any]) -> None:
    path = os.path.join(output_dir, f"{REFERENCE_RECIPE}_summary.json")
    temporary = f"{path}.tmp.{os.getpid()}"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=os.path.join(CODES_ROOT, "datasets"))
    parser.add_argument(
        "--output-dir",
        default=os.path.join(CODES_ROOT, "results", f"reproduce_{REFERENCE_RECIPE}"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--stage", choices=("all", "0", "1"), default="all")
    parser.add_argument("--parent-ckpt", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--no-reproducibility",
        action="store_true",
        help="do not call torch.manual_seed(42); use the environment RNG state",
    )
    return parser.parse_args()


def main() -> Dict[str, Any]:
    args = parse_args()
    args.data_root = os.path.abspath(os.path.expanduser(args.data_root))
    args.output_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)

    summary: Dict[str, Any] = {
        "recipe": REFERENCE_RECIPE,
        "started": datetime.now().isoformat(),
        "reference": REFERENCE,
        "data_root": args.data_root,
        "output_dir": args.output_dir,
        "data_fingerprints": _preflight_data(args.data_root),
        "stage_results": {},
    }
    parent_ckpt = os.path.abspath(os.path.expanduser(args.parent_ckpt)) if args.parent_ckpt else ""

    if args.stage in ("all", "0"):
        cfg0 = _stage0_config(args)
        _assert_fresh(os.path.join(cfg0.output_dir, cfg0.run_name), args.resume)
        result0 = run_stage0(cfg0)
        summary["stage_results"]["stage0"] = result0
        parent_ckpt = result0["ckpt_path"]
        _write_summary(args.output_dir, summary)

    if args.stage in ("all", "1"):
        if not parent_ckpt or not os.path.isfile(parent_ckpt):
            raise FileNotFoundError("Stage1 requires Stage0 output or --parent-ckpt")
        cfg1 = _stage1_config(args, parent_ckpt)
        _assert_fresh(os.path.join(cfg1.output_dir, cfg1.run_name), args.resume)
        result1 = run_stage1(cfg1)
        result1["sha256"] = _sha256(result1["ckpt_path"])
        summary["stage_results"]["stage1"] = result1
        _write_summary(args.output_dir, summary)

    summary["finished"] = datetime.now().isoformat()
    _write_summary(args.output_dir, summary)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
