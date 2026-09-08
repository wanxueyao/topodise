from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path

from topodise.data import CORN1Dataset, RegionPairDataset
from topodise.data.region_pair_dataset import assert_no_pd_leak

from experiments.topodise_experiments.stage0_corn1_pretrain.config import Stage0Config
from experiments.topodise_experiments.stage0_corn1_pretrain.main import run_single_experiment as run_stage0
from experiments.topodise_experiments.stage1_rscp_finetune.config import Stage1Config
from experiments.topodise_experiments.stage1_rscp_finetune.main import run_single_experiment as run_stage1

from .config import load_config
from .utils import atomic_json_dump, canonical_hash, resolve_codes_path, sha256_file


def _dataclass_kwargs(cls, values: dict) -> dict:
    allowed = {item.name for item in fields(cls)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"unsupported {cls.__name__} options: {sorted(unknown)}")
    return {key: value for key, value in values.items() if key in allowed}


def _resolved_stage0(spec: dict, device_request: str) -> Stage0Config:
    values = dict(spec)
    values["corn1_root"] = str(resolve_codes_path(values["corn1_root"]))
    values["output_dir"] = str(resolve_codes_path(values["output_dir"]))
    values["device"] = device_request
    return Stage0Config(**_dataclass_kwargs(Stage0Config, values))


def _resolved_stage1(spec: dict, parent_checkpoint: Path, device_request: str) -> Stage1Config:
    values = dict(spec)
    values["parent_ckpt"] = str(parent_checkpoint)
    manifest = values.get("region_pair_manifest")
    values["region_pair_manifest"] = str(resolve_codes_path(manifest)) if manifest else None
    values["pd_json"] = str(
        resolve_codes_path(values.get("pd_json", "datasets/pd_dataset_en/patients_dataset_en.json"))
    )
    values["base_path"] = str(resolve_codes_path(values.get("base_path", "datasets")))
    values["output_dir"] = str(resolve_codes_path(values["output_dir"]))
    values["device"] = device_request
    return Stage1Config(**_dataclass_kwargs(Stage1Config, values))


def _checkpoint_for(output_dir: str, run_name: str) -> Path:
    return Path(output_dir).resolve() / run_name / "ckpt" / "ema_best.pt"


def _audit_inputs(stage0: Stage0Config, stage1: Stage1Config) -> dict:
    corn = CORN1Dataset(stage0.corn1_root, load_masks=stage0.topo_target == "gt")
    if stage1.region_pair_manifest:
        regions = RegionPairDataset(
            manifest_path=stage1.region_pair_manifest,
            base_path=stage1.base_path,
            transform=None,
            strict_whitelist=True,
            allow_dm_disease=False,
        )
        source_path = Path(stage1.region_pair_manifest)
        source_kind = "region_pair_manifest"
    else:
        source_path = Path(stage1.pd_json)
        patients = json.loads(source_path.read_text(encoding="utf-8"))
        controls = [
            row for row in patients
            if row.get("group") in tuple(stage1.patient_groups)
        ]
        regions = RegionPairDataset(
            patients=controls,
            base_path=stage1.base_path,
            transform=None,
            strict_whitelist=True,
            allow_dm_disease=False,
        )
        source_kind = "pd_control_patients"
    assert_no_pd_leak(regions)
    paired = sum(bool(row["is_paired"]) for row in regions.pairs)
    if paired < stage1.batch_size:
        raise RuntimeError(
            "Stage1 manifest has too few valid central-whorl pairs for stable region InfoNCE: "
            f"paired={paired}, batch_size={stage1.batch_size}"
        )
    stable_region_rows = [
        {
            "subject_id": row["subject_id"],
            "central": str(Path(row["central"]).resolve()),
            "whorl": str(Path(row["whorl"]).resolve()),
            "is_paired": bool(row["is_paired"]),
        }
        for row in regions.pairs
    ]
    return {
        "corn1_num_images": len(corn),
        "corn1_image_manifest_sha256": canonical_hash(
            [str(Path(path).resolve()) for path in corn.paths]
        ),
        "stage1_source_kind": source_kind,
        "stage1_source_manifest": str(source_path.resolve()),
        "stage1_source_manifest_sha256": sha256_file(source_path),
        "stage1_valid_rows": len(regions),
        "stage1_paired_rows": paired,
        "stage1_unpaired_rows": len(regions) - paired,
        "stage1_skipped_duplicate_filenames": regions.skipped_duplicate_filenames,
        "stage1_effective_manifest_sha256": canonical_hash(stable_region_rows),
        "pd_positive_paths_used": 0,
    }


def pretrain_and_cache(
    final_checkpoint: Path,
    encoder_spec: dict,
    device_request: str = "auto",
    force: bool = False,
) -> Path:
    pretrain = encoder_spec.get("pretrain") or {}
    if "stage0" not in pretrain or "stage1" not in pretrain:
        raise ValueError("TopoDiSE auto-pretrain requires pretrain.stage0 and pretrain.stage1")
    stage0 = _resolved_stage0(pretrain["stage0"], device_request)
    stage0_checkpoint = _checkpoint_for(stage0.output_dir, stage0.run_name)
    stage1 = _resolved_stage1(pretrain["stage1"], stage0_checkpoint, device_request)
    configured_stage1_checkpoint = _checkpoint_for(stage1.output_dir, stage1.run_name)
    if configured_stage1_checkpoint != final_checkpoint.resolve():
        raise ValueError(
            "TopoDiSE checkpoint must match stage1 output location: "
            f"configured={final_checkpoint.resolve()}, generated={configured_stage1_checkpoint}"
        )

    audit = _audit_inputs(stage0, stage1)
    if force or not stage0_checkpoint.is_file():
        print(f"[TopoDiSE] Stage0 CORN-1 pretraining -> {stage0_checkpoint}")
        run_stage0(stage0)
    else:
        print(f"[TopoDiSE] using cached Stage0 checkpoint: {stage0_checkpoint}")
    if not stage0_checkpoint.is_file():
        raise RuntimeError(f"Stage0 finished without expected checkpoint: {stage0_checkpoint}")

    if force or not configured_stage1_checkpoint.is_file():
        print(f"[TopoDiSE] Stage1 unlabeled regional pretraining -> {configured_stage1_checkpoint}")
        run_stage1(stage1)
    else:
        print(f"[TopoDiSE] using cached Stage1 checkpoint: {configured_stage1_checkpoint}")
    if not configured_stage1_checkpoint.is_file():
        raise RuntimeError(
            f"Stage1 finished without expected checkpoint: {configured_stage1_checkpoint}"
        )

    summary = {
        "method": "topodise_stage0_stage1",
        "stage0_checkpoint": str(stage0_checkpoint),
        "stage0_checkpoint_sha256": sha256_file(stage0_checkpoint),
        "stage1_checkpoint": str(configured_stage1_checkpoint),
        "stage1_checkpoint_sha256": sha256_file(configured_stage1_checkpoint),
        "stage0_config": stage0.asdict(),
        "stage1_config": stage1.asdict(),
        "input_audit": audit,
    }
    atomic_json_dump(
        summary,
        Path(stage1.output_dir).resolve() / "pretrain_summary.json",
    )
    return configured_stage1_checkpoint


def ensure_topodise_checkpoint(
    checkpoint: Path,
    encoder_spec: dict,
    device_request: str = "auto",
) -> Path:
    if checkpoint.is_file() and checkpoint.stat().st_size > 0:
        return checkpoint
    if not bool(encoder_spec.get("auto_pretrain_if_missing", True)):
        raise FileNotFoundError(
            f"TopoDiSE Stage1 checkpoint not found and auto-pretrain is disabled: {checkpoint}"
        )
    return pretrain_and_cache(checkpoint, encoder_spec, device_request)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain/cache TopoDiSE Stage0 and Stage1")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    spec = cfg["encoders"]["topodise_stage1"]
    checkpoint = resolve_codes_path(spec["checkpoint"])
    if checkpoint.is_file() and not args.force:
        print(f"[TopoDiSE] using existing Stage1 cache: {checkpoint}")
        return
    pretrain_and_cache(checkpoint, spec, args.device, force=args.force)


if __name__ == "__main__":
    main()
