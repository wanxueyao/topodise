from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from .utils import resolve_codes_path


DEFAULTS: dict[str, Any] = {
    "protocol_version": "core-v1.3-portable",
    "dataset_json": "datasets/pd_dataset_en/patients_dataset_en.json",
    "dataset_root": "datasets",
    "output_root": "results/unified_stage2_encoder_comparison",
    "tasks": ["pd_binary", "pd_stage", "pd_subtype_tdrest"],
    "encoders": {
        "topodise_stage1": {
            "checkpoint": (
                "results/unified_stage2_encoder_comparison/pretrained/topodise/"
                "stage1_rscp/ckpt/ema_best.pt"
            ),
            "use_ema": True,
            "auto_pretrain_if_missing": True,
            "pretrain": {
                "stage0": {
                    "corn1_root": "datasets/CORN-1",
                    "output_dir": "results/unified_stage2_encoder_comparison/pretrained/topodise",
                    "run_name": "stage0_corn1",
                    "image_size": 224,
                    "epochs": 200,
                    "batch_size": 16,
                    "lr": 1.0e-4,
                    "weight_decay": 1.0e-2,
                    "topo_target": "hgrad",
                    "lambda_simple": 1.0,
                    "lambda_topo": 0.1,
                    "num_workers": 4,
                    "persistent_workers": False,
                    "pin_memory": None,
                    "non_blocking_transfers": False,
                    "expected_samples": 1516,
                    "amp": False,
                    "channels_last": False,
                    "fused_adamw": False,
                    "resume": True,
                    "reproducibility": True,
                },
                "stage1": {
                    "pd_json": "datasets/pd_dataset_en/patients_dataset_en.json",
                    "region_pair_manifest": None,
                    "base_path": "datasets",
                    "output_dir": "results/unified_stage2_encoder_comparison/pretrained/topodise",
                    "run_name": "stage1_rscp",
                    "epochs": 80,
                    "batch_size": 16,
                    "lr": 2.0e-5,
                    "weight_decay": 1.0e-2,
                    "lambda_region": 0.5,
                    "lambda_topo": 0.05,
                    "tau_region": 0.2,
                    "topo_target": "hgrad",
                    "parent_encoder_weights": "raw",
                    "train_denoiser": False,
                    "allow_dm_disease": False,
                    "num_workers": 4,
                    "persistent_workers": False,
                    "pin_memory": None,
                    "non_blocking_transfers": False,
                    "expected_region_pairs": 182,
                    "amp": False,
                    "channels_last": False,
                    "fused_adamw": False,
                    "resume": True,
                    "reproducibility": True,
                },
            },
        },
        "simclr_resnet18": {
            "checkpoint": "results/unified_stage2_encoder_comparison/pretrained/simclr_resnet18.pt",
            "auto_pretrain_if_missing": True,
            "pretrain": {
                "initialization": False,
                "batch_size": 32,
                "lr": 1.0e-3,
                "temperature": 0.5,
                "proj_hidden_dim": 256,
                "proj_out_dim": 128,
                "weight_decay": 1.0e-4,
                "num_workers": 4,
                "amp": True,
                "channels_last": True,
                "persistent_workers": True,
                "pin_memory": False,
                "fused_adamw": True,
                "drop_last": True,
                "seed": 20260831,
                "stages": [
                    {
                        "name": "stage_a_corn1",
                        "source": "corn1_raw",
                        "source_root": "datasets/CORN-1",
                        "epochs": 200,
                        "lr": 1.0e-3,
                    },
                    {
                        "name": "stage_b_regions",
                        "source": "region_manifest",
                        "manifest": "datasets/unlabeled_dataset_en/unlabeled_dataset_en.json",
                        "dataset_root": "datasets",
                        "epochs": 80,
                        "lr": 2.0e-4,
                    },
                ],
            },
        },
        "dinov2_vits14": {"checkpoint": "", "model_name": "vit_small_patch14_dinov2"},
    },
    "splits": {"outer_folds": 5, "seeds": [42, 202, 3407], "inner_val_ratio": 0.2},
    "pairing": {"mode": "cartesian", "cap": 8, "seed": 20260831},
    "model": {
        "common_dim": 512,
        "adapter_rank": 32,
        "attention_hidden": 128,
        "attention_dropout": 0.25,
        "morphology_dim": 32,
    },
    "optimization": {
        "epochs": 50,
        "warmup_epochs": 3,
        "lr": 5.0e-4,
        "min_lr": 1.0e-6,
        "weight_decay": 1.0e-2,
        "grad_clip": 1.0,
        "accumulate_patients": 8,
        "patient_batch_size": 8,
        "eval_patient_batch_size": 64,
        "fused_adamw": True,
    },
    "runtime": {
        "device": "auto",
        "amp": True,
        "num_workers": 4,
        "feature_batch_size": 64,
        "pin_memory": False,
        "channels_last": True,
    },
}


def _deep_update(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _deep_update(dst[key], value)
        else:
            dst[key] = value
    return dst


def load_config(path: str | Path) -> dict[str, Any]:
    cfg = deepcopy(DEFAULTS)
    with Path(path).open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    _deep_update(cfg, loaded)
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict[str, Any]) -> None:
    allowed_tasks = {"pd_binary", "pd_stage", "pd_subtype_tdrest"}
    unknown = set(cfg["tasks"]) - allowed_tasks
    if unknown:
        raise ValueError(f"unsupported tasks in locked Core protocol: {sorted(unknown)}")
    if cfg["pairing"]["mode"] != "cartesian":
        raise ValueError("Core protocol requires deterministic Cartesian pairing")
    if int(cfg["pairing"]["cap"]) != 8:
        raise ValueError("Core protocol locks pair cap to 8")
    if int(cfg["model"]["common_dim"]) != 512:
        raise ValueError("Core protocol locks the common feature dimension to 512")
    if int(cfg["splits"]["outer_folds"]) != 5:
        raise ValueError("Core protocol requires patient-level 5-fold CV")
    if int(cfg["optimization"].get("patient_batch_size", 8)) != int(
        cfg["optimization"]["accumulate_patients"]
    ):
        raise ValueError("patient_batch_size must preserve the locked effective batch of 8")
    topo_spec = cfg["encoders"]["topodise_stage1"]
    if not str(topo_spec.get("checkpoint", "")).strip():
        raise ValueError("TopoDiSE checkpoint/cache path must be configured")
    topo_stage1 = topo_spec.get("pretrain", {}).get("stage1", {})
    if bool(topo_stage1.get("allow_dm_disease", False)):
        raise ValueError("Core protocol forbids disease-labelled images in TopoDiSE Stage1")
    simclr = cfg["encoders"]["simclr_resnet18"].get("pretrain", {})
    if simclr.get("initialization", False) is not False:
        raise ValueError("Core protocol trains the SimCLR comparator from scratch")
    simclr_sources = [stage.get("source") for stage in simclr.get("stages", [])]
    if simclr_sources != ["corn1_raw", "region_manifest"]:
        raise ValueError(
            "Core protocol requires data-matched SimCLR CORN-1 then region-manifest stages"
        )


def resolved_paths(cfg: dict[str, Any]) -> dict[str, Path]:
    return {
        "dataset_json": resolve_codes_path(cfg["dataset_json"]),
        "dataset_root": resolve_codes_path(cfg["dataset_root"]),
        "output_root": resolve_codes_path(cfg["output_root"]),
    }
