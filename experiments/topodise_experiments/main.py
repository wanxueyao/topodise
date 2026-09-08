"""Top-level CLI dispatcher for TopoDiSE 3-stage pipeline.

Examples:
    python -m experiments.topodise_experiments.main --stage 0 \
        --corn1-root datasets/CORN-1 --epochs 5 --device cuda

    python -m experiments.topodise_experiments.main --stage 1 \
        --parent-ckpt results/topodise_stage0/ckpt/ema_best.pt \
        --pd-json datasets/pd_dataset_en/patients_dataset_en.json

    python -m experiments.topodise_experiments.main --stage 2 \
        --parent-ckpt results/topodise_stage1_rscp/ckpt/ema_best.pt \
        --task pd_binary --dataset pd

    python -m experiments.topodise_experiments.main --stage 2 \
        --parent-ckpt <ckpt> --task dm_binary --dataset dm
"""
from __future__ import annotations

import os
import sys
import json
import argparse
import dataclasses

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)


def _override(cfg, overrides: dict):
    for k, v in overrides.items():
        if v is None:
            continue
        if not hasattr(cfg, k):
            raise KeyError(f"unknown config key '{k}' for {type(cfg).__name__}")
        setattr(cfg, k, v)
    return cfg


def _load_yaml(path: str) -> dict:
    if not path:
        return {}
    try:
        import yaml  # type: ignore
    except ImportError as e:
        raise RuntimeError("PyYAML not installed; pip install pyyaml") from e
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


DM_DEFAULT_JSON = "datasets/dm_dataset_en/dm_patients_dataset_en.json"
PD_DEFAULT_JSON = "datasets/pd_dataset_en/patients_dataset_en.json"


def parse_args():
    p = argparse.ArgumentParser(
        description="TopoDiSE 3-stage CLI dispatcher (PD + DM downstream)."
    )
    p.add_argument("--stage", type=int, required=True, choices=[0, 1, 2])
    p.add_argument("--dataset", choices=["pd", "dm"], default="pd",
                   help="Stage 2 only: which downstream dataset family.")
    p.add_argument("--task", default=None,
                   help="Stage 2 only: pd_binary|pd_stage|pd_subtype|dm_binary|dm_subtype.")
    p.add_argument("--parent-ckpt", default=None, help="Required for stages 1/2.")
    p.add_argument("--config-yaml", default=None,
                   help="Optional YAML file with arbitrary stageX config overrides.")
    p.add_argument("--corn1-root", default=None)
    p.add_argument("--pd-json", default=None)
    p.add_argument("--base-path", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--image-size", type=int, default=None)
    p.add_argument("--few-shot-k", type=int, default=None)
    p.add_argument("--num-folds", type=int, default=None)
    p.add_argument("--eval-folds", type=int, default=None)
    p.add_argument("--kfold-mode", choices=["patient", "image"], default=None)
    p.add_argument("--scheduler", choices=["none", "cosine", "plateau"], default=None)
    p.add_argument("--min-lr", type=float, default=None)
    p.add_argument("--warmup-epochs", type=int, default=None)
    p.add_argument("--early-stopping", type=int, choices=[0, 1], default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--min-delta", type=float, default=None)
    p.add_argument("--classifier-dropout", type=float, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--run-name", default=None)
    p.add_argument("--device", default=None)
    p.add_argument(
        "--no-reproducibility", action="store_true",
        help="Stage0/1 only: do not call torch.manual_seed(42)",
    )
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    yaml_overrides = _load_yaml(args.config_yaml) if args.config_yaml else {}

    cli_kwargs = {
        "corn1_root": args.corn1_root,
        "pd_json": args.pd_json,
        "base_path": args.base_path,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "few_shot_k": args.few_shot_k,
        "num_folds": args.num_folds,
        "eval_folds": args.eval_folds,
        "kfold_mode": args.kfold_mode,
        "scheduler": args.scheduler,
        "min_lr": args.min_lr,
        "warmup_epochs": args.warmup_epochs,
        "early_stopping": (bool(args.early_stopping)
                   if args.early_stopping is not None else None),
        "patience": args.patience,
        "min_delta": args.min_delta,
        "classifier_dropout": args.classifier_dropout,
        "lr": args.lr,
        "seed": args.seed,
        "output_dir": args.output_dir,
        "run_name": args.run_name,
        "device": args.device,
    }

    if args.stage == 0:
        from experiments.topodise_experiments.stage0_corn1_pretrain.config import Stage0Config
        from experiments.topodise_experiments.stage0_corn1_pretrain.main import (
            run_single_experiment,
        )
        cfg = Stage0Config()
        if args.smoke:
            cfg.epochs, cfg.batch_size, cfg.image_size = 1, 2, 64
            cfg.encoder_base_channels = 16
            cfg.encoder_channel_mult = (1, 2, 2, 2)
            cfg.encoder_blocks_per_stage = (1, 1, 1, 1)
            cfg.unet_model_channels = 16
            cfg.unet_channel_mult = (1, 2, 2)
            cfg.z_dim = 64
            cfg.T = 50
            cfg.num_workers = 0
            cfg.run_name = "smoke_stage0"
            cfg.smoke = True
        _override(cfg, yaml_overrides.get("stage0", yaml_overrides))
        _override(cfg, {k: v for k, v in cli_kwargs.items()
                        if k in cfg.__dataclass_fields__})
        if args.no_reproducibility:
            cfg.reproducibility = False
    elif args.stage == 1:
        from experiments.topodise_experiments.stage1_rscp_finetune.config import Stage1Config
        from experiments.topodise_experiments.stage1_rscp_finetune.main import (
            run_single_experiment,
        )
        if not args.parent_ckpt:
            raise SystemExit("--parent-ckpt is required for stage 1")
        cfg = Stage1Config()
        cfg.parent_ckpt = args.parent_ckpt
        if args.smoke:
            cfg.epochs, cfg.batch_size = 1, 2
            cfg.num_workers = 0
            cfg.run_name = "smoke_stage1"
            cfg.smoke = True
        _override(cfg, yaml_overrides.get("stage1", yaml_overrides))
        _override(cfg, {k: v for k, v in cli_kwargs.items()
                        if k in cfg.__dataclass_fields__})
        if args.no_reproducibility:
            cfg.reproducibility = False
    elif args.stage == 2:
        from experiments.topodise_experiments.stage2_racf_downstream.config import Stage2Config
        from experiments.topodise_experiments.stage2_racf_downstream.main import (
            run_single_experiment, NUM_CLASSES,
        )
        if not args.parent_ckpt:
            raise SystemExit("--parent-ckpt is required for stage 2")
        if not args.task:
            raise SystemExit("--task is required for stage 2")
        if args.task not in NUM_CLASSES:
            raise SystemExit(f"unknown --task {args.task} (allowed: {list(NUM_CLASSES.keys())})")
        cfg = Stage2Config()
        cfg.parent_ckpt = args.parent_ckpt
        cfg.task = args.task
        if args.dataset == "dm" and not cli_kwargs.get("pd_json"):
            cli_kwargs["pd_json"] = DM_DEFAULT_JSON
        elif args.dataset == "pd" and not cli_kwargs.get("pd_json"):
            cli_kwargs["pd_json"] = PD_DEFAULT_JSON
        if args.smoke:
            cfg.epochs, cfg.batch_size, cfg.image_size = 2, 2, 64
            cfg.adapter_rank = 8
            cfg.val_split = 0.3
            cfg.num_folds = 2
            cfg.eval_folds = 1
            cfg.num_workers = 0
            cfg.run_name = "smoke_stage2"
            cfg.smoke = True
        _override(cfg, yaml_overrides.get("stage2", yaml_overrides))
        _override(cfg, {k: v for k, v in cli_kwargs.items()
                        if k in cfg.__dataclass_fields__})

    print("=" * 72)
    print(f"[topodise.main] running stage {args.stage}")
    print("=" * 72)
    for k, v in dataclasses.asdict(cfg).items():
        print(f"  {k:24s} = {v}")
    print("=" * 72)
    out = run_single_experiment(cfg)
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    main()
