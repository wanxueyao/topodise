from __future__ import annotations

import json

import torch

from .trainer import run_outer_fold
from .utils import codes_root


def main() -> None:
    generator = torch.Generator().manual_seed(7)
    bags = {}
    for index in range(20):
        label = index % 2
        signal = float(label) * 0.5
        pid = f"S{index:03d}"
        bags[pid] = {
            "patient_id": pid,
            "label": label,
            "central": torch.randn(3, 16, generator=generator) + signal,
            "whorl": torch.randn(3, 16, generator=generator) + signal,
            "morphology": torch.randn(21, generator=generator) + signal,
        }
    fold = {
        "fold": 0,
        "outer_train": [f"S{i:03d}" for i in range(16)],
        "inner_train": [f"S{i:03d}" for i in range(12)],
        "inner_val": [f"S{i:03d}" for i in range(12, 16)],
        "outer_test": [f"S{i:03d}" for i in range(16, 20)],
    }
    output = codes_root() / "results" / "unified_stage2_encoder_comparison" / "smoke"
    summary = run_outer_fold(
        raw_dim=16,
        num_classes=2,
        model_cfg={
            "common_dim": 32, "adapter_rank": 4, "attention_hidden": 8,
            "attention_dropout": 0.0, "morphology_dim": 4,
        },
        optim_cfg={
            "epochs": 2, "warmup_epochs": 1, "lr": 1e-3, "min_lr": 1e-6,
            "weight_decay": 1e-2, "grad_clip": 1.0, "accumulate_patients": 4,
        },
        bags=bags,
        fold_spec=fold,
        task="pd_binary",
        encoder_name="synthetic",
        split_seed=7,
        device=torch.device("cpu"),
        amp=False,
        run_dir=output,
        provenance={
            "smoke": True,
            "protocol_version": "smoke",
            "protocol_config_sha256": "smoke",
        },
    )
    print(json.dumps(summary["outer_metrics"], indent=2))


if __name__ == "__main__":
    main()
