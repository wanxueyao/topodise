from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

from .cohort import TASK_CLASSES, audit_and_save
from .config import load_config, resolved_paths
from .encoders import build_frozen_encoder
from .encoders.factory import SUPPORTED_ENCODERS, encoder_spec
from .feature_cache import build_patient_bags, prepare_feature_cache
from .manifests import materialize_manifests
from .trainer import run_outer_fold
from .utils import atomic_json_dump, canonical_hash, pick_device


def _lock_protocol(cfg: dict, output_root: Path, accept_change: bool) -> str:
    protocol_path = output_root / "protocol" / "resolved_config.json"
    protocol_hash = canonical_hash(cfg)
    if protocol_path.is_file():
        existing = json.loads(protocol_path.read_text(encoding="utf-8"))
        if existing.get("config_sha256") != protocol_hash and not accept_change:
            raise RuntimeError(
                "resolved config differs from the existing protocol lock; use a new output_root or "
                "pass --accept-protocol-change after deliberate review"
            )
    atomic_json_dump({"config_sha256": protocol_hash, "config": cfg}, protocol_path)
    return protocol_hash


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one fair frozen-encoder Stage-2 fold")
    parser.add_argument("--config", required=True)
    parser.add_argument("--encoder", required=True, choices=SUPPORTED_ENCODERS)
    parser.add_argument("--task", required=True, choices=list(TASK_CLASSES))
    parser.add_argument("--split-seed", required=True, type=int)
    parser.add_argument("--fold", required=True, type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--accept-cohort-change", action="store_true")
    parser.add_argument("--accept-protocol-change", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = resolved_paths(cfg)
    output_root = paths["output_root"]
    protocol_hash = _lock_protocol(cfg, output_root, args.accept_protocol_change)
    records, audit = audit_and_save(
        paths["dataset_json"], paths["dataset_root"],
        output_root / "protocol" / "cohort_audit.json", args.accept_cohort_change,
    )
    manifests = materialize_manifests(records, cfg, output_root)
    split = manifests["splits"].get((args.task, args.split_seed))
    if split is None:
        raise ValueError("requested task/seed is not present in the locked config")
    if not 0 <= args.fold < len(split["folds"]):
        raise ValueError(f"fold must be in [0, {len(split['folds']) - 1}]")
    run_dir = output_root / "runs" / args.encoder / args.task / f"seed{args.split_seed}" / f"fold{args.fold}"
    if (run_dir / "DONE.json").is_file() and not args.force:
        print(f"[skip] completed run: {run_dir}")
        return
    if run_dir.exists() and any(run_dir.iterdir()) and not args.force:
        raise RuntimeError(f"incomplete run directory exists; inspect it or rerun with --force: {run_dir}")

    device = pick_device(str(cfg["runtime"]["device"]))
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    encoder = build_frozen_encoder(
        args.encoder, encoder_spec(cfg, args.encoder), str(cfg["runtime"]["device"])
    )
    encoder_metadata = encoder.metadata()
    raw_dim = encoder.raw_dim
    model_cfg = {**cfg["model"], "input_mode": encoder.feature_mode}
    features, feature_dir = prepare_feature_cache(
        encoder, manifests["pair"], output_root, device,
        batch_size=max(1, int(cfg["runtime"].get("feature_batch_size", 64))),
        amp=bool(cfg["runtime"]["amp"]),
        num_workers=max(0, int(cfg["runtime"].get("num_workers", 8))),
        pin_memory=bool(cfg["runtime"].get("pin_memory", False)),
        channels_last=bool(cfg["runtime"].get("channels_last", True)),
    )
    del encoder
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    bags = build_patient_bags(manifests["pair"], features, split["patient_labels"])
    provenance = {
        "protocol_version": cfg["protocol_version"],
        "protocol_config_sha256": protocol_hash,
        "cohort_json_sha256": audit["dataset_json_sha256"],
        "pair_manifest_sha256": manifests["pair"]["manifest_sha256"],
        "split_manifest_sha256": split["manifest_sha256"],
        "encoder_metadata": encoder_metadata,
        "feature_cache": str(feature_dir),
        "device": str(device),
    }
    summary = run_outer_fold(
        raw_dim=raw_dim,
        num_classes=len(TASK_CLASSES[args.task]),
        model_cfg=model_cfg,
        optim_cfg=cfg["optimization"],
        bags=bags,
        fold_spec=split["folds"][args.fold],
        task=args.task,
        encoder_name=args.encoder,
        split_seed=args.split_seed,
        device=device,
        amp=bool(cfg["runtime"]["amp"]),
        run_dir=run_dir,
        provenance=provenance,
    )
    print(json.dumps({"run_dir": str(run_dir), "outer_metrics": summary["outer_metrics"]}, indent=2))


if __name__ == "__main__":
    main()
