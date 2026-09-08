from __future__ import annotations

import argparse
import gc
import json
import time

import torch
from core.progress import ETAEstimator, format_duration

from .bag_store import PackedBagStore
from .cohort import TASK_CLASSES, audit_and_save
from .config import load_config, resolved_paths
from .encoders import build_frozen_encoder
from .encoders.factory import SUPPORTED_ENCODERS, encoder_spec
from .feature_cache import build_patient_bags, prepare_feature_cache
from .manifests import materialize_manifests
from .run import _lock_protocol
from .trainer import run_outer_fold
from .utils import append_jsonl, atomic_json_dump, pick_device


ENCODER_NAMES = SUPPORTED_ENCODERS


def _pending_runs(
    output_root, encoder_name: str, tasks: list[str], seeds: list[int], folds: list[int]
) -> list[tuple]:
    pending = []
    for task in tasks:
        for seed in seeds:
            for fold in folds:
                run_dir = output_root / "runs" / encoder_name / task / f"seed{seed}" / f"fold{fold}"
                if not (run_dir / "DONE.json").is_file():
                    pending.append((task, int(seed), fold, run_dir))
    return pending


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load one encoder once and run all requested task/seed/fold jobs"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--encoder", required=True, choices=ENCODER_NAMES)
    parser.add_argument("--tasks", nargs="*", default=[])
    parser.add_argument("--split-seeds", nargs="*", type=int, default=[])
    parser.add_argument("--folds", nargs="*", type=int, default=[])
    parser.add_argument("--max-folds", type=int, default=0)
    parser.add_argument("--worker-id", default="")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--accept-cohort-change", action="store_true")
    parser.add_argument("--accept-protocol-change", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    cfg = load_config(args.config)
    tasks = args.tasks or list(cfg["tasks"])
    unknown = set(tasks) - set(cfg["tasks"])
    if unknown:
        raise ValueError(f"tasks are not present in the locked config: {sorted(unknown)}")
    seeds = args.split_seeds or [int(seed) for seed in cfg["splits"]["seeds"]]
    unknown_seeds = set(seeds) - {int(seed) for seed in cfg["splits"]["seeds"]}
    if unknown_seeds:
        raise ValueError(f"split seeds are not present in the locked config: {sorted(unknown_seeds)}")
    folds = args.folds or list(range(int(cfg["splits"]["outer_folds"])))
    invalid_folds = [
        fold for fold in folds if not 0 <= fold < int(cfg["splits"]["outer_folds"])
    ]
    if invalid_folds:
        raise ValueError(f"folds are outside the locked range: {sorted(invalid_folds)}")
    if len(set(seeds)) != len(seeds) or len(set(folds)) != len(folds):
        raise ValueError("split seed and fold filters must not contain duplicates")
    paths = resolved_paths(cfg)
    output_root = paths["output_root"]
    protocol_hash = _lock_protocol(cfg, output_root, args.accept_protocol_change)
    records, audit = audit_and_save(
        paths["dataset_json"], paths["dataset_root"],
        output_root / "protocol" / "cohort_audit.json", args.accept_cohort_change,
    )
    manifests = materialize_manifests(records, cfg, output_root)
    naturally_pending = _pending_runs(output_root, args.encoder, tasks, seeds, folds)
    total_requested = len(tasks) * len(seeds) * len(folds)
    already_completed = total_requested - len(naturally_pending)
    pending = naturally_pending
    if args.force:
        already_completed = 0
        pending = [
            (task, int(seed), fold,
             output_root / "runs" / args.encoder / task / f"seed{seed}" / f"fold{fold}")
            for task in tasks
            for seed in seeds
            for fold in folds
        ]
    total_pending_for_eta = len(pending)
    if args.max_folds > 0:
        pending = pending[:args.max_folds]
    if not pending:
        print(f"[skip] all requested folds are complete for {args.encoder}")
        return
    for _, _, _, run_dir in pending:
        if run_dir.exists() and any(run_dir.iterdir()) and not args.force:
            raise RuntimeError(f"incomplete run directory exists; inspect it or use --force: {run_dir}")

    progress_dir = output_root / "runs" / args.encoder
    if args.worker_id:
        progress_dir = progress_dir / "workers" / args.worker_id
    progress_path = progress_dir / "execution_progress.jsonl"
    progress_latest_path = progress_dir / "progress_latest.json"
    current_fold: dict = {}

    def emit_progress(event: dict) -> None:
        record = {
            "wall_time": time.time(),
            "encoder": args.encoder,
            **current_fold,
            **event,
        }
        append_jsonl(record, progress_path)
        atomic_json_dump(record, progress_latest_path)
        kind = record.get("kind")
        if kind == "feature_progress":
            print(
                f"[progress][{args.encoder}] feature/{record['region']} "
                f"{record['completed_units']}/{record['total_units']} "
                f"({record['progress_pct']:.1f}%) "
                f"ETA={format_duration(record.get('eta_seconds'))} "
                f"finish={record.get('estimated_finish_at')}",
                flush=True,
            )
        elif kind == "feature_cache_hit":
            print(
                f"[progress][{args.encoder}] feature cache hit: {record['cache_dir']}",
                flush=True,
            )
        elif kind == "stage2_epoch_progress":
            epoch = int(record["epoch"])
            epochs = int(record.get("epochs", record["total_units"]))
            if epoch == 1 or epoch == epochs or epoch % 5 == 0:
                print(
                    f"[progress][{args.encoder}] {record['task']} seed={record['seed']} "
                    f"fold={record['fold']} phase={record['phase']} "
                    f"epoch={epoch}/{epochs} ({record['progress_pct']:.1f}%) "
                    f"ETA={format_duration(record.get('eta_seconds'))} "
                    f"finish={record.get('estimated_finish_at')}",
                    flush=True,
                )

    emit_progress({
        "kind": "encoder_process_start",
        "already_completed_folds": already_completed,
        "pending_folds_total": total_pending_for_eta,
        "folds_planned_this_process": len(pending),
        "total_requested_folds": total_requested,
    })

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
        num_workers=max(0, int(cfg["runtime"].get("num_workers", 10))),
        pin_memory=bool(cfg["runtime"].get("pin_memory", False)),
        channels_last=bool(cfg["runtime"].get("channels_last", True)),
        progress_callback=emit_progress,
    )
    del encoder
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    pending_keys = {(task, seed, fold) for task, seed, fold, _ in pending}
    summaries = []
    fold_progress = ETAEstimator(max(total_pending_for_eta, 1), window=5)
    for task in tasks:
        reference_split = manifests["splits"][(task, int(cfg["splits"]["seeds"][0]))]
        bags = build_patient_bags(manifests["pair"], features, reference_split["patient_labels"])
        store = PackedBagStore.from_bags(bags, device)
        if store.feature_mode != encoder_metadata["feature_mode"]:
            raise RuntimeError(
                "feature cache/store mode does not match the frozen encoder: "
                f"{store.feature_mode} != {encoder_metadata['feature_mode']}"
            )
        for seed in seeds:
            split = manifests["splits"][(task, int(seed))]
            for fold_spec in split["folds"]:
                fold = int(fold_spec["fold"])
                if fold not in folds:
                    continue
                if (task, int(seed), fold) not in pending_keys:
                    continue
                current_fold.clear()
                current_fold.update({"task": task, "seed": int(seed), "fold": fold})
                run_dir = output_root / "runs" / args.encoder / task / f"seed{seed}" / f"fold{fold}"
                fold_started = time.perf_counter()
                print(
                    f"[progress][{args.encoder}] start fold "
                    f"{len(summaries)+1}/{total_pending_for_eta}: "
                    f"task={task} seed={seed} fold={fold}",
                    flush=True,
                )
                emit_progress({
                    "kind": "fold_start",
                    "completed_new_folds": len(summaries),
                    "pending_folds_total": total_pending_for_eta,
                })
                provenance = {
                    "protocol_version": cfg["protocol_version"],
                    "protocol_config_sha256": protocol_hash,
                    "cohort_json_sha256": audit["dataset_json_sha256"],
                    "pair_manifest_sha256": manifests["pair"]["manifest_sha256"],
                    "split_manifest_sha256": split["manifest_sha256"],
                    "encoder_metadata": encoder_metadata,
                    "feature_cache": str(feature_dir),
                    "device": str(device),
                    "execution_mode": "single_encoder_process_reuse",
                }
                summary = run_outer_fold(
                    raw_dim=raw_dim,
                    num_classes=len(TASK_CLASSES[task]),
                    model_cfg=model_cfg,
                    optim_cfg=cfg["optimization"],
                    bags=store,
                    fold_spec=fold_spec,
                    task=task,
                    encoder_name=args.encoder,
                    split_seed=int(seed),
                    device=device,
                    amp=bool(cfg["runtime"]["amp"]),
                    run_dir=run_dir,
                    provenance=provenance,
                    progress_callback=emit_progress,
                )
                summaries.append({
                    "task": task,
                    "seed": int(seed),
                    "fold": fold,
                    "elapsed_seconds": summary["elapsed_seconds"],
                })
                print(json.dumps(summaries[-1]))
                fold_state = fold_progress.update(
                    len(summaries),
                    interval_seconds=time.perf_counter() - fold_started,
                )
                overall_completed = already_completed + len(summaries)
                emit_progress({
                    "kind": "fold_complete",
                    "fold_elapsed_seconds": summary["elapsed_seconds"],
                    "overall_completed_folds": overall_completed,
                    "total_requested_folds": total_requested,
                    **fold_state,
                })
                print(
                    f"[ETA][{args.encoder}] folds={overall_completed}/{total_requested} "
                    f"session_new={len(summaries)}/{total_pending_for_eta} "
                    f"last={format_duration(summary['elapsed_seconds'])} "
                    f"rolling_fold={format_duration(fold_state['rolling_unit_seconds'])} "
                    f"remaining={format_duration(fold_state['eta_seconds'])} "
                    f"finish={fold_state['estimated_finish_at']} "
                    f"quality={fold_state['eta_quality']}",
                    flush=True,
                )
        del store, bags
        if device.type == "cuda":
            torch.cuda.empty_cache()
    total_elapsed = time.perf_counter() - started
    report = {
        "encoder": args.encoder,
        "completed_in_this_process": summaries,
        "num_completed": len(summaries),
        "total_elapsed_seconds": total_elapsed,
    }
    atomic_json_dump(report, progress_dir / "process_summary.json")
    current_fold.clear()
    emit_progress({"kind": "encoder_process_complete", **report})
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
