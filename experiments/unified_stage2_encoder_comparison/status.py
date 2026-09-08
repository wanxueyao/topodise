from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.progress import format_duration

from .config import load_config, resolved_paths
from .utils import resolve_codes_path


def _last_jsonl(path: Path, kinds: set[str] | None = None) -> dict | None:
    if not path.is_file():
        return None
    latest = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            if kinds is None or row.get("kind") in kinds:
                latest = row
    return latest


def _topodise_stage_status(stage_spec: dict, stage_name: str) -> dict:
    run_dir = resolve_codes_path(stage_spec["output_dir"]) / stage_spec["run_name"]
    latest = _last_jsonl(run_dir / "events.jsonl", {"step_progress", "epoch_end"})
    checkpoint = run_dir / "ckpt" / "ema_best.pt"
    return {
        "name": stage_name,
        "run_dir": str(run_dir),
        "checkpoint_ready": checkpoint.is_file(),
        "latest": latest,
    }


def collect_status(cfg: dict, encoders: tuple[str, ...]) -> dict:
    output_root = resolved_paths(cfg)["output_root"]
    result: dict = {
        "protocol_version": cfg["protocol_version"],
        "output_root": str(output_root),
        "encoders": {},
    }
    if "topodise_stage1" in encoders:
        pretrain = cfg["encoders"]["topodise_stage1"]["pretrain"]
        result["topodise_pretrain"] = [
            _topodise_stage_status(pretrain["stage0"], "topodise_stage0"),
            _topodise_stage_status(pretrain["stage1"], "topodise_stage1"),
        ]
    if "simclr_resnet18" in encoders:
        checkpoint = resolve_codes_path(cfg["encoders"]["simclr_resnet18"]["checkpoint"])
        progress_path = checkpoint.with_name(f"{checkpoint.stem}_progress.jsonl")
        result["simclr_pretrain"] = {
            "checkpoint_ready": checkpoint.is_file(),
            "progress_path": str(progress_path),
            "latest": _last_jsonl(progress_path),
        }

    total_per_encoder = (
        len(cfg["tasks"])
        * len(cfg["splits"]["seeds"])
        * int(cfg["splits"]["outer_folds"])
    )
    for encoder in encoders:
        encoder_root = output_root / "runs" / encoder
        completed = sum(
            (
                encoder_root / task / f"seed{seed}" / f"fold{fold}" / "DONE.json"
            ).is_file()
            for task in cfg["tasks"]
            for seed in cfg["splits"]["seeds"]
            for fold in range(int(cfg["splits"]["outer_folds"]))
        )
        latest_path = encoder_root / "progress_latest.json"
        latest = (
            json.loads(latest_path.read_text(encoding="utf-8"))
            if latest_path.is_file() else None
        )
        workers = {}
        for worker_path in sorted((encoder_root / "workers").glob("*/progress_latest.json")):
            worker = worker_path.parent.name
            workers[worker] = json.loads(worker_path.read_text(encoding="utf-8"))
        if workers:
            worker_latest = max(workers.values(), key=lambda row: float(row.get("wall_time", 0)))
            if latest is None or float(worker_latest.get("wall_time", 0)) > float(
                latest.get("wall_time", 0)
            ):
                latest = worker_latest
        result["encoders"][encoder] = {
            "completed_folds": completed,
            "total_folds": total_per_encoder,
            "progress_pct": round(100.0 * completed / total_per_encoder, 2),
            "latest": latest,
            "workers": workers,
        }
    return result


def _print_latest(label: str, latest: dict | None) -> None:
    if latest is None:
        print(f"  {label}: not started")
        return
    completed = latest.get("completed_units")
    total = latest.get("total_units")
    progress = latest.get("progress_pct")
    unit_text = f" {completed}/{total}" if completed is not None and total is not None else ""
    pct_text = f" ({progress:.1f}%)" if isinstance(progress, (int, float)) else ""
    print(
        f"  {label}:{unit_text}{pct_text} kind={latest.get('kind')} "
        f"ETA={format_duration(latest.get('eta_seconds'))} "
        f"finish={latest.get('estimated_finish_at')} "
        f"quality={latest.get('eta_quality')}"
    )


def print_human(status: dict) -> None:
    print(f"protocol: {status['protocol_version']}")
    print(f"results:  {status['output_root']}")
    if "topodise_pretrain" in status:
        print("TopoDiSE pretraining:")
        for stage in status["topodise_pretrain"]:
            _print_latest(stage["name"], stage["latest"])
            print(f"    checkpoint_ready={stage['checkpoint_ready']}")
    if "simclr_pretrain" in status:
        print("SimCLR pretraining:")
        _print_latest("simclr", status["simclr_pretrain"]["latest"])
        print(f"    checkpoint_ready={status['simclr_pretrain']['checkpoint_ready']}")
    print("Stage2:")
    for encoder, row in status["encoders"].items():
        print(
            f"  {encoder}: folds={row['completed_folds']}/{row['total_folds']} "
            f"({row['progress_pct']:.1f}%)"
        )
        _print_latest("latest", row["latest"])
        for worker, latest in row.get("workers", {}).items():
            _print_latest(f"worker/{worker}", latest)


def main() -> None:
    parser = argparse.ArgumentParser(description="Show live pretraining and Stage2 ETA status")
    parser.add_argument("--config", required=True)
    parser.add_argument("--encoders", nargs="*", default=[])
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.encoders:
        encoders = tuple(args.encoders)
    else:
        encoders = tuple(cfg["encoders"])
    status = collect_status(cfg, encoders)
    if args.json:
        print(json.dumps(status, ensure_ascii=False, indent=2))
    else:
        print_human(status)


if __name__ == "__main__":
    main()
