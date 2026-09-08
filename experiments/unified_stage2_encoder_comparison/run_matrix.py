from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .config import load_config, resolved_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the encoder matrix on the current machine"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--encoders", nargs="*", default=[])
    parser.add_argument("--tasks", nargs="*", default=[])
    parser.add_argument("--split-seeds", nargs="*", type=int, default=[])
    parser.add_argument("--folds", nargs="*", type=int, default=[])
    parser.add_argument("--max-folds-per-encoder", type=int, default=0)
    parser.add_argument("--worker-id", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    paths = resolved_paths(cfg)
    encoders = args.encoders or list(cfg["encoders"])
    tasks = args.tasks or list(cfg["tasks"])
    seeds = args.split_seeds or [int(seed) for seed in cfg["splits"]["seeds"]]
    folds = args.folds or list(range(int(cfg["splits"]["outer_folds"])))
    commands = []
    for encoder in encoders:
        pending = 0
        for task in tasks:
            for seed in seeds:
                for fold in folds:
                    done = (
                        paths["output_root"] / "runs" / encoder / task
                        / f"seed{seed}" / f"fold{fold}" / "DONE.json"
                    )
                    pending += int(not done.is_file())
        if pending == 0:
            continue
        command = [
            sys.executable,
            "-m",
            "experiments.unified_stage2_encoder_comparison.run_encoder",
            "--config",
            str(Path(args.config).resolve()),
            "--encoder",
            encoder,
        ]
        if args.tasks:
            command.extend(["--tasks", *tasks])
        if args.split_seeds:
            command.extend(["--split-seeds", *map(str, args.split_seeds)])
        if args.folds:
            command.extend(["--folds", *map(str, args.folds)])
        if args.worker_id:
            command.extend(["--worker-id", args.worker_id])
        if args.max_folds_per_encoder > 0:
            command.extend(["--max-folds", str(args.max_folds_per_encoder)])
        commands.append((pending, command))
    print(f"pending encoder processes: {len(commands)}")
    print(f"pending folds: {sum(pending for pending, _ in commands)}")
    for pending, command in commands:
        print(f"[{pending} folds] {' '.join(command)}")
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
