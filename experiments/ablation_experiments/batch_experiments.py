"""Edit the settings below, then run this file (or use --dry-run / --phase)."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict
import os
from pathlib import Path
import queue
import subprocess
import sys

if sys.version_info < (3, 10):
    raise SystemExit("Python >= 3.10 is required by the existing training modules.")

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments._stage_configs import Stage0Config, Stage1Config
from experiments.ablation_experiments.protocol import (
    CODES, ENCODER, ROUTES, checkpoint, lock, run_root, source_hashes,
)

# All six routes use these same settings. Do not put per-route hyperparameters
# here: topology source is the only factor varied by this experiment family.
EXPERIMENT_NAME = "topology_targets_v2"
DEVICES = ["auto"]                 # e.g. ["cuda:0", "cuda:1"] on the training server
PRETRAIN_SEEDS = [42]               # Independent Stage0/1 runs if more seeds are added
DATASET_ROOT = CODES / "datasets"
CORN1_ROOT = DATASET_ROOT / "CORN-1"
PD_JSON = DATASET_ROOT / "pd_dataset_en/patients_dataset_en.json"
STAGE1_MANIFEST = DATASET_ROOT / "unlabeled_dataset_en/unlabeled_dataset_en.json"
STAGE0_SETTINGS = {"epochs": 200, "batch_size": 16, "num_workers": 4,
                   "expected_samples": 1516, "reproducibility": "strict"}
STAGE1_SETTINGS = {"epochs": 80, "batch_size": 16, "num_workers": 4,
                   "expected_region_pairs": None, "reproducibility": "strict"}
# Unspecified Stage2 fields inherit the existing Unified Core implementation.
STAGE2_SETTINGS = {
    "tasks": ["pd_binary", "pd_stage", "pd_subtype_tdrest"],
    "splits": {"outer_folds": 5, "seeds": [42, 202, 3407], "inner_val_ratio": 0.2},
}


def build_plan(root, dataset_root, corn_root, pd_json, manifest, seeds):
    """Pure plan construction: no training imports, directories or GPU access."""
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("pretraining seeds must be nonempty and unique")
    jobs = {"stage0": [], "stage1": [], "stage2": []}
    for seed in seeds:
        base = root / f"pretrain_seed{seed}"
        for target in ("gt", "hgrad_dual", "frangi"):
            cfg = asdict(Stage0Config(**STAGE0_SETTINGS))
            cfg.update(corn1_root=str(corn_root), seed=seed, topo_target=target,
                       output_dir=str(base / "stage0"), run_name=target)
            jobs["stage0"].append({"id": f"s0_{target}_seed{seed}", "stage": "stage0", "config": cfg})
        for name, (s0, s1) in ROUTES.items():
            cfg = asdict(Stage1Config(**STAGE1_SETTINGS))
            cfg.update(parent_ckpt=str(checkpoint(base / "stage0" / s0)), seed=seed,
                       topo_target=s1, region_pair_manifest=str(manifest),
                       pd_json=str(pd_json), base_path=str(dataset_root),
                       output_dir=str(base / "stage1"), run_name=name,
                       allow_dm_disease=False, train_denoiser=False)
            jobs["stage1"].append({"id": f"s1_{name}_seed{seed}", "stage": "stage1", "config": cfg})
            core = deepcopy(STAGE2_SETTINGS)
            core.update(dataset_root=str(dataset_root), dataset_json=str(pd_json),
                        output_root=str(base / "stage2" / name),
                        encoders={"topodise_stage1": {
                            "checkpoint": str(checkpoint(base / "stage1" / name)),
                            "use_ema": True, "auto_pretrain_if_missing": False,
                        }})
            jobs["stage2"].append({"id": f"s2_{name}_seed{seed}", "stage": "stage2",
                                    "encoder": ENCODER, "config": core})
    return jobs


def selected_jobs(jobs, phase, variants):
    stages = {"all": ("stage0", "stage1", "stage2"),
              "pretrain": ("stage0", "stage1")}.get(phase, (phase,))
    parents = {ROUTES[name][0] for name in variants}
    return {stage: [job for job in jobs[stage] if
                   (job["config"]["topo_target"] in parents if stage == "stage0" else
                    (job["config"]["run_name"] if stage == "stage1" else
                     Path(job["config"]["output_root"]).name) in variants)]
            for stage in stages}


def execute_phase(jobs, root, devices, retry_incomplete=False, max_folds=0):
    slots = queue.Queue()
    for device in devices:
        slots.put(device)

    def execute(job):
        device = slots.get()
        try:
            config_path = root / "configs" / f"{job['id']}.json"
            log = root / "logs" / f"{job['id']}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, "-u", "-m", "experiments.ablation_experiments.worker",
                       "--job", str(config_path), "--device", device]
            if retry_incomplete:
                command.append("--retry-incomplete")
            if max_folds:
                command += ["--max-folds", str(max_folds)]
            print(f"[start] {job['id']} on {device}; log: {log}", flush=True)
            with log.open("a", encoding="utf-8") as stream:
                result = subprocess.run(command, cwd=CODES, stdout=stream,
                                        stderr=subprocess.STDOUT, check=False,
                                        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
            if result.returncode:
                raise RuntimeError(f"{job['id']} failed ({result.returncode}); inspect {log}")
            print(f"[done] {job['id']}", flush=True)
        finally:
            slots.put(device)

    # Phase barriers ensure Stage0 parents finish before Stage1, and Stage1
    # parents before Stage2. Each device has at most one active subprocess.
    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        for result in pool.map(execute, jobs):
            pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default=EXPERIMENT_NAME)
    parser.add_argument("--phase", choices=("all", "pretrain", "stage0", "stage1", "stage2", "prepare", "analyze"), default="all")
    parser.add_argument("--variants", nargs="+", choices=tuple(ROUTES), default=list(ROUTES))
    parser.add_argument("--devices", nargs="+", default=DEVICES)
    parser.add_argument("--pretrain-seeds", nargs="+", type=int, default=PRETRAIN_SEEDS)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--corn1-root", type=Path, default=None)
    parser.add_argument("--pd-json", type=Path, default=None)
    parser.add_argument("--stage1-manifest", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Print the full plan without writing files or loading torch")
    parser.add_argument("--retry-incomplete", action="store_true", help="Archive interrupted Stage2 folds, then rerun only unfinished folds")
    parser.add_argument("--max-folds", type=int, default=0, help="Limit Stage2 pending folds per route for a pilot (0=all)")
    args = parser.parse_args()
    if len(set(args.devices)) != len(args.devices) or ("auto" in args.devices and len(args.devices) != 1):
        parser.error("Use distinct explicit devices for parallel jobs, or a single auto device")
    if any(d != "auto" and d != "cpu" and d != "mps" and not (d.startswith("cuda:") and d[5:].isdigit()) for d in args.devices):
        parser.error("Devices must be auto, cpu, mps, or cuda:N")
    if args.max_folds < 0:
        parser.error("--max-folds must be nonnegative")
    root = run_root(args.name)
    if args.phase == "analyze":
        from experiments.ablation_experiments.analyze import summarize
        summarize(root)
        return
    dataset = args.dataset_root.expanduser().resolve()
    default_root = dataset == DATASET_ROOT.resolve()
    corn = (args.corn1_root or (CORN1_ROOT if default_root else dataset / "CORN-1")).expanduser().resolve()
    pd = (args.pd_json or (PD_JSON if default_root else dataset / "pd_dataset_en/patients_dataset_en.json")).expanduser().resolve()
    manifest = (args.stage1_manifest or (STAGE1_MANIFEST if default_root else dataset / "unlabeled_dataset_en/unlabeled_dataset_en.json")).expanduser().resolve()
    jobs = build_plan(root, dataset, corn, pd, manifest, args.pretrain_seeds)
    chosen = selected_jobs(jobs, "all" if args.phase == "prepare" else args.phase, args.variants)
    print(f"Output: {root}\nStage2: {ENCODER}; frozen encoder + cached features")
    for stage, items in chosen.items():
        for job in items:
            print(f"  {job['id']}")
    if args.dry_run:
        return
    # Lock all six, even for a filtered invocation; filters never change the
    # scientific protocol or allow stale checkpoints from another recipe.
    contract = {"version": "topology-targets-v2", "jobs": jobs, "source_sha256": source_hashes()}
    lock(root / "protocol.json", contract)
    for items in jobs.values():
        for job in items:
            lock(root / "configs" / f"{job['id']}.json", job)
    from experiments.ablation_experiments.audit import audit_inputs
    audit_inputs(root, jobs)
    if args.phase == "prepare":
        print("Inputs audited and configurations saved. No training started.")
        return
    for items in chosen.values():
        execute_phase(items, root, args.devices, args.retry_incomplete, args.max_folds)
    if args.phase in ("all", "stage2"):
        from experiments.ablation_experiments.analyze import summarize
        summarize(root)


if __name__ == "__main__":
    main()
