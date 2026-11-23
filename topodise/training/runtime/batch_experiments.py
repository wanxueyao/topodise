"""Shared full Stage0/1/2 scheduler; experiment families supply only their routes."""
from __future__ import annotations
import argparse
from dataclasses import asdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import os
import queue
import subprocess
import sys

from experiments._stage_configs import Stage0Config, Stage1Config
from .protocol import CODES, ROOT, ROUTES, checkpoint, lock, run_root, source_hashes, read
from .full_stage2 import TASKS, build_full_config, summarize_full

STAGE0_SETTINGS = {"epochs": 200, "batch_size": 16, "num_workers": 4,
                   "expected_samples": 1516, "reproducibility": "strict"}
STAGE1_SETTINGS = {"epochs": 80, "batch_size": 16, "num_workers": 4,
                   "expected_region_pairs": None, "reproducibility": "strict"}


def build_plan(root, dataset_root, corn_root, pd_json, manifest, seeds,
               routes=ROUTES, tasks=TASKS, repeats=3, num_workers=0, dm_json=None,
               bootstrap_samples=2000, bootstrap_seed=20260915):
    if not seeds or len(seeds) != len(set(seeds)) or repeats < 1:
        raise ValueError("Use unique pretraining seeds and positive repeat count")
    jobs = {"stage0": [], "stage1": [], "stage2": []}
    for seed in seeds:
        base = root / f"pretrain_seed{seed}"
        for target in dict.fromkeys(pair[0] for pair in routes.values()):
            cfg = asdict(Stage0Config(**STAGE0_SETTINGS))
            cfg.update(corn1_root=str(corn_root), seed=seed, topo_target=target,
                       output_dir=str(base / "stage0"), run_name=target)
            jobs["stage0"].append({"id": f"s0_{target}_seed{seed}", "stage": "stage0", "config": cfg})
        for name, (s0, s1) in routes.items():
            cfg = asdict(Stage1Config(**STAGE1_SETTINGS))
            cfg.update(parent_ckpt=str(checkpoint(base / "stage0" / s0)), seed=seed,
                       topo_target=s1, region_pair_manifest=str(manifest), pd_json=str(pd_json),
                       base_path=str(dataset_root), output_dir=str(base / "stage1"), run_name=name,
                       allow_dm_disease=False, train_denoiser=False)
            jobs["stage1"].append({"id": f"s1_{name}_seed{seed}", "stage": "stage1", "config": cfg})
            cfg = build_full_config(checkpoint(base / "stage1" / name), base / "stage2" / name,
                pd_json, dataset_root, repeats, num_workers, tasks, dm_json, bootstrap_samples, bootstrap_seed)
            jobs["stage2"].append({"id": f"s2_{name}_seed{seed}", "stage": "stage2", "backend": "full",
                "variant": name, "pretrain_seed": seed, "config": cfg})
    return jobs


def selected_jobs(jobs, phase, variants, routes=ROUTES):
    stages = {"all": ("stage0", "stage1", "stage2"), "pretrain": ("stage0", "stage1")}.get(phase, (phase,))
    parents = {routes[name][0] for name in variants}
    return {stage: [j for j in jobs[stage] if
        (j["config"]["topo_target"] in parents if stage == "stage0" else
         (j["config"]["run_name"] if stage == "stage1" else j["variant"]) in variants)] for stage in stages}


def execute_phase(jobs, root, devices, worker_module="topodise.training.runtime.worker"):
    slots = queue.Queue()
    for device in devices:
        slots.put(device)
    def execute(job):
        device = slots.get()
        try:
            log = root / "logs" / f"{job['id']}.log"; log.parent.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, "-u", "-m", worker_module,
                       "--job", str(root / "configs" / f"{job['id']}.json"), "--device", device]
            print(f"[start] {job['id']} on {device}; log: {log}", flush=True)
            with log.open("a", encoding="utf-8") as stream:
                result = subprocess.run(command, cwd=CODES, stdout=stream, stderr=subprocess.STDOUT,
                    env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
            if result.returncode:
                raise RuntimeError(f"{job['id']} failed ({result.returncode}); inspect {log}")
            print(f"[done] {job['id']}", flush=True)
        finally:
            slots.put(device)
    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        for _ in pool.map(execute, jobs):
            pass


def main(argv=None, routes=ROUTES, home=ROOT, default_name="gt_hgrad_dual_full_v1",
         default_tasks=TASKS):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default=default_name)
    parser.add_argument("--phase", choices=("all", "pretrain", "stage0", "stage1", "stage2", "prepare", "analyze"), default="all")
    parser.add_argument("--variants", nargs="+", choices=tuple(routes), default=list(routes))
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(default_tasks))
    parser.add_argument("--section", choices=("all", "main_tasks", "dm_transfer"), default=None)
    parser.add_argument("--devices", nargs="+", default=["auto"])
    parser.add_argument("--pretrain-seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--full-stage2-repeats", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=0, help="Stage2 data workers; pretrain settings are fixed")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260915)
    parser.add_argument("--dataset-root", type=Path, default=CODES / "datasets")
    parser.add_argument("--corn1-root", type=Path)
    parser.add_argument("--pd-json", type=Path)
    parser.add_argument("--dm-json", type=Path)
    parser.add_argument("--stage1-manifest", type=Path)
    parser.add_argument("--pretrained-from", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if len(set(args.devices)) != len(args.devices) or ("auto" in args.devices and len(args.devices) != 1):
        parser.error("Use distinct explicit devices or a single auto device")
    if any(d not in ("auto", "cpu", "mps") and not (d.startswith("cuda:") and d[5:].isdigit()) for d in args.devices):
        parser.error("Devices must be auto, cpu, mps, or cuda:N")
    if min(args.full_stage2_repeats, args.bootstrap_samples) < 1 or args.num_workers < 0:
        parser.error("Repeats/bootstrap must be positive; workers must be nonnegative")
    root = run_root(args.name, home)
    if args.phase == "analyze":
        return summarize_full(root, read(root / "protocol.json"))
    if args.pretrained_from and args.phase not in ("all", "prepare", "stage2"):
        parser.error("--pretrained-from requires all, stage2 or prepare")
    tasks = args.tasks
    if args.section:
        tasks = list(TASKS) if args.section == "all" else (["dm_binary"] if args.section == "dm_transfer" else list(TASKS[:3]))
    data = args.dataset_root.expanduser().resolve()
    pd = (args.pd_json or data / "pd_dataset_en/patients_dataset_en.json").resolve()
    dm = (args.dm_json or data / "dm_dataset_en/dm_patients_dataset_en.json").resolve()
    corn = (args.corn1_root or data / "CORN-1").resolve()
    manifest = (args.stage1_manifest or data / "unlabeled_dataset_en/unlabeled_dataset_en.json").resolve()
    jobs = build_plan(root, data, corn, pd, manifest, args.pretrain_seeds, routes, tasks,
                      args.full_stage2_repeats, args.num_workers, dm, args.bootstrap_samples, args.bootstrap_seed)
    if args.pretrained_from:
        from .reuse_pretrain import reuse_parents
        reuse_parents(jobs, args.pretrained_from.expanduser().resolve(), args.variants, verify=not args.dry_run)
    chosen = selected_jobs(jobs, "all" if args.phase == "prepare" else args.phase, args.variants, routes)
    if args.pretrained_from:
        chosen = {"stage2": chosen["stage2"]}
    print(f"Output: {root}\nFull Stage2 tasks: {', '.join(tasks)}")
    for items in chosen.values():
        for job in items:
            print(f"  {job['id']}")
    if args.dry_run:
        return jobs
    contract = {"version": "full-stage2-ci95-v1", "jobs": jobs, "source_sha256": source_hashes()}
    lock(root / "protocol.json", contract)
    for items in jobs.values():
        for job in items:
            lock(root / "configs" / f"{job['id']}.json", job)
    from .audit import audit_inputs
    audit = audit_inputs(root, jobs)
    if args.pretrained_from:
        previous = read(args.pretrained_from.expanduser().resolve() / "input_audit.json")
        # New DM images may be added; every previously audited shared input must agree.
        current = audit["input_file_sha256"]
        if any(current.get(k) != v for k, v in previous["input_file_sha256"].items()):
            raise RuntimeError("Reuse source data differs from current audit")
    if args.phase == "prepare":
        print("Configurations and data audited. No training started.")
        return jobs
    for items in chosen.values():
        execute_phase(items, root, args.devices)
    if args.phase in ("all", "stage2"):
        summarize_full(root, contract)
    return jobs
