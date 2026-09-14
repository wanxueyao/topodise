"""Subprocess entry: isolate training-module overrides to one experiment."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
from importlib import import_module
import os
from pathlib import Path
import sys
import time

from .protocol import (ENCODER, checkpoint, file_hash, lock, read,
                       require_complete, source_hashes, write)


@contextmanager
def execution_lock(path):
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"This job is already running: {path}") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


@contextmanager
def training_overrides(module, stage):
    from .topo_targets import compute_topo_target
    original_target = module.compute_topo_target
    original_dataset = getattr(module, "CORN1Dataset", None)
    module.compute_topo_target = compute_topo_target
    if stage == "stage0":
        from .aligned_corn import AlignedCORN1Dataset
        module.CORN1Dataset = AlignedCORN1Dataset
    try:
        yield
    finally:
        module.compute_topo_target = original_target
        if stage == "stage0":
            module.CORN1Dataset = original_dataset


def run_pretrain(job, device):
    import torch
    from experiments._stage_configs import Stage0Config, Stage1Config

    stage = job["stage"]
    values = dict(job["config"])
    output = Path(values["output_dir"]) / values["run_name"]
    parent_hash = None
    if stage == "stage1":
        parent = require_complete(Path(values["parent_ckpt"]).parent.parent)
        parent_hash = file_hash(parent)
    lock(output / "ablation_job.json", {"job": job, "parent_checkpoint_sha256": parent_hash})
    if (output / "ABLATION_DONE.json").exists():
        require_complete(output)
        print(f"[skip] completed {job['id']}", flush=True)
        return
    # Do not silently resume an interrupted run from its earlier best epoch.
    if checkpoint(output).exists() and not (output / "ckpt/last.pt").exists():
        raise RuntimeError(f"Best checkpoint exists but last.pt is missing: {output}")
    values["device"] = device
    if stage == "stage0":
        config = Stage0Config(**values)
        module_name = "experiments.topodise_experiments.stage0_corn1_pretrain.main"
    else:
        config = Stage1Config(**values)
        module_name = "experiments.topodise_experiments.stage1_rscp_finetune.main"
    module = import_module(module_name)
    with training_overrides(module, stage):
        result = module.run_single_experiment(config)
    last = torch.load(output / "ckpt/last.pt", map_location="cpu", weights_only=False)
    if int(last["epoch"]) != int(config.epochs):
        raise RuntimeError(f"Training did not finish requested epochs: {job['id']}")
    del last
    write(output / "ABLATION_DONE.json", {
        "job_id": job["id"], "epochs": config.epochs,
        "checkpoint_sha256": file_hash(checkpoint(output)),
        "parent_checkpoint_sha256": parent_hash, "result": result,
    })


def archive_incomplete(output, encoder, cfg):
    """Keep interrupted artifacts; existing Core runner resumes completed folds."""
    archived = []
    stamp = f"{time.time_ns()}"
    runs = output / "runs" / encoder
    for task in cfg["tasks"]:
        for seed in cfg["splits"]["seeds"]:
            for fold in range(cfg["splits"]["outer_folds"]):
                path = runs / task / f"seed{seed}" / f"fold{fold}"
                if path.exists() and any(path.iterdir()) and not (path / "DONE.json").exists():
                    destination = output / "interrupted" / stamp / path.relative_to(output)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    path.rename(destination)
                    archived.append(str(destination))
    return archived


def run_stage2(job, device, retry_incomplete=False, max_folds=0):
    if job.get("backend") == "full":
        if max_folds:
            raise ValueError("Full Stage2 must run complete configured folds")
        from .full_stage2 import run_full_stage2
        return run_full_stage2(job, device)
    from experiments.unified_stage2_encoder_comparison.config import load_config
    from experiments.unified_stage2_encoder_comparison.run_encoder import main as core_main

    if job["encoder"] != ENCODER:
        raise ValueError("All routes must use the same frozen native z_sem encoder")
    values = job["config"]
    checkpoint_path = Path(values["encoders"]["topodise_stage1"]["checkpoint"])
    require_complete(checkpoint_path.parent.parent)
    output = Path(values["output_root"])
    lock(output / "ablation_job.json", {
        "job": job, "checkpoint_sha256": file_hash(checkpoint_path),
    })
    # JSON is valid YAML. Resolve and persist the full existing Core defaults;
    # there is no reimplementation of the Stage2 training loop here.
    override_path = output / "core_overrides.json"
    lock(override_path, values)
    cfg = load_config(override_path)
    cfg["runtime"]["device"] = device
    resolved = output / "core_config.json"
    lock(resolved, cfg)
    if retry_incomplete:
        for path in archive_incomplete(output, ENCODER, cfg):
            print(f"[archived interrupted fold] {path}", flush=True)
    original_argv = sys.argv
    try:
        sys.argv = ["unified_stage2.run_encoder", "--config", str(resolved), "--encoder", ENCODER]
        if max_folds:
            sys.argv += ["--max-folds", str(max_folds)]
        core_main()
    finally:
        sys.argv = original_argv


def normalize_device(device):
    # Logical cuda:N respects an existing CUDA_VISIBLE_DEVICES assignment.
    # Each child exposes one device as cuda so its locked config is independent
    # of which GPU slot the scheduler assigns on a later continuation.
    if device.startswith("cuda:"):
        index = int(device.split(":")[1])
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible is not None:
            choices = visible.split(",")
            if index >= len(choices):
                raise ValueError(f"{device} is outside CUDA_VISIBLE_DEVICES={visible}")
            selected = choices[index]
        else:
            selected = str(index)
        os.environ["CUDA_VISIBLE_DEVICES"] = selected
        return "cuda"
    return device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--retry-incomplete", action="store_true")
    parser.add_argument("--max-folds", type=int, default=0)
    args = parser.parse_args()
    device = normalize_device(args.device)  # Before torch is imported
    job = read(args.job)
    protocol = read(args.job.parent.parent / "protocol.json")
    if job not in protocol["jobs"][job["stage"]] or protocol["source_sha256"] != source_hashes():
        raise RuntimeError("Job or source no longer matches the locked ablation protocol")
    # An ambient c8 checkpoint/portable override would invalidate the seed plan.
    os.environ.pop("TOPODISE_REPRODUCIBILITY", None)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    with execution_lock(args.job.with_suffix(".running.lock")):
        if job["stage"] == "stage2":
            run_stage2(job, device, args.retry_incomplete, args.max_folds)
        else:
            run_pretrain(job, device)


if __name__ == "__main__":
    main()
