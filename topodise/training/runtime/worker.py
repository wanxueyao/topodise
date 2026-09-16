"""Isolated execution of locked Stage0, Stage1 and full Stage2 jobs."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
from importlib import import_module
import os
from pathlib import Path

from .protocol import (checkpoint, file_hash, lock, read,
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
    lock(output / "training_job.json", {"job": job, "parent_checkpoint_sha256": parent_hash})
    if (output / "TRAINING_DONE.json").exists() or (output / "ABLATION_DONE.json").exists():
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
    result = module.run_single_experiment(config)
    last = torch.load(output / "ckpt/last.pt", map_location="cpu", weights_only=False)
    if int(last["epoch"]) != int(config.epochs):
        raise RuntimeError(f"Training did not finish requested epochs: {job['id']}")
    del last
    write(output / "TRAINING_DONE.json", {
        "job_id": job["id"], "epochs": config.epochs,
        "checkpoint_sha256": file_hash(checkpoint(output)),
        "parent_checkpoint_sha256": parent_hash, "result": result,
    })


def run_stage2(job, device):
    from .full_stage2 import run_full_stage2
    return run_full_stage2(job, device)


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
    args = parser.parse_args()
    device = normalize_device(args.device)  # Before torch is imported
    job = read(args.job)
    protocol = read(args.job.parent.parent / "protocol.json")
    if job not in protocol["jobs"][job["stage"]] or protocol["source_sha256"] != source_hashes():
        raise RuntimeError("Job or source no longer matches the locked training protocol")
    # An ambient c8 checkpoint/portable override would invalidate the seed plan.
    os.environ.pop("TOPODISE_REPRODUCIBILITY", None)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    with execution_lock(args.job.with_suffix(".running.lock")):
        if job["stage"] == "stage2":
            run_stage2(job, device)
        else:
            run_pretrain(job, device)


if __name__ == "__main__":
    main()
