"""Compare TopoDiSE, SimCLR and DINOv2 with the full shared Stage2."""
from __future__ import annotations
import argparse
from copy import deepcopy
from pathlib import Path
from .config import load_config
from .utils import resolve_codes_path
from topodise.training.runtime.batch_experiments import build_plan, execute_phase as shared_execute_phase
from topodise.training.runtime.full_stage2 import TASKS, build_full_config, summarize_full
from topodise.training.runtime.protocol import read, lock, run_root, file_hash

from ..common.execution import source_hashes

def execute_phase(jobs,root,devices):
    return shared_execute_phase(jobs,root,devices,worker_module='ablation_experiments.G01_G05_encoder_comparison.worker')

HOME = Path(__file__).resolve().parent
ENCODERS = ("topodise", "simclr_resnet18", "dinov2_vits14")


def build_comparison(root, config, selected=ENCODERS):
    data = resolve_codes_path(config["dataset_root"])
    jobs = build_plan(root, data, data / "CORN-1", data / "pd_dataset_en/patients_dataset_en.json",
                      data / "unlabeled_dataset_en/unlabeled_dataset_en.json", [42],
                      tasks=config["tasks"], repeats=config["repeats"],
                      bootstrap_samples=config["bootstrap_samples"], bootstrap_seed=config["bootstrap_seed"])
    native = jobs["stage2"][0]
    if "topodise" not in selected:
        jobs = {"stage0": [], "stage1": [], "stage2": []}
    else:
        native["variant"] = "topodise"
    for name in selected:
        if name == "topodise":
            continue
        spec = config["encoders"][name]
        checkpoint = resolve_codes_path(spec["checkpoint"])
        cfg = build_full_config("", root / "pretrain_seed42/stage2" / name,
            data / "pd_dataset_en/patients_dataset_en.json", data,
            repeats=config["repeats"], tasks=config["tasks"],
            bootstrap_samples=config["bootstrap_samples"], bootstrap_seed=config["bootstrap_seed"],
            encoder_backend=name, encoder_checkpoint=checkpoint,
            encoder_model_name=spec.get("model_name", "vit_small_patch14_dinov2"))
        jobs["stage2"].append({"id": f"s2_{name}_seed42", "stage": "stage2", "backend": "full",
                              "variant": name, "pretrain_seed": 42, "config": cfg})
    return jobs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--name", default="encoders_full_v1")
    parser.add_argument("--encoders", nargs="+", choices=ENCODERS, default=list(ENCODERS))
    parser.add_argument("--phase", choices=("all", "pretrain", "stage2", "prepare", "analyze"), default="all")
    parser.add_argument("--devices", nargs="+", default=["auto"])
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--topodise-pretrained-from", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if len(set(args.encoders)) != len(args.encoders): parser.error("Duplicate encoder selection")
    if len(set(args.devices)) != len(args.devices) or ("auto" in args.devices and len(args.devices) != 1):
        parser.error("Use distinct explicit devices or a single auto device")
    if any(d not in ("auto", "cpu", "mps") and not (d.startswith("cuda:") and d[5:].isdigit()) for d in args.devices):
        parser.error("Devices must be auto, cpu, mps, or cuda:N")
    root = run_root(args.name, HOME)
    if args.phase == "analyze": return summarize_full(root, read(root / "protocol.json"))
    cfg = load_config(args.config)
    if args.dataset_root: cfg["dataset_root"] = str(args.dataset_root.resolve())
    if cfg["repeats"] < 1 or cfg["bootstrap_samples"] < 1: parser.error("Repeats/bootstrap must be positive")
    data = resolve_codes_path(cfg["dataset_root"])
    # Match the main method's source pools; do not silently fall back to PD SSL.
    spec = cfg["encoders"]["simclr_resnet18"]["pretrain"]
    for stage in spec.get("stages", []):
        if stage["source"] == "corn1_raw": stage["source_root"] = str(data / "CORN-1")
        if stage["source"] == "region_manifest":
            stage.update(manifest=str(data / "unlabeled_dataset_en/unlabeled_dataset_en.json"), dataset_root=str(data))
    jobs = build_comparison(root, cfg, args.encoders)
    if args.topodise_pretrained_from:
        if "topodise" not in args.encoders: parser.error("Pretraining reuse requires topodise")
        from topodise.training.runtime.reuse_pretrain import reuse_parents
        native_jobs = {**jobs, "stage2": [j for j in jobs["stage2"] if j["variant"] == "topodise"]}
        reuse_parents(native_jobs, args.topodise_pretrained_from.resolve(), ["gt_hgrad_dual"], verify=not args.dry_run)
    print(f"Output: {root}\nEncoders: {args.encoders}; full Stage2 tasks: {cfg['tasks']}")
    if args.dry_run: return jobs
    contract = {"version": "encoder-full-ci95-v1", "jobs": jobs, "encoder_settings": cfg,
                "source_sha256": source_hashes()}
    lock(root / "protocol.json", contract)
    for stage in jobs.values():
        for job in stage: lock(root / "configs" / f"{job['id']}.json", job)
    from topodise.training.runtime.audit import audit_inputs, audit_stage2
    if jobs["stage0"]:
        audit = audit_inputs(root, jobs)
        if args.topodise_pretrained_from:
            old = read(args.topodise_pretrained_from.resolve() / "input_audit.json")["input_file_sha256"]
            if any(audit["input_file_sha256"].get(k) != v for k,v in old.items()):
                raise RuntimeError("Pretraining reuse data differs")
    else:
        files = audit_stage2(root, [j["config"] for j in jobs["stage2"]])
        lock(root / "input_audit.json", {"input_file_sha256": {str(Path(p).resolve()): file_hash(p) for p in sorted(files)}})
    # Fail before expensive training if external DINO weights are unavailable or incompatible.
    if "dinov2_vits14" in args.encoders:
        from .encoders.dinov2 import DINOv2ViTS14Frozen
        dino = cfg["encoders"]["dinov2_vits14"]
        model = DINOv2ViTS14Frozen(resolve_codes_path(dino["checkpoint"]), dino["model_name"])
        del model
    if "simclr_resnet18" in args.encoders:
        from .pretrain_simclr import ensure_simclr_checkpoint, collect_pretrain_images
        simclr = cfg["encoders"]["simclr_resnet18"]
        checkpoint = resolve_codes_path(simclr["checkpoint"])
        if checkpoint.is_file():
            ensure_simclr_checkpoint(checkpoint, simclr, args.devices[0])
            from .encoders.simclr import SimCLRResNet18Frozen
            model = SimCLRResNet18Frozen(checkpoint)
            del model
        else:
            if args.phase == "stage2" or not simclr.get("auto_pretrain_if_missing", True):
                raise FileNotFoundError(f"SimCLR pretraining checkpoint required: {checkpoint}")
            for stage in simclr["pretrain"].get("stages", []):
                collect_pretrain_images({**simclr["pretrain"], **stage})
    if args.phase == "prepare": return jobs
    if args.phase in ("all", "pretrain"):
        if jobs["stage0"] and not args.topodise_pretrained_from:
            execute_phase(jobs["stage0"], root, args.devices)
            execute_phase(jobs["stage1"], root, args.devices)
    if "simclr_resnet18" in args.encoders:
        from .pretrain_simclr import ensure_simclr_checkpoint
        simclr = deepcopy(cfg["encoders"]["simclr_resnet18"])
        if args.phase == "stage2": simclr["auto_pretrain_if_missing"] = False
        ensure_simclr_checkpoint(resolve_codes_path(simclr["checkpoint"]), simclr, args.devices[0])
    if args.phase in ("all", "stage2"):
        execute_phase(jobs["stage2"], root, args.devices)
        summarize_full(root, contract)
    return jobs

if __name__ == "__main__":
    main()
