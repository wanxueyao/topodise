"""Reuse completed six-route pretraining across a Stage2-only protocol change."""
from pathlib import Path
from .protocol import checkpoint, digest, read, require_complete, file_hash


def reuse_parents(jobs, source, variants, verify=True):
    stage1_jobs = {j["id"]: j for j in jobs["stage1"]}
    for job in jobs["stage2"]:
        name = Path(job["config"]["output_root"]).name
        current = stage1_jobs[job["id"].replace("s2_", "s1_", 1)]
        seed = current["config"]["seed"]
        old_s1 = source / f"pretrain_seed{seed}/stage1" / name
        parent = checkpoint(old_s1)
        if verify and name in variants:
            require_complete(old_s1)
            old = read(old_s1 / "ablation_job.json")["job"]
            def normalized(cfg):
                return {k: v for k, v in cfg.items() if k not in ("parent_ckpt", "output_dir", "run_name", "device")}
            if digest(normalized(old["config"])) != digest(normalized(current["config"])):
                raise RuntimeError(f"Stage1 pretraining recipe differs: {old_s1}")
            target = Path(current["config"]["parent_ckpt"]).parent.parent.name
            expected = next(j for j in jobs["stage0"] if j["config"]["seed"] == seed and j["config"]["topo_target"] == target)
            old_s0 = source / f"pretrain_seed{seed}/stage0" / target
            s0_path = require_complete(old_s0)
            old0 = read(old_s0 / "ablation_job.json")["job"]
            if digest(normalized(old0["config"])) != digest(normalized(expected["config"])):
                raise RuntimeError(f"Stage0 pretraining recipe differs: {old_s0}")
            if read(old_s1 / "ABLATION_DONE.json")["parent_checkpoint_sha256"] != file_hash(s0_path):
                raise RuntimeError(f"Stage1/Stage0 provenance mismatch: {old_s1}")
        if job.get("backend") == "full":
            job["config"]["parent_ckpt"] = str(parent)
            for recipe in job["config"]["recipes"]:
                for candidate in recipe["candidates"]:
                    candidate["config"]["parent_ckpt"] = str(parent)
        else:
            job["config"]["encoders"]["topodise_stage1"]["checkpoint"] = str(parent)
