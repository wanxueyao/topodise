"""Audit the exact common source pools before any training starts."""
from pathlib import Path
from copy import deepcopy

from experiments._stage_configs import Stage0Config, Stage1Config, Stage2Config
from experiments.unified_stage2_encoder_comparison.pretrain_topodise import _audit_inputs
from experiments.unified_stage2_encoder_comparison.cohort import load_cohort, build_audit, validate_audit
from experiments.unified_stage2_encoder_comparison.config import DEFAULTS, _deep_update, validate_config
from experiments.unified_stage2_encoder_comparison.manifests import (
    build_pair_manifest, build_split_manifest, validate_split_manifest,
)
from topodise.data import CORN1Dataset, RegionPairDataset
from topodise.data.corn1_dataset import _derive_label_path
from .protocol import file_hash, lock


def audit_inputs(root, jobs):
    s0 = Stage0Config(**jobs["stage0"][0]["config"])
    s1 = Stage1Config(**jobs["stage1"][0]["config"])
    audit = _audit_inputs(s0, s1)
    if s0.expected_samples is not None and audit["corn1_num_images"] != s0.expected_samples:
        raise ValueError(f"CORN-1 count {audit['corn1_num_images']} != {s0.expected_samples}")
    corn = CORN1Dataset(s0.corn1_root)
    missing = [p for p in corn.paths if _derive_label_path(p) is None]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} GT masks; first: {missing[0]}")
    if len(corn) < s0.batch_size:
        raise ValueError("Stage0 has fewer samples than one full batch")
    regions = RegionPairDataset(manifest_path=s1.region_pair_manifest, base_path=s1.base_path)
    # This family uses the independent unlabeled manifest for ALL six routes.
    # Retain the historical basename exclusion rule; do not silently substitute
    # the 182 PD-control pairs when a manifest is missing.
    if any("unlabeled_dataset_en" not in Path(r[k]).parts
           for r in regions.pairs for k in ("central", "whorl")):
        raise ValueError("This protocol requires Stage1 images from unlabeled_dataset_en only")
    # Catch downstream cohort/config/split errors before expensive pretraining.
    records, exclusions = load_cohort(Path(s1.pd_json), Path(s1.base_path))
    cohort_audit = build_audit(Path(s1.pd_json), records, exclusions)
    validate_audit(cohort_audit)
    audit["stage2_cohort"] = cohort_audit
    if jobs["stage2"][0].get("backend") == "full":
        from experiments.topodise_experiments.stage2_racf_downstream.main import _make_dataset, _build_kfold_splits
        for recipe in jobs["stage2"][0]["config"]["recipes"]:
            for candidate in recipe["candidates"]:
                cfg = Stage2Config(**candidate["config"])
                ds = _make_dataset(cfg, None)
                folds = []
                for fold, (train, val) in enumerate(_build_kfold_splits(ds, cfg), 1):
                    train_ids = sorted({str(ds.samples[i][3]) for i in train})
                    val_ids = sorted({str(ds.samples[i][3]) for i in val})
                    if set(train_ids) & set(val_ids):
                        raise RuntimeError("Full Stage2 patient split overlap")
                    folds.append({"fold": fold, "train_patients": train_ids, "val_patients": val_ids})
                lock(root / "full_stage2_manifests" / f"{candidate['name']}.json",
                     {"task": cfg.task, "samples": ds.samples, "folds": folds,
                      "evaluation": "original validation-selected CV"})
    else:
        for job in jobs["stage2"]:
            core = _deep_update(deepcopy(DEFAULTS), job["config"])
            validate_config(core)
        core = _deep_update(deepcopy(DEFAULTS), jobs["stage2"][0]["config"])
        lock(root / "common_stage2_manifests/pairs.json", build_pair_manifest(
            records, int(core["pairing"]["seed"]), int(core["pairing"]["cap"])))
        for task in core["tasks"]:
            for seed in core["splits"]["seeds"]:
                split = build_split_manifest(records, task, seed, core["splits"]["outer_folds"],
                                             core["splits"]["inner_val_ratio"])
                validate_split_manifest(split)
                lock(root / "common_stage2_manifests" / f"splits_{task}_seed{seed}.json", split)
    files = set(corn.paths) | {_derive_label_path(p) for p in corn.paths}
    files.update(r[k] for r in regions.pairs for k in ("central", "whorl"))
    files.update(str(p) for record in records for p in (*record.central, *record.whorl))
    files.add(s1.pd_json)
    # Content hashes catch changed images or masks even if filenames stay fixed.
    audit["input_file_sha256"] = {str(Path(p).resolve()): file_hash(p) for p in sorted(files)}
    audit["gt_masks"] = len(corn)
    audit["duplicate_filter"] = "historical PD/DM basename exclusion; not a patient-identity audit"
    lock(root / "input_audit.json", audit)
    print(f"[audit] CORN-1={len(corn)}, masks={len(corn)}, "
          f"Stage1={audit['stage1_valid_rows']} "
          f"(paired={audit['stage1_paired_rows']}, unpaired={audit['stage1_unpaired_rows']})", flush=True)
    return audit
