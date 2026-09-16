"""Audit actual pretraining images and full Stage2 patient splits before training."""
from pathlib import Path
from experiments._stage_configs import Stage0Config, Stage1Config, Stage2Config
from topodise.data import CORN1Dataset, RegionPairDataset
from topodise.data.corn1_dataset import _derive_label_path
from topodise.data.region_pair_dataset import assert_no_pd_leak
from .protocol import file_hash, lock


def audit_stage2(root, configs):
    from topodise.training.stage2 import _make_dataset, _build_kfold_splits
    import json
    files, manifests = set(), {}
    for bundle in configs:
        for recipe in bundle["recipes"]:
            for candidate in recipe["candidates"]:
                cfg = Stage2Config(**candidate["config"])
                # Validate declared paths too: the original DM loader drops missing
                # images, which must not silently change this comparison cohort.
                for patient in json.loads(Path(cfg.pd_json).read_text()):
                    image_paths = patient.get("image_paths", [])
                    groups = image_paths.values() if isinstance(image_paths, dict) else [image_paths]
                    for paths in groups:
                        files.update(str((Path(cfg.base_path) / p).resolve()) for p in paths)
                ds = _make_dataset(cfg, None)
                folds = []
                for fold, (train, val) in enumerate(_build_kfold_splits(ds, cfg), 1):
                    train_ids = sorted({str(ds.samples[i][3]) for i in train})
                    val_ids = sorted({str(ds.samples[i][3]) for i in val})
                    if set(train_ids) & set(val_ids):
                        raise RuntimeError("Stage2 train/validation patients overlap")
                    folds.append({"fold": fold, "train_patients": train_ids, "val_patients": val_ids})
                manifest = {"task": cfg.task, "samples": ds.samples, "folds": folds,
                            "evaluation": "validation-selected repeated patient CV"}
                key = candidate["name"]
                if key in manifests and manifests[key] != manifest:
                    raise RuntimeError(f"Ablation patient splits differ: {key}")
                manifests[key] = manifest
                lock(root / "full_stage2_manifests" / f"{key}.json", manifest)
                files.add(cfg.pd_json)
                for sample in ds.samples:
                    files.update(sample[:2])
    missing = [p for p in files if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Stage2 images/manifests: {missing[:5]}")
    return files


def audit_inputs(root, jobs):
    s0 = Stage0Config(**jobs["stage0"][0]["config"])
    s1 = Stage1Config(**jobs["stage1"][0]["config"])
    corn = CORN1Dataset(s0.corn1_root)
    if s0.expected_samples is not None and len(corn) != s0.expected_samples:
        raise ValueError(f"CORN-1 count {len(corn)} != {s0.expected_samples}")
    masks = [_derive_label_path(p) for p in corn.paths]
    if any(p is None for p in masks):
        raise FileNotFoundError("CORN-1 GT masks are incomplete")
    if len(corn) < s0.batch_size:
        raise ValueError("Stage0 has fewer samples than a complete batch")
    regions = RegionPairDataset(manifest_path=s1.region_pair_manifest, base_path=s1.base_path,
                                strict_whitelist=True, allow_dm_disease=False)
    assert_no_pd_leak(regions)
    if any("unlabeled_dataset_en" not in Path(r[k]).parts for r in regions.pairs for k in ("central", "whorl")):
        raise ValueError("Stage1 must use the independent unlabeled manifest")
    paired = sum(bool(r["is_paired"]) for r in regions.pairs)
    if paired < s1.batch_size:
        raise ValueError("Too few paired Stage1 rows for region contrastive loss")
    files = set(corn.paths) | set(masks) | {s1.pd_json, s1.region_pair_manifest}
    files.update(r[k] for r in regions.pairs for k in ("central", "whorl"))
    files.update(audit_stage2(root, [j["config"] for j in jobs["stage2"]]))
    # Audit all PD images even when only a PD-positive task is selected; this
    # keeps pretraining reuse tied to the same original cohort manifest.
    import json
    for patient in json.loads(Path(s1.pd_json).read_text()):
        for paths in patient.get("image_paths", {}).values():
            files.update(str((Path(s1.base_path) / p).resolve()) for p in paths)
    audit = {"corn1_num_images": len(corn), "gt_masks": len(masks),
             "stage1_valid_rows": len(regions), "stage1_paired_rows": paired,
             "stage1_unpaired_rows": len(regions) - paired,
             "stage1_skipped_duplicate_filenames": regions.skipped_duplicate_filenames,
             "duplicate_filter": "historical PD/DM basename exclusion; not a patient-identity audit",
             "input_file_sha256": {str(Path(p).resolve()): file_hash(p) for p in sorted(files)}}
    lock(root / "input_audit.json", audit)
    print(f"[audit] CORN-1={len(corn)}, masks={len(masks)}, Stage1={len(regions)} (paired={paired})", flush=True)
    return audit
