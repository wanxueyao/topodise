from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

from .cohort import PatientRecord, task_records
from .utils import atomic_json_dump, canonical_hash


def _stable_path_id(path: Path) -> str:
    parts = path.parts
    if "pd_dataset_en" in parts:
        return "/".join(parts[parts.index("pd_dataset_en"):])
    return "/".join(parts[-4:])


def _pair_score(pair_seed: int, patient_id: str, central: Path, whorl: Path) -> str:
    raw = (
        f"{pair_seed}|{patient_id}|{_stable_path_id(central)}|{_stable_path_id(whorl)}"
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def build_pair_manifest(
    records: list[PatientRecord], pair_seed: int = 20260831, cap: int = 8
) -> dict:
    patients = []
    for record in sorted(records, key=lambda x: x.patient_id):
        all_pairs = [(c, w) for c in record.central for w in record.whorl]
        ranked = sorted(all_pairs, key=lambda pair: _pair_score(pair_seed, record.patient_id, *pair))
        selected = ranked[:cap]
        patients.append({
            "patient_id": record.patient_id,
            "available_pairs": len(all_pairs),
            "selected_pairs": [
                {
                    "central": str(c),
                    "whorl": str(w),
                    "central_id": _stable_path_id(c),
                    "whorl_id": _stable_path_id(w),
                    "rank_sha256": _pair_score(pair_seed, record.patient_id, c, w),
                }
                for c, w in selected
            ],
        })
    body = {"pairing": "cartesian_sha256_topk", "pair_seed": pair_seed, "cap": cap, "patients": patients}
    stable_contract = {
        "pairing": body["pairing"], "pair_seed": pair_seed, "cap": cap,
        "patients": [
            {
                "patient_id": patient["patient_id"],
                "available_pairs": patient["available_pairs"],
                "selected_pairs": [
                    {key: pair[key] for key in ("central_id", "whorl_id", "rank_sha256")}
                    for pair in patient["selected_pairs"]
                ],
            }
            for patient in patients
        ],
    }
    body["manifest_sha256"] = canonical_hash(stable_contract)
    return body


def build_split_manifest(
    records: list[PatientRecord],
    task: str,
    seed: int,
    outer_folds: int = 5,
    inner_val_ratio: float = 0.2,
) -> dict:
    eligible = sorted(task_records(records, task), key=lambda x: x.patient_id)
    patient_ids = np.array([p.patient_id for p in eligible])
    labels = np.array([int(p.label(task)) for p in eligible])
    splitter = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=seed)
    folds = []
    for fold, (outer_train_idx, outer_test_idx) in enumerate(splitter.split(patient_ids, labels)):
        outer_train_ids = patient_ids[outer_train_idx]
        outer_train_labels = labels[outer_train_idx]
        inner_seed = seed * 100 + fold
        inner_train_ids, inner_val_ids = train_test_split(
            outer_train_ids,
            test_size=inner_val_ratio,
            random_state=inner_seed,
            stratify=outer_train_labels,
        )
        folds.append({
            "fold": fold,
            "inner_seed": inner_seed,
            "outer_train": sorted(outer_train_ids.tolist()),
            "inner_train": sorted(inner_train_ids.tolist()),
            "inner_val": sorted(inner_val_ids.tolist()),
            "outer_test": sorted(patient_ids[outer_test_idx].tolist()),
        })
    body = {
        "task": task,
        "seed": seed,
        "outer_folds": outer_folds,
        "inner_val_ratio": inner_val_ratio,
        "patient_labels": {p.patient_id: int(p.label(task)) for p in eligible},
        "folds": folds,
    }
    body["manifest_sha256"] = canonical_hash(body)
    return body


def validate_split_manifest(manifest: dict) -> None:
    labels = manifest["patient_labels"]
    all_ids = set(labels)
    test_seen: list[str] = []
    for fold in manifest["folds"]:
        outer_train = set(fold["outer_train"])
        inner_train = set(fold["inner_train"])
        inner_val = set(fold["inner_val"])
        outer_test = set(fold["outer_test"])
        if outer_train & outer_test:
            raise AssertionError("outer train/test patient leakage")
        if inner_train & inner_val:
            raise AssertionError("inner train/validation patient leakage")
        if inner_train | inner_val != outer_train:
            raise AssertionError("inner split does not partition outer train")
        if outer_train | outer_test != all_ids:
            raise AssertionError("outer split does not cover the task cohort")
        test_seen.extend(outer_test)
    if sorted(test_seen) != sorted(all_ids):
        raise AssertionError("each patient must appear in outer test exactly once")


def materialize_manifests(records: list[PatientRecord], cfg: dict, root: Path) -> dict:
    manifest_dir = root / "protocol" / "manifests"
    pair = build_pair_manifest(records, int(cfg["pairing"]["seed"]), int(cfg["pairing"]["cap"]))
    atomic_json_dump(pair, manifest_dir / "pairs.json")
    splits = {}
    for task in cfg["tasks"]:
        for seed in cfg["splits"]["seeds"]:
            split = build_split_manifest(
                records, task, int(seed), int(cfg["splits"]["outer_folds"]),
                float(cfg["splits"]["inner_val_ratio"]),
            )
            validate_split_manifest(split)
            atomic_json_dump(split, manifest_dir / f"splits_{task}_seed{seed}.json")
            splits[(task, int(seed))] = split
    return {"pair": pair, "splits": splits}
