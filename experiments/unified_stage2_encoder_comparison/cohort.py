from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .utils import atomic_json_dump, sha256_file


TASK_CLASSES = {
    "pd_binary": ["control", "pd"],
    "pd_stage": ["stage_1", "stage_2", "stage_3"],
    "pd_subtype_tdrest": ["TD", "non_TD"],
}

EXPECTED_AUDIT = {
    "json_patients": 114,
    "paired_eligible": 110,
    "excluded_patient_ids": ["C0017", "C0018", "C0019", "C0020"],
    "task_counts": {
        "pd_binary": {"0": 47, "1": 63},
        "pd_stage": {"0": 22, "1": 30, "2": 11},
        "pd_subtype_tdrest": {"0": 23, "1": 40},
    },
}


@dataclass(frozen=True)
class PatientRecord:
    patient_id: str
    group: str
    stage: int
    subtype: str | None
    central: tuple[Path, ...]
    whorl: tuple[Path, ...]

    def label(self, task: str) -> int | None:
        if task == "pd_binary":
            if self.group in ("control", "control_cataract"):
                return 0
            if self.group == "pd":
                return 1
            return None
        if task == "pd_stage":
            return self.stage - 1 if self.group == "pd" and self.stage in (1, 2, 3) else None
        if task == "pd_subtype_tdrest":
            if self.group != "pd":
                return None
            if self.subtype == "TD":
                return 0
            if self.subtype in ("PIGD", "mixed"):
                return 1
            return None
        raise KeyError(task)


def load_cohort(json_path: Path, dataset_root: Path) -> tuple[list[PatientRecord], list[dict]]:
    with json_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    records: list[PatientRecord] = []
    exclusions: list[dict] = []
    for item in raw:
        paths = item.get("image_paths") or {}
        central = tuple((dataset_root / p).resolve() for p in paths.get("central", []))
        whorl = tuple((dataset_root / p).resolve() for p in paths.get("whorl", []))
        missing_c = [str(p) for p in central if not p.is_file()]
        missing_w = [str(p) for p in whorl if not p.is_file()]
        reasons = []
        if not central:
            reasons.append("no_central_paths")
        if not whorl:
            reasons.append("no_whorl_paths")
        if missing_c:
            reasons.append("missing_central_files")
        if missing_w:
            reasons.append("missing_whorl_files")
        if reasons:
            exclusions.append({
                "patient_id": str(item["patient_id"]),
                "reasons": reasons,
                "missing_central": missing_c,
                "missing_whorl": missing_w,
            })
            continue
        records.append(PatientRecord(
            patient_id=str(item["patient_id"]),
            group=str(item["group"]),
            stage=int(item.get("stage") or 0),
            subtype=item.get("subtype"),
            central=central,
            whorl=whorl,
        ))
    return records, exclusions


def task_records(records: Iterable[PatientRecord], task: str) -> list[PatientRecord]:
    return [record for record in records if record.label(task) is not None]


def build_audit(
    json_path: Path,
    records: list[PatientRecord],
    exclusions: list[dict],
) -> dict:
    with json_path.open("r", encoding="utf-8") as handle:
        json_count = len(json.load(handle))
    task_counts = {}
    for task in TASK_CLASSES:
        counts: dict[str, int] = {}
        for record in task_records(records, task):
            key = str(record.label(task))
            counts[key] = counts.get(key, 0) + 1
        task_counts[task] = counts
    return {
        "dataset_json": str(json_path),
        "dataset_json_sha256": sha256_file(json_path),
        "json_patients": json_count,
        "paired_eligible": len(records),
        "excluded_patient_ids": sorted(x["patient_id"] for x in exclusions),
        "exclusions": exclusions,
        "task_counts": task_counts,
    }


def validate_audit(audit: dict, accept_cohort_change: bool = False) -> None:
    observed = {key: audit[key] for key in EXPECTED_AUDIT}
    if observed != EXPECTED_AUDIT and not accept_cohort_change:
        raise RuntimeError(
            "cohort differs from the locked protocol; inspect cohort_audit.json and "
            "rerun with --accept-cohort-change only after deliberate review\n"
            f"expected={EXPECTED_AUDIT}\nobserved={observed}"
        )


def audit_and_save(
    json_path: Path,
    dataset_root: Path,
    output_path: Path,
    accept_cohort_change: bool = False,
) -> tuple[list[PatientRecord], dict]:
    records, exclusions = load_cohort(json_path, dataset_root)
    audit = build_audit(json_path, records, exclusions)
    atomic_json_dump(audit, output_path)
    validate_audit(audit, accept_cohort_change=accept_cohort_change)
    return records, audit
