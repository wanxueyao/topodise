from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .config import load_config, resolved_paths
from .metrics import classification_metrics
from .utils import atomic_json_dump


def _read_predictions(root: Path) -> list[dict]:
    rows = []
    for path in sorted((root / "runs").glob("*/*/seed*/fold*/predictions.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    keys = [(r["encoder"], r["task"], r["split_seed"], r["patient_id"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError("duplicate OOF patient predictions detected; clean/inspect repeated fold outputs")
    protocol_hashes = {row.get("protocol_config_sha256") for row in rows}
    if None in protocol_hashes or len(protocol_hashes) != 1:
        raise RuntimeError(
            "predictions contain missing or mixed protocol hashes; do not combine pre-optimization "
            "and runs produced under a different protocol"
        )
    return rows


def _metric(rows: list[dict]) -> dict:
    return classification_metrics(
        [int(row["label"]) for row in rows], [row["probabilities"] for row in rows]
    )


def _cluster_bootstrap_metric_ci(rows: list[dict], n_boot: int, seed: int) -> dict:
    """Stratified patient bootstrap, retaining all split-seed predictions per patient."""
    by_patient = defaultdict(list)
    for row in rows:
        by_patient[str(row["patient_id"])].append(row)
    labels = {}
    by_class = defaultdict(list)
    for patient_id, patient_rows in by_patient.items():
        patient_labels = {int(row["label"]) for row in patient_rows}
        if len(patient_labels) != 1:
            raise RuntimeError(f"inconsistent labels for patient {patient_id}")
        label = patient_labels.pop()
        labels[patient_id] = label
        by_class[label].append(patient_id)

    rng = np.random.default_rng(seed)
    accuracy_samples = np.empty(n_boot, dtype=float)
    f1_samples = np.empty(n_boot, dtype=float)
    for index in range(n_boot):
        sampled_rows = []
        for patient_ids in by_class.values():
            sampled_ids = rng.choice(patient_ids, size=len(patient_ids), replace=True)
            for patient_id in sampled_ids:
                sampled_rows.extend(by_patient[str(patient_id)])
        metric = _metric(sampled_rows)
        accuracy_samples[index] = metric["accuracy"]
        f1_samples[index] = metric["macro_f1"]

    accuracy_ci = np.quantile(accuracy_samples, [0.025, 0.975])
    f1_ci = np.quantile(f1_samples, [0.025, 0.975])
    return {
        "accuracy_ci95": [float(accuracy_ci[0]), float(accuracy_ci[1])],
        "macro_f1_ci95": [float(f1_ci[0]), float(f1_ci[1])],
        "ci_method": "stratified_patient_cluster_bootstrap_with_all_split_seeds",
        "n_bootstrap": int(n_boot),
    }


def _holm_adjust(items: list[dict]) -> None:
    order = sorted(range(len(items)), key=lambda i: items[i]["p_value"])
    running = 0.0
    m = len(items)
    for rank, index in enumerate(order):
        adjusted = min(1.0, (m - rank) * items[index]["p_value"])
        running = max(running, adjusted)
        items[index]["p_holm"] = running


def _paired_cluster_bootstrap(
    rows_a: list[dict], rows_b: list[dict], n_boot: int, seed: int
) -> dict:
    def key(row: dict) -> tuple[int, str]:
        return int(row["split_seed"]), row["patient_id"]

    a, b = {key(r): r for r in rows_a}, {key(r): r for r in rows_b}
    if set(a) != set(b):
        raise RuntimeError("paired comparison has non-aligned seed/patient predictions")
    patient_ids = sorted({pid for _, pid in a})
    labels = {pid: int(next(row["label"] for k, row in a.items() if k[1] == pid)) for pid in patient_ids}
    by_class = defaultdict(list)
    for pid in patient_ids:
        by_class[labels[pid]].append(pid)
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(n_boot):
        sampled = []
        for pids in by_class.values():
            sampled.extend(rng.choice(pids, size=len(pids), replace=True).tolist())
        sample_a, sample_b = [], []
        for pid in sampled:
            seed_values = sorted(s for s, p in a if p == pid)
            sample_a.extend(a[(s, pid)] for s in seed_values)
            sample_b.extend(b[(s, pid)] for s in seed_values)
        differences.append(_metric(sample_a)["macro_f1"] - _metric(sample_b)["macro_f1"])
    diff = np.asarray(differences)
    observed = _metric(rows_a)["macro_f1"] - _metric(rows_b)["macro_f1"]
    lower = (int((diff <= 0).sum()) + 1) / (n_boot + 1)
    upper = (int((diff >= 0).sum()) + 1) / (n_boot + 1)
    p_value = min(1.0, 2.0 * min(lower, upper))
    return {
        "macro_f1_difference_a_minus_b": observed,
        "ci95": [float(np.quantile(diff, 0.025)), float(np.quantile(diff, 0.975))],
        "p_value": p_value,
        "bootstrap_unit": "patient_cluster_with_all_split_seeds",
        "n_bootstrap": n_boot,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate OOF metrics and paired patient bootstrap tests")
    parser.add_argument("--config", required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()
    cfg = load_config(args.config)
    output_root = resolved_paths(cfg)["output_root"]
    rows = _read_predictions(output_root)
    if not rows:
        raise RuntimeError(f"no predictions found under {output_root / 'runs'}")
    by_seed = defaultdict(list)
    by_method_task = defaultdict(list)
    for row in rows:
        by_seed[(row["encoder"], row["task"], int(row["split_seed"]))].append(row)
        by_method_task[(row["encoder"], row["task"])].append(row)
    summary_rows = []
    grouped_metrics = defaultdict(list)
    for (encoder, task, seed), group in sorted(by_seed.items()):
        metric = _metric(group)
        grouped_metrics[(encoder, task)].append(metric)
        summary_rows.append({"encoder": encoder, "task": task, "split_seed": seed, **metric})
    aggregate = []
    for (encoder, task), metrics in sorted(grouped_metrics.items()):
        accuracy_mean = float(np.mean([m["accuracy"] for m in metrics]))
        macro_f1_mean = float(np.mean([m["macro_f1"] for m in metrics]))
        ci = _cluster_bootstrap_metric_ci(
            by_method_task[(encoder, task)], args.bootstrap, args.seed,
        )
        aggregate.append({
            "encoder": encoder,
            "task": task,
            "num_split_seeds": len(metrics),
            "macro_f1_mean": macro_f1_mean,
            "macro_f1_std": float(np.std([m["macro_f1"] for m in metrics], ddof=1)) if len(metrics) > 1 else 0.0,
            "balanced_accuracy_mean": float(np.mean([m["balanced_accuracy"] for m in metrics])),
            "accuracy_mean": accuracy_mean,
            "accuracy_ci95_half_width": max(
                accuracy_mean - ci["accuracy_ci95"][0],
                ci["accuracy_ci95"][1] - accuracy_mean,
            ),
            "macro_f1_ci95_half_width": max(
                macro_f1_mean - ci["macro_f1_ci95"][0],
                ci["macro_f1_ci95"][1] - macro_f1_mean,
            ),
            **ci,
        })
    analysis_dir = output_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    with (analysis_dir / "metrics_by_seed.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["encoder", "task", "split_seed", "n", "accuracy", "balanced_accuracy", "macro_f1", "macro_auc_ovr"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary_rows)
    with (analysis_dir / "aggregate.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)
    comparisons = []
    for task in cfg["tasks"]:
        available = sorted(encoder for encoder, current_task in by_method_task if current_task == task)
        for encoder_a, encoder_b in itertools.combinations(available, 2):
            result = _paired_cluster_bootstrap(
                by_method_task[(encoder_a, task)], by_method_task[(encoder_b, task)],
                args.bootstrap, args.seed,
            )
            comparisons.append({"task": task, "encoder_a": encoder_a, "encoder_b": encoder_b, **result})
    _holm_adjust(comparisons)
    atomic_json_dump({"aggregate": aggregate, "paired_comparisons": comparisons}, analysis_dir / "analysis.json")
    print(f"wrote analysis to {analysis_dir}")


if __name__ == "__main__":
    main()
