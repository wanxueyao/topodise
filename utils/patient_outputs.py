"""Patient-level prediction aggregation and persistence helpers."""
import csv
import json
import math
import os
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def task_labels_for_indices(dataset, indices: Sequence[int], task: str) -> Tuple[List[int], List[Any]]:
    """Return valid labels and patient ids for ``indices`` in dataloader order."""
    labels: List[int] = []
    patient_ids: List[Any] = []
    for idx in indices:
        raw_idx = int(idx)
        if task == 'pd':
            label = int(dataset.PD_labels[raw_idx])
        elif task == 'stage':
            label = int(dataset.stage_labels[raw_idx])
        elif task == 'subtype':
            label = int(dataset.subtype_labels[raw_idx])
        elif task == 'pd_subtype_tdrest':
            subtype = int(dataset.subtype_labels[raw_idx])
            label = 0 if subtype == 0 else (1 if subtype > 0 else -1)
        elif task == 'pd_subtype_pigd_mixed':
            subtype = int(dataset.subtype_labels[raw_idx])
            label = 0 if subtype == 1 else (1 if subtype == 2 else -1)
        else:
            label = -1
        if label < 0:
            continue
        labels.append(label)
        patient_ids.append(_sample_patient_id(dataset, raw_idx))
    return labels, patient_ids


def save_fold_patient_outputs(
    output_dir: str,
    task: str,
    suffix: str,
    fold: int,
    patient_ids: Sequence[Any],
    labels: Sequence[int],
    probs: Optional[Sequence[Sequence[float]]] = None,
    logits: Optional[Sequence[Sequence[float]]] = None,
) -> Dict[str, Any]:
    """Aggregate image predictions to patient level, save JSON/CSV, and return metrics."""
    os.makedirs(output_dir, exist_ok=True)
    rows = aggregate_patient_outputs(patient_ids, labels, probs=probs, logits=logits)
    safe_suffix = suffix.strip('_')
    prefix = f"{task}_{safe_suffix}" if safe_suffix else task
    json_path = os.path.join(output_dir, f"{prefix}_fold{fold}_patient_logits.json")
    csv_path = os.path.join(output_dir, f"{prefix}_fold{fold}_patient_logits.csv")

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    fields = ['patient_id', 'n_images', 'label', 'pred', 'correct', 'patient_logits', 'patient_probs']
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                'patient_id': row['patient_id'],
                'n_images': row['n_images'],
                'label': row['label'],
                'pred': row['pred'],
                'correct': int(row['correct']),
                'patient_logits': json.dumps(row['patient_logits']),
                'patient_probs': json.dumps(row['patient_probs']),
            })

    patient_acc = float(np.mean([r['correct'] for r in rows])) if rows else 0.0
    return {
        'patient_acc': patient_acc,
        'n_patients': len(rows),
        'patient_logits_json': json_path,
        'patient_logits_csv': csv_path,
        'patient_outputs': rows,
    }


def aggregate_patient_outputs(
    patient_ids: Sequence[Any],
    labels: Sequence[int],
    probs: Optional[Sequence[Sequence[float]]] = None,
    logits: Optional[Sequence[Sequence[float]]] = None,
) -> List[Dict[str, Any]]:
    if probs is None and logits is None:
        raise ValueError('Either probs or logits must be provided')

    prob_arr = _as_2d_array(probs) if probs is not None else None
    logit_arr = _as_2d_array(logits) if logits is not None else None
    if prob_arr is None:
        prob_arr = _softmax(logit_arr)
    if logit_arr is None or logit_arr.shape[1] != prob_arr.shape[1]:
        logit_arr = np.log(np.clip(prob_arr, 1e-12, 1.0))

    if len(patient_ids) != len(labels) or len(labels) != prob_arr.shape[0]:
        raise ValueError(
            f'prediction alignment mismatch: patient_ids={len(patient_ids)}, '
            f'labels={len(labels)}, probs={prob_arr.shape[0]}'
        )
    valid = np.asarray(labels, dtype=int) >= 0
    patient_ids = [pid for pid, keep in zip(patient_ids, valid.tolist()) if keep]
    labels = [int(label) for label, keep in zip(labels, valid.tolist()) if keep]
    prob_arr = prob_arr[valid]
    logit_arr = logit_arr[valid]

    grouped: Dict[str, Dict[str, Any]] = defaultdict(lambda: {'labels': [], 'probs': [], 'logits': []})
    pid_display: Dict[str, Any] = {}
    for pid, label, prob, logit in zip(patient_ids, labels, prob_arr, logit_arr):
        key = str(pid)
        pid_display[key] = pid
        grouped[key]['labels'].append(int(label))
        grouped[key]['probs'].append(np.asarray(prob, dtype=float))
        grouped[key]['logits'].append(np.asarray(logit, dtype=float))

    rows: List[Dict[str, Any]] = []
    for key in sorted(grouped.keys(), key=_natural_key):
        item = grouped[key]
        mean_probs = np.mean(np.stack(item['probs']), axis=0)
        mean_logits = np.mean(np.stack(item['logits']), axis=0)
        label = _majority_label(item['labels'])
        pred = int(np.argmax(mean_probs))
        rows.append({
            'patient_id': pid_display[key],
            'n_images': len(item['labels']),
            'label': label,
            'pred': pred,
            'correct': bool(pred == label),
            'patient_logits': [float(v) for v in mean_logits],
            'patient_probs': [float(v) for v in mean_probs],
        })
    return rows


def _sample_patient_id(dataset, sample_idx: int) -> Any:
    pids = getattr(dataset, 'patient_ids', None)
    if pids is None:
        return sample_idx
    pid = pids[sample_idx]
    patient_records = getattr(dataset, 'patients', None)
    if patient_records is not None:
        try:
            record = patient_records[int(pid)]
            return record.get('patient_id', pid)
        except Exception:
            return pid
    return pid


def _as_2d_array(values: Sequence[Sequence[float]]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f'expected a 2-D prediction array, got shape {arr.shape}')
    return arr


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _majority_label(labels: Iterable[int]) -> int:
    counts: Dict[int, int] = defaultdict(int)
    for label in labels:
        counts[int(label)] += 1
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def _natural_key(value: str):
    try:
        number = float(value)
        if math.isfinite(number):
            return (0, number)
    except ValueError:
        pass
    return (1, value)