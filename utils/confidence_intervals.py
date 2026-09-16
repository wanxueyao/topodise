"""Patient-cluster bootstrap for repeated out-of-fold predictions.

The point estimate is the mean of complete-repeat OOF metrics (not a mean
of folds or a vote across repeats). Each bootstrap draw resamples patients
within true classes once and applies those same weights to every repeat.
Intervals condition on trained models and do not remove validation-selection
bias or quantify pretraining/training-seed uncertainty.
"""
from __future__ import annotations

import numpy as np

METRICS = ("accuracy", "bacc", "macro_f1", "auc")


def _auc(y, scores, weights):
    order = np.argsort(scores, kind="stable")
    y, scores, weights = y[order], scores[order], weights[order]
    starts = np.r_[0, np.flatnonzero(np.diff(scores)) + 1]
    pos = np.add.reduceat(weights * y, starts)
    neg = np.add.reduceat(weights * (1 - y), starts)
    denominator = pos.sum() * neg.sum()
    if denominator == 0:
        return np.nan
    return float(np.sum(pos * (np.cumsum(neg) - 0.5 * neg)) / denominator)


def scores(y, pred, probs, classes, weights=None, evaluated_labels=None):
    y, pred = np.asarray(y, dtype=int), np.asarray(pred, dtype=int)
    probs = np.asarray(probs, dtype=float)
    w = np.ones(len(y)) if weights is None else np.asarray(weights, dtype=float)
    cm = np.bincount(y * classes + pred, weights=w, minlength=classes**2).reshape(classes, classes)
    support, predicted, tp = cm.sum(1), cm.sum(0), cm.diagonal()
    recall = np.divide(tp, support, out=np.zeros(classes), where=support > 0)
    f1 = np.divide(2 * tp, support + predicted, out=np.zeros(classes), where=(support + predicted) > 0)
    evaluated = np.arange(classes) if evaluated_labels is None else np.asarray(evaluated_labels, dtype=int)
    auc = _auc((y == 1).astype(int), probs[:, 1], w) if classes == 2 else float(np.mean([
        _auc((y == c).astype(int), probs[:, c], w) for c in evaluated]))
    return dict(zip(METRICS, [float(tp.sum() / w.sum()),
                            float(recall[evaluated][support[evaluated] > 0].mean()), float(f1[evaluated].mean()), auc]))


def report(rows, classes, folds_per_repeat=5, bootstrap_samples=2000, seed=20260915, evaluated_labels=None):
    if not rows or bootstrap_samples < 1 or folds_per_repeat < 1:
        raise ValueError("Predictions, positive bootstrap_samples and folds_per_repeat are required")
    if evaluated_labels is not None:
        if not evaluated_labels or len(set(evaluated_labels)) != len(evaluated_labels) or any(c < 0 or c >= classes for c in evaluated_labels):
            raise ValueError("Invalid evaluation labels")
    repeats = {}
    for row in rows:
        repeat = int(row.get("repeat", (int(row["fold"]) - 1) // folds_per_repeat))
        pid = str(row["patient_id"])
        if pid in repeats.setdefault(repeat, {}):
            raise ValueError(f"Patient {pid} has multiple OOF predictions in repeat {repeat}")
        repeats[repeat][pid] = row
    ids = sorted(next(iter(repeats.values())))
    if any(set(r) != set(ids) for r in repeats.values()):
        raise ValueError("Every repeat must cover exactly the same patients")
    arrays = []
    for repeat, by_id in sorted(repeats.items()):
        records = [by_id[pid] for pid in ids]
        y = np.array([r["y_true"] for r in records], dtype=int)
        pred = np.array([r["y_pred"] for r in records], dtype=int)
        probs = np.array([r.get("ensemble_probs", r.get("patient_probs")) for r in records], dtype=float)
        if probs.shape != (len(ids), classes) or not np.isfinite(probs).all():
            raise ValueError("Missing or invalid patient probability vectors; AUC requires probabilities")
        if np.any(probs < 0) or np.any(probs > 1) or not np.allclose(probs.sum(1), 1, atol=1e-5):
            raise ValueError("Patient probabilities must be normalized")
        if np.any(y < 0) or np.any(y >= classes) or np.any(pred < 0) or np.any(pred >= classes):
            raise ValueError("Class labels outside the task definition")
        if evaluated_labels is not None and not set(y).issubset(evaluated_labels):
            raise ValueError("Ground truth outside declared evaluation labels")
        if arrays and not np.array_equal(y, arrays[0][0]):
            raise ValueError("Patient labels changed between repeats")
        arrays.append((y, pred, probs))
    y = arrays[0][0]
    per_repeat = [scores(*a, classes, evaluated_labels=evaluated_labels) for a in arrays]
    rng = np.random.default_rng(seed)
    strata = [np.flatnonzero(y == c) for c in np.unique(y)]
    draws = {key: [] for key in METRICS}
    for _ in range(bootstrap_samples):
        selected = np.concatenate([rng.choice(indices, len(indices), replace=True) for indices in strata])
        weights = np.bincount(selected, minlength=len(ids))
        values = [scores(*a, classes, weights=weights, evaluated_labels=evaluated_labels) for a in arrays]
        for key in METRICS:
            draws[key].append(float(np.mean([v[key] for v in values])))
    metrics = {}
    for key in METRICS:
        estimate = float(np.mean([v[key] for v in per_repeat]))
        valid = np.asarray(draws[key]); valid = valid[np.isfinite(valid)]
        bounds = np.quantile(valid, [0.025, 0.975]).tolist() if len(valid) else [None, None]
        metrics[key] = {"estimate": estimate if np.isfinite(estimate) else None,
                        "ci95_low": bounds[0], "ci95_high": bounds[1],
                        "valid_bootstrap_samples": len(valid)}
    return {"output_classes": classes, "evaluated_labels": list(range(classes)) if evaluated_labels is None else list(evaluated_labels),
            "metrics": metrics, "n_patients": len(ids), "n_repeats": len(arrays),
            "bootstrap_samples": bootstrap_samples, "bootstrap_seed": seed,
            "point_estimate": "mean of repeat-level out-of-fold patient metrics",
            "ci_method": "true-class-stratified patient-cluster percentile bootstrap; same draw across repeats",
            "auc_definition": "class-1 ROC AUC" if classes == 2 else "macro one-vs-rest ROC AUC",
            "limitation": "conditional on trained models; does not correct validation-based epoch selection"}
