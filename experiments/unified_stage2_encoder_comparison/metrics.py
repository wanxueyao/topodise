from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def classification_metrics(y_true: list[int], probabilities: list[list[float]]) -> dict:
    y = np.asarray(y_true, dtype=int)
    probs = np.asarray(probabilities, dtype=float)
    pred = probs.argmax(axis=1)
    classes = list(range(probs.shape[1]))
    cm = confusion_matrix(y, pred, labels=classes)
    sensitivity, specificity = [], []
    for index in classes:
        tp = cm[index, index]
        fn = cm[index].sum() - tp
        fp = cm[:, index].sum() - tp
        tn = cm.sum() - tp - fn - fp
        sensitivity.append(float(tp / max(tp + fn, 1)))
        specificity.append(float(tn / max(tn + fp, 1)))
    try:
        if probs.shape[1] == 2:
            auc = float(roc_auc_score(y, probs[:, 1]))
        else:
            auc = float(roc_auc_score(y, probs, multi_class="ovr", average="macro"))
    except ValueError:
        auc = None
    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=classes, average="macro", zero_division=0)),
        "macro_auc_ovr": auc,
        "sensitivity_per_class": sensitivity,
        "specificity_per_class": specificity,
        "confusion_matrix": cm.tolist(),
    }
