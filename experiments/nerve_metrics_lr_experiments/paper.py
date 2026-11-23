"""Evaluate Nerve-LR with the paper's patient folds, C grid and summary statistics."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .main import ROOT, load_features, task_label


def paper_folds(labels, seed):
    """Reproduce class order and within-fold shuffles used by the archived inner CV."""
    rng = np.random.RandomState(seed)
    groups = defaultdict(list)
    for index, label in enumerate(labels):
        groups[label].append(index)
    validation = [[] for _ in range(5)]
    for group in groups.values():
        rng.shuffle(group)
        for i, index in enumerate(group):
            validation[i % 5].append(index)
    for fold in validation:
        rng.shuffle(fold)
    for fold in validation:
        train = np.setdiff1d(np.arange(len(labels)), fold)
        val = np.array(fold)
        rng.shuffle(train)
        rng.shuffle(val)
        yield train, val


def evaluate(rows, features, task):
    patients = [p for p in rows if p['patient_id'] in features and task_label(p, task) is not None]
    if task == 'pd_binary':
        # The archived diagnosis table concatenated controls then PD, in manifest order.
        patients.sort(key=lambda p: task_label(p, task))
    y = np.array([task_label(p, task) for p in patients], dtype=int)
    if min(np.bincount(y, minlength=3 if task == 'pd_stage' else 2)) < 5:
        raise ValueError(f'{task} requires at least five patients in each class')
    x = np.stack([features[p['patient_id']] for p in patients])
    results = []
    for seed in (42, 1042, 2042):
        for fold, (train, val) in enumerate(paper_folds(y, seed), 1):
            pipe = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scale', StandardScaler()),
                ('lr', LogisticRegression(class_weight='balanced', solver='lbfgs',
                                          random_state=42, max_iter=10000)),
            ])
            inner = StratifiedKFold(3, shuffle=True, random_state=seed + fold)
            model = GridSearchCV(pipe, {'lr__C': [0.01, 0.1, 1, 10]},
                                 scoring='f1_macro', cv=inner, error_score='raise')
            model.fit(x[train], y[train])
            prediction = model.predict(x[val])
            results.append(dict(seed=seed, fold=fold, selected_C=model.best_params_['lr__C'],
                                accuracy=accuracy_score(y[val], prediction),
                                macro_f1=f1_score(y[val], prediction, average='macro')))
    summary = {}
    for metric in ('accuracy', 'macro_f1'):
        values = [f[metric] for f in results]
        summary[metric] = dict(mean=float(np.mean(values)),
                               ci95_half_width=float(1.96 * np.std(values, ddof=1) / np.sqrt(len(values))))
    return dict(n_patients=len(y), folds=results, summary=summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, default=ROOT / 'datasets')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/nerve_metrics_lr/paper.json')
    args = parser.parse_args()
    data = args.dataset_root / 'pd_dataset_en'
    records = json.loads((data / 'patients_dataset_en.json').read_text())
    features, _ = load_features([data / 'pd_patient_feature_measurements.csv',
                                 data / 'non_pd_patient_feature_measurements.csv'])
    ids = {r['patient_id'] for r in records}
    if len(ids) != len(records) or features.keys() - ids:
        raise ValueError('Measurement IDs must match the unique patients in the manifest')
    results = {task: evaluate(records, features, task) for task in ('pd_binary', 'pd_stage', 'pd_subtype')}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
