"""Patient-level 11 nerve metrics + LR; preprocessing is fitted inside each fold."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from utils.patient_outputs import save_fold_patient_outputs
from utils.experiment_reporting import with_patient_report

ROOT = Path(__file__).resolve().parents[2]
FEATURES = ('cnfd', 'cnbd', 'cnfl', 'ctbd', 'cnfa', 'cnfw', 'cnfractal',
            'iwl', 'iwfa', 'iwfw', 'iwfrd')


def task_label(patient, task):
    group = patient['group']
    if task == 'pd_binary':
        return 1 if group == 'pd' else (0 if group in ('control', 'control_cataract') else None)
    if task == 'pd_stage':
        return int(patient['stage']) - 1 if group == 'pd' and patient.get('stage') in (1, 2, 3) else None
    if task == 'pd_subtype':
        return (0 if patient['subtype'] == 'TD' else 1) if group == 'pd' and patient.get('subtype') in ('TD', 'PIGD', 'mixed') else None
    if task == 'dm_binary':
        return 0 if group == 'control' else (1 if group in ('NDR', 'NPDR') else None)
    raise ValueError(f'Unknown task: {task}')


def load_features(files):
    """Average repeated measurements within each known patient, without labels."""
    grouped, sources = {}, []
    for file in files:
        path = Path(file).expanduser().resolve()
        sources.append({'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        with path.open(newline='', encoding='utf-8-sig') as stream:
            reader = csv.DictReader(stream)
            names = {name.lower(): name for name in reader.fieldnames or []}
            if 'patient_id' not in names:
                raise ValueError(f'{path}: requires explicit patient_id; names or row order are not identity mappings')
            cols = [names.get(f, names.get(f + '_mean')) for f in FEATURES]
            if any(c is None for c in cols):
                raise ValueError(f'{path}: requires all 11 measurements: {FEATURES}')
            for row in reader:
                pid = row[names['patient_id']].strip()
                if not pid:
                    raise ValueError(f'{path}: unaligned measurement; patient_id is empty')
                values = np.array([float(row[c]) if row[c].strip() else np.nan for c in cols])
                if np.isinf(values).any():
                    raise ValueError(f'{path}: infinite measurement')
                grouped.setdefault(pid, []).append(values)
    averaged = {}
    for pid, rows in grouped.items():
        values = np.stack(rows)
        count = np.isfinite(values).sum(0)
        averaged[pid] = np.divide(np.nansum(values, axis=0), count,
                                 out=np.full(len(FEATURES), np.nan), where=count > 0)
    return averaged, sources


@with_patient_report
def run_paper_experiment(exp, output_dir, device='cpu', num_workers=0):
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    dataset = Path(exp.get('dataset_root', ROOT / 'datasets')).expanduser().resolve()
    tasks = exp['_run_tasks']
    if ('dm_binary' in tasks) and len(tasks) != 1:
        raise ValueError('DM uses a separate clinical table and recipe')
    default_json = dataset / ('dm_dataset_en/dm_patients_dataset_en.json' if tasks == ['dm_binary'] else 'pd_dataset_en/patients_dataset_en.json')
    json_path = Path(exp.get('patient_json', default_json)).expanduser().resolve()
    raw = json.loads(json_path.read_text(encoding='utf-8'))
    patients = {str(p['patient_id']): p for p in raw}
    if len(patients) != len(raw):
        raise ValueError('Duplicate patient_id in clinical manifest')
    files = exp.get('feature_csvs')
    if not files:
        if tasks == ['dm_binary']:
            raise ValueError('DM nerve measurements are not provided; set feature_csvs to an aligned DM table')
        files = [dataset / 'pd_dataset_en/pd_patient_feature_measurements.csv',
                 dataset / 'pd_dataset_en/non_pd_patient_feature_measurements.csv']
    features, sources = load_features(files)
    unknown = set(features) - set(patients)
    if unknown:
        raise ValueError(f'{len(unknown)} measurement patient IDs absent from the clinical manifest')
    n_folds = int(exp.get('num_folds', 5)); repeats = int(exp.get('kfold_repeats', 3))
    seed = int(exp.get('seed', 42)); results = {}; audit = {'features': list(FEATURES),
        'measurement_sources': sources, 'patient_manifest_sha256': hashlib.sha256(json_path.read_bytes()).hexdigest(),
        'preprocessing': 'within-patient mean, then train-fold median imputation and train-fold standardization', 'tasks': {}}
    for task in tasks:
        eligible = {pid: task_label(p, task) for pid, p in patients.items() if task_label(p, task) is not None}
        missing = sorted(set(eligible) - set(features))
        if missing and exp.get('require_complete_cohort', False):
            raise ValueError(f'{task}: {len(missing)} eligible patients lack nerve measurements')
        ids = sorted(set(eligible) & set(features)); y = np.array([eligible[pid] for pid in ids], dtype=int)
        x = np.stack([features[pid] for pid in ids]) if ids else np.empty((0, 11))
        classes = 3 if task == 'pd_stage' else 2
        counts = np.bincount(y, minlength=classes)
        if len(counts) != classes or counts.min() < n_folds:
            raise ValueError(f'{task}: each class requires at least {n_folds} patients, got {counts.tolist()}')
        audit['tasks'][task] = {'n_eligible': len(eligible), 'n_included': len(ids),
            'class_counts': counts.tolist(), 'excluded_missing_measurements': missing,
            'cohort_complete': not missing, 'folds': []}
        folds = []
        for repeat in range(repeats):
            cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed + 1000 * repeat)
            for fold, (train, val) in enumerate(cv.split(x, y), 1):
                if np.isnan(x[train]).all(0).any():
                    raise ValueError('A training fold has an entirely missing feature column')
                model = Pipeline([('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler()), ('lr', LogisticRegression(C=float(exp.get('C', 1.0)),
                    class_weight=exp.get('class_weight', 'balanced'), solver='lbfgs',
                    max_iter=int(exp.get('max_iter', 2000)), random_state=seed + 1000 * repeat))])
                model.fit(x[train], y[train]); probs = model.predict_proba(x[val])
                if not np.array_equal(model.classes_, np.arange(classes)):
                    raise RuntimeError('Unexpected classifier class order')
                selected = save_fold_patient_outputs(str(output / task / f'repeat{repeat}'), task, '', fold,
                    [ids[i] for i in val], y[val].tolist(), probs=probs)
                folds.append({'repeat': repeat, 'fold': fold, 'patient_logits_json': selected['patient_logits_json']})
                audit['tasks'][task]['folds'].append({'repeat': repeat, 'fold': fold,
                    'train_ids': [ids[i] for i in train], 'validation_ids': [ids[i] for i in val]})
        results[task] = {'fold_results': folds}
    (output / 'cohort_audit.json').write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding='utf-8')
    return results


if __name__ == '__main__':
    from experiments._execution import single_main
    single_main('experiments.nerve_metrics_lr_experiments.main')
