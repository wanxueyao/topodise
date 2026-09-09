"""Baseline task names and patient-level TD-vs-rest binary evaluation."""
import csv
import json
from pathlib import Path


def resolve_baseline_tasks(exp):
    """Translate public task names at the training boundary without mutating the recipe."""
    result = dict(exp)
    aliases = {'pd_binary': 'pd', 'pd_stage': 'stage', 'pd_subtype': 'pd_subtype_tdrest',
               'dm_binary': 'dm_binary'}
    tasks = exp.get('_run_tasks', [])
    unknown = set(tasks) - aliases.keys()
    if unknown:
        raise ValueError(f'Unknown baseline tasks: {sorted(unknown)}')
    result['_run_tasks'] = [aliases[t] for t in tasks]
    if exp.get('_method') == 'unet_dual':
        result['tasks'] = result['_run_tasks']
    if 'pd_subtype' in tasks:
        if exp.get('_subtype_protocol') != 'td_rest' or tasks != ['pd_subtype']:
            raise ValueError('pd_subtype requires a dedicated td_rest recipe')
        result['_subtype_td_rest'] = True
    return result


def td_rest_rows(patients, patient_ids, predictions, probabilities, repeat, fold):
    rows = []
    for pid in patient_ids:
        subtype = patients[pid]['subtype']
        if subtype not in ('TD', 'PIGD', 'mixed'):
            raise ValueError(f'Unknown subtype: {subtype}')
        # Missing predictions must fail, never silently substitute Rest.
        pred, prob = int(predictions[pid]), float(probabilities[pid])
        if pred not in (0, 1) or not 0 <= prob <= 1:
            raise ValueError(f'Invalid TD-rest prediction for {pid}')
        rows.append({'repeat': repeat, 'fold': fold, 'patient_id': pid,
                     'source_subtype': subtype, 'label': 0 if subtype == 'TD' else 1,
                     'prediction': pred, 'prob_rest': prob})
    return rows


def td_rest_fold(rows):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
    if not rows:
        raise ValueError('No patients in TD-rest validation fold')
    labels = [r['label'] for r in rows]
    preds = [r['prediction'] for r in rows]
    return {'repeat': rows[0]['repeat'], 'fold': rows[0]['fold'], 'n_patients': len(rows),
            'acc': float(accuracy_score(labels, preds)),
            'balanced_acc': float(balanced_accuracy_score(labels, preds)),
            'f1': float(f1_score(labels, preds, average='macro', labels=[0, 1], zero_division=0)),
            'auc': float(roc_auc_score(labels, [r['prob_rest'] for r in rows])) if len(set(labels)) == 2 else None,
            'confusion_matrix': confusion_matrix(labels, preds, labels=[0, 1]).tolist()}


def save_td_rest_results(output_dir, folds, rows, n_splits, repeats, seed, prefix='pd_subtype'):
    import numpy as np
    if len(folds) != n_splits * repeats:
        raise ValueError('Incomplete TD-rest cross-validation')
    summary = {'task': 'pd_subtype', 'subtype_protocol': 'td_rest',
               'class_names': ['TD', 'Rest (PIGD + mixed)'], 'positive_class': 1,
               'num_folds': n_splits, 'repeats': repeats, 'seed': seed, 'fold_results': folds}
    for metric in ('acc', 'balanced_acc', 'f1', 'auc'):
        values = [f[metric] for f in folds if f[metric] is not None]
        summary[metric + '_mean'] = float(np.mean(values)) if values else None
        summary[metric + '_std'] = float(np.std(values, ddof=1)) if len(values) > 1 else (0.0 if values else None)
    summary['final_results'] = {'avg_acc': summary['acc_mean'], 'std_acc': summary['acc_std'],
                                'avg_f1': summary['f1_mean'], 'std_f1': summary['f1_std'],
                                'auc_scores': {'TD_rest': summary['auc_mean']} if summary['auc_mean'] is not None else {}}
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / f'{prefix}_predictions.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['repeat','fold','patient_id','source_subtype','label','prediction','prob_rest'])
        writer.writeheader()
        writer.writerows(rows)
    summary['prediction_csv'] = str(csv_path)
    (output / f'{prefix}_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"pd_subtype (TD-rest): acc={summary['acc_mean']:.4f}, macro F1={summary['f1_mean']:.4f}")
    return summary
