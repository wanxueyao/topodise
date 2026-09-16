"""Automatic patient OOF reports at the public experiment boundary.

Legacy training summaries may contain image metrics or averages of fold metrics.
This module follows the selected prediction files, never recomputes a result from
those scalar averages, and never mixes different configurations in a batch.
"""
from __future__ import annotations
import csv
from functools import wraps
import hashlib
import json
from pathlib import Path
import shutil
from .confidence_intervals import METRICS, report

ALIASES = {'pd': 'pd_binary', 'pd_binary': 'pd_binary', 'stage': 'pd_stage',
           'pd_stage': 'pd_stage', 'pd_subtype': 'pd_subtype',
           'pd_subtype_tdrest': 'pd_subtype', 'dm_binary': 'dm_binary'}
CLASS_LABELS = {'pd_binary': [0, 1], 'pd_stage': [0, 1, 2],
                'pd_subtype': [0, 1], 'dm_binary': [0, 1]}


def snapshot_selected_predictions(metrics, destination):
    """Freeze prediction files when an epoch is selected, before next evaluation."""
    result = dict(metrics)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for key, value in metrics.items():
        if key.endswith(('patient_logits_json', 'patient_logits_csv')) and value:
            source = Path(value)
            target = destination / source.name
            if source.resolve() != target.resolve():
                shutil.copy2(source, target)
            result[key] = str(target)
    return result


def collect_predictions(result, recipe, output):
    tasks = recipe['_run_tasks']
    collected = {t: [] for t in tasks}
    seen = set()

    def visit(value, task=None, fold=1, repeat=0):
        if isinstance(value, list):
            for index, item in enumerate(value, 1):
                visit(item, task, index, repeat)
            return
        if not isinstance(value, dict):
            return
        task = ALIASES.get(value.get('task'), task)
        repeat = int(value.get('repeat', repeat))
        if 'repeat' in value and 'fold' in value:
            fold = int(value['fold'])
        if value.get('prediction_csv') and task in collected:
            path = Path(value['prediction_csv']).resolve()
            if path in seen:
                return
            seen.add(path)
            with path.open(newline='', encoding='utf-8') as stream:
                for row in csv.DictReader(stream):
                    prob = float(row['prob_rest'])
                    collected[task].append({'patient_id': str(row['patient_id']),
                        'fold': int(row['fold']), 'repeat': int(row['repeat']),
                        'y_true': int(row['label']), 'y_pred': int(row['prediction']),
                        'patient_probs': [1 - prob, prob]})
            return
        for key, item in value.items():
            if key.endswith('patient_logits_json') and item:
                prefix = key.removesuffix('_patient_logits_json')
                target_task = ALIASES.get(prefix, task)
                if target_task is None and len(tasks) == 1:
                    target_task = tasks[0]
                if target_task not in collected:
                    continue
                path = Path(item).resolve()
                identity = (target_task, path)
                if identity in seen:
                    continue
                seen.add(identity)
                for row in json.loads(path.read_text(encoding='utf-8')):
                    collected[target_task].append({'patient_id': str(row['patient_id']),
                        'fold': fold, 'repeat': repeat, 'y_true': int(row['label']),
                        'y_pred': int(row['pred']), 'patient_probs': row['patient_probs']})
            elif isinstance(item, (dict, list)) and key not in ('config', 'meta', 'patient_report'):
                visit(item, ALIASES.get(key, task), fold, repeat)
    visit(result)
    # Dual-view historically returns scalar summaries and writes the fold index separately.
    if recipe.get('_method') == 'unet_dual' and any(not v for v in collected.values()):
        path = Path(output) / 'fold_results.json'
        if path.exists():
            visit(json.loads(path.read_text(encoding='utf-8')))
    return collected


def summarize_predictions(predictions, recipe):
    # MTL's original PD/DM runner calls its fold setting 'folds'; TD-rest uses num_folds.
    if recipe.get('_method') == 'mtl' and recipe['_run_tasks'] != ['pd_subtype']:
        n_folds = int(recipe.get('folds', 5))
    else:
        n_folds = int(recipe.get('num_folds', recipe.get('NUM_FOLDS', 5)))
    expected_repeats = int(recipe.get('report_repeats', recipe.get('kfold_repeats', 3)
                           if recipe['_run_tasks'] == ['pd_subtype'] else 1))
    reports = {}
    for task in recipe['_run_tasks']:
        rows = predictions.get(task, [])
        repeats = {int(r['repeat']) for r in rows}
        if len(repeats) != expected_repeats:
            raise ValueError(f'{task}: expected {expected_repeats} complete repeats, got {len(repeats)}')
        for repeat in repeats:
            folds = {int(r['fold']) for r in rows if int(r['repeat']) == repeat}
            if len(folds) != n_folds:
                raise ValueError(f'{task}: repeat {repeat} has {len(folds)}/{n_folds} folds')
        width = len(rows[0]['patient_probs'])
        reports[task] = report(rows, width, folds_per_repeat=n_folds,
            bootstrap_samples=int(recipe.get('bootstrap_samples', 2000)),
            seed=int(recipe.get('bootstrap_seed', 20260915)), evaluated_labels=CLASS_LABELS[task])
        reports[task]['folds_per_repeat'] = n_folds
        reports[task]['probability_handling'] = ('Original patient probabilities; extra classifier outputs '
            'are retained and erroneous predictions remain errors. Macro metrics use declared task labels.')
    return reports


def write_reports(output, reports, *, filename='metrics_ci95'):
    """Reports is a list of {method, group, task, ...report metadata}."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / f'{filename}.json').write_text(json.dumps(reports, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    fields = ['method', 'group', 'task', 'n_patients', 'n_repeats']
    fields += [f'{metric}_{part}' for metric in METRICS for part in ('estimate', 'ci95_low', 'ci95_high')]
    with (output / f'{filename}.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for result in reports:
            flat = {key: result.get(key, '') for key in fields[:5]}
            flat.update({f'{m}_{p}': result['metrics'][m][p] for m in METRICS for p in ('estimate', 'ci95_low', 'ci95_high')})
            writer.writerow(flat)
    lines = ['| Method / group | Task | Patients | Repeats | Accuracy (95% CI) | BACC (95% CI) | Macro-F1 (95% CI) | AUC (95% CI) |',
             '|---|---|---:|---:|---|---|---|---|']
    def fmt(metric):
        if metric['estimate'] is None:
            return 'NA (undefined for this cohort)'
        if metric['ci95_low'] is None:
            return f"{metric['estimate']:.4f} (CI undefined)"
        return f"{metric['estimate']:.4f} [{metric['ci95_low']:.4f}, {metric['ci95_high']:.4f}]"
    for result in reports:
        cells = [f"{result.get('method', '')} / {result.get('group', '')}", result['task'],
                 str(result['n_patients']), str(result['n_repeats'])]
        cells += [fmt(result['metrics'][m]) for m in METRICS]
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines += ['', 'Point estimates: mean of repeat-level patient OOF metrics. CI: class-stratified patient-cluster bootstrap,',
              'with the same patient resampling across repeats. Conditional on trained models; validation-based epoch',
              'selection is not corrected. Undefined AUC is NA, never zero. Historical image/fold averages are not this report.']
    (output / f'{filename}.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return output / f'{filename}.csv'


def finalize_result(result, recipe, output):
    predictions = collect_predictions(result, recipe, output)
    summaries = summarize_predictions(predictions, recipe)
    rows = [dict(method=recipe.get('_method', 'unknown'), group=recipe.get('_group', 'single'),
                 task=task, **summary) for task, summary in summaries.items()]
    write_reports(output, rows)
    Path(output, 'oof_predictions.json').write_text(json.dumps(predictions, indent=2, ensure_ascii=False), encoding='utf-8')
    print(Path(output, 'metrics_ci95.md').read_text(encoding='utf-8'), flush=True)
    return {'reports': rows, 'metrics_csv': str(Path(output, 'metrics_ci95.csv'))}


def with_patient_report(function):
    @wraps(function)
    def wrapped(exp, output_dir, device='cpu', num_workers=0):
        # Reporting options never leak into model configuration constructors.
        training = {k: v for k, v in exp.items() if k not in ('bootstrap_samples', 'bootstrap_seed', 'report_repeats')}
        import time
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        previous = [output / name for name in ('metrics_ci95.json', 'metrics_ci95.csv', 'metrics_ci95.md', 'oof_predictions.json') if (output / name).exists()]
        if previous:
            archive = output / 'previous_reports' / str(time.time_ns())
            archive.mkdir(parents=True)
            for path in previous:
                path.rename(archive / path.name)
        status_path = output / 'report_status.json'
        status_path.write_text(json.dumps({'status': 'running'}))
        try:
            result = function(training, output_dir, device, num_workers)
            if isinstance(result, dict) and (result.get('error') or result.get('status') in ('failed', 'error')):
                raise RuntimeError(f'Experiment did not complete: {result}')
            patient_report = finalize_result(result, exp, output_dir)
        except Exception as error:
            status_path.write_text(json.dumps({'status': 'failed', 'error': str(error)}))
            raise
        status_path.write_text(json.dumps({'status': 'complete', 'tasks': exp['_run_tasks']}))
        if isinstance(result, dict):
            result['patient_report'] = patient_report
        return result
    return wrapped


def finalize_batch(output, recipes, statuses):
    """Publish each recipe and pool only identical configurations across seeds."""
    output = Path(output)
    by_group = {r['group']: r for r in statuses}
    all_reports, families, expected = [], {}, {}
    for recipe in recipes:
        identity = {k: v for k, v in recipe.items() if k not in ('_group', '_desc', 'seed', 'SEED')}
        key = json.dumps(identity, sort_keys=True)
        expected[key] = expected.get(key, 0) + 1
    for recipe in recipes:
        status = by_group[recipe['_group']]
        if status['status'] != 'success':
            continue
        directory = Path(status['output_dir'])
        rows = json.loads((directory / 'metrics_ci95.json').read_text(encoding='utf-8'))
        all_reports.extend(rows)
        identity = {k: v for k, v in recipe.items() if k not in ('_group', '_desc', 'seed', 'SEED')}
        key = json.dumps(identity, sort_keys=True)
        families.setdefault(key, []).append((recipe, directory))
    write_reports(output, all_reports)
    combined = []
    for key, members in families.items():
        if len(members) != expected[key]:
            continue  # A partial seed family is never published as a completed pooled result.
        recipe = dict(members[0][0])
        predictions = {task: [] for task in recipe['_run_tasks']}
        offsets = {task: 0 for task in predictions}
        seeds = [r.get('seed', r.get('SEED', 42)) for r, _ in members]
        if len(set(seeds)) != len(seeds):
            raise ValueError('Identical configurations contain duplicate training seeds')
        for member, directory in members:
            values = json.loads((directory / 'oof_predictions.json').read_text(encoding='utf-8'))
            for task, rows in values.items():
                mapping = {r: i + offsets[task] for i, r in enumerate(sorted({v['repeat'] for v in rows}))}
                predictions[task].extend(dict(v, repeat=mapping[v['repeat']]) for v in rows)
                offsets[task] += len(mapping)
        recipe['report_repeats'] = next(iter(offsets.values()))
        for task, summary in summarize_predictions(predictions, recipe).items():
            combined.append(dict(method=recipe.get('_method'), group='config_' + hashlib.sha256(key.encode()).hexdigest()[:10],
                task=task, source_groups=[r['_group'] for r, _ in members], seeds=seeds,
                incomplete_batch=any(s['status'] != 'success' for s in statuses), **summary))
    write_reports(output, combined, filename='combined_metrics_ci95')
