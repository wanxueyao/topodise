"""Edit EXPERIMENTS or use the CLI; outputs include patient metrics and 95% CI."""
from pathlib import Path
import argparse
import copy
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments._execution import run_batch

EXPERIMENTS = [{'_method': 'nerve_metrics_lr', '_group': 'nerve11_lr',
    '_run_tasks': ['pd_binary', 'pd_stage', 'pd_subtype'], 'num_folds': 5,
    'kfold_repeats': 3, 'report_repeats': 3, 'seed': 42, 'C': 1.0,
    'class_weight': 'balanced', 'max_iter': 2000, 'bootstrap_samples': 2000,
    'require_complete_cohort': False}]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path)
    parser.add_argument('--patient-json', type=Path)
    parser.add_argument('--feature-csvs', nargs='+', type=Path)
    parser.add_argument('--tasks', nargs='+', choices=('pd_binary', 'pd_stage', 'pd_subtype', 'dm_binary'))
    parser.add_argument('--output-dir', default='results/nerve_metrics_lr')
    parser.add_argument('--require-complete-cohort', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(); recipes = copy.deepcopy(EXPERIMENTS)
    for recipe in recipes:
        for field in ('dataset_root', 'patient_json'):
            if getattr(args, field): recipe[field] = str(getattr(args, field).expanduser().resolve())
        if args.feature_csvs: recipe['feature_csvs'] = [str(p.expanduser().resolve()) for p in args.feature_csvs]
        if args.tasks: recipe['_run_tasks'] = args.tasks
        if args.require_complete_cohort: recipe['require_complete_cohort'] = True
    if args.dry_run:
        import json
        print(json.dumps(recipes, indent=2)); return
    run_batch('experiments.nerve_metrics_lr_experiments.main', recipes, gpus='cpu',
              concurrency_per_gpu=1, output_dir=args.output_dir, num_workers=0)


if __name__ == '__main__': main()
