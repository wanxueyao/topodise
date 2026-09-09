"""Process execution only; experiment choices and defaults belong to each batch file."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]


def _csv(value):
    return [x.strip() for x in value.split(',') if x.strip()]


def output_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def single_main(module_name, argv=None):
    """Execute the exact configuration supplied by batch_experiments.py."""
    parser = argparse.ArgumentParser(description='执行一组明确配置；编辑和选择实验请使用 batch_experiments.py')
    parser.add_argument('--config', required=True, help='单组实验 JSON；由 batch 自动保存并传入')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--num-workers', type=int, default=0)
    args = parser.parse_args(argv)
    if args.num_workers < 0:
        parser.error('--num-workers must be nonnegative')
    if args.device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    elif not args.device.startswith('cuda'):
        parser.error('--device must be cpu or cuda[:index]')
    recipe = json.loads(Path(args.config).read_text(encoding='utf-8'))
    os.chdir(ROOT)
    import torch
    torch.set_num_threads(1)
    if args.device.startswith('cuda'):
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA was requested but is unavailable')
        torch.cuda.set_device(torch.device(args.device))
    output = output_path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    module = importlib.import_module(module_name)
    result = module.run_paper_experiment(recipe, str(output), args.device, args.num_workers)
    if isinstance(result, dict) and (result.get('status') in ('failed', 'error') or result.get('error')):
        raise RuntimeError(f'Experiment failed: {result}')


def _gpu_slots(args):
    if args.concurrency_per_gpu < 1:
        raise ValueError('--concurrency-per-gpu must be positive')
    if args.gpus is not None:
        gpus = [] if args.gpus.lower() in ('cpu', 'none', '') else _csv(args.gpus)
    elif os.getenv('RUN_SITE', 'auto').lower() in ('local', 'cpu', 'mac', 'macos'):
        gpus = []
    else:
        import torch
        visible = os.getenv('CUDA_VISIBLE_DEVICES')
        gpus = _csv(visible) if visible is not None else [str(i) for i in range(torch.cuda.device_count())]
    if len(gpus) != len(set(gpus)):
        raise ValueError('Duplicate GPU ids')
    return [gpu for gpu in gpus for _ in range(args.concurrency_per_gpu)] or [None]


def _run_slot(module_name, recipes, gpu, output, num_workers):
    results = []
    for recipe in recipes:
        group = recipe['_group']
        exp_dir = output / group
        exp_dir.mkdir(parents=True, exist_ok=True)
        (exp_dir / 'experiment.json').write_text(json.dumps(recipe, indent=2, ensure_ascii=False), encoding='utf-8')
        command = [sys.executable, '-m', module_name,
                   '--config', str(exp_dir / 'experiment.json'), '--output-dir', str(exp_dir),
                   '--device', 'cpu' if gpu is None else 'cuda:0', '--num-workers', str(num_workers)]
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = '' if gpu is None else gpu
        # Selection is already resolved; a child must run exactly the assigned recipe.
        for key in ('RUN_INCLUDE_GROUPS', 'RUN_INCLUDE_GROUP_PREFIXES', 'RUN_EXCLUDE_GROUPS', 'RUN_EXCLUDE_GROUP_PREFIXES'):
            env.pop(key, None)
        for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            env[key] = '1'
        started = time.monotonic()
        print(f"Starting {group} on {'CPU' if gpu is None else 'GPU ' + gpu}", flush=True)
        with (exp_dir / 'process.log').open('a', encoding='utf-8') as log:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        row = {'group': group, 'method': recipe.get('_method', module_name), 'gpu_id': gpu,
               'status': 'success' if result.returncode == 0 else 'failed',
               'returncode': result.returncode, 'elapsed_s': time.monotonic() - started,
               'output_dir': str(exp_dir)}
        (exp_dir / 'run_status.json').write_text(json.dumps(row, indent=2), encoding='utf-8')
        results.append(row)
        print(f"{group}: {row['status']} ({row['elapsed_s']:.1f}s)", flush=True)
    return results



def run_batch(module_name, recipes, *, gpus, concurrency_per_gpu, output_dir, num_workers):
    """Schedule the supplied recipes and preserve every edited parameter in worker input."""
    args = argparse.Namespace(gpus=gpus, concurrency_per_gpu=concurrency_per_gpu)
    slots = _gpu_slots(args)
    output = output_path(output_dir) / datetime.now().strftime('batch_%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'experiments.json').write_text(json.dumps(recipes, indent=2, ensure_ascii=False), encoding='utf-8')
    with ThreadPoolExecutor(max_workers=len(slots)) as pool:
        futures = [pool.submit(_run_slot, module_name, recipes[i::len(slots)], gpu, output, num_workers)
                   for i, gpu in enumerate(slots)]
        results = [row for future in futures for row in future.result()]
    (output / 'batch_summary.json').write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')
    if any(row['status'] != 'success' for row in results):
        raise SystemExit(f'Experiment failures; see {output}/batch_summary.json')
    print(f'Completed {len(results)} recipes: {output}')
