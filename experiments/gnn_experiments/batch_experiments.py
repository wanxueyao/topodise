"""论文选定的 UNet-only 基线实验。

仅运行二分类、分期和亚型任务的最佳配置。
"""
import os
import sys
import time
import json
import re
import traceback
import multiprocessing
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from utils.misc import Tee  # noqa: E402


# ==================== 在这里配置实验 ====================

# 运行场所（可选: 'server' / 'gx10' / 'local'）
RUN_SITE = 'local'

SITE_DEFAULT_GPUS = {
    'server': [2, 3],
    'gx10': [],
    'local': [],
}

SITE_DEFAULT_CONCURRENCY_PER_GPU = {
    'server': 2,
    'gx10': 0,
    'local': 1,
}

# 可用显卡编号（优先级低于 RUN_AVAILABLE_GPUS 环境变量）
AVAILABLE_GPUS = []

# 每卡并发运行实验数（<=0 表示自动按显存估算）
CONCURRENCY_PER_GPU = 0

# 自动并发估算参数
ESTIMATED_EXPERIMENT_GPU_MEM_MB = 6000
GPU_MEMORY_UTILIZATION_RATIO = 0.90
GPU_MEMORY_RESERVE_MB = 2048
MAX_CONCURRENCY_PER_GPU = 2

# 并发场景将 DataLoader worker 固定为 0 更稳
FORCE_NUM_WORKERS_ZERO = True

# 分组过滤
RUN_INCLUDE_GROUPS = []
RUN_INCLUDE_GROUP_PREFIXES = []
RUN_EXCLUDE_GROUPS = []
RUN_EXCLUDE_GROUP_PREFIXES = []

# 过拟合约束（TrainAcc - ValAcc）
MAX_OVERFIT_ACC_GAP = 0.12

ENV_RUN_INCLUDE_GROUPS = os.environ.get('RUN_INCLUDE_GROUPS', '')
ENV_RUN_INCLUDE_GROUP_PREFIXES = os.environ.get('RUN_INCLUDE_GROUP_PREFIXES', '')
ENV_RUN_EXCLUDE_GROUPS = os.environ.get('RUN_EXCLUDE_GROUPS', '')
ENV_RUN_EXCLUDE_GROUP_PREFIXES = os.environ.get('RUN_EXCLUDE_GROUP_PREFIXES', '')
ENV_RUN_AVAILABLE_GPUS = os.environ.get('RUN_AVAILABLE_GPUS', '')
ENV_RUN_CONCURRENCY_PER_GPU = os.environ.get('RUN_CONCURRENCY_PER_GPU', '')
ENV_RUN_ESTIMATED_EXP_GPU_MEM_MB = os.environ.get('RUN_ESTIMATED_EXP_GPU_MEM_MB', '')
ENV_RUN_GPU_MEMORY_UTILIZATION_RATIO = os.environ.get('RUN_GPU_MEMORY_UTILIZATION_RATIO', '')
ENV_RUN_GPU_MEMORY_RESERVE_MB = os.environ.get('RUN_GPU_MEMORY_RESERVE_MB', '')
ENV_RUN_MAX_CONCURRENCY_PER_GPU = os.environ.get('RUN_MAX_CONCURRENCY_PER_GPU', '')
ENV_RUN_OUTPUT_DIR = os.environ.get('RUN_OUTPUT_DIR', '')
ENV_RUN_SITE = os.environ.get('RUN_SITE', '')
ENV_RUN_ENVIRONMENT = os.environ.get('RUN_ENVIRONMENT', '')


from experiments.paper_selection import load_paper_experiments  # noqa: E402

EXPERIMENTS = load_paper_experiments("unet_only")

# 输出目录
OUTPUT_DIR = 'results/unet_only_batch'


# ==================== 辅助函数 ====================

def _parse_csv_list(raw: str) -> list:
    if not raw:
        return []
    return [x.strip() for x in raw.split(',') if x.strip()]


def _resolve_run_site() -> str:
    raw = (ENV_RUN_SITE or ENV_RUN_ENVIRONMENT or RUN_SITE or '').strip().lower()
    alias_map = {
        'server': 'server', 'srv': 'server',
        'gx10': 'gx10', 'dgx': 'gx10', 'dgx_spark': 'gx10',
        'auto': 'gx10', 'cuda': 'gx10',
        'local': 'local', 'mac': 'local', 'macos': 'local', 'cpu': 'local',
    }
    run_site = alias_map.get(raw, raw)
    if run_site not in ('server', 'gx10', 'local'):
        raise ValueError("RUN_SITE 仅支持 'server'、'gx10' 或 'local'")
    return run_site


def _resolve_available_gpus(run_site: str) -> list:
    env_gpus = (ENV_RUN_AVAILABLE_GPUS or '').strip()
    if env_gpus:
        if env_gpus.lower() in ('cpu', 'none'):
            return []
        return [int(token) for token in _parse_csv_list(env_gpus)]

    if run_site == 'local':
        return []

    if AVAILABLE_GPUS:
        return list(AVAILABLE_GPUS)

    site_default_gpus = SITE_DEFAULT_GPUS.get(run_site, [])
    if site_default_gpus:
        return list(site_default_gpus)

    try:
        import torch
        gpu_count = int(torch.cuda.device_count())
        return list(range(gpu_count)) if gpu_count > 0 else []
    except Exception:
        return []


def _resolve_concurrency_per_gpu(available_gpus: list, run_site: str) -> tuple:
    raw = (ENV_RUN_CONCURRENCY_PER_GPU or '').strip()
    if raw:
        val = max(1, int(raw))
        return val, {'mode': 'fixed_env', 'reason': 'RUN_CONCURRENCY_PER_GPU', 'value': val}

    if int(CONCURRENCY_PER_GPU) > 0:
        val = max(1, int(CONCURRENCY_PER_GPU))
        return val, {'mode': 'fixed_code', 'reason': 'CONCURRENCY_PER_GPU', 'value': val}

    site_default_conc = int(SITE_DEFAULT_CONCURRENCY_PER_GPU.get(run_site, 0))
    if site_default_conc > 0:
        val = max(1, site_default_conc)
        return val, {'mode': 'fixed_site_default', 'reason': f'RUN_SITE={run_site}', 'value': val}

    if not available_gpus:
        return 1, {'mode': 'cpu', 'reason': 'no_gpu_detected', 'value': 1}

    est_exp_mem_mb = int((ENV_RUN_ESTIMATED_EXP_GPU_MEM_MB or str(ESTIMATED_EXPERIMENT_GPU_MEM_MB)).strip())
    util_ratio = float((ENV_RUN_GPU_MEMORY_UTILIZATION_RATIO or str(GPU_MEMORY_UTILIZATION_RATIO)).strip())
    reserve_mb = int((ENV_RUN_GPU_MEMORY_RESERVE_MB or str(GPU_MEMORY_RESERVE_MB)).strip())
    max_conc = int((ENV_RUN_MAX_CONCURRENCY_PER_GPU or str(MAX_CONCURRENCY_PER_GPU)).strip())

    try:
        import torch
        per_gpu_total_mb = []
        for gid in available_gpus:
            total_mb = int(torch.cuda.get_device_properties(gid).total_memory / (1024 * 1024))
            per_gpu_total_mb.append(total_mb)

        min_total_mb = min(per_gpu_total_mb)
        usable_mb = int(min_total_mb * max(0.1, min(util_ratio, 1.0))) - max(0, reserve_mb)
        auto_conc = max(1, usable_mb // max(1, est_exp_mem_mb))
        auto_conc = min(max(1, max_conc), auto_conc)

        return auto_conc, {
            'mode': 'auto_gpu_mem',
            'reason': 'memory_estimation',
            'value': auto_conc,
            'per_gpu_total_mb': per_gpu_total_mb,
            'min_total_mb': min_total_mb,
            'estimated_exp_mem_mb': est_exp_mem_mb,
            'util_ratio': util_ratio,
            'reserve_mb': reserve_mb,
            'max_concurrency_per_gpu': max_conc,
            'usable_mb': usable_mb,
        }
    except Exception as e:
        return 1, {'mode': 'fallback', 'reason': f'auto_estimation_failed: {e}', 'value': 1}


def _resolve_output_dir() -> str:
    return (ENV_RUN_OUTPUT_DIR or OUTPUT_DIR).strip()


def _resolve_group_filters() -> dict:
    include_groups = _parse_csv_list(ENV_RUN_INCLUDE_GROUPS) or RUN_INCLUDE_GROUPS
    include_prefixes = _parse_csv_list(ENV_RUN_INCLUDE_GROUP_PREFIXES) or RUN_INCLUDE_GROUP_PREFIXES
    exclude_groups = _parse_csv_list(ENV_RUN_EXCLUDE_GROUPS) or RUN_EXCLUDE_GROUPS
    exclude_prefixes = _parse_csv_list(ENV_RUN_EXCLUDE_GROUP_PREFIXES) or RUN_EXCLUDE_GROUP_PREFIXES
    return {
        'include_groups': include_groups,
        'include_prefixes': include_prefixes,
        'exclude_groups': exclude_groups,
        'exclude_prefixes': exclude_prefixes,
    }


def _normalize_group_filters(rules: dict, experiments: list) -> dict:
    all_groups = [exp.get('_group', '') for exp in experiments if exp.get('_group', '')]
    all_group_set = set(all_groups)

    def _normalize_tokens(exact_tokens: list, prefix_tokens: list) -> tuple:
        resolved_exact = []
        resolved_prefix = list(prefix_tokens)
        for token in exact_tokens:
            if token in all_group_set:
                resolved_exact.append(token)
            elif any(group.startswith(token) for group in all_groups):
                if token not in resolved_prefix:
                    resolved_prefix.append(token)
        return resolved_exact, resolved_prefix

    include_exact, include_prefix = _normalize_tokens(rules['include_groups'], rules['include_prefixes'])
    exclude_exact, exclude_prefix = _normalize_tokens(rules['exclude_groups'], rules['exclude_prefixes'])

    return {
        'include_groups': include_exact,
        'include_prefixes': include_prefix,
        'exclude_groups': exclude_exact,
        'exclude_prefixes': exclude_prefix,
    }


def _filter_experiments_by_group(experiments: list) -> tuple:
    rules = _normalize_group_filters(_resolve_group_filters(), experiments)
    include_groups = set(rules['include_groups'])
    include_prefixes = tuple(rules['include_prefixes'])
    exclude_groups = set(rules['exclude_groups'])
    exclude_prefixes = tuple(rules['exclude_prefixes'])

    def _is_selected(group: str) -> bool:
        if include_groups or include_prefixes:
            if (group not in include_groups) and (not group.startswith(include_prefixes)):
                return False
        if (group in exclude_groups) or group.startswith(exclude_prefixes):
            return False
        return True

    filtered = [exp for exp in experiments if _is_selected(exp.get('_group', ''))]
    return filtered, rules


def _build_exp_name(exp: dict) -> str:
    backbone = exp.get('backbone', 'resnet18')
    architecture = exp.get('architecture', 'dual_stream')
    pretrained = exp.get('pretrained', 'Half')
    num_epochs = exp.get('epochs', 40)
    learning_rate = exp.get('learning_rate', 2e-4)
    weight_decay = exp.get('weight_decay', 3e-3)
    kfold_mode = exp.get('kfold_mode', 'patient')
    num_folds = exp.get('num_folds', 2)
    group = exp.get('_group', '')

    pt = 'Half' if pretrained == 'Half' else ('PT' if pretrained else 'Scratch')
    arch_short = {'single_stream': 'SS', 'mixed_single_stream': 'MS', 'dual_stream': 'DS'}.get(
        architecture, architecture)

    suffix = (f"unet_{backbone}_{arch_short}_{pt}_"
              f"ep{num_epochs}_lr{learning_rate}_wd{weight_decay}_"
              f"{kfold_mode}_{num_folds}fold")
    return f"{group}__{suffix}" if group else suffix


def _build_worker_task(exp: dict, batch_dir: str, gpu_id: int) -> dict:
    name = _build_exp_name(exp)
    return {
        'gpu_id': gpu_id,
        'exp': exp,
        'name': name,
        'group': exp.get('_group', ''),
        'desc': exp.get('_desc', ''),
        'exp_dir': os.path.join(batch_dir, name),
        'worker_log_path': os.path.join(batch_dir, name, 'worker.log'),
    }


def _parse_output_metrics(output_path: str) -> dict:
    if not os.path.exists(output_path):
        return {}

    pattern = re.compile(
        r"^\s*📊\s+([A-Za-z_]+):\s*TrainAcc=([0-9.]+),\s*ValAcc=([0-9.]+)(?:±([0-9.]+))?",
        re.UNICODE,
    )
    metrics = {}
    with open(output_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            m = pattern.match(line.strip())
            if not m:
                continue
            task_name = m.group(1)
            train_acc = float(m.group(2))
            val_acc = float(m.group(3))
            val_std = float(m.group(4)) if m.group(4) else None
            gap = train_acc - val_acc
            metrics[task_name] = {
                'train_acc': train_acc,
                'val_acc': val_acc,
                'val_std': val_std,
                'gap': gap,
            }
    return metrics


def _build_overfit_screening(batch_dir: str, all_results: list) -> dict:
    task_order = ['PD_Classification', 'Stage_Classification', 'Subtype_Classification']
    per_experiment = []

    for res in all_results:
        if res.get('status') != 'success':
            continue
        exp_dir = res.get('exp_dir', '')
        output_path = os.path.join(exp_dir, 'output.txt')
        metrics = _parse_output_metrics(output_path)
        if not metrics:
            continue

        gaps = [metrics[t]['gap'] for t in task_order if t in metrics]
        if not gaps:
            continue

        max_gap = max(gaps)
        mean_gap = sum(gaps) / len(gaps)
        pass_overfit = all(g < MAX_OVERFIT_ACC_GAP for g in gaps)

        pd_val = metrics.get('PD_Classification', {}).get('val_acc', 0.0)
        stage_val = metrics.get('Stage_Classification', {}).get('val_acc', 0.0)
        subtype_val = metrics.get('Subtype_Classification', {}).get('val_acc', 0.0)
        # 与训练默认损失权重一致：PD 1.0, Stage 0.5, Subtype 0.5
        weighted_score = pd_val + 0.5 * stage_val + 0.5 * subtype_val

        per_experiment.append({
            'name': res.get('name'),
            'group': res.get('group'),
            'gpu_id': res.get('gpu_id'),
            'exp_dir': exp_dir,
            'max_gap': max_gap,
            'mean_gap': mean_gap,
            'pass_overfit': pass_overfit,
            'metrics': metrics,
            'weighted_score': weighted_score,
        })

    qualified = [x for x in per_experiment if x['pass_overfit']]
    qualified_sorted = sorted(
        qualified,
        key=lambda x: (x['weighted_score'], x['metrics'].get('PD_Classification', {}).get('val_acc', 0.0)),
        reverse=True,
    )

    best = qualified_sorted[0] if qualified_sorted else None
    return {
        'threshold_gap': MAX_OVERFIT_ACC_GAP,
        'task_order': task_order,
        'total_success_experiments': len(per_experiment),
        'qualified_count': len(qualified_sorted),
        'best_qualified': best,
        'qualified_ranking': qualified_sorted,
        'all_experiments': sorted(per_experiment, key=lambda x: x['weighted_score'], reverse=True),
    }


def _worker(task: dict) -> dict:
    gpu_id = task['gpu_id']
    exp = task['exp']
    exp_name = task['name']
    exp_dir = task['exp_dir']
    worker_log_path = task['worker_log_path']
    start_t = time.time()

    os.makedirs(exp_dir, exist_ok=True)
    worker_log = open(worker_log_path, 'a', encoding='utf-8', buffering=1)

    def _wlog(msg: str):
        device_tag = f"GPU {gpu_id}" if gpu_id is not None else "CPU"
        line = f"[{datetime.now().strftime('%H:%M:%S')}] [{device_tag}] [{exp_name}] {msg}"
        worker_log.write(line + "\n")

    try:
        os.chdir(PROJECT_ROOT)

        if gpu_id is None:
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
        os.environ['NUMEXPR_NUM_THREADS'] = '1'

        from experiments.gnn_experiments.main import (
            GNNExperimentConfig,
            run_single_experiment,
        )
        import torch

        torch.set_num_threads(1)

        cfg = GNNExperimentConfig()
        cfg.BACKBONE = exp.get('backbone', 'resnet18')
        cfg.ARCHITECTURE = exp.get('architecture', 'dual_stream')
        cfg.PRETRAINED = exp.get('pretrained', 'Half')
        cfg.BATCH_SIZE = exp.get('batch_size', cfg.BATCH_SIZE)
        cfg.NUM_EPOCHS = exp.get('epochs', cfg.NUM_EPOCHS)
        cfg.LEARNING_RATE = exp.get('learning_rate', cfg.LEARNING_RATE)
        cfg.WEIGHT_DECAY = exp.get('weight_decay', cfg.WEIGHT_DECAY)
        cfg.LR_SCHEDULER = exp.get('lr_scheduler', cfg.LR_SCHEDULER)
        cfg.LR_STEP_SIZE = exp.get('lr_step_size', cfg.LR_STEP_SIZE)
        cfg.LR_GAMMA = exp.get('lr_gamma', cfg.LR_GAMMA)
        cfg.BACKBONE_LR_RATIO = exp.get('backbone_lr_ratio', cfg.BACKBONE_LR_RATIO)
        cfg.OPTIMIZER = exp.get('optimizer', cfg.OPTIMIZER)
        cfg.EARLY_STOPPING = exp.get('early_stopping', cfg.EARLY_STOPPING)
        cfg.PATIENCE = exp.get('patience', cfg.PATIENCE)
        cfg.DROPOUT = exp.get('dropout', cfg.DROPOUT)
        cfg.PD_LOSS_TYPE = exp.get('pd_loss_type', cfg.PD_LOSS_TYPE)
        cfg.STAGE_LOSS_TYPE = exp.get('stage_loss_type', cfg.STAGE_LOSS_TYPE)
        cfg.SUBTYPE_LOSS_TYPE = exp.get('subtype_loss_type', cfg.SUBTYPE_LOSS_TYPE)
        cfg.LABEL_SMOOTH_EPS = exp.get('label_smooth_eps', cfg.LABEL_SMOOTH_EPS)
        cfg.DATASET_MODE = exp.get('dataset_mode', 'pd_all')
        cfg.NUM_FOLDS = exp.get('num_folds', 2)
        cfg.SEED = exp.get('seed', cfg.SEED)
        cfg.KFOLD_MODE = exp.get('kfold_mode', 'patient')
        cfg.DATA_AUGMENTATION = exp.get('data_augmentation', True)
        cfg.AUG_HORIZONTAL_FLIP = exp.get('aug_horizontal_flip', cfg.AUG_HORIZONTAL_FLIP)
        cfg.AUG_ROTATION = exp.get('aug_rotation', cfg.AUG_ROTATION)
        cfg.AUG_BRIGHTNESS = exp.get('aug_brightness', cfg.AUG_BRIGHTNESS)
        cfg.AUG_CONTRAST = exp.get('aug_contrast', cfg.AUG_CONTRAST)
        cfg.AUG_SATURATION = exp.get('aug_saturation', cfg.AUG_SATURATION)
        cfg.AUG_HUE = exp.get('aug_hue', cfg.AUG_HUE)

        if FORCE_NUM_WORKERS_ZERO:
            cfg.NUM_WORKERS = 0

        cfg.DEVICE = 'cuda:0' if (gpu_id is not None and torch.cuda.is_available()) else 'cpu'

        if exp.get('_subtype_hierarchy'):
            from experiments.gnn_experiments.unet_subtype_hierarchy_runner import (
                run_unet_subtype_hierarchy_experiment,
            )
            output_path = os.path.join(exp_dir, 'output.txt')
            output_file = open(output_path, 'w', encoding='utf-8', buffering=1)
            original_stdout = sys.stdout
            terminal_stdout = getattr(sys, '__stdout__', None) or original_stdout
            sys.stdout = Tee(terminal_stdout, output_file)

            try:
                summary = run_unet_subtype_hierarchy_experiment(
                    cfg,
                    output_dir=exp_dir,
                    seed=cfg.SEED,
                    n_splits=cfg.NUM_FOLDS,
                    repeats=int(exp.get('_sh_repeats', 3)),
                )
            finally:
                sys.stdout = original_stdout
                output_file.close()

            with open(os.path.join(exp_dir, 'hierarchy_result.json'), 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
        else:
            run_single_experiment(cfg, tasks=exp.get('_run_tasks'), output_dir=exp_dir)

        duration = time.time() - start_t
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        _wlog(f"任务完成，耗时 {duration / 60.0:.1f}min")
        return {
            'name': exp_name,
            'group': task['group'],
            'desc': task['desc'],
            'gpu_id': gpu_id if gpu_id is not None else 'cpu',
            'status': 'success',
            'duration_minutes': duration / 60.0,
            'worker_log_path': worker_log_path,
            'exp_dir': exp_dir,
        }
    except Exception as e:
        duration = time.time() - start_t
        _wlog(f"任务失败: {e}")
        _wlog(traceback.format_exc())
        return {
            'name': exp_name,
            'group': task['group'],
            'desc': task['desc'],
            'gpu_id': gpu_id if gpu_id is not None else 'cpu',
            'status': 'failed',
            'duration_minutes': duration / 60.0,
            'error': str(e),
            'traceback': traceback.format_exc(),
            'worker_log_path': worker_log_path,
            'exp_dir': exp_dir,
        }
    finally:
        worker_log.close()


def run():
    os.chdir(PROJECT_ROOT)

    run_site = _resolve_run_site()
    available_gpus = _resolve_available_gpus(run_site)
    concurrency_per_gpu, concurrency_meta = _resolve_concurrency_per_gpu(available_gpus, run_site)

    selected_experiments, rules = _filter_experiments_by_group(EXPERIMENTS)
    if not selected_experiments:
        all_groups = sorted({exp.get('_group', '') for exp in EXPERIMENTS if exp.get('_group', '')})
        raise ValueError(
            "分组过滤后没有可运行实验。请检查 RUN_INCLUDE_* / RUN_EXCLUDE_* 配置或环境变量。"
            f" 当前过滤规则: {rules}; 可用分组: {all_groups}"
        )

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    batch_dir = os.path.join(PROJECT_ROOT, _resolve_output_dir(), f'batch_{timestamp}')
    os.makedirs(batch_dir, exist_ok=True)
    master_log_path = os.path.join(batch_dir, 'master.log')

    def _mlog(msg: str):
        line = f"[{datetime.now().strftime('%H:%M:%S')}] [MASTER] {msg}"
        print(line, flush=True)
        with open(master_log_path, 'a', encoding='utf-8') as f:
            f.write(line + "\n")

    _mlog('=' * 70)
    _mlog('GNN 实验（pd_dataset_unet_en）并行启动')
    _mlog(f'原始实验数: {len(EXPERIMENTS)}')
    _mlog(f'筛选后实验数: {len(selected_experiments)}')
    _mlog(f'输出目录: {batch_dir}')
    _mlog(f'运行场所: {run_site}')
    _mlog(f"可用显卡: {available_gpus if available_gpus else ['cpu']}")
    _mlog(f'每卡并发: {concurrency_per_gpu}')
    _mlog(f"并发模式: {concurrency_meta.get('mode')} ({concurrency_meta.get('reason')})")
    if concurrency_meta.get('mode') == 'auto_gpu_mem':
        _mlog(
            '并发估算: '
            f"min_total={concurrency_meta.get('min_total_mb')}MB, "
            f"usable={concurrency_meta.get('usable_mb')}MB, "
            f"exp_est={concurrency_meta.get('estimated_exp_mem_mb')}MB"
        )
    _mlog(f"过滤规则: include_groups={rules['include_groups']}, "
          f"include_prefixes={rules['include_prefixes']}, "
          f"exclude_groups={rules['exclude_groups']}, "
          f"exclude_prefixes={rules['exclude_prefixes']}")
    _mlog('=' * 70)

    tasks = []
    for i, exp in enumerate(selected_experiments):
        gpu_id = available_gpus[i % len(available_gpus)] if available_gpus else None
        tasks.append(_build_worker_task(exp, batch_dir, gpu_id))

    gpu_slots = max(1, len(available_gpus))
    num_workers = gpu_slots * concurrency_per_gpu

    total_tasks = len(tasks)
    completed = 0
    all_results = []
    t_start = time.time()

    multiprocessing.set_start_method('spawn', force=True)
    with multiprocessing.Pool(processes=num_workers, maxtasksperchild=1) as pool:
        for res in pool.imap_unordered(_worker, tasks):
            completed += 1
            all_results.append(res)

            elapsed = time.time() - t_start
            avg_sec = elapsed / completed
            remain = avg_sec * (total_tasks - completed)
            eta_h = int(remain // 3600)
            eta_m = int((remain % 3600) // 60)

            status = 'OK' if res['status'] == 'success' else f"FAIL: {res.get('error', 'unknown')}"
            _mlog(
                f"进度 {completed:02d}/{total_tasks} | GPU{res['gpu_id']} | "
                f"{res['group']} | 状态: {status} | "
                f"耗时: {res['duration_minutes']:.1f}min | ETA: {eta_h}h{eta_m}m"
            )

    total_duration = time.time() - t_start
    summary_path = os.path.join(batch_dir, 'parallel_summary_log.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': timestamp,
            'run_site': run_site,
            'available_gpus': available_gpus,
            'concurrency_per_gpu': concurrency_per_gpu,
            'concurrency_meta': concurrency_meta,
            'filter_rules': rules,
            'raw_experiment_count': len(EXPERIMENTS),
            'selected_experiment_count': len(selected_experiments),
            'total_duration_minutes': total_duration / 60.0,
            'master_log_path': master_log_path,
            'results': all_results,
        }, f, indent=2, ensure_ascii=False)

    screening = _build_overfit_screening(batch_dir, all_results)
    screening_path = os.path.join(batch_dir, 'overfit_screening.json')
    with open(screening_path, 'w', encoding='utf-8') as f:
        json.dump(screening, f, indent=2, ensure_ascii=False)

    _mlog(f'并行汇总日志: {summary_path}')
    _mlog(f'过拟合筛选日志: {screening_path}')
    _mlog(
        f"过拟合约束: Gap < {MAX_OVERFIT_ACC_GAP:.2f} | "
        f"通过实验数: {screening.get('qualified_count', 0)}/{screening.get('total_success_experiments', 0)}"
    )
    best = screening.get('best_qualified')
    if best:
        pd_val = best['metrics'].get('PD_Classification', {}).get('val_acc', 0.0)
        _mlog(
            f"最佳(满足约束): {best.get('group')} | PD ValAcc={pd_val:.4f} | "
            f"Score={best.get('weighted_score', 0.0):.4f} | max_gap={best.get('max_gap', 0.0):.4f}"
        )
    else:
        _mlog('未找到满足过拟合约束的实验，请放宽阈值或增强正则。')
    _mlog(f'主控日志: {master_log_path}')
    _mlog(f'所有实验结果已保存至: {batch_dir}')
    _mlog(f'完成！结果: {batch_dir}')


if __name__ == '__main__':
    run()
