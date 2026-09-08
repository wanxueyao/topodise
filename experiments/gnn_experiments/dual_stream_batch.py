"""论文选定的 UNet-dual 基线实验。

仅运行二分类、分期和亚型任务的最佳配置。
"""

import os
import sys
import time
import json
import traceback
import multiprocessing
from datetime import datetime
from typing import Dict, List, Any

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from utils.misc import Tee  # noqa: E402


# ==================== 运行配置 ====================

RUN_SITE = 'local'

SITE_DEFAULT_GPUS = {
    'server': [0, 1],
    'gx10': [],
    'local': [],
}

SITE_DEFAULT_CONCURRENCY_PER_GPU = {
    'server': 2,
    'gx10': 0,
    'local': 1,
}

AVAILABLE_GPUS = []
CONCURRENCY_PER_GPU = 0

ESTIMATED_EXPERIMENT_GPU_MEM_MB = 8000
GPU_MEMORY_UTILIZATION_RATIO = 0.90
GPU_MEMORY_RESERVE_MB = 2048
MAX_CONCURRENCY_PER_GPU = 2

FORCE_NUM_WORKERS_ZERO = True

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

EXPERIMENTS = load_paper_experiments("unet_dual")

OUTPUT_DIR = 'results/dual_stream_batch'


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


def _build_worker_task(exp: dict, batch_dir: str, gpu_id: int) -> dict:
    name = exp.get('name', exp.get('_group', 'unknown'))
    return {
        'gpu_id': gpu_id,
        'exp': exp,
        'name': name,
        'group': exp.get('_group', ''),
        'desc': exp.get('_desc', ''),
        'exp_dir': os.path.join(batch_dir, name),
        'worker_log_path': os.path.join(batch_dir, name, 'worker.log'),
    }


def _read_json_if_exists(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def _build_overfit_screening(batch_dir: str, all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    ranking = []

    for item in all_results:
        if item.get('status') != 'success':
            continue

        exp_dir = item.get('exp_dir', '')
        summary_json = _read_json_if_exists(os.path.join(exp_dir, 'summary.json'))
        summary = summary_json.get('summary', {}) if isinstance(summary_json, dict) else {}
        if not summary:
            continue

        # summary 格式为 {task: {train_acc_mean, val_acc_mean, ...}}
        pd_s = summary.get('pd', {})
        stage_s = summary.get('stage', {})
        sub_s = summary.get('subtype', {})

        train_pd = pd_s.get('train_acc_mean')
        val_pd = pd_s.get('val_acc_mean')
        train_stage = stage_s.get('train_acc_mean')
        val_stage = stage_s.get('val_acc_mean')
        train_sub = sub_s.get('train_acc_mean')
        val_sub = sub_s.get('val_acc_mean')

        # pd_control 实验 stage/subtype 可能为空，用 0 填充以允许过拟合检查仅看 PD
        if train_pd is None or val_pd is None:
            continue
        train_stage = train_stage if train_stage is not None else 0.0
        val_stage = val_stage if val_stage is not None else 0.0
        train_sub = train_sub if train_sub is not None else 0.0
        val_sub = val_sub if val_sub is not None else 0.0

        gap_pd = float(train_pd - val_pd)
        gap_stage = float(train_stage - val_stage)
        gap_sub = float(train_sub - val_sub)
        max_gap = max(gap_pd, gap_stage, gap_sub)

        row = {
            'name': item.get('name'),
            'group': item.get('group'),
            'exp_dir': exp_dir,
            'val_pd_acc': float(val_pd),
            'val_stage_acc': float(val_stage),
            'val_subtype_acc': float(val_sub),
            'gap_pd': gap_pd,
            'gap_stage': gap_stage,
            'gap_subtype': gap_sub,
            'max_gap': max_gap,
            'qualified': bool(
                gap_pd < MAX_OVERFIT_ACC_GAP and
                gap_stage < MAX_OVERFIT_ACC_GAP and
                gap_sub < MAX_OVERFIT_ACC_GAP
            ),
            'weighted_score': float(val_pd + 0.5 * val_stage + 0.5 * val_sub),
        }
        ranking.append(row)

    ranking.sort(key=lambda x: (x['qualified'], x['weighted_score']), reverse=True)
    qualified = [x for x in ranking if x['qualified']]

    out = {
        'max_overfit_acc_gap': MAX_OVERFIT_ACC_GAP,
        'checked_experiment_count': len(ranking),
        'qualified_count': len(qualified),
        'best_qualified': qualified[0] if qualified else None,
        'qualified_ranking': qualified,
        'all_ranking': ranking,
    }

    out_path = os.path.join(batch_dir, 'overfit_screening.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    return out


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
        
        from experiments.gnn_experiments.dual_stream_main import run_single_experiment, DualStreamExperimentConfig
        import torch
        
        torch.set_num_threads(1)
        
        # 构建配置
        cfg = DualStreamExperimentConfig()
        for key, val in exp.items():
            if key.startswith('_'):
                continue
            if hasattr(cfg, key):
                setattr(cfg, key, val)
        
        if FORCE_NUM_WORKERS_ZERO:
            cfg.num_workers = 0
        
        # 运行实验
        if exp.get('_subtype_hierarchy'):
            from experiments.gnn_experiments.dual_stream_subtype_hierarchy_runner import (
                run_dual_stream_subtype_hierarchy_experiment,
            )
            output_path = os.path.join(exp_dir, 'output.txt')
            output_file = open(output_path, 'w', encoding='utf-8', buffering=1)
            original_stdout = sys.stdout
            terminal_stdout = getattr(sys, '__stdout__', None) or original_stdout
            sys.stdout = Tee(terminal_stdout, output_file)

            try:
                run_dual_stream_subtype_hierarchy_experiment(
                    cfg,
                    output_dir=exp_dir,
                    seed=exp.get('seed', cfg.seed),
                    n_splits=exp.get('_sh_n_splits', 5),
                    repeats=exp.get('_sh_repeats', 3),
                )
            finally:
                sys.stdout = original_stdout
                output_file.close()
        else:
            run_single_experiment(cfg, output_dir=exp_dir)
        
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
    _mlog('双视角 CNN 批量实验 并行启动')
    _mlog(f'原始实验数: {len(EXPERIMENTS)}')
    _mlog(f'筛选后实验数: {len(selected_experiments)}')
    _mlog(f'输出目录: {batch_dir}')
    _mlog(f'运行场所: {run_site}')
    _mlog(f"可用显卡: {available_gpus if available_gpus else ['cpu']}")
    _mlog(f'每卡并发: {concurrency_per_gpu}')
    _mlog(f"并发模式: {concurrency_meta.get('mode')} ({concurrency_meta.get('reason')})")
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
    
    _mlog(f'并行汇总日志: {summary_path}')
    _mlog(f"过拟合筛选: qualified={screening.get('qualified_count', 0)} / "
          f"checked={screening.get('checked_experiment_count', 0)}, "
          f"阈值={MAX_OVERFIT_ACC_GAP}")
    if screening.get('best_qualified'):
        best = screening['best_qualified']
        _mlog(f"最佳合格实验: {best.get('group')} | score={best.get('weighted_score'):.4f} | "
              f"max_gap={best.get('max_gap'):.4f}")
    _mlog(f'主控日志: {master_log_path}')
    _mlog(f'所有实验结果已保存至: {batch_dir}')
    _mlog(f'完成！结果: {batch_dir}')


if __name__ == '__main__':
    run()
