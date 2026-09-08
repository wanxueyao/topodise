"""论文选定的原型学习实验。

仅运行二分类、分期和亚型任务的最佳配置。
可通过 RUN_* 环境变量选择实验组、设备和输出路径。
"""
import os
import sys
import time
import json
import traceback
import multiprocessing
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)


# ==================== 运行场所配置 ====================

RUN_SITE = 'local'  # 可选 server、gx10 或 local

SITE_DEFAULT_GPUS = {
    'server': [4, 5],
    'gx10': [],
    'local': [],
}

SITE_DEFAULT_CONCURRENCY_PER_GPU = {
    'server': 4,
    'gx10': 0,
    'local': 1,
}

AVAILABLE_GPUS = []
CONCURRENCY_PER_GPU = 0

ESTIMATED_EXPERIMENT_GPU_MEM_MB = 6000
GPU_MEMORY_UTILIZATION_RATIO = 0.90
GPU_MEMORY_RESERVE_MB = 2048
MAX_CONCURRENCY_PER_GPU = 3

RUN_TASKS = ['pd', 'stage', 'subtype']  # 可选 pd、stage 或 subtype

# ==================== 分组过滤开关（A/B/C/D 组严格区分） ====================
# 说明:
# 1) 默认只跑 I-group，优先产出新的 subtype hierarchy 三分类主结果
# 2) INCLUDE 与 INCLUDE_PREFIX 只要命中其一即可保留
# 3) EXCLUDE 与 EXCLUDE_PREFIX 优先级最高（命中即剔除）
# 4) 可通过环境变量临时覆盖，避免频繁改代码
#
# 示例:
#   仅跑 B-group: RUN_INCLUDE_GROUP_PREFIXES = ['B']
#   仅跑 A-group: RUN_INCLUDE_GROUP_PREFIXES = ['A']
#   仅跑 B1/B2:   RUN_INCLUDE_GROUP_PREFIXES = ['B1', 'B2']
#   跳过 B0:      RUN_EXCLUDE_GROUPS = ['B0_baseline_current']
RUN_INCLUDE_GROUPS = []
RUN_INCLUDE_GROUP_PREFIXES = []
RUN_EXCLUDE_GROUPS = []
RUN_EXCLUDE_GROUP_PREFIXES = []

ENV_RUN_INCLUDE_GROUPS = os.environ.get('RUN_INCLUDE_GROUPS', '')
ENV_RUN_INCLUDE_GROUP_PREFIXES = os.environ.get('RUN_INCLUDE_GROUP_PREFIXES', '')
ENV_RUN_EXCLUDE_GROUPS = os.environ.get('RUN_EXCLUDE_GROUPS', '')
ENV_RUN_EXCLUDE_GROUP_PREFIXES = os.environ.get('RUN_EXCLUDE_GROUP_PREFIXES', '')
ENV_RUN_AVAILABLE_GPUS = os.environ.get('RUN_AVAILABLE_GPUS', '')
ENV_RUN_CONCURRENCY_PER_GPU = os.environ.get('RUN_CONCURRENCY_PER_GPU', '')
ENV_RUN_TASKS = os.environ.get('RUN_TASKS', '')
ENV_RUN_OUTPUT_DIR = os.environ.get('RUN_OUTPUT_DIR', '')
ENV_RUN_ESTIMATED_EXP_GPU_MEM_MB = os.environ.get('RUN_ESTIMATED_EXP_GPU_MEM_MB', '')
ENV_RUN_GPU_MEMORY_UTILIZATION_RATIO = os.environ.get('RUN_GPU_MEMORY_UTILIZATION_RATIO', '')
ENV_RUN_GPU_MEMORY_RESERVE_MB = os.environ.get('RUN_GPU_MEMORY_RESERVE_MB', '')
ENV_RUN_MAX_CONCURRENCY_PER_GPU = os.environ.get('RUN_MAX_CONCURRENCY_PER_GPU', '')
ENV_RUN_SITE = os.environ.get('RUN_SITE', '')
ENV_RUN_ENVIRONMENT = os.environ.get('RUN_ENVIRONMENT', '')

FORCE_NUM_WORKERS_ZERO = True
VERBOSE_WORKER_STDERR = False

OUTPUT_DIR = 'results/proto_batch'


from experiments.paper_selection import load_paper_experiments  # noqa: E402

EXPERIMENTS = load_paper_experiments("prototype")

# ==================== 公共函数 ====================

def _build_exp_name(backbone: str, n_way: int, n_shot: int,
                   embedding_dim: int, lr: float, group: str = '') -> str:
    suffix = f"{backbone}_way{n_way}_shot{n_shot}_emb{embedding_dim}_lr{lr:.0e}"
    return f"{group}__{suffix}" if group else suffix


def _parse_csv_list(raw: str) -> list:
    if not raw:
        return []
    return [x.strip() for x in raw.split(',') if x.strip()]


def _resolve_run_site() -> str:
    raw = (ENV_RUN_SITE or ENV_RUN_ENVIRONMENT or RUN_SITE or '').strip().lower()
    alias_map = {
        'server': 'server',
        'srv': 'server',
        'gx10': 'gx10',
        'dgx': 'gx10',
        'dgx_spark': 'gx10',
        'auto': 'gx10',
        'cuda': 'gx10',
        'local': 'local',
        'mac': 'local',
        'macos': 'local',
        'cpu': 'local',
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


def _resolve_run_tasks() -> list:
    env_tasks = _parse_csv_list(ENV_RUN_TASKS)
    return env_tasks or list(RUN_TASKS)


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
        resolved_exact, resolved_prefix, auto_promoted, unmatched = [], list(prefix_tokens), [], []
        for token in exact_tokens:
            if token in all_group_set:
                resolved_exact.append(token)
            elif any(group.startswith(token) for group in all_groups):
                if token not in resolved_prefix:
                    resolved_prefix.append(token)
                auto_promoted.append(token)
            else:
                unmatched.append(token)
        return resolved_exact, resolved_prefix, auto_promoted, unmatched

    ie, ip, ia, iu = _normalize_tokens(rules['include_groups'], rules['include_prefixes'])
    ee, ep, ea, eu = _normalize_tokens(rules['exclude_groups'], rules['exclude_prefixes'])
    return {
        'include_groups': ie,
        'include_prefixes': ip,
        'exclude_groups': ee,
        'exclude_prefixes': ep,
        'auto_promoted_include_prefixes': ia,
        'auto_promoted_exclude_prefixes': ea,
        'unmatched_include_groups': iu,
        'unmatched_exclude_groups': eu,
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
    backbone = exp.get('backbone', 'densenet121')
    n_way = exp.get('n_way', 3)
    n_shot = exp.get('n_shot', 3)
    embedding_dim = exp.get('embedding_dim', 128)
    lr = exp.get('learning_rate', 5e-4)
    group = exp.get('_group', '')

    name = _build_exp_name(backbone, n_way, n_shot, embedding_dim, lr, group)
    return {
        'gpu_id': gpu_id,
        'exp': exp,
        'name': name,
        'desc': exp.get('_desc', ''),
        'group': group,
        'exp_dir': os.path.join(batch_dir, name),
        'worker_log_path': os.path.join(batch_dir, name, 'worker.log'),
    }


def _worker(task: dict, run_tasks: list) -> dict:
    gpu_id = task['gpu_id']
    exp = task['exp']
    exp_name = task['name']
    exp_dir = task['exp_dir']
    worker_log_path = task['worker_log_path']
    start_t = time.time()

    os.makedirs(exp_dir, exist_ok=True)
    worker_log = open(worker_log_path, 'a', encoding='utf-8', buffering=1)

    def _wlog(msg: str, to_stderr: bool = False):
        device_tag = f"GPU {gpu_id}" if gpu_id is not None else "CPU"
        line = f"[{datetime.now().strftime('%H:%M:%S')}] [{device_tag}] [{exp_name}] {msg}"
        worker_log.write(line + "\n")
        if to_stderr:
            sys.stderr.write(line + "\n")
            sys.stderr.flush()

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

        _wlog("启动原型学习任务", to_stderr=VERBOSE_WORKER_STDERR)

        from experiments.prototype_experiments.main import ProtoExperimentConfig, run_single_experiment
        import torch

        torch.set_num_threads(1)

        cfg = ProtoExperimentConfig()
        cfg.BACKBONE = exp.get('backbone', 'densenet121')
        cfg.ARCHITECTURE = exp.get('architecture', 'single_stream')
        cfg.PRETRAINED = exp.get('pretrained', True)
        cfg.EMBEDDING_DIM = exp.get('embedding_dim', 128)
        cfg.FEATURE_DROPOUT = exp.get('feature_dropout', 0.2)

        cfg.N_WAY = exp.get('n_way', 3)
        cfg.N_SHOT = exp.get('n_shot', 3)
        cfg.N_QUERY = exp.get('n_query', 4)
        cfg.TRAIN_EPISODES_PER_EPOCH = exp.get('train_episodes_per_epoch', 80)
        cfg.VAL_EPISODES = exp.get('val_episodes', 80)

        cfg.NUM_EPOCHS = exp.get('num_epochs', 40)
        cfg.LEARNING_RATE = exp.get('learning_rate', 5e-4)
        cfg.WEIGHT_DECAY = exp.get('weight_decay', 1e-4)
        cfg.LR_SCHEDULER = exp.get('lr_scheduler', 'cosine')
        cfg.KFOLD_MODE = exp.get('kfold_mode', 'patient')
        cfg.PATIENCE = exp.get('patience', cfg.PATIENCE)
        cfg.NUM_FOLDS = exp.get('num_folds', cfg.NUM_FOLDS)
        cfg.KFOLD_REPEATS = exp.get('kfold_repeats', cfg.KFOLD_REPEATS)
        cfg.SEED = exp.get('seed', cfg.SEED)
        cfg.DATASET_MODE = exp.get('dataset_mode', cfg.DATASET_MODE)
        cfg.DATASET_JSON = exp.get('dataset_json', cfg.DATASET_JSON)

        if FORCE_NUM_WORKERS_ZERO:
            cfg.NUM_WORKERS = 0

        cfg.DEVICE = 'cuda:0' if (gpu_id is not None and torch.cuda.is_available()) else 'cpu'

        devnull_fd = open(os.devnull, 'w')
        orig_stdout = sys.stdout
        sys.stdout = devnull_fd

        try:
            # 支持实验特定的任务列表（用于任务感知的参数扫描）
            # 如果实验配置中指定了 _run_tasks，就使用它；否则使用全局的 run_tasks
            exp_run_tasks = exp.get('_run_tasks', run_tasks)
            _wlog(f"配置 N_WAY={cfg.N_WAY}，运行任务: {exp_run_tasks}")
            run_single_experiment(cfg, tasks=exp_run_tasks, output_dir=exp_dir)
        finally:
            sys.stdout = orig_stdout
            devnull_fd.close()

        duration = time.time() - start_t
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        _wlog(f"任务完成，耗时 {duration/60.0:.1f}min",
              to_stderr=VERBOSE_WORKER_STDERR)

        return {
            'name': exp_name,
            'group': task['group'],
            'desc': task['desc'],
            'gpu_id': gpu_id if gpu_id is not None else 'cpu',
            'status': 'success',
            'duration_minutes': duration / 60.0,
            'worker_log_path': worker_log_path,
        }

    except Exception as e:
        duration = time.time() - start_t
        _wlog("任务失败", to_stderr=True)
        _wlog(f"错误: {e}")
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
        }
    finally:
        worker_log.close()


def _worker_entry(payload: tuple) -> dict:
    task, run_tasks = payload
    return _worker(task, run_tasks)


def run_parallel():
    os.chdir(PROJECT_ROOT)

    run_site = _resolve_run_site()
    available_gpus = _resolve_available_gpus(run_site)
    concurrency_per_gpu, concurrency_meta = _resolve_concurrency_per_gpu(available_gpus, run_site)
    run_tasks = _resolve_run_tasks()

    selected_experiments, filter_rules = _filter_experiments_by_group(EXPERIMENTS)
    if not selected_experiments:
        all_groups = sorted({exp.get('_group', '') for exp in EXPERIMENTS if exp.get('_group', '')})
        raise ValueError(
            "分组过滤后没有可运行实验。请检查 RUN_INCLUDE_* / RUN_EXCLUDE_* 配置或环境变量。"
            f" 当前过滤规则: {filter_rules}; 可用分组: {all_groups}"
        )

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    batch_dir = os.path.join(PROJECT_ROOT, _resolve_output_dir(), f"batch_{timestamp}")
    os.makedirs(batch_dir, exist_ok=True)
    master_log_path = os.path.join(batch_dir, 'master.log')

    def _mlog(msg: str):
        line = f"[{datetime.now().strftime('%H:%M:%S')}] [MASTER] {msg}"
        print(line, flush=True)
        with open(master_log_path, 'a', encoding='utf-8') as f:
            f.write(line + "\n")

    _mlog("=" * 70)
    _mlog("原型学习并行实验启动（4 Backbone 标准版）")
    _mlog(f"原始实验数: {len(EXPERIMENTS)}")
    _mlog(f"筛选后实验数: {len(selected_experiments)}")
    _mlog(f"输出目录: {batch_dir}")
    _mlog(f"运行场所: {run_site}")
    _mlog(f"可用显卡: {available_gpus if available_gpus else ['cpu']}")
    _mlog(f"每卡并发: {concurrency_per_gpu}")
    _mlog(f"并发模式: {concurrency_meta.get('mode')} ({concurrency_meta.get('reason')})")
    if concurrency_meta.get('mode') == 'auto_gpu_mem':
        _mlog(
            "并发估算: "
            f"min_total={concurrency_meta.get('min_total_mb')}MB, "
            f"usable={concurrency_meta.get('usable_mb')}MB, "
            f"exp_est={concurrency_meta.get('estimated_exp_mem_mb')}MB, "
            f"ratio={concurrency_meta.get('util_ratio')}, "
            f"reserve={concurrency_meta.get('reserve_mb')}MB, "
            f"max={concurrency_meta.get('max_concurrency_per_gpu')}"
        )
    _mlog(f"任务列表: {run_tasks}")
    _mlog(f"include_groups: {filter_rules['include_groups']}")
    _mlog(f"include_prefixes: {filter_rules['include_prefixes']}")
    _mlog(f"exclude_groups: {filter_rules['exclude_groups']}")
    _mlog(f"exclude_prefixes: {filter_rules['exclude_prefixes']}")
    _mlog("=" * 70)

    groups_seen = []
    for exp in selected_experiments:
        g = exp.get('_group', '')
        if g and g not in groups_seen:
            groups_seen.append(g)
    _mlog(f"实验分组: {', '.join(groups_seen)}")

    tasks = []
    for i, exp in enumerate(selected_experiments):
        gpu_id = available_gpus[i % len(available_gpus)] if available_gpus else None
        tasks.append(_build_worker_task(exp, batch_dir, gpu_id))

    payloads = [(task, run_tasks) for task in tasks]

    gpu_slots = max(1, len(available_gpus))
    num_workers = gpu_slots * concurrency_per_gpu
    total_tasks = len(payloads)
    completed = 0
    all_results = []
    t_start = time.time()

    multiprocessing.set_start_method('spawn', force=True)
    with multiprocessing.Pool(processes=num_workers, maxtasksperchild=1) as pool:
        for res in pool.imap_unordered(_worker_entry, payloads):
            completed += 1
            all_results.append(res)

            elapsed = time.time() - t_start
            avg_sec = elapsed / completed
            remain = avg_sec * (total_tasks - completed)
            eta_h = int(remain // 3600)
            eta_m = int((remain % 3600) // 60)

            status = "OK" if res['status'] == 'success' else f"FAIL: {res.get('error', 'unknown')}"
            _mlog(
                f"进度 {completed:02d}/{total_tasks} | GPU{res['gpu_id']} | "
                f"{res['group']} | {status} | "
                f"{res['duration_minutes']:.1f}min | ETA: {eta_h}h{eta_m}m"
            )

    _mlog("=" * 78)
    _mlog(f"{'组':35s}  {'状态':10s}  {'耗时'}")
    _mlog("-" * 68)
    all_results.sort(key=lambda x: x['name'])
    for r in all_results:
        status = 'OK' if r['status'] == 'success' else 'FAIL'
        _mlog(f"{r['group']:35s}  {status:10s}  {r['duration_minutes']:.1f}min")
    _mlog("=" * 78)

    total_duration = time.time() - t_start
    summary_path = os.path.join(batch_dir, 'parallel_summary_log.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': timestamp,
            'run_site': run_site,
            'available_gpus': available_gpus,
            'concurrency_per_gpu': concurrency_per_gpu,
            'concurrency_meta': concurrency_meta,
            'tasks': run_tasks,
            'filter_rules': filter_rules,
            'raw_experiment_count': len(EXPERIMENTS),
            'selected_experiment_count': len(selected_experiments),
            'total_duration_minutes': total_duration / 60.0,
            'master_log_path': master_log_path,
            'results': all_results,
        }, f, indent=2, ensure_ascii=False)

    _mlog(f"并行汇总日志: {summary_path}")
    _mlog(f"主控日志: {master_log_path}")
    _mlog(f"所有实验结果已保存至: {batch_dir}")
    _mlog(f"完成！结果: {batch_dir}")


if __name__ == '__main__':
    run_parallel()
