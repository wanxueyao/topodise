"""论文选定的 MTL-MAE 实验。

仅运行二分类、分期、亚型和 DM 迁移的最佳配置。
可通过 RUN_* 环境变量选择实验组、设备和输出路径。
"""
from __future__ import annotations

import json
import multiprocessing
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

# ── 项目根目录 ─────────────────────────────────────────────────────────
_THIS_FILE = Path(__file__).resolve()
CODES_ROOT = str(_THIS_FILE.parents[2])
REPO_ROOT = str(_THIS_FILE.parents[3])
sys.path.insert(0, CODES_ROOT)
sys.path.insert(0, REPO_ROOT)

# ══════════════════════════════════════════════════════════════════════
# 全局运行配置（可通过环境变量覆盖）
# ══════════════════════════════════════════════════════════════════════

RUN_SITE = 'local'                    # 可选 server、gx10 或 local

SITE_DEFAULT_GPUS = {
    'server': [0, 1],
    'gx10':   [],
    'local':  [],
}
SITE_DEFAULT_CONCURRENCY_PER_GPU = {
    'server': 2,
    'gx10':   0,
    'local':  1,
}

AVAILABLE_GPUS: list = []
CONCURRENCY_PER_GPU: int = 0

ESTIMATED_EXPERIMENT_GPU_MEM_MB   = 8000
GPU_MEMORY_UTILIZATION_RATIO      = 0.90
GPU_MEMORY_RESERVE_MB             = 2048
MAX_CONCURRENCY_PER_GPU           = 3

FORCE_NUM_WORKERS_ZERO = True

# 分组过滤（默认运行 F/G 进阶抗过拟合组）
RUN_INCLUDE_GROUPS:          list = []
RUN_INCLUDE_GROUP_PREFIXES:  list = []
RUN_EXCLUDE_GROUPS:          list = []
RUN_EXCLUDE_GROUP_PREFIXES:  list = []

OUTPUT_DIR = 'results/mtl_batch'

# ── 环境变量读取 ────────────────────────────────────────────────────────
ENV_RUN_INCLUDE_GROUPS           = os.environ.get('RUN_INCLUDE_GROUPS', '')
ENV_RUN_INCLUDE_GROUP_PREFIXES   = os.environ.get('RUN_INCLUDE_GROUP_PREFIXES', '')
ENV_RUN_EXCLUDE_GROUPS           = os.environ.get('RUN_EXCLUDE_GROUPS', '')
ENV_RUN_EXCLUDE_GROUP_PREFIXES   = os.environ.get('RUN_EXCLUDE_GROUP_PREFIXES', '')
ENV_RUN_AVAILABLE_GPUS           = os.environ.get('RUN_AVAILABLE_GPUS', '')
ENV_RUN_CONCURRENCY_PER_GPU      = os.environ.get('RUN_CONCURRENCY_PER_GPU', '')
ENV_RUN_ESTIMATED_EXP_GPU_MEM_MB = os.environ.get('RUN_ESTIMATED_EXP_GPU_MEM_MB', '')
ENV_RUN_GPU_MEMORY_UTILIZATION_RATIO = os.environ.get('RUN_GPU_MEMORY_UTILIZATION_RATIO', '')
ENV_RUN_GPU_MEMORY_RESERVE_MB    = os.environ.get('RUN_GPU_MEMORY_RESERVE_MB', '')
ENV_RUN_MAX_CONCURRENCY_PER_GPU  = os.environ.get('RUN_MAX_CONCURRENCY_PER_GPU', '')
ENV_RUN_OUTPUT_DIR               = os.environ.get('RUN_OUTPUT_DIR', '')
ENV_RUN_SITE                     = os.environ.get('RUN_SITE', '')
ENV_RUN_NUM_FOLDS                = os.environ.get('RUN_NUM_FOLDS', '')

# Fold 数量开关：5 = 正式实验，2 = 快速测试
# 可通过 RUN_NUM_FOLDS=2 环境变量覆盖
NUM_FOLDS: int = 5


def _resolve_num_folds() -> int:
    raw = (ENV_RUN_NUM_FOLDS or '').strip()
    if raw:
        val = int(raw)
        if val not in (2, 5):
            raise ValueError(f"RUN_NUM_FOLDS 仅支持 2 或 5，got: {val}")
        return val
    return int(NUM_FOLDS)


# ══════════════════════════════════════════════════════════════════════
# 实验定义
# ══════════════════════════════════════════════════════════════════════

from experiments.paper_selection import load_paper_experiments  # noqa: E402

EXPERIMENTS = load_paper_experiments("mtl")

# 并行运行框架（与 basic_experiments/batch_experiments.py 风格一致）
# ══════════════════════════════════════════════════════════════════════

def _parse_csv_list(raw: str) -> list:
    if not raw:
        return []
    return [x.strip() for x in raw.split(',') if x.strip()]


def _resolve_run_site() -> str:
    raw = (ENV_RUN_SITE or RUN_SITE or '').strip().lower()
    alias = {'server': 'server', 'srv': 'server',
             'gx10': 'gx10', 'dgx': 'gx10', 'dgx_spark': 'gx10',
             'auto': 'gx10', 'cuda': 'gx10',
             'local': 'local', 'mac': 'local', 'macos': 'local', 'cpu': 'local'}
    site = alias.get(raw, raw)
    if site not in ('server', 'gx10', 'local'):
        raise ValueError(f"RUN_SITE 仅支持 'server'/'gx10'/'local'，got: {raw!r}")
    return site


def _resolve_available_gpus(run_site: str) -> list:
    env_gpus = (ENV_RUN_AVAILABLE_GPUS or '').strip()
    if env_gpus:
        if env_gpus.lower() in ('cpu', 'none'):
            return []
        return [int(t) for t in _parse_csv_list(env_gpus)]
    if run_site == 'local':
        return []
    if AVAILABLE_GPUS:
        return list(AVAILABLE_GPUS)
    default_gpus = SITE_DEFAULT_GPUS.get(run_site, [])
    if default_gpus:
        return list(default_gpus)
    try:
        import torch
        n = int(torch.cuda.device_count())
        return list(range(n)) if n > 0 else []
    except Exception:
        return []


def _resolve_concurrency(available_gpus: list, run_site: str) -> tuple:
    raw = (ENV_RUN_CONCURRENCY_PER_GPU or '').strip()
    if raw:
        val = max(1, int(raw))
        return val, {'mode': 'fixed_env', 'value': val}

    if int(CONCURRENCY_PER_GPU) > 0:
        val = max(1, int(CONCURRENCY_PER_GPU))
        return val, {'mode': 'fixed_code', 'value': val}

    site_default = int(SITE_DEFAULT_CONCURRENCY_PER_GPU.get(run_site, 0))
    if site_default > 0:
        return site_default, {'mode': 'site_default', 'value': site_default}

    if not available_gpus:
        return 1, {'mode': 'cpu', 'value': 1}

    est  = int((ENV_RUN_ESTIMATED_EXP_GPU_MEM_MB or str(ESTIMATED_EXPERIMENT_GPU_MEM_MB)).strip())
    util = float((ENV_RUN_GPU_MEMORY_UTILIZATION_RATIO or str(GPU_MEMORY_UTILIZATION_RATIO)).strip())
    rsv  = int((ENV_RUN_GPU_MEMORY_RESERVE_MB or str(GPU_MEMORY_RESERVE_MB)).strip())
    maxc = int((ENV_RUN_MAX_CONCURRENCY_PER_GPU or str(MAX_CONCURRENCY_PER_GPU)).strip())

    try:
        import torch
        total_mb = min(
            int(torch.cuda.get_device_properties(g).total_memory / (1024**2))
            for g in available_gpus
        )
        usable = int(total_mb * max(0.1, util)) - max(0, rsv)
        auto   = min(maxc, max(1, usable // max(1, est)))
        return auto, {'mode': 'auto_mem', 'value': auto, 'total_mb': total_mb, 'usable': usable}
    except Exception as e:
        return 1, {'mode': 'fallback', 'reason': str(e), 'value': 1}


def _resolve_group_filters() -> dict:
    return {
        'include_groups':   _parse_csv_list(ENV_RUN_INCLUDE_GROUPS)  or RUN_INCLUDE_GROUPS,
        'include_prefixes': _parse_csv_list(ENV_RUN_INCLUDE_GROUP_PREFIXES) or RUN_INCLUDE_GROUP_PREFIXES,
        'exclude_groups':   _parse_csv_list(ENV_RUN_EXCLUDE_GROUPS)  or RUN_EXCLUDE_GROUPS,
        'exclude_prefixes': _parse_csv_list(ENV_RUN_EXCLUDE_GROUP_PREFIXES) or RUN_EXCLUDE_GROUP_PREFIXES,
    }


def _filter_experiments(experiments: list) -> tuple:
    rules = _resolve_group_filters()
    inc_g  = set(rules['include_groups'])
    inc_px = tuple(rules['include_prefixes'])
    exc_g  = set(rules['exclude_groups'])
    exc_px = tuple(rules['exclude_prefixes'])

    def _ok(grp: str) -> bool:
        if inc_g or inc_px:
            if grp not in inc_g and not grp.startswith(inc_px):
                return False
        if grp in exc_g or grp.startswith(exc_px):
            return False
        return True

    selected = [e for e in experiments if _ok(e.get('_group', ''))]
    return selected, rules


def _build_exp_name(exp: dict) -> str:
    group  = exp.get('_group', 'X')
    task   = exp.get('branch_a_task', 'seg')
    ep     = exp.get('epochs', 40)
    lr     = exp.get('lr', 2e-4)
    wd     = exp.get('wd', 3e-3)
    pt     = exp.get('pretrained', 'Half')
    aug    = 'aug' if exp.get('data_augmentation', True) else 'noaug'
    return f"{group}__mtl_{task}_resnet18_{pt}_ep{ep}_lr{lr:.0e}_wd{wd:.0e}_{aug}_patient"


def _worker(task: dict) -> dict:
    gpu_id        = task['gpu_id']
    exp           = task['exp']
    exp_name      = task['name']
    exp_dir       = task['exp_dir']
    worker_log_path = task['worker_log_path']
    start_t       = time.time()

    os.makedirs(exp_dir, exist_ok=True)
    wlog_f = open(worker_log_path, 'a', encoding='utf-8', buffering=1)

    def _wlog(msg: str):
        tag  = f"GPU{gpu_id}" if gpu_id is not None else "CPU"
        line = f"[{datetime.now().strftime('%H:%M:%S')}] [{tag}] [{exp_name}] {msg}"
        wlog_f.write(line + "\n")
        wlog_f.flush()

    _orig_stdout = sys.stdout
    _orig_stderr = sys.stderr

    # 将 worker 子进程的所有 print() 输出重定向到 worker.log
    # 这样 main.py 中每个 epoch 的日志都会写入文件，避免多进程交错和静默丢失
    sys.stdout = wlog_f
    sys.stderr = wlog_f

    try:
        os.chdir(CODES_ROOT)

        if gpu_id is None:
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

        for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                  'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            os.environ[k] = '1'

        # 延迟导入（确保 CUDA_VISIBLE_DEVICES 已生效）
        sys.path.insert(0, CODES_ROOT)
        sys.path.insert(0, REPO_ROOT)

        from experiments.mtl_experiments.main import (
            MTLExperimentConfig, run_single_mtl_experiment
        )
        import torch
        torch.set_num_threads(1)

        config = MTLExperimentConfig()
        config.BRANCH_A_TASK    = exp.get('branch_a_task', 'segmentation')
        config.NUM_EPOCHS       = exp.get('epochs', 40)
        config.LEARNING_RATE    = exp.get('lr', 2e-4)
        config.WEIGHT_DECAY     = exp.get('wd', 3e-3)
        config.BATCH_SIZE       = exp.get('batch', 16)
        config.DROPOUT          = exp.get('dropout', 0.0)
        config.PRETRAINED       = exp.get('pretrained', 'Half')
        config.DATA_AUGMENTATION = exp.get('data_augmentation', True)
        config.NUM_FOLDS        = exp.get('folds', 5)
        config.SEED             = exp.get('seed', 42)
        config.RUN_TASKS        = exp.get('_run_tasks')
        config.DEVICE           = 'cuda:0' if (gpu_id is not None and
                                                 torch.cuda.is_available()) else 'cpu'

        # 迁移实验（DM）特有：encoder 微调比例（0=冻结线性探针）
        if 'backbone_lr_ratio' in exp:
            config.BACKBONE_LR_RATIO = float(exp['backbone_lr_ratio'])

        # 可选的损失任务权重覆盖（F/G 组使用）
        _weight_overrides = {
            'pd_weight':      'PD_WEIGHT',
            'stage_weight':   'STAGE_WEIGHT',
            'subtype_weight': 'SUBTYPE_WEIGHT',
            'seg_weight':     'SEG_WEIGHT',
            'mae_weight':     'MAE_WEIGHT',
            'pseudo_weight':  'PSEUDO_WEIGHT',
        }
        for _k, _attr in _weight_overrides.items():
            if _k in exp:
                setattr(config, _attr, float(exp[_k]))

        if FORCE_NUM_WORKERS_ZERO:
            config.NUM_WORKERS = 0

        # Subtype 层级三分类公平重算分发：只跑 subtype 任务，TD-vs-rest + PIGD-vs-Mixed
        if exp.get('_subtype_hierarchy'):
            from experiments.mtl_experiments.subtype_hierarchy_runner import (
                run_mtl_subtype_hierarchy_experiment
            )
            run_mtl_subtype_hierarchy_experiment(
                config, output_dir=exp_dir,
                seed=exp.get('seed', config.SEED),
                n_splits=exp.get('_sh_n_splits', 5),
                repeats=exp.get('_sh_repeats', 3),
            )
        # DM 迁移实验分发：复用 pd encoder，仅重训 DM 下游分类头
        elif exp.get('_transfer') == 'dm':
            from experiments.mtl_experiments.dm_transfer import (
                run_single_dm_transfer_experiment
            )
            run_single_dm_transfer_experiment(
                config, output_dir=exp_dir,
                tasks=exp.get('_run_tasks'),
            )
        else:
            run_single_mtl_experiment(config, output_dir=exp_dir)

        duration = time.time() - start_t
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        _wlog(f"完成 耗时={duration/60:.1f}min")
        return {
            'name': exp_name, 'group': exp.get('_group', ''),
            'gpu_id': gpu_id, 'status': 'success',
            'duration_minutes': duration / 60.0,
        }

    except Exception as exc:
        duration = time.time() - start_t
        tb = traceback.format_exc()
        _wlog(f"失败: {exc}\n{tb}")
        return {
            'name': exp_name, 'group': exp.get('_group', ''),
            'gpu_id': gpu_id, 'status': 'failed',
            'duration_minutes': duration / 60.0,
            'error': str(exc), 'traceback': tb,
        }
    finally:
        sys.stdout = _orig_stdout
        sys.stderr = _orig_stderr
        wlog_f.close()


def run():
    os.chdir(CODES_ROOT)

    run_site       = _resolve_run_site()
    avail_gpus     = _resolve_available_gpus(run_site)
    conc, conc_meta = _resolve_concurrency(avail_gpus, run_site)

    selected, rules = _filter_experiments(EXPERIMENTS)
    if not selected:
        all_groups = sorted({e.get('_group', '') for e in EXPERIMENTS})
        raise ValueError(
            f"过滤后无可运行实验。规则={rules}  可用分组={all_groups}"
        )

    ts        = datetime.now().strftime('%Y%m%d_%H%M%S')
    batch_dir = os.path.join(CODES_ROOT, ENV_RUN_OUTPUT_DIR or OUTPUT_DIR, f'batch_{ts}')
    os.makedirs(batch_dir, exist_ok=True)
    master_log = os.path.join(batch_dir, 'master.log')

    def _mlog(msg: str):
        line = f"[{datetime.now().strftime('%H:%M:%S')}] [MASTER] {msg}"
        print(line, flush=True)
        with open(master_log, 'a', encoding='utf-8') as f:
            f.write(line + '\n')

    _mlog('=' * 70)
    _mlog('MTL 批量实验启动')
    _mlog(f'总实验数: {len(EXPERIMENTS)}  筛选后: {len(selected)}')
    _mlog(f'输出目录: {batch_dir}')
    _mlog(f'运行场所: {run_site}')
    _mlog(f'GPU列表: {avail_gpus or ["cpu"]}')
    _mlog(f'每卡并发: {conc}  (mode={conc_meta.get("mode")})')
    _mlog(f'过滤规则: {rules}')
    _mlog('=' * 70)

    tasks = []
    for i, exp in enumerate(selected):
        gpu_id = avail_gpus[i % len(avail_gpus)] if avail_gpus else None
        name   = _build_exp_name(exp)
        tasks.append({
            'gpu_id':          gpu_id,
            'exp':             exp,
            'name':            name,
            'group':           exp.get('_group', ''),
            'exp_dir':         os.path.join(batch_dir, name),
            'worker_log_path': os.path.join(batch_dir, name, 'worker.log'),
        })

    num_workers = max(1, len(avail_gpus)) * conc
    total       = len(tasks)
    completed   = 0
    all_results = []
    t_start     = time.time()

    multiprocessing.set_start_method('spawn', force=True)
    with multiprocessing.Pool(processes=num_workers, maxtasksperchild=1) as pool:
        for res in pool.imap_unordered(_worker, tasks):
            completed += 1
            all_results.append(res)

            elapsed = time.time() - t_start
            avg_sec = elapsed / completed
            remain  = avg_sec * (total - completed)
            eta_h, eta_m = int(remain // 3600), int((remain % 3600) // 60)

            status = 'OK' if res['status'] == 'success' else f"FAIL: {res.get('error','?')}"
            _mlog(
                f"进度 {completed:02d}/{total} | GPU{res['gpu_id']} | "
                f"{res['group']} | {status} | "
                f"{res['duration_minutes']:.1f}min | ETA {eta_h}h{eta_m}m"
            )

    total_dur = time.time() - t_start
    summary_path = os.path.join(batch_dir, 'parallel_summary_log.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp':             ts,
            'run_site':              run_site,
            'available_gpus':        avail_gpus,
            'concurrency_per_gpu':   conc,
            'filter_rules':          rules,
            'total_experiments':     len(EXPERIMENTS),
            'selected_experiments':  len(selected),
            'total_duration_minutes': total_dur / 60.0,
            'results':               all_results,
        }, f, indent=2, ensure_ascii=False)

    _mlog(f'汇总日志: {summary_path}')
    _mlog(f'全部完成！结果目录: {batch_dir}')


if __name__ == '__main__':
    run()
