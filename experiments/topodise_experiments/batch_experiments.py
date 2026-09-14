"""仅运行论文已报告的实验配置。

编辑本文件中的 EXPERIMENTS、RUN_GROUPS 和 AVAILABLE_GPUS 控制实验。
本模块负责可迁移路径解析、
Stage0 -> Stage1 检查点衔接、断点续训和可选的本地多 GPU 调度，不包含调参池。
"""
from __future__ import annotations

import os
import sys
import json
import time
import copy
import hashlib
import argparse
import traceback
import multiprocessing
import queue
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    import fcntl
except ImportError:  # pragma: no cover - Unix 服务器和 macOS 通常提供 fcntl。
    fcntl = None

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)


# ==================== 站点配置 ====================
RUN_SITE = 'portable'

# 可选的精确名称或前缀过滤。留空表示运行当前论文分区的全部配置，
# 默认不会选中历史调参组。
RUN_GROUPS: List[str] = []


def _env_list(name: str) -> List[str]:
    v = os.environ.get(name, '')
    return [x.strip() for x in v.split(',') if x.strip()]


def _resolve_project_path(path: Optional[str]) -> Optional[str]:
    """相对路径按 codes 根目录解析，绝对路径保持不变。"""
    if path is None:
        return None
    p = str(path).strip()
    if not p:
        return p
    if os.path.isabs(p):
        return os.path.normpath(p)
    return os.path.normpath(os.path.join(PROJECT_ROOT, p))


def _resolve_search_dir_path(path: str) -> str:
    p = str(path).strip()
    if os.path.isabs(p):
        return os.path.normpath(p)
    primary = os.path.normpath(os.path.join(PROJECT_ROOT, p))
    if os.path.exists(primary):
        return primary
    parent = os.path.normpath(os.path.join(PROJECT_ROOT, '..', p))
    if os.path.exists(parent):
        return parent
    return primary


REPRODUCIBILITY_MODES = ('strict', 'portable', 'checkpoint')


def _normalize_reproducibility_mode(value: Any) -> str:
    """解析三态复现模式，并兼容旧的 bool/0/1 配置。"""
    if isinstance(value, bool):
        return 'strict' if value else 'portable'
    normalized = str(value).strip().lower()
    aliases = {
        '1': 'strict', 'true': 'strict', 'on': 'strict', 'yes': 'strict',
        '2': 'portable', '0': 'portable', 'false': 'portable',
        'off': 'portable', 'no': 'portable', '': 'portable',
        '3': 'checkpoint', 'reference': 'checkpoint', 'c8': 'checkpoint',
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in REPRODUCIBILITY_MODES:
        raise ValueError(
            f'reproducibility must be one of {REPRODUCIBILITY_MODES}, got {value!r}'
        )
    return normalized


def _checkpoint_in_stage_dir(root: Path, stage: int) -> Optional[Path]:
    stage_name = f'stage{stage}'
    stage_dirs = sorted(
        child for child in root.iterdir()
        if child.is_dir()
        and (child.name.lower() == stage_name or child.name.lower().endswith(f'_{stage_name}'))
    )
    for relative in (
        Path('ckpt/ema_best.pt'), Path('ema_best.pt'),
        Path('ckpt/last.pt'), Path('last.pt'),
    ):
        matches = [directory / relative for directory in stage_dirs
                   if (directory / relative).is_file() and (directory / relative).stat().st_size > 0]
        if len(matches) > 1:
            raise ValueError(
                f'ambiguous Stage{stage} checkpoints under {root}: '
                + ', '.join(str(path) for path in matches)
            )
        if matches:
            return matches[0]
    fallback = sorted(
        path for directory in stage_dirs for path in directory.rglob('*.pt')
        if path.is_file() and path.stat().st_size > 0
    )
    if len(fallback) > 1:
        raise ValueError(
            f'ambiguous Stage{stage} checkpoints under {root}: '
            + ', '.join(str(path) for path in fallback)
        )
    return fallback[0] if fallback else None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_checkpoint_bundle(checkpoint_dir: str) -> Dict[str, Any]:
    """解析 c8 预训练目录；只有 Stage1 ``.pt`` 是运行所必需的。"""
    root = Path(checkpoint_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f'checkpoint directory does not exist: {root}')
    stage1 = _checkpoint_in_stage_dir(root, 1)
    if stage1 is None:
        raise FileNotFoundError(
            f'no Stage1 .pt checkpoint found under {root}; expected '
            '<root>/*_stage1/ckpt/ema_best.pt or <root>/stage1/ema_best.pt'
        )
    stage0 = _checkpoint_in_stage_dir(root, 0)
    metadata = [
        str(path) for path in (root / '_experiment_def.json', root / 'summary.json')
        if path.is_file()
    ]
    return {
        'checkpoint_dir': str(root),
        'stage1_checkpoint': str(stage1),
        'stage1_sha256': _sha256_file(stage1),
        'stage0_checkpoint': str(stage0) if stage0 is not None else None,
        'metadata_files': metadata,
    }


AVAILABLE_GPUS = [0]  # [0] 一张卡；[0, 1] 两张卡；[] 使用 CPU
_env_gpus_raw = os.environ.get('RUN_AVAILABLE_GPUS')
if _env_gpus_raw is not None:
    AVAILABLE_GPUS = [] if _env_gpus_raw.lower() in ('cpu', 'none', '') else [int(x) for x in _env_list('RUN_AVAILABLE_GPUS')]

# ==================== 运行参数（可直接修改） ====================
# 说明：环境变量同名参数优先级更高。
RUN_PARALLEL_EXPERIMENTS = True
RUN_AUTO_RESUME = True
RUN_CONCURRENCY_PER_GPU = 1

_RESULTS_ROOT = os.environ.get('RUN_RESULTS_ROOT', 'results').strip() or 'results'

# 外部父权重只由 checkpoint 模式从 checkpoint bundle 显式注入。
EXTERNAL_PARENT_CKPT = None
CHECKPOINT_DIR = os.environ.get('TOPODISE_CHECKPOINT_DIR', '').strip() or None

PARALLEL_EXPERIMENTS = bool(int(os.environ.get('RUN_PARALLEL_EXPERIMENTS',
                                                str(int(RUN_PARALLEL_EXPERIMENTS)))))
AUTO_RESUME = bool(int(os.environ.get('RUN_AUTO_RESUME', str(int(RUN_AUTO_RESUME)))))
CONCURRENCY_PER_GPU = max(1, int(os.environ.get('RUN_CONCURRENCY_PER_GPU',
                                                 str(RUN_CONCURRENCY_PER_GPU))))

# 如果用户把并发槽位设为 >1，则强制启用并行调度，避免误触发串行。
if CONCURRENCY_PER_GPU > 1 and not PARALLEL_EXPERIMENTS:
    PARALLEL_EXPERIMENTS = True

DATASETS_ROOT = _resolve_project_path(os.environ.get('RUN_DATASETS_ROOT', 'datasets'))
NUM_WORKERS = int(os.environ.get('RUN_NUM_WORKERS', '4'))

INCLUDE_GROUPS = _env_list('RUN_INCLUDE_GROUPS')
_env_include_prefixes = _env_list('RUN_INCLUDE_GROUP_PREFIXES')
INCLUDE_PREFIXES = _env_include_prefixes if _env_include_prefixes else list(RUN_GROUPS)
EXCLUDE_GROUPS = _env_list('RUN_EXCLUDE_GROUPS')
EXCLUDE_PREFIXES = _env_list('RUN_EXCLUDE_GROUP_PREFIXES')

# 阶段过滤使用逗号分隔的索引；留空时使用实验自身的阶段列表。
_env_stage_filter = _env_list('RUN_STAGE_FILTER')
STAGE_FILTER: Optional[List[int]] = [int(x) for x in _env_stage_filter] if _env_stage_filter else None

OUTPUT_BASE_DIR = _resolve_project_path(os.environ.get(
    'RUN_OUTPUT_DIR',
    os.path.join(_RESULTS_ROOT, 'topodise'),
))
_env_resume_search_dirs = _env_list('RUN_RESUME_SEARCH_DIRS')
RESUME_SEARCH_DIRS = [
    _resolve_search_dir_path(p) for p in _env_resume_search_dirs
] if _env_resume_search_dirs else [OUTPUT_BASE_DIR]
_env_batch_tag = os.environ.get('RUN_BATCH_TAG', '').strip()
if _env_batch_tag:
    BATCH_TAG = _env_batch_tag
else:
    BATCH_TAG = datetime.now().strftime('batch_%Y%m%d_%H%M%S')
OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, BATCH_TAG)


# DATASETS_ROOT 下的数据集 JSON 路径
PD_JSON = os.path.join(DATASETS_ROOT, 'pd_dataset_en', 'patients_dataset_en.json')
CORN1_ROOT = os.path.join(DATASETS_ROOT, 'CORN-1')


# ==================== Stage0/1 默认配方（auto-fallback 用）====================
# 当某个 Stage2-only 实验的父 ckpt 为空 / 不存在时，run_experiment 会自动补跑
# Stage0/1 重新生成父检查点。补跑时若该实验本身没写 stage0/1 override，则使用
# 以下默认配方（统一为 stage/subtype 最佳 c8bf6b548656 配方）。
DEFAULT_STAGE0_OVERRIDES: Dict[str, Any] = {
    'corn1_root': CORN1_ROOT,
    'image_size': 224,
    'encoder_base_channels': 64,
    'unet_model_channels': 64,
    'z_dim': 512,
    'T': 1000,
    'epochs': 200,
    'batch_size': 16,
    'num_workers': NUM_WORKERS,
    'lr': 1e-4,
    'lambda_simple': 1.0,
    'lambda_topo': 0.1,
    'z_dropout': 0.1,
    'topo_target': 'hgrad',
    'reproducibility': 'portable',
    'amp': False,
    'channels_last': False,
    'fused_adamw': False,
}

DEFAULT_STAGE1_OVERRIDES: Dict[str, Any] = {
    'pd_json': PD_JSON,
    'base_path': DATASETS_ROOT,
    'image_size': 224,
    'epochs': 80,
    'batch_size': 16,
    'num_workers': NUM_WORKERS,
    'lr': 2e-5,
    'lambda_region': 0.5,
    'lambda_topo': 0.05,
    'tau_region': 0.2,
    'topo_target': 'hgrad',
    'parent_encoder_weights': 'raw',
    'patient_groups': ('control', 'control_cataract'),
    'reproducibility': 'portable',
    'amp': False,
    'channels_last': False,
    'fused_adamw': False,
}


# ==================== 最佳配置和完整消融配置 ====================
# 直接编辑这些字典；默认 main_tasks，消融和 DM 可通过 RUN_SECTION 或 --section 选择。
RUN_SECTION = 'main_tasks'  # main_tasks / ablation / dm_transfer / all
MAIN_TASKS_EXPERIMENTS = [{'_group': 'PAPER_BINARY_US0_stage2_on_c8bf6b548656',
  '_desc': 'Paper US0 binary Stage2 head on the unified c8bf6b548656 Stage0/1 parent.',
  'dataset': 'pd',
  'stages': [2],
  'stage0_overrides': {},
  'stage1_overrides': {},
  'stage2_overrides': {'pd_json': 'datasets/pd_dataset_en/patients_dataset_en.json',
                       'base_path': 'datasets',
                       'image_size': 224,
                       'epochs': 50,
                       'batch_size': 16,
                       'num_workers': 4,
                       'lr': 0.0005,
                       'adapter_rank': 32,
                       'lambda_ce': 1.0,
                       'lambda_supcon': 1.0,
                       'lambda_recon': 0.05,
                       'tau_supcon': 0.1,
                       'kfold_mode': 'patient',
                       'num_folds': 5,
                       'eval_folds': 5,
                       'scheduler': 'cosine',
                       'early_stopping': True,
                       'patience': 8,
                       'classifier_dropout': 0.35,
                       'label_smooth_eps': 0.05,
                       'mode': 'mil',
                       'pair_mode': 'cartesian',
                       'max_pairs_per_patient': 8,
                       'finetune_fusion': True,
                       'fusion_lr_scale': 0.1,
                       'use_morphology_features': True,
                       'morph_embed_dim': 32,
                       'eval_patient_agg': True,
                       'select_metric': 'patient',
                       'kfold_repeats': 3},
  'stage2_tasks': ['pd_binary'],
  'parent_ckpt_override': None,
  'paper_report_method': 'single_model'},
 {'_group': 'PAPER_STAGE_entropy_routed_U0_U1_U2',
  '_desc': 'PD stage result: label-free per-exam minimum-entropy routing over U0/U1/U2.',
  'dataset': 'pd',
  'stages': [2],
  'stage2_tasks': ['pd_stage'],
  'stage0_overrides': {},
  'stage1_overrides': {},
  'parent_ckpt_override': None,
  'oracle_experiments': [{'_group': 'USTAGELOGITS_U0_stage_lse_nofdiff_anchor_15fold',
                          '_desc': 'USTAGELOGITS U0: stage support-LSE no-fdiff rerun for true patient '
                                   'logits, 5x3',
                          'dataset': 'pd',
                          'stages': [2],
                          'stage2_overrides': {'pd_json': 'datasets/pd_dataset_en/patients_dataset_en.json',
                                               'base_path': 'datasets',
                                               'image_size': 224,
                                               'epochs': 50,
                                               'batch_size': 16,
                                               'num_workers': 4,
                                               'lr': 0.0005,
                                               'adapter_rank': 32,
                                               'lambda_ce': 1.0,
                                               'lambda_supcon': 1.0,
                                               'lambda_recon': 0.05,
                                               'tau_supcon': 0.1,
                                               'kfold_mode': 'patient',
                                               'num_folds': 5,
                                               'eval_folds': 5,
                                               'scheduler': 'cosine',
                                               'early_stopping': True,
                                               'patience': 8,
                                               'classifier_dropout': 0.35,
                                               'label_smooth_eps': 0.05,
                                               'pair_mode': 'cartesian',
                                               'max_pairs_per_patient': 8,
                                               'finetune_fusion': True,
                                               'fusion_lr_scale': 0.1,
                                               'kfold_repeats': 3,
                                               'use_morphology_features': True,
                                               'morph_embed_dim': 32,
                                               'use_feature_diffusion': False,
                                               'fdiff_class_cond': True,
                                               'fdiff_gate_mode': 'soft_cos',
                                               'fdiff_gate_threshold': 0.62,
                                               'fdiff_gate_temperature': 0.12,
                                               'fdiff_T_aug': 100,
                                               'fdiff_k_aug': 1,
                                               'fdiff_start_epoch': 3,
                                               'lambda_fdiff': 0.2,
                                               'lambda_ce_aug': 0.15,
                                               'use_prototype': True,
                                               'proto_dim': 128,
                                               'proto_dropout': 0.65,
                                               'proto_metric': 'euclidean',
                                               'proto_temp': 0.5,
                                               'lambda_proto': 0.8,
                                               'proto_min_per_class': 2,
                                               'proto_eval_blend': 1.0,
                                               'proto_eval_mode': 'support_lse',
                                               'proto_support_k': 5,
                                               'proto_fdiff_support_aug': 0,
                                               'mil_balanced_bags': True,
                                               'mode': 'mil',
                                               'eval_patient_agg': True,
                                               'select_metric': 'patient'},
                          'parent_ckpt_override': None,
                          'stage2_tasks': ['pd_stage']},
                         {'_group': 'USTAGELOGITS_U1_stage_lse_gentle_global_15fold',
                          '_desc': 'USTAGELOGITS U1: stage gentle fdiff global support aug x1 rerun for true '
                                   'patient logits, 5x3',
                          'dataset': 'pd',
                          'stages': [2],
                          'stage2_overrides': {'pd_json': 'datasets/pd_dataset_en/patients_dataset_en.json',
                                               'base_path': 'datasets',
                                               'image_size': 224,
                                               'epochs': 50,
                                               'batch_size': 16,
                                               'num_workers': 4,
                                               'lr': 0.0005,
                                               'adapter_rank': 32,
                                               'lambda_ce': 1.0,
                                               'lambda_supcon': 1.0,
                                               'lambda_recon': 0.05,
                                               'tau_supcon': 0.1,
                                               'kfold_mode': 'patient',
                                               'num_folds': 5,
                                               'eval_folds': 5,
                                               'scheduler': 'cosine',
                                               'early_stopping': True,
                                               'patience': 8,
                                               'classifier_dropout': 0.35,
                                               'label_smooth_eps': 0.05,
                                               'pair_mode': 'cartesian',
                                               'max_pairs_per_patient': 8,
                                               'finetune_fusion': True,
                                               'fusion_lr_scale': 0.1,
                                               'kfold_repeats': 3,
                                               'use_morphology_features': True,
                                               'morph_embed_dim': 32,
                                               'use_feature_diffusion': True,
                                               'fdiff_class_cond': True,
                                               'fdiff_gate_mode': 'soft_cos',
                                               'fdiff_gate_threshold': 0.72,
                                               'fdiff_gate_temperature': 0.08,
                                               'fdiff_T_aug': 45,
                                               'fdiff_k_aug': 1,
                                               'fdiff_start_epoch': 3,
                                               'lambda_fdiff': 0.1,
                                               'lambda_ce_aug': 0.12,
                                               'use_prototype': True,
                                               'proto_dim': 128,
                                               'proto_dropout': 0.65,
                                               'proto_metric': 'euclidean',
                                               'proto_temp': 0.5,
                                               'lambda_proto': 0.8,
                                               'proto_min_per_class': 2,
                                               'proto_eval_blend': 1.0,
                                               'proto_eval_mode': 'support_lse',
                                               'proto_support_k': 5,
                                               'proto_fdiff_support_aug': 1,
                                               'mil_balanced_bags': True,
                                               'mode': 'mil',
                                               'eval_patient_agg': True,
                                               'select_metric': 'patient'},
                          'parent_ckpt_override': None,
                          'stage2_tasks': ['pd_stage']},
                         {'_group': 'USTAGELOGITS_U2_stage_lse_mid_global_15fold',
                          '_desc': 'USTAGELOGITS U2: stage mid fdiff global support aug x2 rerun for true '
                                   'patient logits, 5x3',
                          'dataset': 'pd',
                          'stages': [2],
                          'stage2_overrides': {'pd_json': 'datasets/pd_dataset_en/patients_dataset_en.json',
                                               'base_path': 'datasets',
                                               'image_size': 224,
                                               'epochs': 50,
                                               'batch_size': 16,
                                               'num_workers': 4,
                                               'lr': 0.0005,
                                               'adapter_rank': 32,
                                               'lambda_ce': 1.0,
                                               'lambda_supcon': 1.0,
                                               'lambda_recon': 0.05,
                                               'tau_supcon': 0.1,
                                               'kfold_mode': 'patient',
                                               'num_folds': 5,
                                               'eval_folds': 5,
                                               'scheduler': 'cosine',
                                               'early_stopping': True,
                                               'patience': 8,
                                               'classifier_dropout': 0.35,
                                               'label_smooth_eps': 0.05,
                                               'pair_mode': 'cartesian',
                                               'max_pairs_per_patient': 8,
                                               'finetune_fusion': True,
                                               'fusion_lr_scale': 0.1,
                                               'kfold_repeats': 3,
                                               'use_morphology_features': True,
                                               'morph_embed_dim': 32,
                                               'use_feature_diffusion': True,
                                               'fdiff_class_cond': True,
                                               'fdiff_gate_mode': 'soft_cos',
                                               'fdiff_gate_threshold': 0.62,
                                               'fdiff_gate_temperature': 0.12,
                                               'fdiff_T_aug': 80,
                                               'fdiff_k_aug': 2,
                                               'fdiff_start_epoch': 3,
                                               'lambda_fdiff': 0.16,
                                               'lambda_ce_aug': 0.18,
                                               'use_prototype': True,
                                               'proto_dim': 128,
                                               'proto_dropout': 0.65,
                                               'proto_metric': 'euclidean',
                                               'proto_temp': 0.5,
                                               'lambda_proto': 0.8,
                                               'proto_min_per_class': 2,
                                               'proto_eval_blend': 1.0,
                                               'proto_eval_mode': 'support_lse',
                                               'proto_support_k': 5,
                                               'proto_fdiff_support_aug': 2,
                                               'mil_balanced_bags': True,
                                               'mode': 'mil',
                                               'eval_patient_agg': True,
                                               'select_metric': 'patient'},
                          'parent_ckpt_override': None,
                          'stage2_tasks': ['pd_stage']}],
  'oracle_mode': 'standard',
  'oracle_select_field': 'best_patient_acc',
  'auto_patient_logit_ensemble': True,
  'paper_report_method': 'min_entropy_routing'},
 {'_group': 'PAPER_SUBTYPE_majority_TDREST0_1_2_5_7',
  '_desc': 'Paper subtype result: label-free majority vote over TDREST0/1/2/5/7.',
  'dataset': 'pd',
  'stages': [2],
  'stage2_tasks': ['pd_subtype_tdrest'],
  'stage0_overrides': {},
  'stage1_overrides': {},
  'parent_ckpt_override': None,
  'oracle_experiments': [{'_group': 'TDREST15LOGITS_TDREST0_focal_g3_anchor_15fold',
                          '_desc': 'TDREST15LOGITS TDREST0: class weights + focal gamma3 no-fdiff anchor, '
                                   '5x3',
                          'dataset': 'pd',
                          'stages': [2],
                          'stage2_overrides': {'pd_json': 'datasets/pd_dataset_en/patients_dataset_en.json',
                                               'base_path': 'datasets',
                                               'image_size': 224,
                                               'epochs': 50,
                                               'batch_size': 16,
                                               'num_workers': 4,
                                               'lr': 0.0005,
                                               'adapter_rank': 32,
                                               'lambda_ce': 1.0,
                                               'lambda_supcon': 1.0,
                                               'lambda_recon': 0.05,
                                               'tau_supcon': 0.1,
                                               'kfold_mode': 'patient',
                                               'num_folds': 5,
                                               'eval_folds': 5,
                                               'scheduler': 'cosine',
                                               'early_stopping': True,
                                               'patience': 8,
                                               'classifier_dropout': 0.35,
                                               'label_smooth_eps': 0.05,
                                               'pair_mode': 'cartesian',
                                               'max_pairs_per_patient': 8,
                                               'kfold_repeats': 3,
                                               'use_class_weights': True,
                                               'use_focal': True,
                                               'focal_gamma': 3.0,
                                               'mode': 'mil',
                                               'eval_patient_agg': True,
                                               'select_metric': 'patient'},
                          'parent_ckpt_override': None,
                          'stage2_tasks': ['pd_subtype_tdrest']},
                         {'_group': 'TDREST15LOGITS_TDREST1_focal_g2_anchor_15fold',
                          '_desc': 'TDREST15LOGITS TDREST1: softer focal gamma2 anchor, 5x3',
                          'dataset': 'pd',
                          'stages': [2],
                          'stage2_overrides': {'pd_json': 'datasets/pd_dataset_en/patients_dataset_en.json',
                                               'base_path': 'datasets',
                                               'image_size': 224,
                                               'epochs': 50,
                                               'batch_size': 16,
                                               'num_workers': 4,
                                               'lr': 0.0005,
                                               'adapter_rank': 32,
                                               'lambda_ce': 1.0,
                                               'lambda_supcon': 1.0,
                                               'lambda_recon': 0.05,
                                               'tau_supcon': 0.1,
                                               'kfold_mode': 'patient',
                                               'num_folds': 5,
                                               'eval_folds': 5,
                                               'scheduler': 'cosine',
                                               'early_stopping': True,
                                               'patience': 8,
                                               'classifier_dropout': 0.35,
                                               'label_smooth_eps': 0.05,
                                               'pair_mode': 'cartesian',
                                               'max_pairs_per_patient': 8,
                                               'kfold_repeats': 3,
                                               'use_class_weights': True,
                                               'use_focal': True,
                                               'focal_gamma': 2.0,
                                               'mode': 'mil',
                                               'eval_patient_agg': True,
                                               'select_metric': 'patient'},
                          'parent_ckpt_override': None,
                          'stage2_tasks': ['pd_subtype_tdrest']},
                         {'_group': 'TDREST15LOGITS_TDREST2_mid_fdiff_patient_15fold',
                          '_desc': 'TDREST15LOGITS TDREST2: moderate all-class fdiff with patient selection, '
                                   '5x3',
                          'dataset': 'pd',
                          'stages': [2],
                          'stage2_overrides': {'pd_json': 'datasets/pd_dataset_en/patients_dataset_en.json',
                                               'base_path': 'datasets',
                                               'image_size': 224,
                                               'epochs': 50,
                                               'batch_size': 16,
                                               'num_workers': 4,
                                               'lr': 0.0005,
                                               'adapter_rank': 32,
                                               'lambda_ce': 1.0,
                                               'lambda_supcon': 1.0,
                                               'lambda_recon': 0.05,
                                               'tau_supcon': 0.1,
                                               'kfold_mode': 'patient',
                                               'num_folds': 5,
                                               'eval_folds': 5,
                                               'scheduler': 'cosine',
                                               'early_stopping': True,
                                               'patience': 8,
                                               'classifier_dropout': 0.35,
                                               'label_smooth_eps': 0.05,
                                               'pair_mode': 'cartesian',
                                               'max_pairs_per_patient': 8,
                                               'kfold_repeats': 3,
                                               'use_feature_diffusion': True,
                                               'fdiff_class_cond': True,
                                               'fdiff_gate_mode': 'soft_cos',
                                               'fdiff_gate_threshold': 0.68,
                                               'fdiff_gate_temperature': 0.12,
                                               'fdiff_T_aug': 90,
                                               'fdiff_k_aug': 2,
                                               'fdiff_start_epoch': 3,
                                               'lambda_fdiff': 0.15,
                                               'lambda_ce_aug': 0.2,
                                               'use_class_weights': True,
                                               'mode': 'mil',
                                               'eval_patient_agg': True,
                                               'select_metric': 'patient'},
                          'parent_ckpt_override': None,
                          'stage2_tasks': ['pd_subtype_tdrest']},
                         {'_group': 'TDREST15LOGITS_TDREST5_minor_fdiff_focal_g15_balanced_15fold',
                          '_desc': 'TDREST15LOGITS TDREST5: minority fdiff + focal gamma1.5 + balanced '
                                   'selection, 5x3',
                          'dataset': 'pd',
                          'stages': [2],
                          'stage2_overrides': {'pd_json': 'datasets/pd_dataset_en/patients_dataset_en.json',
                                               'base_path': 'datasets',
                                               'image_size': 224,
                                               'epochs': 50,
                                               'batch_size': 16,
                                               'num_workers': 4,
                                               'lr': 0.0005,
                                               'adapter_rank': 32,
                                               'lambda_ce': 1.0,
                                               'lambda_supcon': 1.0,
                                               'lambda_recon': 0.05,
                                               'tau_supcon': 0.1,
                                               'kfold_mode': 'patient',
                                               'num_folds': 5,
                                               'eval_folds': 5,
                                               'scheduler': 'cosine',
                                               'early_stopping': True,
                                               'patience': 8,
                                               'classifier_dropout': 0.35,
                                               'label_smooth_eps': 0.05,
                                               'pair_mode': 'cartesian',
                                               'max_pairs_per_patient': 8,
                                               'kfold_repeats': 3,
                                               'use_feature_diffusion': True,
                                               'fdiff_class_cond': True,
                                               'fdiff_gate_mode': 'soft_cos',
                                               'fdiff_gate_threshold': 0.7,
                                               'fdiff_gate_temperature': 0.1,
                                               'fdiff_T_aug': 70,
                                               'fdiff_k_aug': 2,
                                               'fdiff_start_epoch': 3,
                                               'lambda_fdiff': 0.12,
                                               'lambda_ce_aug': 0.18,
                                               'fdiff_minority_only': True,
                                               'use_class_weights': True,
                                               'use_focal': True,
                                               'focal_gamma': 1.5,
                                               'select_metric': 'balanced_patient',
                                               'mode': 'mil',
                                               'eval_patient_agg': True},
                          'parent_ckpt_override': None,
                          'stage2_tasks': ['pd_subtype_tdrest']},
                         {'_group': 'TDREST15LOGITS_TDREST7_mid_tdweight15_balanced_15fold',
                          '_desc': 'TDREST15LOGITS TDREST7: mid fdiff + TD class weight x1.5 + balanced '
                                   'selection, 5x3',
                          'dataset': 'pd',
                          'stages': [2],
                          'stage2_overrides': {'pd_json': 'datasets/pd_dataset_en/patients_dataset_en.json',
                                               'base_path': 'datasets',
                                               'image_size': 224,
                                               'epochs': 50,
                                               'batch_size': 16,
                                               'num_workers': 4,
                                               'lr': 0.0005,
                                               'adapter_rank': 32,
                                               'lambda_ce': 1.0,
                                               'lambda_supcon': 1.0,
                                               'lambda_recon': 0.05,
                                               'tau_supcon': 0.1,
                                               'kfold_mode': 'patient',
                                               'num_folds': 5,
                                               'eval_folds': 5,
                                               'scheduler': 'cosine',
                                               'early_stopping': True,
                                               'patience': 8,
                                               'classifier_dropout': 0.35,
                                               'label_smooth_eps': 0.05,
                                               'pair_mode': 'cartesian',
                                               'max_pairs_per_patient': 8,
                                               'kfold_repeats': 3,
                                               'use_feature_diffusion': True,
                                               'fdiff_class_cond': True,
                                               'fdiff_gate_mode': 'soft_cos',
                                               'fdiff_gate_threshold': 0.68,
                                               'fdiff_gate_temperature': 0.12,
                                               'fdiff_T_aug': 90,
                                               'fdiff_k_aug': 2,
                                               'fdiff_start_epoch': 3,
                                               'lambda_fdiff': 0.15,
                                               'lambda_ce_aug': 0.2,
                                               'use_class_weights': True,
                                               'class_weight_multipliers': [1.5, 1.0],
                                               'select_metric': 'balanced_patient',
                                               'mode': 'mil',
                                               'eval_patient_agg': True},
                          'parent_ckpt_override': None,
                          'stage2_tasks': ['pd_subtype_tdrest']}],
  'oracle_mode': 'standard',
  'oracle_select_field': 'best_patient_acc',
  'auto_patient_logit_ensemble': True,
  'paper_report_method': 'majority_vote_tie_avg_prob'}]

ABLATION_EXPERIMENTS = [{'_group': 'ABLTOPO_Full_binary_5f',
  '_desc': 'Topology representation anchor: full US0 Stage0/Stage1 with binary downstream and probe-ready '
           'checkpoints',
  'dataset': 'pd',
  'stages': [0, 1, 2],
  'stage0_overrides': {'corn1_root': 'datasets/CORN-1',
                       'image_size': 224,
                       'encoder_base_channels': 64,
                       'unet_model_channels': 64,
                       'z_dim': 512,
                       'T': 1000,
                       'epochs': 200,
                       'batch_size': 16,
                       'num_workers': 4,
                       'lr': 0.0001,
                       'lambda_simple': 1.0,
                       'lambda_topo': 0.1,
                       'z_dropout': 0.1,
                       'topo_target': 'gt'},
  'stage1_overrides': {'pd_json': 'datasets/pd_dataset_en/patients_dataset_en.json',
                       'base_path': 'datasets',
                       'image_size': 224,
                       'epochs': 80,
                       'batch_size': 16,
                       'num_workers': 4,
                       'lr': 2e-05,
                       'lambda_region': 0.5,
                       'lambda_topo': 0.05,
                       'tau_region': 0.2,
                       'region_pair_manifest': 'datasets/unlabeled_dataset_en/unlabeled_dataset_en.json',
                       'topo_target': 'frangi'},
  'stage2_overrides': {'pd_json': 'datasets/pd_dataset_en/patients_dataset_en.json',
                       'base_path': 'datasets',
                       'image_size': 224,
                       'epochs': 50,
                       'batch_size': 16,
                       'num_workers': 4,
                       'lr': 0.0005,
                       'adapter_rank': 32,
                       'lambda_ce': 1.0,
                       'lambda_supcon': 1.0,
                       'lambda_recon': 0.05,
                       'tau_supcon': 0.1,
                       'kfold_mode': 'patient',
                       'num_folds': 5,
                       'eval_folds': 5,
                       'scheduler': 'cosine',
                       'early_stopping': True,
                       'patience': 8,
                       'classifier_dropout': 0.35,
                       'label_smooth_eps': 0.05,
                       'pair_mode': 'cartesian',
                       'max_pairs_per_patient': 8,
                       'finetune_fusion': True,
                       'fusion_lr_scale': 0.1,
                       'kfold_repeats': 1,
                       'use_morphology_features': True,
                       'morph_embed_dim': 32,
                       'mode': 'mil',
                       'eval_patient_agg': True,
                       'select_metric': 'patient'},
  'stage2_tasks': ['pd_binary'],
  'parent_ckpt_override': None},
 {'_group': 'ABLTOPO_NoDiff_binary_5f',
  '_desc': 'Topology representation ablation: remove Stage0 diffusion reconstruction loss',
  'dataset': 'pd',
  'stages': [0, 1, 2],
  'stage0_overrides': {'corn1_root': 'datasets/CORN-1',
                       'image_size': 224,
                       'encoder_base_channels': 64,
                       'unet_model_channels': 64,
                       'z_dim': 512,
                       'T': 1000,
                       'epochs': 200,
                       'batch_size': 16,
                       'num_workers': 4,
                       'lr': 0.0001,
                       'lambda_simple': 0.0,
                       'lambda_topo': 0.1,
                       'z_dropout': 0.1,
                       'topo_target': 'gt'},
  'stage1_overrides': {'pd_json': 'datasets/pd_dataset_en/patients_dataset_en.json',
                       'base_path': 'datasets',
                       'image_size': 224,
                       'epochs': 80,
                       'batch_size': 16,
                       'num_workers': 4,
                       'lr': 2e-05,
                       'lambda_region': 0.5,
                       'lambda_topo': 0.05,
                       'tau_region': 0.2,
                       'region_pair_manifest': 'datasets/unlabeled_dataset_en/unlabeled_dataset_en.json',
                       'topo_target': 'frangi'},
  'stage2_overrides': {'pd_json': 'datasets/pd_dataset_en/patients_dataset_en.json',
                       'base_path': 'datasets',
                       'image_size': 224,
                       'epochs': 50,
                       'batch_size': 16,
                       'num_workers': 4,
                       'lr': 0.0005,
                       'adapter_rank': 32,
                       'lambda_ce': 1.0,
                       'lambda_supcon': 1.0,
                       'lambda_recon': 0.05,
                       'tau_supcon': 0.1,
                       'kfold_mode': 'patient',
                       'num_folds': 5,
                       'eval_folds': 5,
                       'scheduler': 'cosine',
                       'early_stopping': True,
                       'patience': 8,
                       'classifier_dropout': 0.35,
                       'label_smooth_eps': 0.05,
                       'pair_mode': 'cartesian',
                       'max_pairs_per_patient': 8,
                       'finetune_fusion': True,
                       'fusion_lr_scale': 0.1,
                       'kfold_repeats': 1,
                       'use_morphology_features': True,
                       'morph_embed_dim': 32,
                       'mode': 'mil',
                       'eval_patient_agg': True,
                       'select_metric': 'patient'},
  'stage2_tasks': ['pd_binary'],
  'parent_ckpt_override': None},
 {'_group': 'ABLTOPO_NoTopo_binary_5f',
  '_desc': 'Topology representation ablation: remove Stage0 topology supervision',
  'dataset': 'pd',
  'stages': [0, 1, 2],
  'stage0_overrides': {'corn1_root': 'datasets/CORN-1',
                       'image_size': 224,
                       'encoder_base_channels': 64,
                       'unet_model_channels': 64,
                       'z_dim': 512,
                       'T': 1000,
                       'epochs': 200,
                       'batch_size': 16,
                       'num_workers': 4,
                       'lr': 0.0001,
                       'lambda_simple': 1.0,
                       'lambda_topo': 0.0,
                       'z_dropout': 0.1,
                       'topo_target': 'gt'},
  'stage1_overrides': {'pd_json': 'datasets/pd_dataset_en/patients_dataset_en.json',
                       'base_path': 'datasets',
                       'image_size': 224,
                       'epochs': 80,
                       'batch_size': 16,
                       'num_workers': 4,
                       'lr': 2e-05,
                       'lambda_region': 0.5,
                       'lambda_topo': 0.05,
                       'tau_region': 0.2,
                       'region_pair_manifest': 'datasets/unlabeled_dataset_en/unlabeled_dataset_en.json',
                       'topo_target': 'frangi'},
  'stage2_overrides': {'pd_json': 'datasets/pd_dataset_en/patients_dataset_en.json',
                       'base_path': 'datasets',
                       'image_size': 224,
                       'epochs': 50,
                       'batch_size': 16,
                       'num_workers': 4,
                       'lr': 0.0005,
                       'adapter_rank': 32,
                       'lambda_ce': 1.0,
                       'lambda_supcon': 1.0,
                       'lambda_recon': 0.05,
                       'tau_supcon': 0.1,
                       'kfold_mode': 'patient',
                       'num_folds': 5,
                       'eval_folds': 5,
                       'scheduler': 'cosine',
                       'early_stopping': True,
                       'patience': 8,
                       'classifier_dropout': 0.35,
                       'label_smooth_eps': 0.05,
                       'pair_mode': 'cartesian',
                       'max_pairs_per_patient': 8,
                       'finetune_fusion': True,
                       'fusion_lr_scale': 0.1,
                       'kfold_repeats': 1,
                       'use_morphology_features': True,
                       'morph_embed_dim': 32,
                       'mode': 'mil',
                       'eval_patient_agg': True,
                       'select_metric': 'patient'},
  'stage2_tasks': ['pd_binary'],
  'parent_ckpt_override': None},
 {'_group': 'ABLTOPO_Stage0OnlyFull_binary_5f',
  '_desc': 'Topology representation ablation: full Stage0 directly to Stage2, no Stage1 RSCP confound',
  'dataset': 'pd',
  'stages': [0, 2],
  'stage0_overrides': {'topo_target': 'gt'},
  'stage2_overrides': {'pd_json': 'datasets/pd_dataset_en/patients_dataset_en.json',
                       'base_path': 'datasets',
                       'image_size': 224,
                       'epochs': 50,
                       'batch_size': 16,
                       'num_workers': 4,
                       'lr': 0.0005,
                       'adapter_rank': 32,
                       'lambda_ce': 1.0,
                       'lambda_supcon': 1.0,
                       'lambda_recon': 0.05,
                       'tau_supcon': 0.1,
                       'kfold_mode': 'patient',
                       'num_folds': 5,
                       'eval_folds': 5,
                       'scheduler': 'cosine',
                       'early_stopping': True,
                       'patience': 8,
                       'classifier_dropout': 0.35,
                       'label_smooth_eps': 0.05,
                       'pair_mode': 'cartesian',
                       'max_pairs_per_patient': 8,
                       'finetune_fusion': True,
                       'fusion_lr_scale': 0.1,
                       'kfold_repeats': 1,
                       'use_morphology_features': True,
                       'morph_embed_dim': 32,
                       'mode': 'mil',
                       'eval_patient_agg': True,
                       'select_metric': 'patient'},
  'stage2_tasks': ['pd_binary'],
  'parent_ckpt_override': None},
 {'_group': 'ABLTOPO_Stage1NoTopo_binary_5f',
  '_desc': 'Topology representation ablation: remove Stage1 topology regularization after full Stage0',
  'dataset': 'pd',
  'stages': [0, 1, 2],
  'stage0_overrides': {'corn1_root': 'datasets/CORN-1',
                       'image_size': 224,
                       'encoder_base_channels': 64,
                       'unet_model_channels': 64,
                       'z_dim': 512,
                       'T': 1000,
                       'epochs': 200,
                       'batch_size': 16,
                       'num_workers': 4,
                       'lr': 0.0001,
                       'lambda_simple': 1.0,
                       'lambda_topo': 0.1,
                       'z_dropout': 0.1,
                       'topo_target': 'gt'},
  'stage1_overrides': {'pd_json': 'datasets/pd_dataset_en/patients_dataset_en.json',
                       'base_path': 'datasets',
                       'image_size': 224,
                       'epochs': 80,
                       'batch_size': 16,
                       'num_workers': 4,
                       'lr': 2e-05,
                       'lambda_region': 0.5,
                       'lambda_topo': 0.0,
                       'tau_region': 0.2,
                       'region_pair_manifest': 'datasets/unlabeled_dataset_en/unlabeled_dataset_en.json',
                       'topo_target': 'frangi'},
  'stage2_overrides': {'pd_json': 'datasets/pd_dataset_en/patients_dataset_en.json',
                       'base_path': 'datasets',
                       'image_size': 224,
                       'epochs': 50,
                       'batch_size': 16,
                       'num_workers': 4,
                       'lr': 0.0005,
                       'adapter_rank': 32,
                       'lambda_ce': 1.0,
                       'lambda_supcon': 1.0,
                       'lambda_recon': 0.05,
                       'tau_supcon': 0.1,
                       'kfold_mode': 'patient',
                       'num_folds': 5,
                       'eval_folds': 5,
                       'scheduler': 'cosine',
                       'early_stopping': True,
                       'patience': 8,
                       'classifier_dropout': 0.35,
                       'label_smooth_eps': 0.05,
                       'pair_mode': 'cartesian',
                       'max_pairs_per_patient': 8,
                       'finetune_fusion': True,
                       'fusion_lr_scale': 0.1,
                       'kfold_repeats': 1,
                       'use_morphology_features': True,
                       'morph_embed_dim': 32,
                       'mode': 'mil',
                       'eval_patient_agg': True,
                       'select_metric': 'patient'},
  'stage2_tasks': ['pd_binary'],
  'parent_ckpt_override': None},
 {'_group': 'ABLTOPO_TopoFrangi_binary_5f',
  '_desc': 'Topology representation ablation: Stage0 topology target changed to Frangi vesselness proxy',
  'dataset': 'pd',
  'stages': [0, 1, 2],
  'stage0_overrides': {'corn1_root': 'datasets/CORN-1',
                       'image_size': 224,
                       'encoder_base_channels': 64,
                       'unet_model_channels': 64,
                       'z_dim': 512,
                       'T': 1000,
                       'epochs': 200,
                       'batch_size': 16,
                       'num_workers': 4,
                       'lr': 0.0001,
                       'lambda_simple': 1.0,
                       'lambda_topo': 0.1,
                       'z_dropout': 0.1,
                       'topo_target': 'frangi'},
  'stage1_overrides': {'pd_json': 'datasets/pd_dataset_en/patients_dataset_en.json',
                       'base_path': 'datasets',
                       'image_size': 224,
                       'epochs': 80,
                       'batch_size': 16,
                       'num_workers': 4,
                       'lr': 2e-05,
                       'lambda_region': 0.5,
                       'lambda_topo': 0.05,
                       'tau_region': 0.2,
                       'region_pair_manifest': 'datasets/unlabeled_dataset_en/unlabeled_dataset_en.json',
                       'topo_target': 'frangi'},
  'stage2_overrides': {'pd_json': 'datasets/pd_dataset_en/patients_dataset_en.json',
                       'base_path': 'datasets',
                       'image_size': 224,
                       'epochs': 50,
                       'batch_size': 16,
                       'num_workers': 4,
                       'lr': 0.0005,
                       'adapter_rank': 32,
                       'lambda_ce': 1.0,
                       'lambda_supcon': 1.0,
                       'lambda_recon': 0.05,
                       'tau_supcon': 0.1,
                       'kfold_mode': 'patient',
                       'num_folds': 5,
                       'eval_folds': 5,
                       'scheduler': 'cosine',
                       'early_stopping': True,
                       'patience': 8,
                       'classifier_dropout': 0.35,
                       'label_smooth_eps': 0.05,
                       'pair_mode': 'cartesian',
                       'max_pairs_per_patient': 8,
                       'finetune_fusion': True,
                       'fusion_lr_scale': 0.1,
                       'kfold_repeats': 1,
                       'use_morphology_features': True,
                       'morph_embed_dim': 32,
                       'mode': 'mil',
                       'eval_patient_agg': True,
                       'select_metric': 'patient'},
  'stage2_tasks': ['pd_binary'],
  'parent_ckpt_override': None},
 {'_group': 'ABLTOPO_TopoHGrad_binary_5f',
  '_desc': 'Topology representation ablation: Stage0 topology target changed to high-gradient proxy',
  'dataset': 'pd',
  'stages': [0, 1, 2],
  'stage0_overrides': {'corn1_root': 'datasets/CORN-1',
                       'image_size': 224,
                       'encoder_base_channels': 64,
                       'unet_model_channels': 64,
                       'z_dim': 512,
                       'T': 1000,
                       'epochs': 200,
                       'batch_size': 16,
                       'num_workers': 4,
                       'lr': 0.0001,
                       'lambda_simple': 1.0,
                       'lambda_topo': 0.1,
                       'z_dropout': 0.1,
                       'topo_target': 'hgrad'},
  'stage1_overrides': {'pd_json': 'datasets/pd_dataset_en/patients_dataset_en.json',
                       'base_path': 'datasets',
                       'image_size': 224,
                       'epochs': 80,
                       'batch_size': 16,
                       'num_workers': 4,
                       'lr': 2e-05,
                       'lambda_region': 0.5,
                       'lambda_topo': 0.05,
                       'tau_region': 0.2,
                       'region_pair_manifest': 'datasets/unlabeled_dataset_en/unlabeled_dataset_en.json',
                       'topo_target': 'frangi'},
  'stage2_overrides': {'pd_json': 'datasets/pd_dataset_en/patients_dataset_en.json',
                       'base_path': 'datasets',
                       'image_size': 224,
                       'epochs': 50,
                       'batch_size': 16,
                       'num_workers': 4,
                       'lr': 0.0005,
                       'adapter_rank': 32,
                       'lambda_ce': 1.0,
                       'lambda_supcon': 1.0,
                       'lambda_recon': 0.05,
                       'tau_supcon': 0.1,
                       'kfold_mode': 'patient',
                       'num_folds': 5,
                       'eval_folds': 5,
                       'scheduler': 'cosine',
                       'early_stopping': True,
                       'patience': 8,
                       'classifier_dropout': 0.35,
                       'label_smooth_eps': 0.05,
                       'pair_mode': 'cartesian',
                       'max_pairs_per_patient': 8,
                       'finetune_fusion': True,
                       'fusion_lr_scale': 0.1,
                       'kfold_repeats': 1,
                       'use_morphology_features': True,
                       'morph_embed_dim': 32,
                       'mode': 'mil',
                       'eval_patient_agg': True,
                       'select_metric': 'patient'},
  'stage2_tasks': ['pd_binary'],
  'parent_ckpt_override': None},
 {'_group': 'ABLTOPO_TopoSobel_binary_5f',
  '_desc': 'Topology representation ablation: Stage0 topology target changed to Sobel edge proxy',
  'dataset': 'pd',
  'stages': [0, 1, 2],
  'stage0_overrides': {'corn1_root': 'datasets/CORN-1',
                       'image_size': 224,
                       'encoder_base_channels': 64,
                       'unet_model_channels': 64,
                       'z_dim': 512,
                       'T': 1000,
                       'epochs': 200,
                       'batch_size': 16,
                       'num_workers': 4,
                       'lr': 0.0001,
                       'lambda_simple': 1.0,
                       'lambda_topo': 0.1,
                       'z_dropout': 0.1,
                       'topo_target': 'sobel'},
  'stage1_overrides': {'pd_json': 'datasets/pd_dataset_en/patients_dataset_en.json',
                       'base_path': 'datasets',
                       'image_size': 224,
                       'epochs': 80,
                       'batch_size': 16,
                       'num_workers': 4,
                       'lr': 2e-05,
                       'lambda_region': 0.5,
                       'lambda_topo': 0.05,
                       'tau_region': 0.2,
                       'region_pair_manifest': 'datasets/unlabeled_dataset_en/unlabeled_dataset_en.json',
                       'topo_target': 'frangi'},
  'stage2_overrides': {'pd_json': 'datasets/pd_dataset_en/patients_dataset_en.json',
                       'base_path': 'datasets',
                       'image_size': 224,
                       'epochs': 50,
                       'batch_size': 16,
                       'num_workers': 4,
                       'lr': 0.0005,
                       'adapter_rank': 32,
                       'lambda_ce': 1.0,
                       'lambda_supcon': 1.0,
                       'lambda_recon': 0.05,
                       'tau_supcon': 0.1,
                       'kfold_mode': 'patient',
                       'num_folds': 5,
                       'eval_folds': 5,
                       'scheduler': 'cosine',
                       'early_stopping': True,
                       'patience': 8,
                       'classifier_dropout': 0.35,
                       'label_smooth_eps': 0.05,
                       'pair_mode': 'cartesian',
                       'max_pairs_per_patient': 8,
                       'finetune_fusion': True,
                       'fusion_lr_scale': 0.1,
                       'kfold_repeats': 1,
                       'use_morphology_features': True,
                       'morph_embed_dim': 32,
                       'mode': 'mil',
                       'eval_patient_agg': True,
                       'select_metric': 'patient'},
  'stage2_tasks': ['pd_binary'],
  'parent_ckpt_override': None},
 {'_group': 'ABLBIN_NoRegion_binary_5f',
  '_desc': 'Binary pretrain ablation: Stage1 without region InfoNCE',
  'dataset': 'pd',
  'stages': [0, 1, 2],
  'stage0_overrides': {'corn1_root': 'datasets/CORN-1',
                       'image_size': 224,
                       'encoder_base_channels': 64,
                       'unet_model_channels': 64,
                       'z_dim': 512,
                       'T': 1000,
                       'epochs': 200,
                       'batch_size': 16,
                       'num_workers': 4,
                       'lr': 0.0001,
                       'lambda_simple': 1.0,
                       'lambda_topo': 0.1,
                       'z_dropout': 0.1,
                       'topo_target': 'gt'},
  'stage1_overrides': {'pd_json': 'datasets/pd_dataset_en/patients_dataset_en.json',
                       'base_path': 'datasets',
                       'image_size': 224,
                       'epochs': 80,
                       'batch_size': 16,
                       'num_workers': 4,
                       'lr': 2e-05,
                       'lambda_region': 0.0,
                       'lambda_topo': 0.05,
                       'tau_region': 0.2,
                       'region_pair_manifest': 'datasets/unlabeled_dataset_en/unlabeled_dataset_en.json',
                       'topo_target': 'frangi'},
  'stage2_overrides': {'pd_json': 'datasets/pd_dataset_en/patients_dataset_en.json',
                       'base_path': 'datasets',
                       'image_size': 224,
                       'epochs': 50,
                       'batch_size': 16,
                       'num_workers': 4,
                       'lr': 0.0005,
                       'adapter_rank': 32,
                       'lambda_ce': 1.0,
                       'lambda_supcon': 1.0,
                       'lambda_recon': 0.05,
                       'tau_supcon': 0.1,
                       'kfold_mode': 'patient',
                       'num_folds': 5,
                       'eval_folds': 5,
                       'scheduler': 'cosine',
                       'early_stopping': True,
                       'patience': 8,
                       'classifier_dropout': 0.35,
                       'label_smooth_eps': 0.05,
                       'pair_mode': 'cartesian',
                       'max_pairs_per_patient': 8,
                       'finetune_fusion': True,
                       'fusion_lr_scale': 0.1,
                       'kfold_repeats': 1,
                       'use_morphology_features': True,
                       'morph_embed_dim': 32,
                       'mode': 'mil',
                       'eval_patient_agg': True,
                       'select_metric': 'patient'},
  'stage2_tasks': ['pd_binary'],
  'parent_ckpt_override': None},
 {'_group': 'ABLBIN_Stage1DenoiserOn_binary_5f',
  '_desc': 'Binary pretrain ablation: Stage1 enables denoiser gradient',
  'dataset': 'pd',
  'stages': [0, 1, 2],
  'stage0_overrides': {'corn1_root': 'datasets/CORN-1',
                       'image_size': 224,
                       'encoder_base_channels': 64,
                       'unet_model_channels': 64,
                       'z_dim': 512,
                       'T': 1000,
                       'epochs': 200,
                       'batch_size': 16,
                       'num_workers': 4,
                       'lr': 0.0001,
                       'lambda_simple': 1.0,
                       'lambda_topo': 0.1,
                       'z_dropout': 0.1,
                       'topo_target': 'gt'},
  'stage1_overrides': {'pd_json': 'datasets/pd_dataset_en/patients_dataset_en.json',
                       'base_path': 'datasets',
                       'image_size': 224,
                       'epochs': 80,
                       'batch_size': 16,
                       'num_workers': 4,
                       'lr': 2e-05,
                       'lambda_region': 0.5,
                       'lambda_topo': 0.05,
                       'tau_region': 0.2,
                       'region_pair_manifest': 'datasets/unlabeled_dataset_en/unlabeled_dataset_en.json',
                       'topo_target': 'frangi',
                       'train_denoiser': True,
                       'lambda_simple': 0.5},
  'stage2_overrides': {'pd_json': 'datasets/pd_dataset_en/patients_dataset_en.json',
                       'base_path': 'datasets',
                       'image_size': 224,
                       'epochs': 50,
                       'batch_size': 16,
                       'num_workers': 4,
                       'lr': 0.0005,
                       'adapter_rank': 32,
                       'lambda_ce': 1.0,
                       'lambda_supcon': 1.0,
                       'lambda_recon': 0.05,
                       'tau_supcon': 0.1,
                       'kfold_mode': 'patient',
                       'num_folds': 5,
                       'eval_folds': 5,
                       'scheduler': 'cosine',
                       'early_stopping': True,
                       'patience': 8,
                       'classifier_dropout': 0.35,
                       'label_smooth_eps': 0.05,
                       'pair_mode': 'cartesian',
                       'max_pairs_per_patient': 8,
                       'finetune_fusion': True,
                       'fusion_lr_scale': 0.1,
                       'kfold_repeats': 1,
                       'use_morphology_features': True,
                       'morph_embed_dim': 32,
                       'mode': 'mil',
                       'eval_patient_agg': True,
                       'select_metric': 'patient'},
  'stage2_tasks': ['pd_binary'],
  'parent_ckpt_override': None},
 {'_group': 'ABLFINE_NoDiffStageU1_stage_5f',
  '_desc': 'Fine-grained pretraining ablation: remove Stage0 diffusion reconstruction, then evaluate '
           'representative U1 support-LSE stage config',
  'dataset': 'pd',
  'stages': [2],
  'stage0_overrides': {'corn1_root': 'datasets/CORN-1',
                       'image_size': 224,
                       'encoder_base_channels': 64,
                       'unet_model_channels': 64,
                       'z_dim': 512,
                       'T': 1000,
                       'epochs': 200,
                       'batch_size': 16,
                       'num_workers': 4,
                       'lr': 0.0001,
                       'lambda_simple': 0.0,
                       'lambda_topo': 0.1,
                       'z_dropout': 0.1,
                       'topo_target': 'gt'},
  'stage1_overrides': {'pd_json': 'datasets/pd_dataset_en/patients_dataset_en.json',
                       'base_path': 'datasets',
                       'image_size': 224,
                       'epochs': 80,
                       'batch_size': 16,
                       'num_workers': 4,
                       'lr': 2e-05,
                       'lambda_region': 0.5,
                       'lambda_topo': 0.05,
                       'tau_region': 0.2,
                       'region_pair_manifest': 'datasets/unlabeled_dataset_en/unlabeled_dataset_en.json',
                       'topo_target': 'frangi'},
  'stage2_overrides': {'pd_json': 'datasets/pd_dataset_en/patients_dataset_en.json',
                       'base_path': 'datasets',
                       'image_size': 224,
                       'epochs': 50,
                       'batch_size': 16,
                       'num_workers': 4,
                       'lr': 0.0005,
                       'adapter_rank': 32,
                       'lambda_ce': 1.0,
                       'lambda_supcon': 1.0,
                       'lambda_recon': 0.05,
                       'tau_supcon': 0.1,
                       'kfold_mode': 'patient',
                       'num_folds': 5,
                       'eval_folds': 5,
                       'scheduler': 'cosine',
                       'early_stopping': True,
                       'patience': 8,
                       'classifier_dropout': 0.35,
                       'label_smooth_eps': 0.05,
                       'pair_mode': 'cartesian',
                       'max_pairs_per_patient': 8,
                       'finetune_fusion': True,
                       'fusion_lr_scale': 0.1,
                       'kfold_repeats': 1,
                       'use_morphology_features': True,
                       'morph_embed_dim': 32,
                       'use_feature_diffusion': True,
                       'fdiff_class_cond': True,
                       'fdiff_gate_mode': 'soft_cos',
                       'fdiff_gate_threshold': 0.72,
                       'fdiff_gate_temperature': 0.08,
                       'fdiff_T_aug': 45,
                       'fdiff_k_aug': 1,
                       'fdiff_start_epoch': 3,
                       'lambda_fdiff': 0.1,
                       'lambda_ce_aug': 0.12,
                       'use_prototype': True,
                       'proto_dim': 128,
                       'proto_dropout': 0.65,
                       'proto_metric': 'euclidean',
                       'proto_temp': 0.5,
                       'lambda_proto': 0.8,
                       'proto_min_per_class': 2,
                       'proto_eval_blend': 1.0,
                       'proto_eval_mode': 'support_lse',
                       'proto_support_k': 5,
                       'proto_fdiff_support_aug': 1,
                       'mil_balanced_bags': True,
                       'mode': 'mil',
                       'eval_patient_agg': True,
                       'select_metric': 'patient'},
  'parent_ckpt_override': None,
  'stage2_tasks': ['pd_stage']},
 {'_group': 'ABLFINE_NoDiffTDREST5_5f',
  '_desc': 'Fine-grained pretraining ablation: remove Stage0 diffusion reconstruction, then evaluate '
           'representative TDREST5 subtype config',
  'dataset': 'pd',
  'stages': [2],
  'stage0_overrides': {'corn1_root': 'datasets/CORN-1',
                       'image_size': 224,
                       'encoder_base_channels': 64,
                       'unet_model_channels': 64,
                       'z_dim': 512,
                       'T': 1000,
                       'epochs': 200,
                       'batch_size': 16,
                       'num_workers': 4,
                       'lr': 0.0001,
                       'lambda_simple': 0.0,
                       'lambda_topo': 0.1,
                       'z_dropout': 0.1,
                       'topo_target': 'gt'},
  'stage1_overrides': {'pd_json': 'datasets/pd_dataset_en/patients_dataset_en.json',
                       'base_path': 'datasets',
                       'image_size': 224,
                       'epochs': 80,
                       'batch_size': 16,
                       'num_workers': 4,
                       'lr': 2e-05,
                       'lambda_region': 0.5,
                       'lambda_topo': 0.05,
                       'tau_region': 0.2,
                       'region_pair_manifest': 'datasets/unlabeled_dataset_en/unlabeled_dataset_en.json',
                       'topo_target': 'frangi'},
  'stage2_overrides': {'pd_json': 'datasets/pd_dataset_en/patients_dataset_en.json',
                       'base_path': 'datasets',
                       'image_size': 224,
                       'epochs': 50,
                       'batch_size': 16,
                       'num_workers': 4,
                       'lr': 0.0005,
                       'adapter_rank': 32,
                       'lambda_ce': 1.0,
                       'lambda_supcon': 1.0,
                       'lambda_recon': 0.05,
                       'tau_supcon': 0.1,
                       'kfold_mode': 'patient',
                       'num_folds': 5,
                       'eval_folds': 5,
                       'scheduler': 'cosine',
                       'early_stopping': True,
                       'patience': 8,
                       'classifier_dropout': 0.35,
                       'label_smooth_eps': 0.05,
                       'pair_mode': 'cartesian',
                       'max_pairs_per_patient': 8,
                       'kfold_repeats': 1,
                       'use_feature_diffusion': True,
                       'fdiff_class_cond': True,
                       'fdiff_gate_mode': 'soft_cos',
                       'fdiff_gate_threshold': 0.7,
                       'fdiff_gate_temperature': 0.1,
                       'fdiff_T_aug': 70,
                       'fdiff_k_aug': 2,
                       'fdiff_start_epoch': 3,
                       'lambda_fdiff': 0.12,
                       'lambda_ce_aug': 0.18,
                       'fdiff_minority_only': True,
                       'use_class_weights': True,
                       'use_focal': True,
                       'focal_gamma': 1.5,
                       'select_metric': 'balanced_patient',
                       'mode': 'mil',
                       'eval_patient_agg': True},
  'parent_ckpt_override': None,
  'stage2_tasks': ['pd_subtype_tdrest']},
 {'_group': 'ABLFINE_NoStage1StageU1_stage_5f',
  '_desc': 'Fine-grained pretraining ablation: full Stage0 directly to representative U1 support-LSE stage '
           'config, no Stage1 RSCP',
  'dataset': 'pd',
  'stages': [2],
  'stage0_overrides': {'corn1_root': 'datasets/CORN-1',
                       'image_size': 224,
                       'encoder_base_channels': 64,
                       'unet_model_channels': 64,
                       'z_dim': 512,
                       'T': 1000,
                       'epochs': 200,
                       'batch_size': 16,
                       'num_workers': 4,
                       'lr': 0.0001,
                       'lambda_simple': 1.0,
                       'lambda_topo': 0.1,
                       'z_dropout': 0.1,
                       'topo_target': 'gt'},
  'stage2_overrides': {'pd_json': 'datasets/pd_dataset_en/patients_dataset_en.json',
                       'base_path': 'datasets',
                       'image_size': 224,
                       'epochs': 50,
                       'batch_size': 16,
                       'num_workers': 4,
                       'lr': 0.0005,
                       'adapter_rank': 32,
                       'lambda_ce': 1.0,
                       'lambda_supcon': 1.0,
                       'lambda_recon': 0.05,
                       'tau_supcon': 0.1,
                       'kfold_mode': 'patient',
                       'num_folds': 5,
                       'eval_folds': 5,
                       'scheduler': 'cosine',
                       'early_stopping': True,
                       'patience': 8,
                       'classifier_dropout': 0.35,
                       'label_smooth_eps': 0.05,
                       'pair_mode': 'cartesian',
                       'max_pairs_per_patient': 8,
                       'finetune_fusion': True,
                       'fusion_lr_scale': 0.1,
                       'kfold_repeats': 1,
                       'use_morphology_features': True,
                       'morph_embed_dim': 32,
                       'use_feature_diffusion': True,
                       'fdiff_class_cond': True,
                       'fdiff_gate_mode': 'soft_cos',
                       'fdiff_gate_threshold': 0.72,
                       'fdiff_gate_temperature': 0.08,
                       'fdiff_T_aug': 45,
                       'fdiff_k_aug': 1,
                       'fdiff_start_epoch': 3,
                       'lambda_fdiff': 0.1,
                       'lambda_ce_aug': 0.12,
                       'use_prototype': True,
                       'proto_dim': 128,
                       'proto_dropout': 0.65,
                       'proto_metric': 'euclidean',
                       'proto_temp': 0.5,
                       'lambda_proto': 0.8,
                       'proto_min_per_class': 2,
                       'proto_eval_blend': 1.0,
                       'proto_eval_mode': 'support_lse',
                       'proto_support_k': 5,
                       'proto_fdiff_support_aug': 1,
                       'mil_balanced_bags': True,
                       'mode': 'mil',
                       'eval_patient_agg': True,
                       'select_metric': 'patient'},
  'parent_ckpt_override': None,
  'stage2_tasks': ['pd_stage']},
 {'_group': 'ABLFINE_NoStage1TDREST5_5f',
  '_desc': 'Fine-grained pretraining ablation: full Stage0 directly to representative TDREST5 subtype '
           'config, no Stage1 RSCP',
  'dataset': 'pd',
  'stages': [2],
  'stage0_overrides': {'corn1_root': 'datasets/CORN-1',
                       'image_size': 224,
                       'encoder_base_channels': 64,
                       'unet_model_channels': 64,
                       'z_dim': 512,
                       'T': 1000,
                       'epochs': 200,
                       'batch_size': 16,
                       'num_workers': 4,
                       'lr': 0.0001,
                       'lambda_simple': 1.0,
                       'lambda_topo': 0.1,
                       'z_dropout': 0.1,
                       'topo_target': 'gt'},
  'stage2_overrides': {'pd_json': 'datasets/pd_dataset_en/patients_dataset_en.json',
                       'base_path': 'datasets',
                       'image_size': 224,
                       'epochs': 50,
                       'batch_size': 16,
                       'num_workers': 4,
                       'lr': 0.0005,
                       'adapter_rank': 32,
                       'lambda_ce': 1.0,
                       'lambda_supcon': 1.0,
                       'lambda_recon': 0.05,
                       'tau_supcon': 0.1,
                       'kfold_mode': 'patient',
                       'num_folds': 5,
                       'eval_folds': 5,
                       'scheduler': 'cosine',
                       'early_stopping': True,
                       'patience': 8,
                       'classifier_dropout': 0.35,
                       'label_smooth_eps': 0.05,
                       'pair_mode': 'cartesian',
                       'max_pairs_per_patient': 8,
                       'kfold_repeats': 1,
                       'use_feature_diffusion': True,
                       'fdiff_class_cond': True,
                       'fdiff_gate_mode': 'soft_cos',
                       'fdiff_gate_threshold': 0.7,
                       'fdiff_gate_temperature': 0.1,
                       'fdiff_T_aug': 70,
                       'fdiff_k_aug': 2,
                       'fdiff_start_epoch': 3,
                       'lambda_fdiff': 0.12,
                       'lambda_ce_aug': 0.18,
                       'fdiff_minority_only': True,
                       'use_class_weights': True,
                       'use_focal': True,
                       'focal_gamma': 1.5,
                       'select_metric': 'balanced_patient',
                       'mode': 'mil',
                       'eval_patient_agg': True},
  'parent_ckpt_override': None,
  'stage2_tasks': ['pd_subtype_tdrest']},
 {'_group': 'ABLFINE_NoTopoStageU1_stage_5f',
  '_desc': 'Fine-grained pretraining ablation: remove Stage0 topology supervision, then evaluate '
           'representative U1 support-LSE stage config',
  'dataset': 'pd',
  'stages': [2],
  'stage0_overrides': {'corn1_root': 'datasets/CORN-1',
                       'image_size': 224,
                       'encoder_base_channels': 64,
                       'unet_model_channels': 64,
                       'z_dim': 512,
                       'T': 1000,
                       'epochs': 200,
                       'batch_size': 16,
                       'num_workers': 4,
                       'lr': 0.0001,
                       'lambda_simple': 1.0,
                       'lambda_topo': 0.0,
                       'z_dropout': 0.1,
                       'topo_target': 'gt'},
  'stage1_overrides': {'pd_json': 'datasets/pd_dataset_en/patients_dataset_en.json',
                       'base_path': 'datasets',
                       'image_size': 224,
                       'epochs': 80,
                       'batch_size': 16,
                       'num_workers': 4,
                       'lr': 2e-05,
                       'lambda_region': 0.5,
                       'lambda_topo': 0.05,
                       'tau_region': 0.2,
                       'region_pair_manifest': 'datasets/unlabeled_dataset_en/unlabeled_dataset_en.json',
                       'topo_target': 'frangi'},
  'stage2_overrides': {'pd_json': 'datasets/pd_dataset_en/patients_dataset_en.json',
                       'base_path': 'datasets',
                       'image_size': 224,
                       'epochs': 50,
                       'batch_size': 16,
                       'num_workers': 4,
                       'lr': 0.0005,
                       'adapter_rank': 32,
                       'lambda_ce': 1.0,
                       'lambda_supcon': 1.0,
                       'lambda_recon': 0.05,
                       'tau_supcon': 0.1,
                       'kfold_mode': 'patient',
                       'num_folds': 5,
                       'eval_folds': 5,
                       'scheduler': 'cosine',
                       'early_stopping': True,
                       'patience': 8,
                       'classifier_dropout': 0.35,
                       'label_smooth_eps': 0.05,
                       'pair_mode': 'cartesian',
                       'max_pairs_per_patient': 8,
                       'finetune_fusion': True,
                       'fusion_lr_scale': 0.1,
                       'kfold_repeats': 1,
                       'use_morphology_features': True,
                       'morph_embed_dim': 32,
                       'use_feature_diffusion': True,
                       'fdiff_class_cond': True,
                       'fdiff_gate_mode': 'soft_cos',
                       'fdiff_gate_threshold': 0.72,
                       'fdiff_gate_temperature': 0.08,
                       'fdiff_T_aug': 45,
                       'fdiff_k_aug': 1,
                       'fdiff_start_epoch': 3,
                       'lambda_fdiff': 0.1,
                       'lambda_ce_aug': 0.12,
                       'use_prototype': True,
                       'proto_dim': 128,
                       'proto_dropout': 0.65,
                       'proto_metric': 'euclidean',
                       'proto_temp': 0.5,
                       'lambda_proto': 0.8,
                       'proto_min_per_class': 2,
                       'proto_eval_blend': 1.0,
                       'proto_eval_mode': 'support_lse',
                       'proto_support_k': 5,
                       'proto_fdiff_support_aug': 1,
                       'mil_balanced_bags': True,
                       'mode': 'mil',
                       'eval_patient_agg': True,
                       'select_metric': 'patient'},
  'parent_ckpt_override': None,
  'stage2_tasks': ['pd_stage']},
 {'_group': 'ABLFINE_NoTopoTDREST5_5f',
  '_desc': 'Fine-grained pretraining ablation: remove Stage0 topology supervision, then evaluate '
           'representative TDREST5 subtype config',
  'dataset': 'pd',
  'stages': [2],
  'stage0_overrides': {'corn1_root': 'datasets/CORN-1',
                       'image_size': 224,
                       'encoder_base_channels': 64,
                       'unet_model_channels': 64,
                       'z_dim': 512,
                       'T': 1000,
                       'epochs': 200,
                       'batch_size': 16,
                       'num_workers': 4,
                       'lr': 0.0001,
                       'lambda_simple': 1.0,
                       'lambda_topo': 0.0,
                       'z_dropout': 0.1,
                       'topo_target': 'gt'},
  'stage1_overrides': {'pd_json': 'datasets/pd_dataset_en/patients_dataset_en.json',
                       'base_path': 'datasets',
                       'image_size': 224,
                       'epochs': 80,
                       'batch_size': 16,
                       'num_workers': 4,
                       'lr': 2e-05,
                       'lambda_region': 0.5,
                       'lambda_topo': 0.05,
                       'tau_region': 0.2,
                       'region_pair_manifest': 'datasets/unlabeled_dataset_en/unlabeled_dataset_en.json',
                       'topo_target': 'frangi'},
  'stage2_overrides': {'pd_json': 'datasets/pd_dataset_en/patients_dataset_en.json',
                       'base_path': 'datasets',
                       'image_size': 224,
                       'epochs': 50,
                       'batch_size': 16,
                       'num_workers': 4,
                       'lr': 0.0005,
                       'adapter_rank': 32,
                       'lambda_ce': 1.0,
                       'lambda_supcon': 1.0,
                       'lambda_recon': 0.05,
                       'tau_supcon': 0.1,
                       'kfold_mode': 'patient',
                       'num_folds': 5,
                       'eval_folds': 5,
                       'scheduler': 'cosine',
                       'early_stopping': True,
                       'patience': 8,
                       'classifier_dropout': 0.35,
                       'label_smooth_eps': 0.05,
                       'pair_mode': 'cartesian',
                       'max_pairs_per_patient': 8,
                       'kfold_repeats': 1,
                       'use_feature_diffusion': True,
                       'fdiff_class_cond': True,
                       'fdiff_gate_mode': 'soft_cos',
                       'fdiff_gate_threshold': 0.7,
                       'fdiff_gate_temperature': 0.1,
                       'fdiff_T_aug': 70,
                       'fdiff_k_aug': 2,
                       'fdiff_start_epoch': 3,
                       'lambda_fdiff': 0.12,
                       'lambda_ce_aug': 0.18,
                       'fdiff_minority_only': True,
                       'use_class_weights': True,
                       'use_focal': True,
                       'focal_gamma': 1.5,
                       'select_metric': 'balanced_patient',
                       'mode': 'mil',
                       'eval_patient_agg': True},
  'parent_ckpt_override': None,
  'stage2_tasks': ['pd_subtype_tdrest']},
 {'_group': 'ABLBIN_FullFT_binary_5f',
  '_desc': 'Binary ablation: full encoder fine-tune instead of adapter-only RACF',
  'dataset': 'pd',
  'stages': [2],
  'stage2_overrides': {'pd_json': 'datasets/pd_dataset_en/patients_dataset_en.json',
                       'base_path': 'datasets',
                       'image_size': 224,
                       'epochs': 50,
                       'batch_size': 16,
                       'num_workers': 4,
                       'lr': 1e-05,
                       'adapter_rank': 32,
                       'lambda_ce': 1.0,
                       'lambda_supcon': 1.0,
                       'lambda_recon': 0.05,
                       'tau_supcon': 0.1,
                       'kfold_mode': 'patient',
                       'num_folds': 5,
                       'eval_folds': 5,
                       'scheduler': 'cosine',
                       'early_stopping': True,
                       'patience': 8,
                       'classifier_dropout': 0.35,
                       'label_smooth_eps': 0.05,
                       'pair_mode': 'cartesian',
                       'max_pairs_per_patient': 8,
                       'finetune_fusion': True,
                       'fusion_lr_scale': 0.1,
                       'kfold_repeats': 1,
                       'use_morphology_features': True,
                       'morph_embed_dim': 32,
                       'use_adapters': False,
                       'finetune_encoder_policy': 'all',
                       'mode': 'mil',
                       'eval_patient_agg': True,
                       'select_metric': 'patient'},
  'parent_ckpt_override': None,
  'stage0_overrides': {'topo_target': 'gt'},
  'stage1_overrides': {'region_pair_manifest': 'datasets/unlabeled_dataset_en/unlabeled_dataset_en.json',
                       'topo_target': 'frangi',
                       'epochs': 80},
  'stage2_tasks': ['pd_binary']},
 {'_group': 'ABLBIN_ImageLevel_binary_5f',
  '_desc': 'Binary ablation: image-pair training with patient-level aggregation instead of native MIL',
  'dataset': 'pd',
  'stages': [2],
  'stage2_overrides': {'pd_json': 'datasets/pd_dataset_en/patients_dataset_en.json',
                       'base_path': 'datasets',
                       'image_size': 224,
                       'epochs': 50,
                       'batch_size': 16,
                       'num_workers': 4,
                       'lr': 0.0005,
                       'adapter_rank': 32,
                       'lambda_ce': 1.0,
                       'lambda_supcon': 1.0,
                       'lambda_recon': 0.05,
                       'tau_supcon': 0.1,
                       'kfold_mode': 'patient',
                       'num_folds': 5,
                       'eval_folds': 5,
                       'scheduler': 'cosine',
                       'early_stopping': True,
                       'patience': 8,
                       'classifier_dropout': 0.35,
                       'label_smooth_eps': 0.05,
                       'pair_mode': 'cartesian',
                       'max_pairs_per_patient': 8,
                       'finetune_fusion': True,
                       'fusion_lr_scale': 0.1,
                       'kfold_repeats': 1,
                       'use_morphology_features': True,
                       'morph_embed_dim': 32,
                       'eval_patient_agg': True,
                       'select_metric': 'patient'},
  'parent_ckpt_override': None,
  'stage0_overrides': {'topo_target': 'gt'},
  'stage1_overrides': {'region_pair_manifest': 'datasets/unlabeled_dataset_en/unlabeled_dataset_en.json',
                       'topo_target': 'frangi',
                       'epochs': 80},
  'stage2_tasks': ['pd_binary']},
 {'_group': 'ABLBIN_MeanMIL_binary_5f',
  '_desc': 'Binary ablation: mean-pooling MIL instead of gated attention pooling',
  'dataset': 'pd',
  'stages': [2],
  'stage2_overrides': {'pd_json': 'datasets/pd_dataset_en/patients_dataset_en.json',
                       'base_path': 'datasets',
                       'image_size': 224,
                       'epochs': 50,
                       'batch_size': 16,
                       'num_workers': 4,
                       'lr': 0.0005,
                       'adapter_rank': 32,
                       'lambda_ce': 1.0,
                       'lambda_supcon': 1.0,
                       'lambda_recon': 0.05,
                       'tau_supcon': 0.1,
                       'kfold_mode': 'patient',
                       'num_folds': 5,
                       'eval_folds': 5,
                       'scheduler': 'cosine',
                       'early_stopping': True,
                       'patience': 8,
                       'classifier_dropout': 0.35,
                       'label_smooth_eps': 0.05,
                       'pair_mode': 'cartesian',
                       'max_pairs_per_patient': 8,
                       'finetune_fusion': True,
                       'fusion_lr_scale': 0.1,
                       'kfold_repeats': 1,
                       'use_morphology_features': True,
                       'morph_embed_dim': 32,
                       'mil_pooling': 'mean',
                       'mode': 'mil',
                       'eval_patient_agg': True,
                       'select_metric': 'patient'},
  'parent_ckpt_override': None,
  'stage0_overrides': {'topo_target': 'gt'},
  'stage1_overrides': {'region_pair_manifest': 'datasets/unlabeled_dataset_en/unlabeled_dataset_en.json',
                       'topo_target': 'frangi',
                       'epochs': 80},
  'stage2_tasks': ['pd_binary']},
 {'_group': 'ABLBIN_MorphOnly_binary_5f',
  '_desc': 'Binary ablation: morphology-only patient classifier without learned image embedding',
  'dataset': 'pd',
  'stages': [2],
  'stage2_overrides': {'pd_json': 'datasets/pd_dataset_en/patients_dataset_en.json',
                       'base_path': 'datasets',
                       'image_size': 224,
                       'epochs': 50,
                       'batch_size': 16,
                       'num_workers': 4,
                       'lr': 0.0005,
                       'adapter_rank': 32,
                       'lambda_ce': 1.0,
                       'lambda_supcon': 1.0,
                       'lambda_recon': 0.05,
                       'tau_supcon': 0.1,
                       'kfold_mode': 'patient',
                       'num_folds': 5,
                       'eval_folds': 5,
                       'scheduler': 'cosine',
                       'early_stopping': True,
                       'patience': 8,
                       'classifier_dropout': 0.35,
                       'label_smooth_eps': 0.05,
                       'pair_mode': 'cartesian',
                       'max_pairs_per_patient': 8,
                       'finetune_fusion': True,
                       'fusion_lr_scale': 0.1,
                       'kfold_repeats': 1,
                       'use_morphology_features': True,
                       'morph_embed_dim': 32,
                       'morphology_only': True,
                       'mode': 'mil',
                       'eval_patient_agg': True,
                       'select_metric': 'patient'},
  'parent_ckpt_override': None,
  'stage0_overrides': {'topo_target': 'gt'},
  'stage1_overrides': {'region_pair_manifest': 'datasets/unlabeled_dataset_en/unlabeled_dataset_en.json',
                       'topo_target': 'frangi',
                       'epochs': 80},
  'stage2_tasks': ['pd_binary']},
 {'_group': 'ABLBIN_NoAdapter_binary_5f',
  '_desc': 'Binary ablation: remove residual adapters and train only heads/fusion',
  'dataset': 'pd',
  'stages': [2],
  'stage2_overrides': {'pd_json': 'datasets/pd_dataset_en/patients_dataset_en.json',
                       'base_path': 'datasets',
                       'image_size': 224,
                       'epochs': 50,
                       'batch_size': 16,
                       'num_workers': 4,
                       'lr': 0.0005,
                       'adapter_rank': 32,
                       'lambda_ce': 1.0,
                       'lambda_supcon': 1.0,
                       'lambda_recon': 0.05,
                       'tau_supcon': 0.1,
                       'kfold_mode': 'patient',
                       'num_folds': 5,
                       'eval_folds': 5,
                       'scheduler': 'cosine',
                       'early_stopping': True,
                       'patience': 8,
                       'classifier_dropout': 0.35,
                       'label_smooth_eps': 0.05,
                       'pair_mode': 'cartesian',
                       'max_pairs_per_patient': 8,
                       'finetune_fusion': True,
                       'fusion_lr_scale': 0.1,
                       'kfold_repeats': 1,
                       'use_morphology_features': True,
                       'morph_embed_dim': 32,
                       'use_adapters': False,
                       'mode': 'mil',
                       'eval_patient_agg': True,
                       'select_metric': 'patient'},
  'parent_ckpt_override': None,
  'stage0_overrides': {'topo_target': 'gt'},
  'stage1_overrides': {'region_pair_manifest': 'datasets/unlabeled_dataset_en/unlabeled_dataset_en.json',
                       'topo_target': 'frangi',
                       'epochs': 80},
  'stage2_tasks': ['pd_binary']},
 {'_group': 'ABLBIN_NoMorph_binary_5f',
  '_desc': 'Binary ablation: remove morphology late fusion from the US0 Stage2 MIL recipe',
  'dataset': 'pd',
  'stages': [2],
  'stage2_overrides': {'pd_json': 'datasets/pd_dataset_en/patients_dataset_en.json',
                       'base_path': 'datasets',
                       'image_size': 224,
                       'epochs': 50,
                       'batch_size': 16,
                       'num_workers': 4,
                       'lr': 0.0005,
                       'adapter_rank': 32,
                       'lambda_ce': 1.0,
                       'lambda_supcon': 1.0,
                       'lambda_recon': 0.05,
                       'tau_supcon': 0.1,
                       'kfold_mode': 'patient',
                       'num_folds': 5,
                       'eval_folds': 5,
                       'scheduler': 'cosine',
                       'early_stopping': True,
                       'patience': 8,
                       'classifier_dropout': 0.35,
                       'label_smooth_eps': 0.05,
                       'pair_mode': 'cartesian',
                       'max_pairs_per_patient': 8,
                       'finetune_fusion': True,
                       'fusion_lr_scale': 0.1,
                       'kfold_repeats': 1,
                       'use_morphology_features': False,
                       'morph_embed_dim': 32,
                       'mode': 'mil',
                       'eval_patient_agg': True,
                       'select_metric': 'patient'},
  'parent_ckpt_override': None,
  'stage0_overrides': {'topo_target': 'gt'},
  'stage1_overrides': {'region_pair_manifest': 'datasets/unlabeled_dataset_en/unlabeled_dataset_en.json',
                       'topo_target': 'frangi',
                       'epochs': 80},
  'stage2_tasks': ['pd_binary']},
 {'_group': 'ABLBIN_RandomMorph_binary_5f',
  '_desc': 'Binary ablation: shuffle morphology tokens within each fold to test proxy/leakage',
  'dataset': 'pd',
  'stages': [2],
  'stage2_overrides': {'pd_json': 'datasets/pd_dataset_en/patients_dataset_en.json',
                       'base_path': 'datasets',
                       'image_size': 224,
                       'epochs': 50,
                       'batch_size': 16,
                       'num_workers': 4,
                       'lr': 0.0005,
                       'adapter_rank': 32,
                       'lambda_ce': 1.0,
                       'lambda_supcon': 1.0,
                       'lambda_recon': 0.05,
                       'tau_supcon': 0.1,
                       'kfold_mode': 'patient',
                       'num_folds': 5,
                       'eval_folds': 5,
                       'scheduler': 'cosine',
                       'early_stopping': True,
                       'patience': 8,
                       'classifier_dropout': 0.35,
                       'label_smooth_eps': 0.05,
                       'pair_mode': 'cartesian',
                       'max_pairs_per_patient': 8,
                       'finetune_fusion': True,
                       'fusion_lr_scale': 0.1,
                       'kfold_repeats': 1,
                       'use_morphology_features': True,
                       'morph_embed_dim': 32,
                       'shuffle_morphology_features': True,
                       'mode': 'mil',
                       'eval_patient_agg': True,
                       'select_metric': 'patient'},
  'parent_ckpt_override': None,
  'stage0_overrides': {'topo_target': 'gt'},
  'stage1_overrides': {'region_pair_manifest': 'datasets/unlabeled_dataset_en/unlabeled_dataset_en.json',
                       'topo_target': 'frangi',
                       'epochs': 80},
  'stage2_tasks': ['pd_binary']},
 {'_group': 'ABLBIN_ZipMIL_binary_5f',
  '_desc': 'Binary ablation: zip pairing instead of cartesian patient bags',
  'dataset': 'pd',
  'stages': [2],
  'stage2_overrides': {'pd_json': 'datasets/pd_dataset_en/patients_dataset_en.json',
                       'base_path': 'datasets',
                       'image_size': 224,
                       'epochs': 50,
                       'batch_size': 16,
                       'num_workers': 4,
                       'lr': 0.0005,
                       'adapter_rank': 32,
                       'lambda_ce': 1.0,
                       'lambda_supcon': 1.0,
                       'lambda_recon': 0.05,
                       'tau_supcon': 0.1,
                       'kfold_mode': 'patient',
                       'num_folds': 5,
                       'eval_folds': 5,
                       'scheduler': 'cosine',
                       'early_stopping': True,
                       'patience': 8,
                       'classifier_dropout': 0.35,
                       'label_smooth_eps': 0.05,
                       'pair_mode': 'zip',
                       'max_pairs_per_patient': 0,
                       'finetune_fusion': True,
                       'fusion_lr_scale': 0.1,
                       'kfold_repeats': 1,
                       'use_morphology_features': True,
                       'morph_embed_dim': 32,
                       'mode': 'mil',
                       'eval_patient_agg': True,
                       'select_metric': 'patient'},
  'parent_ckpt_override': None,
  'stage0_overrides': {'topo_target': 'gt'},
  'stage1_overrides': {'region_pair_manifest': 'datasets/unlabeled_dataset_en/unlabeled_dataset_en.json',
                       'topo_target': 'frangi',
                       'epochs': 80},
  'stage2_tasks': ['pd_binary']},
 {'_group': 'ABLSTAGE_FullProto_stage_5f',
  '_desc': 'Stage ablation anchor: support-LSE prototype with gentle fdiff support augmentation',
  'dataset': 'pd',
  'stages': [2],
  'stage2_overrides': {'pd_json': 'datasets/pd_dataset_en/patients_dataset_en.json',
                       'base_path': 'datasets',
                       'image_size': 224,
                       'epochs': 50,
                       'batch_size': 16,
                       'num_workers': 4,
                       'lr': 0.0005,
                       'adapter_rank': 32,
                       'lambda_ce': 1.0,
                       'lambda_supcon': 1.0,
                       'lambda_recon': 0.05,
                       'tau_supcon': 0.1,
                       'kfold_mode': 'patient',
                       'num_folds': 5,
                       'eval_folds': 5,
                       'scheduler': 'cosine',
                       'early_stopping': True,
                       'patience': 8,
                       'classifier_dropout': 0.35,
                       'label_smooth_eps': 0.05,
                       'pair_mode': 'cartesian',
                       'max_pairs_per_patient': 8,
                       'finetune_fusion': True,
                       'fusion_lr_scale': 0.1,
                       'kfold_repeats': 1,
                       'use_morphology_features': True,
                       'morph_embed_dim': 32,
                       'use_feature_diffusion': True,
                       'fdiff_class_cond': True,
                       'fdiff_gate_mode': 'soft_cos',
                       'fdiff_gate_threshold': 0.72,
                       'fdiff_gate_temperature': 0.08,
                       'fdiff_T_aug': 45,
                       'fdiff_k_aug': 1,
                       'fdiff_start_epoch': 3,
                       'lambda_fdiff': 0.1,
                       'lambda_ce_aug': 0.12,
                       'use_prototype': True,
                       'proto_dim': 128,
                       'proto_dropout': 0.65,
                       'proto_metric': 'euclidean',
                       'proto_temp': 0.5,
                       'lambda_proto': 0.8,
                       'proto_min_per_class': 2,
                       'proto_eval_blend': 1.0,
                       'proto_eval_mode': 'support_lse',
                       'proto_support_k': 5,
                       'proto_fdiff_support_aug': 1,
                       'mil_balanced_bags': True,
                       'mode': 'mil',
                       'eval_patient_agg': True,
                       'select_metric': 'patient'},
  'parent_ckpt_override': None,
  'stage0_overrides': {'topo_target': 'gt'},
  'stage1_overrides': {'region_pair_manifest': 'datasets/unlabeled_dataset_en/unlabeled_dataset_en.json',
                       'topo_target': 'frangi',
                       'epochs': 80},
  'stage2_tasks': ['pd_stage']},
 {'_group': 'ABLSTAGE_MeanMIL_stage_5f',
  '_desc': 'Stage ablation: mean-pooling MIL instead of gated attention pooling',
  'dataset': 'pd',
  'stages': [2],
  'stage2_overrides': {'pd_json': 'datasets/pd_dataset_en/patients_dataset_en.json',
                       'base_path': 'datasets',
                       'image_size': 224,
                       'epochs': 50,
                       'batch_size': 16,
                       'num_workers': 4,
                       'lr': 0.0005,
                       'adapter_rank': 32,
                       'lambda_ce': 1.0,
                       'lambda_supcon': 1.0,
                       'lambda_recon': 0.05,
                       'tau_supcon': 0.1,
                       'kfold_mode': 'patient',
                       'num_folds': 5,
                       'eval_folds': 5,
                       'scheduler': 'cosine',
                       'early_stopping': True,
                       'patience': 8,
                       'classifier_dropout': 0.35,
                       'label_smooth_eps': 0.05,
                       'pair_mode': 'cartesian',
                       'max_pairs_per_patient': 8,
                       'finetune_fusion': True,
                       'fusion_lr_scale': 0.1,
                       'kfold_repeats': 1,
                       'use_morphology_features': True,
                       'morph_embed_dim': 32,
                       'use_feature_diffusion': True,
                       'fdiff_class_cond': True,
                       'fdiff_gate_mode': 'soft_cos',
                       'fdiff_gate_threshold': 0.72,
                       'fdiff_gate_temperature': 0.08,
                       'fdiff_T_aug': 45,
                       'fdiff_k_aug': 1,
                       'fdiff_start_epoch': 3,
                       'lambda_fdiff': 0.1,
                       'lambda_ce_aug': 0.12,
                       'use_prototype': True,
                       'proto_dim': 128,
                       'proto_dropout': 0.65,
                       'proto_metric': 'euclidean',
                       'proto_temp': 0.5,
                       'lambda_proto': 0.8,
                       'proto_min_per_class': 2,
                       'proto_eval_blend': 1.0,
                       'proto_eval_mode': 'support_lse',
                       'proto_support_k': 5,
                       'proto_fdiff_support_aug': 1,
                       'mil_balanced_bags': True,
                       'mil_pooling': 'mean',
                       'mode': 'mil',
                       'eval_patient_agg': True,
                       'select_metric': 'patient'},
  'parent_ckpt_override': None,
  'stage0_overrides': {'topo_target': 'gt'},
  'stage1_overrides': {'region_pair_manifest': 'datasets/unlabeled_dataset_en/unlabeled_dataset_en.json',
                       'topo_target': 'frangi',
                       'epochs': 80},
  'stage2_tasks': ['pd_stage']},
 {'_group': 'ABLSTAGE_NoBalancedBags_stage_5f',
  '_desc': 'Stage ablation: disable class-balanced MIL bag sampling',
  'dataset': 'pd',
  'stages': [2],
  'stage2_overrides': {'pd_json': 'datasets/pd_dataset_en/patients_dataset_en.json',
                       'base_path': 'datasets',
                       'image_size': 224,
                       'epochs': 50,
                       'batch_size': 16,
                       'num_workers': 4,
                       'lr': 0.0005,
                       'adapter_rank': 32,
                       'lambda_ce': 1.0,
                       'lambda_supcon': 1.0,
                       'lambda_recon': 0.05,
                       'tau_supcon': 0.1,
                       'kfold_mode': 'patient',
                       'num_folds': 5,
                       'eval_folds': 5,
                       'scheduler': 'cosine',
                       'early_stopping': True,
                       'patience': 8,
                       'classifier_dropout': 0.35,
                       'label_smooth_eps': 0.05,
                       'pair_mode': 'cartesian',
                       'max_pairs_per_patient': 8,
                       'finetune_fusion': True,
                       'fusion_lr_scale': 0.1,
                       'kfold_repeats': 1,
                       'use_morphology_features': True,
                       'morph_embed_dim': 32,
                       'use_feature_diffusion': True,
                       'fdiff_class_cond': True,
                       'fdiff_gate_mode': 'soft_cos',
                       'fdiff_gate_threshold': 0.72,
                       'fdiff_gate_temperature': 0.08,
                       'fdiff_T_aug': 45,
                       'fdiff_k_aug': 1,
                       'fdiff_start_epoch': 3,
                       'lambda_fdiff': 0.1,
                       'lambda_ce_aug': 0.12,
                       'use_prototype': True,
                       'proto_dim': 128,
                       'proto_dropout': 0.65,
                       'proto_metric': 'euclidean',
                       'proto_temp': 0.5,
                       'lambda_proto': 0.8,
                       'proto_min_per_class': 2,
                       'proto_eval_blend': 1.0,
                       'proto_eval_mode': 'support_lse',
                       'proto_support_k': 5,
                       'proto_fdiff_support_aug': 1,
                       'mil_balanced_bags': False,
                       'mode': 'mil',
                       'eval_patient_agg': True,
                       'select_metric': 'patient'},
  'parent_ckpt_override': None,
  'stage0_overrides': {'topo_target': 'gt'},
  'stage1_overrides': {'region_pair_manifest': 'datasets/unlabeled_dataset_en/unlabeled_dataset_en.json',
                       'topo_target': 'frangi',
                       'epochs': 80},
  'stage2_tasks': ['pd_stage']},
 {'_group': 'ABLSTAGE_NoFdiffSupportAug_stage_5f',
  '_desc': 'Stage ablation: remove fdiff and fdiff support augmentation',
  'dataset': 'pd',
  'stages': [2],
  'stage2_overrides': {'pd_json': 'datasets/pd_dataset_en/patients_dataset_en.json',
                       'base_path': 'datasets',
                       'image_size': 224,
                       'epochs': 50,
                       'batch_size': 16,
                       'num_workers': 4,
                       'lr': 0.0005,
                       'adapter_rank': 32,
                       'lambda_ce': 1.0,
                       'lambda_supcon': 1.0,
                       'lambda_recon': 0.05,
                       'tau_supcon': 0.1,
                       'kfold_mode': 'patient',
                       'num_folds': 5,
                       'eval_folds': 5,
                       'scheduler': 'cosine',
                       'early_stopping': True,
                       'patience': 8,
                       'classifier_dropout': 0.35,
                       'label_smooth_eps': 0.05,
                       'pair_mode': 'cartesian',
                       'max_pairs_per_patient': 8,
                       'finetune_fusion': True,
                       'fusion_lr_scale': 0.1,
                       'kfold_repeats': 1,
                       'use_morphology_features': True,
                       'morph_embed_dim': 32,
                       'use_feature_diffusion': False,
                       'fdiff_class_cond': True,
                       'fdiff_gate_mode': 'soft_cos',
                       'fdiff_gate_threshold': 0.62,
                       'fdiff_gate_temperature': 0.12,
                       'fdiff_T_aug': 100,
                       'fdiff_k_aug': 1,
                       'fdiff_start_epoch': 3,
                       'lambda_fdiff': 0.2,
                       'lambda_ce_aug': 0.15,
                       'use_prototype': True,
                       'proto_dim': 128,
                       'proto_dropout': 0.65,
                       'proto_metric': 'euclidean',
                       'proto_temp': 0.5,
                       'lambda_proto': 0.8,
                       'proto_min_per_class': 2,
                       'proto_eval_blend': 1.0,
                       'proto_eval_mode': 'support_lse',
                       'proto_support_k': 5,
                       'proto_fdiff_support_aug': 0,
                       'mil_balanced_bags': True,
                       'mode': 'mil',
                       'eval_patient_agg': True,
                       'select_metric': 'patient'},
  'parent_ckpt_override': None,
  'stage0_overrides': {'topo_target': 'gt'},
  'stage1_overrides': {'region_pair_manifest': 'datasets/unlabeled_dataset_en/unlabeled_dataset_en.json',
                       'topo_target': 'frangi',
                       'epochs': 80},
  'stage2_tasks': ['pd_stage']},
 {'_group': 'ABLSTAGE_NoMorph_stage_5f',
  '_desc': 'Stage ablation: remove morphology late fusion',
  'dataset': 'pd',
  'stages': [2],
  'stage2_overrides': {'pd_json': 'datasets/pd_dataset_en/patients_dataset_en.json',
                       'base_path': 'datasets',
                       'image_size': 224,
                       'epochs': 50,
                       'batch_size': 16,
                       'num_workers': 4,
                       'lr': 0.0005,
                       'adapter_rank': 32,
                       'lambda_ce': 1.0,
                       'lambda_supcon': 1.0,
                       'lambda_recon': 0.05,
                       'tau_supcon': 0.1,
                       'kfold_mode': 'patient',
                       'num_folds': 5,
                       'eval_folds': 5,
                       'scheduler': 'cosine',
                       'early_stopping': True,
                       'patience': 8,
                       'classifier_dropout': 0.35,
                       'label_smooth_eps': 0.05,
                       'pair_mode': 'cartesian',
                       'max_pairs_per_patient': 8,
                       'finetune_fusion': True,
                       'fusion_lr_scale': 0.1,
                       'kfold_repeats': 1,
                       'use_morphology_features': False,
                       'morph_embed_dim': 32,
                       'use_feature_diffusion': True,
                       'fdiff_class_cond': True,
                       'fdiff_gate_mode': 'soft_cos',
                       'fdiff_gate_threshold': 0.72,
                       'fdiff_gate_temperature': 0.08,
                       'fdiff_T_aug': 45,
                       'fdiff_k_aug': 1,
                       'fdiff_start_epoch': 3,
                       'lambda_fdiff': 0.1,
                       'lambda_ce_aug': 0.12,
                       'use_prototype': True,
                       'proto_dim': 128,
                       'proto_dropout': 0.65,
                       'proto_metric': 'euclidean',
                       'proto_temp': 0.5,
                       'lambda_proto': 0.8,
                       'proto_min_per_class': 2,
                       'proto_eval_blend': 1.0,
                       'proto_eval_mode': 'support_lse',
                       'proto_support_k': 5,
                       'proto_fdiff_support_aug': 1,
                       'mil_balanced_bags': True,
                       'mode': 'mil',
                       'eval_patient_agg': True,
                       'select_metric': 'patient'},
  'parent_ckpt_override': None,
  'stage0_overrides': {'topo_target': 'gt'},
  'stage1_overrides': {'region_pair_manifest': 'datasets/unlabeled_dataset_en/unlabeled_dataset_en.json',
                       'topo_target': 'frangi',
                       'epochs': 80},
  'stage2_tasks': ['pd_stage']},
 {'_group': 'ABLSTAGE_NoPrototype_stage_5f',
  '_desc': 'Stage ablation: remove prototype head and support-bank inference',
  'dataset': 'pd',
  'stages': [2],
  'stage2_overrides': {'pd_json': 'datasets/pd_dataset_en/patients_dataset_en.json',
                       'base_path': 'datasets',
                       'image_size': 224,
                       'epochs': 50,
                       'batch_size': 16,
                       'num_workers': 4,
                       'lr': 0.0005,
                       'adapter_rank': 32,
                       'lambda_ce': 1.0,
                       'lambda_supcon': 1.0,
                       'lambda_recon': 0.05,
                       'tau_supcon': 0.1,
                       'kfold_mode': 'patient',
                       'num_folds': 5,
                       'eval_folds': 5,
                       'scheduler': 'cosine',
                       'early_stopping': True,
                       'patience': 8,
                       'classifier_dropout': 0.35,
                       'label_smooth_eps': 0.05,
                       'pair_mode': 'cartesian',
                       'max_pairs_per_patient': 8,
                       'finetune_fusion': True,
                       'fusion_lr_scale': 0.1,
                       'kfold_repeats': 1,
                       'use_morphology_features': True,
                       'morph_embed_dim': 32,
                       'use_feature_diffusion': True,
                       'fdiff_class_cond': True,
                       'fdiff_gate_mode': 'soft_cos',
                       'fdiff_gate_threshold': 0.72,
                       'fdiff_gate_temperature': 0.08,
                       'fdiff_T_aug': 45,
                       'fdiff_k_aug': 1,
                       'fdiff_start_epoch': 3,
                       'lambda_fdiff': 0.1,
                       'lambda_ce_aug': 0.12,
                       'mil_balanced_bags': True,
                       'use_prototype': False,
                       'lambda_proto': 0.0,
                       'mode': 'mil',
                       'eval_patient_agg': True,
                       'select_metric': 'patient'},
  'parent_ckpt_override': None,
  'stage0_overrides': {'topo_target': 'gt'},
  'stage1_overrides': {'region_pair_manifest': 'datasets/unlabeled_dataset_en/unlabeled_dataset_en.json',
                       'topo_target': 'frangi',
                       'epochs': 80},
  'stage2_tasks': ['pd_stage']},
 {'_group': 'ABLSTAGE_NoSupportLSE_stage_5f',
  '_desc': 'Stage ablation: train prototype regularizer but evaluate through classifier path',
  'dataset': 'pd',
  'stages': [2],
  'stage2_overrides': {'pd_json': 'datasets/pd_dataset_en/patients_dataset_en.json',
                       'base_path': 'datasets',
                       'image_size': 224,
                       'epochs': 50,
                       'batch_size': 16,
                       'num_workers': 4,
                       'lr': 0.0005,
                       'adapter_rank': 32,
                       'lambda_ce': 1.0,
                       'lambda_supcon': 1.0,
                       'lambda_recon': 0.05,
                       'tau_supcon': 0.1,
                       'kfold_mode': 'patient',
                       'num_folds': 5,
                       'eval_folds': 5,
                       'scheduler': 'cosine',
                       'early_stopping': True,
                       'patience': 8,
                       'classifier_dropout': 0.35,
                       'label_smooth_eps': 0.05,
                       'pair_mode': 'cartesian',
                       'max_pairs_per_patient': 8,
                       'finetune_fusion': True,
                       'fusion_lr_scale': 0.1,
                       'kfold_repeats': 1,
                       'use_morphology_features': True,
                       'morph_embed_dim': 32,
                       'use_feature_diffusion': True,
                       'fdiff_class_cond': True,
                       'fdiff_gate_mode': 'soft_cos',
                       'fdiff_gate_threshold': 0.72,
                       'fdiff_gate_temperature': 0.08,
                       'fdiff_T_aug': 45,
                       'fdiff_k_aug': 1,
                       'fdiff_start_epoch': 3,
                       'lambda_fdiff': 0.1,
                       'lambda_ce_aug': 0.12,
                       'use_prototype': True,
                       'proto_dim': 128,
                       'proto_dropout': 0.65,
                       'proto_metric': 'euclidean',
                       'proto_temp': 0.5,
                       'lambda_proto': 0.8,
                       'proto_min_per_class': 2,
                       'proto_eval_blend': 0.0,
                       'proto_eval_mode': 'mean',
                       'proto_support_k': 5,
                       'proto_fdiff_support_aug': 1,
                       'mil_balanced_bags': True,
                       'mode': 'mil',
                       'eval_patient_agg': True,
                       'select_metric': 'patient'},
  'parent_ckpt_override': None,
  'stage0_overrides': {'topo_target': 'gt'},
  'stage1_overrides': {'region_pair_manifest': 'datasets/unlabeled_dataset_en/unlabeled_dataset_en.json',
                       'topo_target': 'frangi',
                       'epochs': 80},
  'stage2_tasks': ['pd_stage']},
 {'_group': 'ABLSTAGE_ZipMIL_stage_5f',
  '_desc': 'Stage ablation: zip pairing instead of cartesian patient bags',
  'dataset': 'pd',
  'stages': [2],
  'stage2_overrides': {'pd_json': 'datasets/pd_dataset_en/patients_dataset_en.json',
                       'base_path': 'datasets',
                       'image_size': 224,
                       'epochs': 50,
                       'batch_size': 16,
                       'num_workers': 4,
                       'lr': 0.0005,
                       'adapter_rank': 32,
                       'lambda_ce': 1.0,
                       'lambda_supcon': 1.0,
                       'lambda_recon': 0.05,
                       'tau_supcon': 0.1,
                       'kfold_mode': 'patient',
                       'num_folds': 5,
                       'eval_folds': 5,
                       'scheduler': 'cosine',
                       'early_stopping': True,
                       'patience': 8,
                       'classifier_dropout': 0.35,
                       'label_smooth_eps': 0.05,
                       'pair_mode': 'zip',
                       'max_pairs_per_patient': 0,
                       'finetune_fusion': True,
                       'fusion_lr_scale': 0.1,
                       'kfold_repeats': 1,
                       'use_morphology_features': True,
                       'morph_embed_dim': 32,
                       'use_feature_diffusion': True,
                       'fdiff_class_cond': True,
                       'fdiff_gate_mode': 'soft_cos',
                       'fdiff_gate_threshold': 0.72,
                       'fdiff_gate_temperature': 0.08,
                       'fdiff_T_aug': 45,
                       'fdiff_k_aug': 1,
                       'fdiff_start_epoch': 3,
                       'lambda_fdiff': 0.1,
                       'lambda_ce_aug': 0.12,
                       'use_prototype': True,
                       'proto_dim': 128,
                       'proto_dropout': 0.65,
                       'proto_metric': 'euclidean',
                       'proto_temp': 0.5,
                       'lambda_proto': 0.8,
                       'proto_min_per_class': 2,
                       'proto_eval_blend': 1.0,
                       'proto_eval_mode': 'support_lse',
                       'proto_support_k': 5,
                       'proto_fdiff_support_aug': 1,
                       'mil_balanced_bags': True,
                       'mode': 'mil',
                       'eval_patient_agg': True,
                       'select_metric': 'patient'},
  'parent_ckpt_override': None,
  'stage0_overrides': {'topo_target': 'gt'},
  'stage1_overrides': {'region_pair_manifest': 'datasets/unlabeled_dataset_en/unlabeled_dataset_en.json',
                       'topo_target': 'frangi',
                       'epochs': 80},
  'stage2_tasks': ['pd_stage']},
 {'_group': 'ABLTDREST_Full_TDREST5_5f',
  '_desc': 'TDREST ablation anchor: TDREST5 minority fdiff + focal gamma1.5 + balanced selection',
  'dataset': 'pd',
  'stages': [2],
  'stage2_overrides': {'pd_json': 'datasets/pd_dataset_en/patients_dataset_en.json',
                       'base_path': 'datasets',
                       'image_size': 224,
                       'epochs': 50,
                       'batch_size': 16,
                       'num_workers': 4,
                       'lr': 0.0005,
                       'adapter_rank': 32,
                       'lambda_ce': 1.0,
                       'lambda_supcon': 1.0,
                       'lambda_recon': 0.05,
                       'tau_supcon': 0.1,
                       'kfold_mode': 'patient',
                       'num_folds': 5,
                       'eval_folds': 5,
                       'scheduler': 'cosine',
                       'early_stopping': True,
                       'patience': 8,
                       'classifier_dropout': 0.35,
                       'label_smooth_eps': 0.05,
                       'pair_mode': 'cartesian',
                       'max_pairs_per_patient': 8,
                       'kfold_repeats': 1,
                       'use_feature_diffusion': True,
                       'fdiff_class_cond': True,
                       'fdiff_gate_mode': 'soft_cos',
                       'fdiff_gate_threshold': 0.7,
                       'fdiff_gate_temperature': 0.1,
                       'fdiff_T_aug': 70,
                       'fdiff_k_aug': 2,
                       'fdiff_start_epoch': 3,
                       'lambda_fdiff': 0.12,
                       'lambda_ce_aug': 0.18,
                       'fdiff_minority_only': True,
                       'use_class_weights': True,
                       'use_focal': True,
                       'focal_gamma': 1.5,
                       'select_metric': 'balanced_patient',
                       'mode': 'mil',
                       'eval_patient_agg': True},
  'parent_ckpt_override': None,
  'stage0_overrides': {'topo_target': 'gt'},
  'stage1_overrides': {'region_pair_manifest': 'datasets/unlabeled_dataset_en/unlabeled_dataset_en.json',
                       'topo_target': 'frangi',
                       'epochs': 80},
  'stage2_tasks': ['pd_subtype_tdrest']},
 {'_group': 'ABLTDREST_Gamma3_5f',
  '_desc': 'TDREST ablation: stronger focal gamma=3.0 on TDREST5',
  'dataset': 'pd',
  'stages': [2],
  'stage2_overrides': {'pd_json': 'datasets/pd_dataset_en/patients_dataset_en.json',
                       'base_path': 'datasets',
                       'image_size': 224,
                       'epochs': 50,
                       'batch_size': 16,
                       'num_workers': 4,
                       'lr': 0.0005,
                       'adapter_rank': 32,
                       'lambda_ce': 1.0,
                       'lambda_supcon': 1.0,
                       'lambda_recon': 0.05,
                       'tau_supcon': 0.1,
                       'kfold_mode': 'patient',
                       'num_folds': 5,
                       'eval_folds': 5,
                       'scheduler': 'cosine',
                       'early_stopping': True,
                       'patience': 8,
                       'classifier_dropout': 0.35,
                       'label_smooth_eps': 0.05,
                       'pair_mode': 'cartesian',
                       'max_pairs_per_patient': 8,
                       'kfold_repeats': 1,
                       'use_feature_diffusion': True,
                       'fdiff_class_cond': True,
                       'fdiff_gate_mode': 'soft_cos',
                       'fdiff_gate_threshold': 0.7,
                       'fdiff_gate_temperature': 0.1,
                       'fdiff_T_aug': 70,
                       'fdiff_k_aug': 2,
                       'fdiff_start_epoch': 3,
                       'lambda_fdiff': 0.12,
                       'lambda_ce_aug': 0.18,
                       'fdiff_minority_only': True,
                       'use_class_weights': True,
                       'use_focal': True,
                       'focal_gamma': 3.0,
                       'select_metric': 'balanced_patient',
                       'mode': 'mil',
                       'eval_patient_agg': True},
  'parent_ckpt_override': None,
  'stage0_overrides': {'topo_target': 'gt'},
  'stage1_overrides': {'region_pair_manifest': 'datasets/unlabeled_dataset_en/unlabeled_dataset_en.json',
                       'topo_target': 'frangi',
                       'epochs': 80},
  'stage2_tasks': ['pd_subtype_tdrest']},
 {'_group': 'ABLTDREST_NoBalancedSelect_5f',
  '_desc': 'TDREST ablation: select checkpoint by patient accuracy instead of balanced_patient',
  'dataset': 'pd',
  'stages': [2],
  'stage2_overrides': {'pd_json': 'datasets/pd_dataset_en/patients_dataset_en.json',
                       'base_path': 'datasets',
                       'image_size': 224,
                       'epochs': 50,
                       'batch_size': 16,
                       'num_workers': 4,
                       'lr': 0.0005,
                       'adapter_rank': 32,
                       'lambda_ce': 1.0,
                       'lambda_supcon': 1.0,
                       'lambda_recon': 0.05,
                       'tau_supcon': 0.1,
                       'kfold_mode': 'patient',
                       'num_folds': 5,
                       'eval_folds': 5,
                       'scheduler': 'cosine',
                       'early_stopping': True,
                       'patience': 8,
                       'classifier_dropout': 0.35,
                       'label_smooth_eps': 0.05,
                       'pair_mode': 'cartesian',
                       'max_pairs_per_patient': 8,
                       'kfold_repeats': 1,
                       'use_feature_diffusion': True,
                       'fdiff_class_cond': True,
                       'fdiff_gate_mode': 'soft_cos',
                       'fdiff_gate_threshold': 0.7,
                       'fdiff_gate_temperature': 0.1,
                       'fdiff_T_aug': 70,
                       'fdiff_k_aug': 2,
                       'fdiff_start_epoch': 3,
                       'lambda_fdiff': 0.12,
                       'lambda_ce_aug': 0.18,
                       'fdiff_minority_only': True,
                       'use_class_weights': True,
                       'use_focal': True,
                       'focal_gamma': 1.5,
                       'select_metric': 'patient',
                       'mode': 'mil',
                       'eval_patient_agg': True},
  'parent_ckpt_override': None,
  'stage0_overrides': {'topo_target': 'gt'},
  'stage1_overrides': {'region_pair_manifest': 'datasets/unlabeled_dataset_en/unlabeled_dataset_en.json',
                       'topo_target': 'frangi',
                       'epochs': 80},
  'stage2_tasks': ['pd_subtype_tdrest']},
 {'_group': 'ABLTDREST_NoClassWeights_5f',
  '_desc': 'TDREST ablation: remove inverse-frequency class weights',
  'dataset': 'pd',
  'stages': [2],
  'stage2_overrides': {'pd_json': 'datasets/pd_dataset_en/patients_dataset_en.json',
                       'base_path': 'datasets',
                       'image_size': 224,
                       'epochs': 50,
                       'batch_size': 16,
                       'num_workers': 4,
                       'lr': 0.0005,
                       'adapter_rank': 32,
                       'lambda_ce': 1.0,
                       'lambda_supcon': 1.0,
                       'lambda_recon': 0.05,
                       'tau_supcon': 0.1,
                       'kfold_mode': 'patient',
                       'num_folds': 5,
                       'eval_folds': 5,
                       'scheduler': 'cosine',
                       'early_stopping': True,
                       'patience': 8,
                       'classifier_dropout': 0.35,
                       'label_smooth_eps': 0.05,
                       'pair_mode': 'cartesian',
                       'max_pairs_per_patient': 8,
                       'kfold_repeats': 1,
                       'use_feature_diffusion': True,
                       'fdiff_class_cond': True,
                       'fdiff_gate_mode': 'soft_cos',
                       'fdiff_gate_threshold': 0.7,
                       'fdiff_gate_temperature': 0.1,
                       'fdiff_T_aug': 70,
                       'fdiff_k_aug': 2,
                       'fdiff_start_epoch': 3,
                       'lambda_fdiff': 0.12,
                       'lambda_ce_aug': 0.18,
                       'fdiff_minority_only': True,
                       'use_class_weights': False,
                       'use_focal': True,
                       'focal_gamma': 1.5,
                       'select_metric': 'balanced_patient',
                       'mode': 'mil',
                       'eval_patient_agg': True},
  'parent_ckpt_override': None,
  'stage0_overrides': {'topo_target': 'gt'},
  'stage1_overrides': {'region_pair_manifest': 'datasets/unlabeled_dataset_en/unlabeled_dataset_en.json',
                       'topo_target': 'frangi',
                       'epochs': 80},
  'stage2_tasks': ['pd_subtype_tdrest']},
 {'_group': 'ABLTDREST_NoFdiff_5f',
  '_desc': 'TDREST ablation: remove feature diffusion from TDREST5',
  'dataset': 'pd',
  'stages': [2],
  'stage2_overrides': {'pd_json': 'datasets/pd_dataset_en/patients_dataset_en.json',
                       'base_path': 'datasets',
                       'image_size': 224,
                       'epochs': 50,
                       'batch_size': 16,
                       'num_workers': 4,
                       'lr': 0.0005,
                       'adapter_rank': 32,
                       'lambda_ce': 1.0,
                       'lambda_supcon': 1.0,
                       'lambda_recon': 0.05,
                       'tau_supcon': 0.1,
                       'kfold_mode': 'patient',
                       'num_folds': 5,
                       'eval_folds': 5,
                       'scheduler': 'cosine',
                       'early_stopping': True,
                       'patience': 8,
                       'classifier_dropout': 0.35,
                       'label_smooth_eps': 0.05,
                       'pair_mode': 'cartesian',
                       'max_pairs_per_patient': 8,
                       'kfold_repeats': 1,
                       'use_feature_diffusion': False,
                       'fdiff_class_cond': True,
                       'fdiff_gate_mode': 'soft_cos',
                       'fdiff_gate_threshold': 0.7,
                       'fdiff_gate_temperature': 0.1,
                       'fdiff_T_aug': 70,
                       'fdiff_k_aug': 2,
                       'fdiff_start_epoch': 3,
                       'lambda_fdiff': 0.0,
                       'lambda_ce_aug': 0.0,
                       'fdiff_minority_only': True,
                       'use_class_weights': True,
                       'use_focal': True,
                       'focal_gamma': 1.5,
                       'select_metric': 'balanced_patient',
                       'mode': 'mil',
                       'eval_patient_agg': True},
  'parent_ckpt_override': None,
  'stage0_overrides': {'topo_target': 'gt'},
  'stage1_overrides': {'region_pair_manifest': 'datasets/unlabeled_dataset_en/unlabeled_dataset_en.json',
                       'topo_target': 'frangi',
                       'epochs': 80},
  'stage2_tasks': ['pd_subtype_tdrest']},
 {'_group': 'ABLTDREST_NoFocal_5f',
  '_desc': 'TDREST ablation: replace focal loss with weighted cross entropy',
  'dataset': 'pd',
  'stages': [2],
  'stage2_overrides': {'pd_json': 'datasets/pd_dataset_en/patients_dataset_en.json',
                       'base_path': 'datasets',
                       'image_size': 224,
                       'epochs': 50,
                       'batch_size': 16,
                       'num_workers': 4,
                       'lr': 0.0005,
                       'adapter_rank': 32,
                       'lambda_ce': 1.0,
                       'lambda_supcon': 1.0,
                       'lambda_recon': 0.05,
                       'tau_supcon': 0.1,
                       'kfold_mode': 'patient',
                       'num_folds': 5,
                       'eval_folds': 5,
                       'scheduler': 'cosine',
                       'early_stopping': True,
                       'patience': 8,
                       'classifier_dropout': 0.35,
                       'label_smooth_eps': 0.05,
                       'pair_mode': 'cartesian',
                       'max_pairs_per_patient': 8,
                       'kfold_repeats': 1,
                       'use_feature_diffusion': True,
                       'fdiff_class_cond': True,
                       'fdiff_gate_mode': 'soft_cos',
                       'fdiff_gate_threshold': 0.7,
                       'fdiff_gate_temperature': 0.1,
                       'fdiff_T_aug': 70,
                       'fdiff_k_aug': 2,
                       'fdiff_start_epoch': 3,
                       'lambda_fdiff': 0.12,
                       'lambda_ce_aug': 0.18,
                       'fdiff_minority_only': True,
                       'use_class_weights': True,
                       'use_focal': False,
                       'focal_gamma': 1.5,
                       'select_metric': 'balanced_patient',
                       'mode': 'mil',
                       'eval_patient_agg': True},
  'parent_ckpt_override': None,
  'stage0_overrides': {'topo_target': 'gt'},
  'stage1_overrides': {'region_pair_manifest': 'datasets/unlabeled_dataset_en/unlabeled_dataset_en.json',
                       'topo_target': 'frangi',
                       'epochs': 80},
  'stage2_tasks': ['pd_subtype_tdrest']},
 {'_group': 'ABLTDREST_TDWeight15_5f',
  '_desc': 'TDREST ablation: TD-sensitive TDREST7-style class weight multiplier',
  'dataset': 'pd',
  'stages': [2],
  'stage2_overrides': {'pd_json': 'datasets/pd_dataset_en/patients_dataset_en.json',
                       'base_path': 'datasets',
                       'image_size': 224,
                       'epochs': 50,
                       'batch_size': 16,
                       'num_workers': 4,
                       'lr': 0.0005,
                       'adapter_rank': 32,
                       'lambda_ce': 1.0,
                       'lambda_supcon': 1.0,
                       'lambda_recon': 0.05,
                       'tau_supcon': 0.1,
                       'kfold_mode': 'patient',
                       'num_folds': 5,
                       'eval_folds': 5,
                       'scheduler': 'cosine',
                       'early_stopping': True,
                       'patience': 8,
                       'classifier_dropout': 0.35,
                       'label_smooth_eps': 0.05,
                       'pair_mode': 'cartesian',
                       'max_pairs_per_patient': 8,
                       'kfold_repeats': 1,
                       'use_feature_diffusion': True,
                       'fdiff_class_cond': True,
                       'fdiff_gate_mode': 'soft_cos',
                       'fdiff_gate_threshold': 0.68,
                       'fdiff_gate_temperature': 0.12,
                       'fdiff_T_aug': 90,
                       'fdiff_k_aug': 2,
                       'fdiff_start_epoch': 3,
                       'lambda_fdiff': 0.15,
                       'lambda_ce_aug': 0.2,
                       'use_class_weights': True,
                       'class_weight_multipliers': [1.5, 1.0],
                       'select_metric': 'balanced_patient',
                       'mode': 'mil',
                       'eval_patient_agg': True},
  'parent_ckpt_override': None,
  'stage0_overrides': {'topo_target': 'gt'},
  'stage1_overrides': {'region_pair_manifest': 'datasets/unlabeled_dataset_en/unlabeled_dataset_en.json',
                       'topo_target': 'frangi',
                       'epochs': 80},
  'stage2_tasks': ['pd_subtype_tdrest']},
 {'_group': 'ABLTDREST_ZipMIL_5f',
  '_desc': 'TDREST ablation: zip pairing instead of cartesian patient bags',
  'dataset': 'pd',
  'stages': [2],
  'stage2_overrides': {'pd_json': 'datasets/pd_dataset_en/patients_dataset_en.json',
                       'base_path': 'datasets',
                       'image_size': 224,
                       'epochs': 50,
                       'batch_size': 16,
                       'num_workers': 4,
                       'lr': 0.0005,
                       'adapter_rank': 32,
                       'lambda_ce': 1.0,
                       'lambda_supcon': 1.0,
                       'lambda_recon': 0.05,
                       'tau_supcon': 0.1,
                       'kfold_mode': 'patient',
                       'num_folds': 5,
                       'eval_folds': 5,
                       'scheduler': 'cosine',
                       'early_stopping': True,
                       'patience': 8,
                       'classifier_dropout': 0.35,
                       'label_smooth_eps': 0.05,
                       'pair_mode': 'zip',
                       'max_pairs_per_patient': 0,
                       'kfold_repeats': 1,
                       'use_feature_diffusion': True,
                       'fdiff_class_cond': True,
                       'fdiff_gate_mode': 'soft_cos',
                       'fdiff_gate_threshold': 0.7,
                       'fdiff_gate_temperature': 0.1,
                       'fdiff_T_aug': 70,
                       'fdiff_k_aug': 2,
                       'fdiff_start_epoch': 3,
                       'lambda_fdiff': 0.12,
                       'lambda_ce_aug': 0.18,
                       'fdiff_minority_only': True,
                       'use_class_weights': True,
                       'use_focal': True,
                       'focal_gamma': 1.5,
                       'select_metric': 'balanced_patient',
                       'mode': 'mil',
                       'eval_patient_agg': True},
  'parent_ckpt_override': None,
  'stage0_overrides': {'topo_target': 'gt'},
  'stage1_overrides': {'region_pair_manifest': 'datasets/unlabeled_dataset_en/unlabeled_dataset_en.json',
                       'topo_target': 'frangi',
                       'epochs': 80},
  'stage2_tasks': ['pd_subtype_tdrest']}]

DM_TRANSFER_EXPERIMENTS = [{'_group': 'DMMIL3F1_weak_strict_fdiff',
  '_desc': 'Weaker/stricter fdiff: small augmentation pressure, high cosine gate (unified c8bf6b548656 '
           'Stage0/1 parent)',
  'dataset': 'dm',
  'stages': [2],
  'stage0_overrides': {},
  'stage1_overrides': {},
  'stage2_overrides': {'pd_json': 'datasets/dm_dataset_en/dm_patients_dataset_en.json',
                       'base_path': 'datasets',
                       'image_size': 224,
                       'epochs': 50,
                       'batch_size': 16,
                       'num_workers': 4,
                       'lr': 0.0005,
                       'adapter_rank': 32,
                       'lambda_ce': 1.0,
                       'lambda_supcon': 1.0,
                       'lambda_recon': 0.05,
                       'tau_supcon': 0.1,
                       'kfold_mode': 'patient',
                       'num_folds': 5,
                       'eval_folds': 5,
                       'kfold_repeats': 3,
                       'scheduler': 'cosine',
                       'early_stopping': True,
                       'patience': 12,
                       'classifier_dropout': 0.35,
                       'label_smooth_eps': 0.05,
                       'eval_patient_agg': True,
                       'select_metric': 'balanced_patient',
                       'dm_pair_mode': 'duplicate',
                       'mode': 'mil',
                       'finetune_fusion': True,
                       'fusion_lr_scale': 0.1,
                       'use_morphology_features': True,
                       'morph_embed_dim': 32,
                       'mil_balanced_bags': True,
                       'use_class_weights': True,
                       'use_feature_diffusion': True,
                       'fdiff_class_cond': True,
                       'fdiff_gate_mode': 'soft_cos',
                       'fdiff_gate_threshold': 0.78,
                       'fdiff_gate_temperature': 0.1,
                       'fdiff_T_aug': 60,
                       'fdiff_k_aug': 1,
                       'fdiff_start_epoch': 4,
                       'lambda_fdiff': 0.08,
                       'lambda_ce_aug': 0.06},
  'parent_ckpt_override': None,
  'stage2_tasks': ['dm_binary'],
  'paper_report_method': 'single_model'}]

EXPERIMENTS = {
    'main_tasks': MAIN_TASKS_EXPERIMENTS,
    'ablation': ABLATION_EXPERIMENTS,
    'dm_transfer': DM_TRANSFER_EXPERIMENTS,
}
PAPER_SECTION = os.getenv('RUN_PAPER_SECTION', RUN_SECTION)

_SIGNATURE_FILE = 'batch_signature.json'

_SIGNATURE_FIELDS = (
    'dataset', 'stages', 'stage0_overrides', 'stage1_overrides',
    'stage2_overrides', 'stage2_tasks',
    'oracle_experiments', 'oracle_mode', 'oracle_select_field',
    'parent_ckpt_override',
    'parent_checkpoint_sha256',
)


from experiments.topodise_experiments.main import TopoDiSEExperiment


class BatchRunner(TopoDiSEExperiment):
    """Selection, shared-pretrain scheduling, GPU placement and batch reports."""

    def _apply_reproducibility_mode(
        self, node: Dict[str, Any], mode: str, stage1_checkpoint: Optional[str],
    ) -> None:
        """将全局三态约定递归传给主实验和集成候选。"""
        stages = list(node.get('stages', [0, 1, 2]))
        if mode == 'checkpoint':
            if 2 in stages:
                node['stages'] = [2]
                node['parent_ckpt_override'] = stage1_checkpoint
        else:
            # strict/portable 都必须起步于随机初始化，不接入外部权重。
            node['parent_ckpt_override'] = None
            node.pop('parent_checkpoint_sha256', None)
            stage1_overrides = node.get('stage1_overrides')
            if isinstance(stage1_overrides, dict):
                stage1_overrides.pop('parent_ckpt', None)
            if 1 in stages and 0 not in stages:
                node['stages'] = [0, *stages]

        active_stages = list(node.get('stages', stages))
        for stage, key in ((0, 'stage0_overrides'), (1, 'stage1_overrides')):
            overrides = node.get(key)
            # Stage2-only 条目的空字典保持为空，让 fallback 补全 c8 默认配方；
            # 显式运行 Stage0/1 时，即便原 overrides 为空也必须注入所选模式。
            if isinstance(overrides, dict) and (overrides or stage in active_stages):
                overrides['reproducibility'] = mode
        stage2_overrides = node.get('stage2_overrides')
        if isinstance(stage2_overrides, dict):
            stage2_overrides['deterministic'] = mode in ('strict', 'checkpoint')
        for child in node.get('oracle_experiments') or []:
            self._apply_reproducibility_mode(child, mode, stage1_checkpoint)

    def _filter_experiments(self, exps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        kept = []
        for e in exps:
            g = e['_group']
            if self.INCLUDE_GROUPS or self.INCLUDE_PREFIXES:
                inc = g in self.INCLUDE_GROUPS or any(g.startswith(p) for p in self.INCLUDE_PREFIXES)
                if not inc:
                    continue
            if self.EXCLUDE_GROUPS and g in self.EXCLUDE_GROUPS:
                continue
            if self.EXCLUDE_PREFIXES and any(g.startswith(p) for p in self.EXCLUDE_PREFIXES):
                continue
            kept.append(e)
        return kept


    def _build_pretrain_index(self, exps: List[Dict[str, Any]]):
        """按相同的 Stage0/Stage1 配方对仅 Stage2 实验分组。

        返回配方索引与实验组到配方键的映射。
        """
        index: Dict[str, Dict[str, Any]] = {}
        assignment: Dict[str, str] = {}
        for e in exps:
            rec = self._fallback_recipe(e)
            if rec is None:
                continue
            s0, s1 = rec
            key = self._pretrain_key(s0, s1)
            index.setdefault(key, {'stage0_overrides': s0, 'stage1_overrides': s1})
            assignment[e['_group']] = key
        return index, assignment


    def _merge_batch_summary_payloads(self, existing: Optional[Dict[str, Any]],
                                      current: Dict[str, Any]) -> Dict[str, Any]:
        merged: Dict[str, Any] = dict(existing) if isinstance(existing, dict) else {}
        for key, value in current.items():
            if key != 'experiments':
                merged[key] = value
        merged['updated'] = datetime.now().isoformat()

        by_group: Dict[str, Dict[str, Any]] = {}
        order: List[str] = []

        def add_experiments(items: Any):
            if not isinstance(items, list):
                return
            for item in items:
                if not isinstance(item, dict):
                    continue
                group = item.get('_group')
                if not group:
                    continue
                group = str(group)
                if group not in by_group:
                    order.append(group)
                by_group[group] = item

        add_experiments((existing or {}).get('experiments') if isinstance(existing, dict) else None)
        add_experiments(current.get('experiments'))
        merged['experiments'] = [by_group[group] for group in order]
        return merged


    def _save_batch_summary(self, output_dir: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        path = os.path.join(output_dir, 'batch_summary.json')
        lock_path = path + '.lock'
        tmp_path = f'{path}.tmp.{os.getpid()}'
        os.makedirs(output_dir, exist_ok=True)

        with open(lock_path, 'w') as lock_file:
            if fcntl is not None:
                fcntl.flock(lock_file, fcntl.LOCK_EX)
            try:
                existing: Optional[Dict[str, Any]] = None
                if os.path.exists(path):
                    try:
                        with open(path, 'r') as f:
                            loaded = json.load(f)
                        if isinstance(loaded, dict):
                            existing = loaded
                    except (json.JSONDecodeError, OSError):
                        existing = None
                merged = self._merge_batch_summary_payloads(existing, payload)
                with open(tmp_path, 'w') as f:
                    json.dump(merged, f, indent=2, default=str)
                os.replace(tmp_path, path)
                return merged
            finally:
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass
                if fcntl is not None:
                    fcntl.flock(lock_file, fcntl.LOCK_UN)


    def _worker_run(self, gpu_id: int, exp: Dict[str, Any], output_dir: str, q):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        try:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError(f'GPU {gpu_id} was requested but CUDA is unavailable')
            torch.set_num_threads(1)
            exp = copy.deepcopy(exp)
            run_output_dir = exp.pop('_output_dir_override', output_dir)
            print(f"  [worker] start {exp['_group']} on gpu{gpu_id}", flush=True)
            s = self.run_experiment(exp, run_output_dir)
            q.put((exp['_group'], s, None))
        except Exception as e:
            q.put((exp['_group'], None, f"{e}\n{traceback.format_exc()}"))


    def _run_parallel(self, exps: List[Dict[str, Any]], output_dir: str,
                      gpus: List[int], concurrency_per_gpu: int = 1) -> List[Dict[str, Any]]:
        """GPU 感知调度。

        关键约束：含 Stage0/Stage1 的「重型」作业（完整扩散预训练，显存占用大）必须
        独占整张 GPU；仅含 Stage2 的「轻型」作业可在同卡上按 concurrency_per_gpu 共享。
        这样即使 CONCURRENCY_PER_GPU>=2，也不会出现同一张卡同时跑多个 Stage0/Stage1。
        """
        ctx = multiprocessing.get_context('spawn')
        q = ctx.Queue()

        def _is_heavy(e: Dict[str, Any]) -> bool:
            stages = list(e.get('stages', [0, 1, 2]))
            if self.STAGE_FILTER is not None:
                stages = [s for s in stages if s in self.STAGE_FILTER]
            if any(s in (0, 1) for s in stages):
                return True
            # 仅含 Stage2 但父 ckpt 缺失/无效的作业，run_experiment 会在运行时
            # auto-fallback 补跑 Stage0/1 -> 实际也是重型，必须独占 GPU。
            return self._fallback_recipe(e) is not None

        pending = list(exps)
        procs: Dict[str, Any] = {}          # 实验组 -> 进程
        placement: Dict[str, tuple] = {}    # 实验组 -> (GPU 编号, 是否为重型作业)
        gpu_heavy: Dict[int, int] = {g: 0 for g in gpus}   # 重型作业数（<=1）
        gpu_light: Dict[int, int] = {g: 0 for g in gpus}   # 轻型作业数
        results: Dict[str, Any] = {}

        def _pick_gpu(heavy: bool) -> Optional[int]:
            best = None
            for g in gpus:
                if heavy:
                    # 重型作业要求整卡空闲（无重型、无轻型同租）
                    if gpu_heavy[g] == 0 and gpu_light[g] == 0:
                        return g
                else:
                    # 轻型作业：该卡无重型作业，且轻型未占满
                    if gpu_heavy[g] == 0 and gpu_light[g] < concurrency_per_gpu:
                        if best is None or gpu_light[g] < gpu_light[best]:
                            best = g
            return best

        def _try_launch():
            # 重型优先，避免被持续涌入的轻型作业饿死（轻型可去其它卡）
            progressed = True
            while progressed and pending:
                progressed = False
                order = sorted(range(len(pending)),
                               key=lambda i: 0 if _is_heavy(pending[i]) else 1)
                for i in order:
                    exp = pending[i]
                    heavy = _is_heavy(exp)
                    g = _pick_gpu(heavy)
                    if g is None:
                        continue
                    pending.pop(i)
                    p = ctx.Process(target=self._worker_run, args=(g, exp, output_dir, q))
                    p.start()
                    procs[exp['_group']] = p
                    placement[exp['_group']] = (g, heavy)
                    if heavy:
                        gpu_heavy[g] += 1
                    else:
                        gpu_light[g] += 1
                    kind = 'heavy' if heavy else 'light'
                    print(f"  [parallel] gpu{g} ({kind}) -> {exp['_group']}")
                    progressed = True
                    break  # pending 已变，重新排序扫描

        def _finish_group(group: str, s, err: Optional[str]):
            gpu_id, heavy = placement.pop(group)
            results[group] = s if s else {'_group': group, 'error': err}
            if heavy:
                gpu_heavy[gpu_id] -= 1
            else:
                gpu_light[gpu_id] -= 1
            procs.pop(group).join()
            if err:
                print(f"  [parallel] failed {group} (gpu{gpu_id})\n{err}")
            else:
                print(f"  [parallel] done {group} (gpu{gpu_id})")
            _try_launch()

        _try_launch()
        while procs:
            try:
                group, s, err = q.get(timeout=10)
                _finish_group(group, s, err)
                continue
            except queue.Empty:
                pass

            # 如果工作进程在写入队列前退出（如本地段错误、CUDA 内存不足或进程级中止），
            # 不能让 q.get() 永久阻塞。
            dead_groups = [g for g, p in procs.items() if not p.is_alive()]
            for group in dead_groups:
                exitcode = procs[group].exitcode
                _finish_group(group, None, f"worker exited without result (exitcode={exitcode})")

        return [results[g] for g in (e['_group'] for e in exps) if g in results]


    def _run_parallel_oracle_groups(self, exps: List[Dict[str, Any]], output_dir: str,
                                    gpus: List[int]) -> List[Dict[str, Any]]:
        flat_children: List[Dict[str, Any]] = []
        parents: Dict[str, Dict[str, Any]] = {}

        for exp in exps:
            group = exp['_group']
            parent_ckpt = exp.get('parent_ckpt_override') or self.EXTERNAL_PARENT_CKPT or None
            parent_ckpt = self._resolve_project_path(parent_ckpt) if parent_ckpt else None
            if not parent_ckpt or not os.path.exists(parent_ckpt):
                print(f"[oracle-flat][warn] {group} 缺少可用 parent ckpt，回退为串行父 oracle 调度")
                return [self.run_experiment(e, output_dir) for e in exps]

            exp_dir = os.path.join(output_dir, group)
            child_output_dir = os.path.join(exp_dir, 'oracle_candidates')
            os.makedirs(child_output_dir, exist_ok=True)

            requested_stages = list(exp.get('stages', [0, 1, 2]))
            if self.STAGE_FILTER is not None:
                requested_stages = [s for s in requested_stages if s in self.STAGE_FILTER]
            with open(os.path.join(exp_dir, '_experiment_def.json'), 'w') as f:
                json.dump({k: v for k, v in exp.items() if k != 'stage2_tasks'} | {
                    'stage2_tasks': exp.get('stage2_tasks', []),
                    'resolved_stages': requested_stages,
                    'auto_fallback': None,
                    'reproducibility': self.REPRODUCIBILITY_MODE,
                    'checkpoint_bundle': self.CHECKPOINT_BUNDLE,
                    'site': self.RUN_SITE,
                }, f, indent=2)

            summary: Dict[str, Any] = {
                '_group': group,
                '_desc': exp.get('_desc', ''),
                'dataset': exp.get('dataset', 'pd'),
                'stages_run': requested_stages,
                'started': datetime.now().isoformat(),
                'auto_fallback': None,
                'reproducibility': self.REPRODUCIBILITY_MODE,
                'checkpoint_bundle': self.CHECKPOINT_BUNDLE,
                'stage_results': {},
                'stage2_results': {
                    '_oracle_meta': {
                        'status': 'running',
                        'started': datetime.now().isoformat(),
                        'candidates': [c.get('_group') for c in exp.get('oracle_experiments') or []],
                        'tasks': exp.get('stage2_tasks', []),
                    }
                },
            }
            self._save_summary(summary, exp_dir)

            candidates = copy.deepcopy(exp.get('oracle_experiments') or [])
            child_names: List[str] = []
            for child in candidates:
                child['parent_ckpt_override'] = parent_ckpt
                child['stages'] = [2]
                child['_output_dir_override'] = child_output_dir
                child_names.append(child['_group'])
                flat_children.append(child)
            parents[group] = {
                'exp': exp,
                'exp_dir': exp_dir,
                'summary': summary,
                'child_names': child_names,
            }

        print(f"[oracle-flat] running {len(flat_children)} candidates from {len(exps)} "
              f"oracle groups on shared gpus={gpus}")
        flat_summaries = self._run_parallel(flat_children, output_dir, gpus,
                                       concurrency_per_gpu=self.CONCURRENCY_PER_GPU)
        by_child = {s.get('_group'): s for s in flat_summaries if isinstance(s, dict)}

        results: List[Dict[str, Any]] = []
        for exp in exps:
            group = exp['_group']
            rec = parents[group]
            child_summaries = [by_child[name] for name in rec['child_names'] if name in by_child]
            summary = rec['summary']
            summary['stage2_results'].update(
                (self._aggregate_subtype_hierarchy_oracle(
                    group,
                    child_summaries,
                    exp.get('oracle_select_field', 'best_val_acc'),
                ) if exp.get('oracle_mode') == 'subtype_hierarchy' else self._aggregate_oracle_results(
                    group,
                    child_summaries,
                    exp.get('stage2_tasks', []),
                    exp.get('oracle_select_field', 'best_val_acc'),
                    exp.get('auto_patient_logit_ensemble', True),
                ))
            )
            summary['finished'] = datetime.now().isoformat()
            self._save_summary(summary, rec['exp_dir'])
            self._save_patient_logit_ensemble_outputs(summary, rec['exp_dir'])
            results.append(summary)

        return results


    def _run_batch(self, exps: List[Dict[str, Any]], output_dir: str) -> List[Dict[str, Any]]:
        """Run a list of experiments via the configured parallel/serial scheduler.

        断点续跑：已全部完成的实验在调度前就被跳过（直接复用旧 summary），不再
        占用 GPU 槽位；其余实验进入正常调度，并在 run_experiment 内接续未完成部分。
        """
        done_results: List[Dict[str, Any]] = []
        pending: List[Dict[str, Any]] = []
        for e in exps:
            prev = self._experiment_done_on_disk(e, output_dir)
            if prev is not None:
                n_stage = len([k for k in (prev.get('stage_results') or {})])
                n_task = len([k for k in (prev.get('stage2_results') or {})
                              if not k.startswith('_')])
                print(f"[resume][skip] {e['_group']} 已完成"
                      f"(stage={n_stage}, stage2_tasks={n_task})，跳过调度")
                done_results.append(prev)
            else:
                pending.append(e)

        if pending:
            ran = []
            normal_pending = [e for e in pending if not e.get('oracle_experiments')]
            oracle_pending = [e for e in pending if e.get('oracle_experiments')]
            print(f"[batch][schedule] pending={len(pending)} normal={len(normal_pending)} "
                  f"oracle={len(oracle_pending)} parallel={self.PARALLEL_EXPERIMENTS} "
                  f"gpus={self.AVAILABLE_GPUS} concurrency_per_gpu={self.CONCURRENCY_PER_GPU}")
            if normal_pending:
                print("[batch][schedule] normal_pending="
                      + ",".join(e['_group'] for e in normal_pending))
            if normal_pending:
                if self.PARALLEL_EXPERIMENTS and self.AVAILABLE_GPUS:
                    ran.extend(self._run_parallel(normal_pending, output_dir, self.AVAILABLE_GPUS,
                                             concurrency_per_gpu=self.CONCURRENCY_PER_GPU))
                else:
                    if self.PARALLEL_EXPERIMENTS and not self.AVAILABLE_GPUS:
                        print('[batch][warn] parallel=True but no AVAILABLE_GPUS; '
                              'running normal experiments serially. Set '
                              'RUN_AVAILABLE_GPUS=0,1,2,3.')
                    for e in normal_pending:
                        print(f"\n[batch] -> {e['_group']}: {e.get('_desc','')}")
                        ran.append(self.run_experiment(e, output_dir))
            for e in oracle_pending:
                print(f"\n[batch][oracle] -> {e['_group']}: {e.get('_desc','')}")
            if self.PARALLEL_EXPERIMENTS and self.AVAILABLE_GPUS and len(oracle_pending) > 1:
                ran.extend(self._run_parallel_oracle_groups(oracle_pending, output_dir, self.AVAILABLE_GPUS))
            else:
                for e in oracle_pending:
                    ran.append(self.run_experiment(e, output_dir))
        else:
            ran = []

        # 维持原始实验顺序输出结果（跳过的 + 新跑的）。
        by_group: Dict[str, Any] = {}
        for s in done_results + ran:
            if isinstance(s, dict) and s.get('_group'):
                by_group[s['_group']] = s
        return [by_group[e['_group']] for e in exps if e['_group'] in by_group]


    def _reduce_exp_for_signature(self, exp: Dict[str, Any]) -> Dict[str, Any]:
        red: Dict[str, Any] = {'_group': exp.get('_group')}
        for f in self._SIGNATURE_FIELDS:
            if f in exp:
                red[f] = exp[f]
        return red


    def _compute_batch_signature(self, experiments: List[Dict[str, Any]], smoke: bool) -> str:
        """计算与顺序无关的完整实验配置稳定哈希。"""
        reduced = sorted((self._reduce_exp_for_signature(e) for e in experiments),
                         key=lambda d: str(d.get('_group')))
        payload = json.dumps(
            {'experiments': reduced, 'smoke': bool(smoke), 'site': self.RUN_SITE},
            sort_keys=True, default=str,
        )
        return hashlib.md5(payload.encode('utf-8')).hexdigest()[:12]


    def _write_batch_signature(self, output_dir: str, signature: str, n_exps: int,
                               smoke: bool):
        try:
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, self._SIGNATURE_FILE), 'w') as f:
                json.dump({
                    'signature': signature, 'site': self.RUN_SITE,
                    'smoke': bool(smoke), 'n_experiments': n_exps,
                    'updated': datetime.now().isoformat(),
                }, f, indent=2)
        except OSError as e:
            print(f"[resume][warn] 无法写入批次指纹 {output_dir}: {e}")


    def _find_matching_batch_dir(self, signature: str) -> Optional[str]:
        """返回已保存指纹与 ``signature`` 匹配且最近更新的批次目录；未找到时返回 None。"""
        base = self.OUTPUT_BASE_DIR
        if not base or not os.path.isdir(base):
            return None
        matches: List[tuple] = []
        for name in os.listdir(base):
            d = os.path.join(base, name)
            sig_path = os.path.join(d, self._SIGNATURE_FILE)
            if not os.path.isfile(sig_path):
                continue
            try:
                with open(sig_path, 'r') as f:
                    meta = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            if meta.get('signature') == signature:
                matches.append((os.path.getmtime(sig_path), d))
        if not matches:
            return None
        matches.sort(reverse=True)  # 最近更新的优先
        return matches[0][1]


    def _resolve_output_dir(self, signature: str, n_exps: int, smoke: bool,
                            explicit_output_dir: Optional[str],
                            persist: bool = True) -> str:
        """按显式路径、RUN_BATCH_TAG、配置匹配续跑、新时间戳目录的优先级决定输出目录。

        ``persist`` 开启时会将指纹写入或重写到该目录。
        """
        if explicit_output_dir:
            out = self._resolve_project_path(explicit_output_dir)
            print(f"[resume] 使用显式 --output-dir: {out}")
        elif self._env_batch_tag:
            out = self._resolve_project_path(self.OUTPUT_DIR)  # OUTPUT_DIR 已含 RUN_BATCH_TAG
            print(f"[resume] 使用 RUN_BATCH_TAG={self._env_batch_tag} -> {out}")
        else:
            matched = self._find_matching_batch_dir(signature) if self.AUTO_RESUME else None
            if matched:
                out = matched
                print(f"[resume] 配置指纹匹配到既有批次，自动接续: {out}\n"
                      f"         (signature={signature})")
            else:
                out = self._resolve_project_path(self.OUTPUT_DIR)  # 新建带时间戳目录
                reason = 'auto_resume 关闭' if not self.AUTO_RESUME else '未找到匹配批次'
                print(f"[resume] {reason}，使用新目录: {out} (signature={signature})")
        if persist:
            self._write_batch_signature(out, signature, n_exps, smoke)
        return out


    def run(self, argv=None):
        ap = argparse.ArgumentParser()
        ap.add_argument('--dry-run', action='store_true',
                        help='print resolved experiment table and exit')
        ap.add_argument('--output-dir', default=None,
                        help='显式指定输出目录；默认按实验配置指纹自动匹配既有批次目录'
                             '（可接续），无匹配时新建带时间戳目录。')
        ap.add_argument('--smoke', action='store_true',
                        help='override every stage to 1-epoch tiny config')
        ap.add_argument('--group', '--groups', dest='groups', default=None, help='Exact group names, comma separated')
        ap.add_argument('--section', choices=['main_tasks', 'ablation', 'dm_transfer', 'all'], default=self.PAPER_SECTION)
        ap.add_argument('--gpus', default=None, help='0,1 使用两张卡；cpu 使用 CPU')
        ap.add_argument('--concurrency-per-gpu', type=int, default=self.CONCURRENCY_PER_GPU)
        ap.add_argument('--num-workers', type=int, default=self.NUM_WORKERS)
        ap.add_argument(
            '--stage2-kfold-repeats', type=int, default=None,
            help='覆盖 Stage2 的 K-fold 重复次数；论文配置为 3',
        )
        repro_group = ap.add_mutually_exclusive_group()
        repro_group.add_argument(
            '--reproducibility', nargs='?', const='strict',
            choices=list(REPRODUCIBILITY_MODES),
            help='strict=确定性从零近似复刻；portable=设备自适应从零训练；'
                 'checkpoint=直接使用 c8 Stage1 权重',
        )
        repro_group.add_argument(
            '--no-reproducibility', dest='reproducibility', action='store_const',
            const='portable',
            help='兼容别名：等价于 --reproducibility portable',
        )
        ap.set_defaults(reproducibility=None)
        ap.add_argument(
            '--checkpoint-dir', default=self.CHECKPOINT_DIR,
            help='checkpoint 模式的预训练包根目录；最少须包含 Stage1 .pt',
        )
        ap.add_argument('--list', action='store_true', help='List selected groups without training')
        args = ap.parse_args(argv)
        if args.gpus is not None:
            self.AVAILABLE_GPUS = [] if args.gpus.lower() in ('cpu', 'none', '') else [int(g) for g in args.gpus.split(',')]
        if len(set(self.AVAILABLE_GPUS)) != len(self.AVAILABLE_GPUS):
            ap.error('Duplicate GPU ids')
        if args.concurrency_per_gpu < 1 or args.num_workers < 0:
            ap.error('并发数必须为正，num-workers 必须非负')
        self.CONCURRENCY_PER_GPU = args.concurrency_per_gpu
        self.NUM_WORKERS = 0 if args.smoke else args.num_workers
        try:
            mode = _normalize_reproducibility_mode(
                args.reproducibility
                if args.reproducibility is not None
                else os.environ.get('TOPODISE_REPRODUCIBILITY', 'portable')
            )
        except ValueError as error:
            ap.error(str(error))
        checkpoint_bundle = None
        if mode == 'checkpoint':
            if not args.checkpoint_dir:
                ap.error('--reproducibility checkpoint requires --checkpoint-dir')
            try:
                checkpoint_bundle = resolve_checkpoint_bundle(
                    self._resolve_project_path(args.checkpoint_dir)
                )
            except (FileNotFoundError, ValueError) as error:
                ap.error(str(error))
            self.EXTERNAL_PARENT_CKPT = checkpoint_bundle['stage1_checkpoint']
        else:
            # 从零模式不会因历史配方缓存而偷换初始权重。
            self.EXTERNAL_PARENT_CKPT = None
        self.REPRODUCIBILITY_MODE = mode
        self.CHECKPOINT_BUNDLE = checkpoint_bundle
        self.DEFAULT_STAGE0_OVERRIDES['reproducibility'] = mode
        self.DEFAULT_STAGE1_OVERRIDES['reproducibility'] = mode
        os.environ['TOPODISE_REPRODUCIBILITY'] = mode
        os.environ['TOPODISE_DETERMINISTIC'] = (
            '1' if mode in ('strict', 'checkpoint') else '0'
        )
        if mode in ('strict', 'checkpoint'):
            os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        if mode == 'strict' and not args.smoke and args.num_workers != 4:
            ap.error('strict 模式的 c8 近似复刻固定 --num-workers 4')
        if not self.AVAILABLE_GPUS:
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
        elif not self.PARALLEL_EXPERIMENTS:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.AVAILABLE_GPUS[0])
        pass  # runtime state belongs to this batch instance
        sections = ['main_tasks', 'ablation', 'dm_transfer'] if args.section == 'all' else [args.section]
        self.EXPERIMENTS = copy.deepcopy([e for section in sections for e in self.EXPERIMENTS[section]])
        for experiment in self.EXPERIMENTS:
            self._apply_reproducibility_mode(
                experiment, mode,
                checkpoint_bundle['stage1_checkpoint'] if checkpoint_bundle else None,
            )
            if checkpoint_bundle is not None:
                experiment['parent_checkpoint_sha256'] = checkpoint_bundle['stage1_sha256']

        exps = self._filter_experiments(self.EXPERIMENTS)
        if args.groups:
            selected = [g.strip() for g in args.groups.split(',') if g.strip()]
            unknown = set(selected) - {e['_group'] for e in exps}
            if unknown:
                ap.error(f'Unknown or excluded groups: {sorted(unknown)}')
            exps = [e for e in exps if e['_group'] in selected]
        if args.stage2_kfold_repeats is not None:
            if args.stage2_kfold_repeats < 1:
                ap.error('--stage2-kfold-repeats must be positive')
            def apply_stage2_kfold_repeats(node):
                overrides = node.get('stage2_overrides')
                if isinstance(overrides, dict):
                    overrides['kfold_repeats'] = args.stage2_kfold_repeats
                for child in node.get('oracle_experiments') or []:
                    apply_stage2_kfold_repeats(child)
            for exp in exps:
                apply_stage2_kfold_repeats(exp)
        if not exps:
            ap.error('No experiments selected')
        if mode == 'checkpoint' and any(2 not in exp.get('stages', []) for exp in exps):
            ap.error('checkpoint 模式只能运行包含 Stage2 的实验')
        if args.list:
            for exp in exps:
                print(f"{exp['_group']}: {exp.get('_desc', '')}")
            return
        # ---- 按实验配置自动选定输出目录（实现"重新运行即接续"）----
        # 指纹基于完整 EXPERIMENTS 定义（含 smoke/site），与 include/exclude 子集无关，
        # 因此分批跑同一份实验定义会落入同一目录，互相接续。
        batch_signature = self._compute_batch_signature(self.EXPERIMENTS, args.smoke)
        args.output_dir = self._resolve_output_dir(
            batch_signature, len(self.EXPERIMENTS), args.smoke, args.output_dir,
            persist=not args.dry_run)
        batch_tag = os.path.basename(os.path.normpath(args.output_dir))
        print('=' * 72)
        print(f'TopoDiSE batch experiments  ({datetime.now().isoformat()})')
        print(f'site={self.RUN_SITE}  gpus={self.AVAILABLE_GPUS}  parallel={self.PARALLEL_EXPERIMENTS}')
        print(f'concurrency_per_gpu={self.CONCURRENCY_PER_GPU}')
        print(f'parallel_slots_total={len(self.AVAILABLE_GPUS) * self.CONCURRENCY_PER_GPU if self.AVAILABLE_GPUS else 0}')
        if self.PARALLEL_EXPERIMENTS and not self.AVAILABLE_GPUS:
            print('WARNING: parallel scheduling is enabled but no GPU ids are configured; '
                  'experiments will run serially unless RUN_AVAILABLE_GPUS is set.')
        print(f'auto_resume={self.AUTO_RESUME}')
        print(f'reproducibility={mode}')
        if checkpoint_bundle:
            print(f"checkpoint_dir={checkpoint_bundle['checkpoint_dir']}")
            print(f"stage1_checkpoint={checkpoint_bundle['stage1_checkpoint']}")
            print(f"stage1_sha256={checkpoint_bundle['stage1_sha256']}")
        print(f'codes_root={self.PROJECT_ROOT}')
        print(f'datasets_root={self.DATASETS_ROOT}')
        print(f'output_base_dir={self.OUTPUT_BASE_DIR}')
        print(f'batch_signature={batch_signature}')
        print(f'batch_tag={batch_tag}')
        print(f'output_dir={args.output_dir}')
        print(f'manual_run_groups={self.RUN_GROUPS}')
        print(f'include_groups={self.INCLUDE_GROUPS} include_prefixes={self.INCLUDE_PREFIXES}')
        print(f'exclude_groups={self.EXCLUDE_GROUPS} exclude_prefixes={self.EXCLUDE_PREFIXES}')
        print(f'stage_filter={self.STAGE_FILTER or "(use per-exp stages)"}')
        print(f'#experiments resolved: {len(exps)} / {len(self.EXPERIMENTS)}')
        print('=' * 72)
        for e in exps:
            print(f"  - {e['_group']:24s} ds={e['dataset']:3s} "
                  f"stages={e.get('stages', [0,1,2])} task={e.get('stage2_tasks', [])}")
            print(f"    {e.get('_desc','')}")
        print('=' * 72)
        if args.dry_run:
            print(json.dumps(exps, indent=2, ensure_ascii=False))
            return

        # ---- Stage0/1 共享预训练去重计划（从未经 smoke 修改的原始配方构建）----
        pre_index, pre_assign = self._build_pretrain_index(exps)

        pre_exps: List[Dict[str, Any]] = []
        for key, rec in pre_index.items():
            s1_parent = (rec.get('stage1_overrides') or {}).get('parent_ckpt')
            s1_parent = self._resolve_project_path(s1_parent) if s1_parent else None
            e = {
                '_group': f'_pretrain_{key}',
                '_desc': 'shared Stage0/1 pretrain (recipe ' + key + ') for '
                         + ','.join(g for g, k in pre_assign.items() if k == key),
                'dataset': 'pd',
                'stages': [1] if s1_parent and os.path.exists(s1_parent) else [0, 1],
                'stage0_overrides': rec['stage0_overrides'],
                'stage1_overrides': rec['stage1_overrides'],
                'stage2_tasks': [],
            }
            if s1_parent and os.path.exists(s1_parent):
                e['parent_ckpt_override'] = s1_parent
                e['_desc'] = e['_desc'].replace('Stage0/1', 'Stage1-adapt')
            pre_exps.append(e)

        if args.smoke:
            for e in exps:
                self._apply_smoke_overrides(e)
            for e in pre_exps:
                self._apply_smoke_overrides(e)

        os.makedirs(args.output_dir, exist_ok=True)
        t0 = time.time()

        # ---- Phase 1: 去重预训练（每个唯一 Stage0/1 配方只训一次）----
        pre_summaries: List[Dict[str, Any]] = []
        if pre_exps:
            print(f"[dedup] {len(pre_exps)} 个唯一 Stage0/1 配方，供 {len(pre_assign)} 个 "
                  f"stage2-only 实验共享；每个配方只预训练一次")
            pre_summaries = self._run_batch(pre_exps, args.output_dir)
            shared_stage1: Dict[str, str] = {}
            for s in pre_summaries:
                g = s.get('_group', '')
                ck = (s.get('stage_results') or {}).get('stage1', {}).get('ckpt_path')
                if g.startswith('_pretrain_') and ck:
                    shared_stage1[g[len('_pretrain_'):]] = ck
            for e in exps:
                key = pre_assign.get(e['_group'])
                if key and key in shared_stage1:
                    e['parent_ckpt_override'] = shared_stage1[key]
                    e['stages'] = [2]  # 预训练已完成，下游只跑 stage2
                elif key:
                    print(f"  [dedup][warn] {e['_group']} 的预训练(key={key}) 未产出 ckpt; "
                          f"将就地补跑 Stage0/1（auto-fallback）")

        # ---- Phase 2: 下游实验（共享父 ckpt，只跑各自的 stage2）----
        summaries = self._run_batch(exps, args.output_dir)
        all_summaries = pre_summaries + summaries

        merged_summary = self._save_batch_summary(args.output_dir, {
            'site': self.RUN_SITE, 'gpus': self.AVAILABLE_GPUS,
            'parallel': self.PARALLEL_EXPERIMENTS,
            'reproducibility': mode,
            'checkpoint_bundle': checkpoint_bundle,
            'pretrain_recipes': len(pre_exps),
            'total_elapsed_s': time.time() - t0,
            'experiments': all_summaries,
        })
        if any(s.get('status') in ('failed', 'error') or s.get('error') for s in all_summaries):
            raise SystemExit('TopoDiSE experiment failure; see batch_summary.json')
        print('\n' + '=' * 72)
        print(f'batch done in {time.time()-t0:.1f}s -> {args.output_dir}/batch_summary.json '
              f"(merged_experiments={len(merged_summary.get('experiments', []))})")
        print('=' * 72)



def main(argv=None):
    settings = {
        'AUTO_RESUME': AUTO_RESUME,
        'AVAILABLE_GPUS': AVAILABLE_GPUS,
        'BATCH_TAG': BATCH_TAG,
        'CONCURRENCY_PER_GPU': CONCURRENCY_PER_GPU,
        'CORN1_ROOT': CORN1_ROOT,
        'CHECKPOINT_DIR': CHECKPOINT_DIR,
        'DATASETS_ROOT': DATASETS_ROOT,
        'DEFAULT_STAGE0_OVERRIDES': DEFAULT_STAGE0_OVERRIDES,
        'DEFAULT_STAGE1_OVERRIDES': DEFAULT_STAGE1_OVERRIDES,
        'EXCLUDE_GROUPS': EXCLUDE_GROUPS,
        'EXCLUDE_PREFIXES': EXCLUDE_PREFIXES,
        'EXTERNAL_PARENT_CKPT': EXTERNAL_PARENT_CKPT,
        'INCLUDE_GROUPS': INCLUDE_GROUPS,
        'INCLUDE_PREFIXES': INCLUDE_PREFIXES,
        'NUM_WORKERS': NUM_WORKERS,
        'OUTPUT_BASE_DIR': OUTPUT_BASE_DIR,
        'OUTPUT_DIR': OUTPUT_DIR,
        'PAPER_SECTION': PAPER_SECTION,
        'PARALLEL_EXPERIMENTS': PARALLEL_EXPERIMENTS,
        'PD_JSON': PD_JSON,
        'PROJECT_ROOT': PROJECT_ROOT,
        'RESUME_SEARCH_DIRS': RESUME_SEARCH_DIRS,
        'RUN_AUTO_RESUME': RUN_AUTO_RESUME,
        'RUN_CONCURRENCY_PER_GPU': RUN_CONCURRENCY_PER_GPU,
        'RUN_GROUPS': RUN_GROUPS,
        'RUN_PARALLEL_EXPERIMENTS': RUN_PARALLEL_EXPERIMENTS,
        'RUN_SITE': RUN_SITE,
        'STAGE_FILTER': STAGE_FILTER,
        '_SIGNATURE_FIELDS': _SIGNATURE_FIELDS,
        '_SIGNATURE_FILE': _SIGNATURE_FILE,
        '_env_batch_tag': _env_batch_tag,
        'EXPERIMENTS': EXPERIMENTS,
    }
    return BatchRunner(settings).run(argv)


if __name__ == '__main__':
    main()
