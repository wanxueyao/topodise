"""
gnn_main.py - GNN 实验单次运行入口

使用 codes/datasets/pd_dataset_unet_en 数据集，
通过 codes/gnn.GNNClassifier 进行 K-fold 训练和评估。

用法:
    cd /path/to/ARIS/codes
    python experiments/gnn_experiments/gnn_main.py
"""

import os
import sys
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any

import numpy as np
import torch

# ─── 路径配置 ─────────────────────────────────────────────────────────────────
_CODES_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _CODES_DIR not in sys.path:
    sys.path.insert(0, _CODES_DIR)

from core.data_loader import (
    load_and_filter_dataset,
    stratified_kfold,
    DATASET_GROUPS,
    SUBTYPE_MAP,
)
from utils.patient_outputs import save_fold_patient_outputs
from gnn.classifier import GNNClassifier, GNNClassifierConfig


# ─── 实验配置 ─────────────────────────────────────────────────────────────────

@dataclass
class GNNExperimentConfig:
    """GNN 单次实验的全部超参数。"""
    name: str = 'gnn_experiment'

    # ── 数据 ────────────────────────────────────────────────────────────────────
    dataset_json: str = 'datasets/pd_dataset_unet_en/patients_dataset_en.json'
    dataset_group: str = 'pd_control'   # 见 DATASET_GROUPS
    num_folds: int = 2
    kfold_mode: str = 'patient'
    graph_cache_dir: Optional[str] = None   # None = 自动使用 results/graph_cache

    # ── 模型 ────────────────────────────────────────────────────────────────────
    gnn_type: str = 'gat'           # 'gat' | 'graphsage'
    hidden_dim: int = 64
    num_layers: int = 2
    num_heads: int = 4
    shared_encoder: bool = False

    # ── 训练 ────────────────────────────────────────────────────────────────────
    lr: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = 'adam'
    lr_scheduler: str = 'cosine'
    warmup_epochs: int = 5
    epochs: int = 50
    patience: int = 10
    batch_size: int = 16
    dropout: float = 0.2
    classifier_dropout: float = 0.3

    # ── 损失 ────────────────────────────────────────────────────────────────────
    pd_loss_weight: float = 1.0
    stage_loss_weight: float = 0.5
    subtype_loss_weight: float = 0.5
    label_smoothing: float = 0.0

    # ── 其他 ────────────────────────────────────────────────────────────────────
    primary_task: str = 'pd'        # 'pd' | 'stage' | 'subtype'
    num_workers: int = 0
    seed: int = 42
    reuse_checkpoints: bool = True


# ─── 日志设置 ─────────────────────────────────────────────────────────────────

def _setup_logging(output_dir: str, name: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, f'{name}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[
            logging.FileHandler(log_path, mode='w', encoding='utf-8'),
            logging.StreamHandler(),
        ],
        force=True,
    )


# ─── 指标汇总 ─────────────────────────────────────────────────────────────────

def _summarize(
    fold_results: List[Dict],
    config: GNNExperimentConfig,
    tasks: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """计算 K-fold 均值 ± 标准差汇总。"""
    selected_tasks = tasks or ['pd', 'stage', 'subtype']
    metric_keys = []
    if 'pd' in selected_tasks:
        metric_keys.extend(['pd_acc', 'pd_patient_acc', 'pd_balanced_acc', 'pd_f1', 'pd_auc'])
    if 'stage' in selected_tasks:
        metric_keys.extend(['stage_acc', 'stage_patient_acc', 'stage_balanced_acc', 'stage_f1', 'stage_auc'])
    if 'subtype' in selected_tasks:
        metric_keys.extend(['subtype_acc', 'subtype_patient_acc', 'subtype_balanced_acc', 'subtype_f1', 'subtype_auc'])
    agg: Dict[str, float] = {}
    for k in metric_keys:
        vals = [r[k] for r in fold_results if k in r]
        if vals:
            agg[f'{k}_mean'] = float(np.mean(vals))
            agg[f'{k}_std']  = float(np.std(vals))

    return {
        'name': config.name,
        'config': asdict(config),
        'num_folds': len(fold_results),
        'fold_results': fold_results,
        **agg,
    }


def _log_summary(summary: Dict, logger: logging.Logger) -> None:
    def fmt(mk, sk):
        return f"{summary.get(mk, 0):.4f} ± {summary.get(sk, 0):.4f}"

    logger.info(
        f"PD     | acc {fmt('pd_acc_mean', 'pd_acc_std')} | "
        f"pat {fmt('pd_patient_acc_mean', 'pd_patient_acc_std')} | "
        f"bacc {fmt('pd_balanced_acc_mean', 'pd_balanced_acc_std')} | "
        f"f1 {fmt('pd_f1_mean', 'pd_f1_std')} | "
        f"auc {fmt('pd_auc_mean', 'pd_auc_std')}"
    )
    logger.info(
        f"Stage  | acc {fmt('stage_acc_mean', 'stage_acc_std')} | "
        f"pat {fmt('stage_patient_acc_mean', 'stage_patient_acc_std')} | "
        f"bacc {fmt('stage_balanced_acc_mean', 'stage_balanced_acc_std')} | "
        f"f1 {fmt('stage_f1_mean', 'stage_f1_std')}"
    )
    logger.info(
        f"Subtype| acc {fmt('subtype_acc_mean', 'subtype_acc_std')} | "
        f"pat {fmt('subtype_patient_acc_mean', 'subtype_patient_acc_std')} | "
        f"bacc {fmt('subtype_balanced_acc_mean', 'subtype_balanced_acc_std')} | "
        f"f1 {fmt('subtype_f1_mean', 'subtype_f1_std')}"
    )


# ─── 主入口 ───────────────────────────────────────────────────────────────────

def run_single_experiment(
    config: GNNExperimentConfig,
    output_dir: str = 'experiment_records',
    tasks: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    运行单次 GNN 实验 (K-fold)。

    Args:
        config:     GNNExperimentConfig
        output_dir: 实验结果保存根目录
        tasks:      要汇总和保存 patient logits 的任务列表；None 表示全部任务

    Returns:
        summary: 包含各 fold 指标及均值 / 标准差的字典
    """
    # ── 准备目录与日志 ───────────────────────────────────────────────────────────
    exp_dir = os.path.join(output_dir, config.name)
    os.makedirs(exp_dir, exist_ok=True)
    _setup_logging(exp_dir, config.name)
    logger = logging.getLogger(__name__)

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    selected_tasks = tasks or ['pd', 'stage', 'subtype']

    logger.info(f"=== GNN 实验: {config.name} ===")
    logger.info(f"配置: {asdict(config)}")

    # ── 加载患者数据 ─────────────────────────────────────────────────────────────
    json_path = os.path.join(_CODES_DIR, config.dataset_json)
    patients = load_and_filter_dataset(json_path, config.dataset_group)
    logger.info(f"患者总数: {len(patients)}  (group={config.dataset_group})")

    # ── 图缓存目录（所有实验共享，避免重复构建）────────────────────────────────────
    cache_dir = config.graph_cache_dir
    if cache_dir is None:
        # 存放在 results/graph_cache 下，与数据集目录隔离，实验间共享
        cache_dir = os.path.join(_CODES_DIR, 'results', 'graph_cache')
    logger.info(f"图缓存目录: {cache_dir}")

    # ── 提取患者级标签（用于 stratified kfold）───────────────────────────────────
    # 每个 patient 当作一个"样本"，patient_ids[i] = i 使得 patient kfold 直接返回患者索引
    pd_labels      = [1 if p['group'] == 'pd' else 0 for p in patients]
    stage_labels   = [int(p.get('stage', 0)) for p in patients]
    subtype_labels = [SUBTYPE_MAP.get(p.get('subtype') or '', -1) for p in patients]
    patient_ids    = list(range(len(patients)))   # 每人唯一索引

    # ── K-fold 划分 ──────────────────────────────────────────────────────────────
    folds = list(stratified_kfold(
        pd_labels=pd_labels,
        stage_labels=stage_labels,
        subtype_labels=subtype_labels,
        n_splits=config.num_folds,
        kfold_mode=config.kfold_mode,
        patient_ids=patient_ids,
    ))
    logger.info(f"K-fold: {config.num_folds} folds (mode={config.kfold_mode})")

    # ── 分类器配置 ───────────────────────────────────────────────────────────────
    clf_cfg = GNNClassifierConfig(
        gnn_type=config.gnn_type,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        shared_encoder=config.shared_encoder,
        lr=config.lr,
        weight_decay=config.weight_decay,
        optimizer=config.optimizer,
        lr_scheduler=config.lr_scheduler,
        warmup_epochs=config.warmup_epochs,
        epochs=config.epochs,
        patience=config.patience,
        batch_size=config.batch_size,
        dropout=config.dropout,
        classifier_dropout=config.classifier_dropout,
        pd_loss_weight=config.pd_loss_weight,
        stage_loss_weight=config.stage_loss_weight,
        subtype_loss_weight=config.subtype_loss_weight,
        label_smoothing=config.label_smoothing,
        primary_task=config.primary_task,
        graph_cache_dir=cache_dir,
        num_workers=config.num_workers,
    )

    base_path = os.path.join(_CODES_DIR, 'datasets')

    # ── 运行 K-fold ──────────────────────────────────────────────────────────────
    fold_results = []
    for fold_idx, (train_indices, val_indices) in enumerate(folds):
        logger.info(f"\n── Fold {fold_idx + 1}/{config.num_folds} "
                    f"(train={len(train_indices)}, val={len(val_indices)}) ──")

        train_patients = [patients[i] for i in train_indices]
        val_patients   = [patients[i] for i in val_indices]

        clf = GNNClassifier(config=clf_cfg)
        checkpoint_path = os.path.join(exp_dir, f'fold_{fold_idx + 1}_best_model.pt')
        result = clf.train_one_fold(
            train_patients, val_patients, base_path, fold_idx=fold_idx + 1,
            checkpoint_path=checkpoint_path,
            reuse_checkpoint=config.reuse_checkpoints,
        )
        predictions = result.pop('predictions', {})
        if predictions:
            for task in selected_tasks:
                task_pred = predictions.get(task, {})
                patient_result = save_fold_patient_outputs(
                    exp_dir, task, '', fold_idx + 1,
                    predictions.get('patient_ids', []),
                    task_pred.get('labels', []),
                    probs=task_pred.get('probs'),
                    logits=task_pred.get('logits'),
                )
                result[f'{task}_patient_acc'] = patient_result['patient_acc']
                result[f'{task}_patient_logits_json'] = patient_result['patient_logits_json']
                result[f'{task}_patient_logits_csv'] = patient_result['patient_logits_csv']
            keep_prefixes = tuple(f'{task}_' for task in selected_tasks)
            result = {k: v for k, v in result.items() if k == 'fold' or k.startswith(keep_prefixes)}
        fold_results.append(result)

        # 保存单 fold 结果
        fold_path = os.path.join(exp_dir, f'fold_{fold_idx + 1}_result.json')
        with open(fold_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    # ── 汇总 ─────────────────────────────────────────────────────────────────────
    summary = _summarize(fold_results, config, tasks=selected_tasks)
    summary_path = os.path.join(exp_dir, 'summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(f"\n=== {config.name} 完成 ===")
    _log_summary(summary, logger)

    return summary


# ─── 命令行入口 ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    cfg = GNNExperimentConfig(
        name='gnn_gat_quick_test',
        gnn_type='gat',
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        epochs=30,
        patience=8,
        num_folds=2,
        kfold_mode='patient',
        dataset_group='pd_control',
    )
    run_single_experiment(
        cfg,
        output_dir=os.path.join(
            os.path.dirname(_CODES_DIR), 'experiment_records'
        ),
    )
