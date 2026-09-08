"""
trainer.py - GNN 训练与评估循环

多任务损失:
  - PD 分类:  交叉熵
  - 分期:     交叉熵 (仅对 stage_label >= 0 的样本)
  - 亚型:     交叉熵 (仅对 subtype_label >= 0 的样本)

评估指标:
  - Balanced accuracy (macro recall)
  - Macro F1
  - AUC (二分类用 AUROC, 多分类用 one-vs-rest macro)
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

from .models import GNNMultiTaskModel
from .graph_dataset import GraphBatch

logger = logging.getLogger(__name__)


# ─── 训练配置 ─────────────────────────────────────────────────────────────────

@dataclass
class GNNTrainerConfig:
    """GNN 训练超参数。"""
    # 优化器
    lr: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = 'adam'         # 'adam' | 'adamw'

    # 学习率调度
    lr_scheduler: str = 'cosine'    # 'cosine' | 'step' | 'none'
    warmup_epochs: int = 5

    # 训练循环
    epochs: int = 50
    patience: int = 10              # early stopping patience

    # 多任务损失权重
    pd_loss_weight: float = 1.0
    stage_loss_weight: float = 0.5
    subtype_loss_weight: float = 0.5

    # 标签平滑
    label_smoothing: float = 0.0


# ─── 损失函数 ─────────────────────────────────────────────────────────────────

def _masked_ce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    smoothing: float = 0.0,
) -> torch.Tensor:
    """交叉熵损失，自动忽略 label == -1 的无效样本。"""
    mask = labels >= 0
    if not mask.any():
        return torch.zeros(1, device=logits.device, requires_grad=True).squeeze()
    return F.cross_entropy(logits[mask], labels[mask], label_smoothing=smoothing)


def compute_multitask_loss(
    pd_out: torch.Tensor,
    stage_out: torch.Tensor,
    subtype_out: torch.Tensor,
    pd_labels: torch.Tensor,
    stage_labels: torch.Tensor,
    subtype_labels: torch.Tensor,
    cfg: GNNTrainerConfig,
) -> torch.Tensor:
    """计算加权多任务损失。"""
    pd_loss      = _masked_ce(pd_out,      pd_labels,      cfg.label_smoothing)
    stage_loss   = _masked_ce(stage_out,   stage_labels,   cfg.label_smoothing)
    subtype_loss = _masked_ce(subtype_out, subtype_labels, cfg.label_smoothing)
    return (
        cfg.pd_loss_weight      * pd_loss
        + cfg.stage_loss_weight   * stage_loss
        + cfg.subtype_loss_weight * subtype_loss
    )


# ─── 设备迁移 ─────────────────────────────────────────────────────────────────

def batch_to_device(gb: GraphBatch, device: torch.device) -> GraphBatch:
    """将 GraphBatch 中所有张量移动到 device。"""
    return GraphBatch(
        x=gb.x.to(device),
        edge_index=gb.edge_index.to(device),
        edge_attr=gb.edge_attr.to(device),
        batch=gb.batch.to(device),
        num_graphs=gb.num_graphs,
        ptr=gb.ptr.to(device),
    )


# ─── 指标计算 ─────────────────────────────────────────────────────────────────

def _compute_metrics(
    preds: np.ndarray,
    labels: np.ndarray,
    probs: Optional[np.ndarray],
    num_classes: int,
) -> dict:
    """计算 acc / balanced_acc / macro-F1 / AUC（自动过滤 label=-1 的样本）。"""
    valid = labels >= 0
    if not valid.any():
        return {'acc': 0.0, 'balanced_acc': 0.0, 'f1': 0.0, 'auc': 0.0}

    y_true = labels[valid]
    y_pred = preds[valid]

    acc = float(np.mean(y_true == y_pred))
    ba = float(balanced_accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, average='macro', zero_division=0))

    auc = 0.0
    if probs is not None:
        try:
            p = probs[valid]
            if num_classes == 2:
                auc = float(roc_auc_score(y_true, p[:, 1]))
            else:
                auc = float(roc_auc_score(
                    y_true, p, multi_class='ovr', average='macro'
                ))
        except Exception:
            auc = 0.0

    return {'acc': acc, 'balanced_acc': ba, 'f1': f1, 'auc': auc}


# ─── 训练一个 Epoch ───────────────────────────────────────────────────────────

def train_epoch(
    model: GNNMultiTaskModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cfg: GNNTrainerConfig,
) -> float:
    """训练一个 epoch，返回平均损失。"""
    model.train()
    total_loss = 0.0
    n = 0

    for batch in loader:
        central_b, whorl_b, pd_ls, stage_ls, sub_ls = batch[:5]

        central_b = batch_to_device(central_b, device)
        whorl_b   = batch_to_device(whorl_b,   device)
        pd_ls     = pd_ls.to(device)
        stage_ls  = stage_ls.to(device)
        sub_ls    = sub_ls.to(device)

        optimizer.zero_grad()
        pd_out, stage_out, sub_out = model(central_b, whorl_b)
        loss = compute_multitask_loss(
            pd_out, stage_out, sub_out, pd_ls, stage_ls, sub_ls, cfg
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item() * pd_ls.size(0)
        n += pd_ls.size(0)

    return total_loss / max(n, 1)


# ─── 评估 ─────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: GNNMultiTaskModel,
    loader: DataLoader,
    device: torch.device,
    cfg: GNNTrainerConfig,
    primary_task: str = 'pd',
) -> dict:
    """
    评估模型，返回指标字典。

    Args:
        model:        待评估模型
        loader:       DataLoader
        device:       推理设备
        cfg:          训练配置（用于损失计算）
        primary_task: 用于模型选择的主要任务 ('pd' | 'stage' | 'subtype')

    Returns:
        dict 包含 loss, pd/stage/subtype 指标, primary_metric
    """
    model.eval()

    all_pd_preds,    all_pd_labels,    all_pd_probs,    all_pd_logits    = [], [], [], []
    all_stage_preds, all_stage_labels, all_stage_probs, all_stage_logits = [], [], [], []
    all_sub_preds,   all_sub_labels,   all_sub_probs,   all_sub_logits   = [], [], [], []
    all_patient_ids = []
    total_loss = 0.0
    n = 0

    for batch in loader:
        central_b, whorl_b, pd_ls, stage_ls, sub_ls = batch[:5]
        patient_ids = batch[5] if len(batch) > 5 else [str(i) for i in range(pd_ls.size(0))]

        central_b = batch_to_device(central_b, device)
        whorl_b   = batch_to_device(whorl_b,   device)
        pd_ls     = pd_ls.to(device)
        stage_ls  = stage_ls.to(device)
        sub_ls    = sub_ls.to(device)

        pd_out, stage_out, sub_out = model(central_b, whorl_b)
        loss = compute_multitask_loss(
            pd_out, stage_out, sub_out, pd_ls, stage_ls, sub_ls, cfg
        )
        total_loss += loss.item() * pd_ls.size(0)
        n += pd_ls.size(0)

        pd_prob    = torch.softmax(pd_out,    dim=-1).cpu().numpy()
        stage_prob = torch.softmax(stage_out, dim=-1).cpu().numpy()
        sub_prob   = torch.softmax(sub_out,   dim=-1).cpu().numpy()

        all_pd_preds.extend(pd_prob.argmax(1).tolist())
        all_pd_labels.extend(pd_ls.cpu().numpy().tolist())
        all_pd_probs.append(pd_prob)
        all_pd_logits.append(pd_out.cpu().numpy())

        all_stage_preds.extend(stage_prob.argmax(1).tolist())
        all_stage_labels.extend(stage_ls.cpu().numpy().tolist())
        all_stage_probs.append(stage_prob)
        all_stage_logits.append(stage_out.cpu().numpy())

        all_sub_preds.extend(sub_prob.argmax(1).tolist())
        all_sub_labels.extend(sub_ls.cpu().numpy().tolist())
        all_sub_probs.append(sub_prob)
        all_sub_logits.append(sub_out.cpu().numpy())
        all_patient_ids.extend([str(pid) for pid in patient_ids])

    avg_loss = total_loss / max(n, 1)

    pd_metrics    = _compute_metrics(
        np.array(all_pd_preds),    np.array(all_pd_labels),
        np.vstack(all_pd_probs)    if all_pd_probs    else None, 2
    )
    stage_metrics = _compute_metrics(
        np.array(all_stage_preds), np.array(all_stage_labels),
        np.vstack(all_stage_probs) if all_stage_probs else None, 3
    )
    sub_metrics   = _compute_metrics(
        np.array(all_sub_preds),   np.array(all_sub_labels),
        np.vstack(all_sub_probs)   if all_sub_probs   else None, 3
    )

    primary_map = {
        'pd':      pd_metrics['balanced_acc'],
        'stage':   stage_metrics['balanced_acc'],
        'subtype': sub_metrics['balanced_acc'],
    }

    return {
        'loss':           avg_loss,
        'pd':             pd_metrics,
        'stage':          stage_metrics,
        'subtype':        sub_metrics,
        'predictions': {
            'patient_ids': all_patient_ids,
            'pd': {
                'labels': all_pd_labels,
                'probs': np.vstack(all_pd_probs).tolist() if all_pd_probs else [],
                'logits': np.vstack(all_pd_logits).tolist() if all_pd_logits else [],
            },
            'stage': {
                'labels': all_stage_labels,
                'probs': np.vstack(all_stage_probs).tolist() if all_stage_probs else [],
                'logits': np.vstack(all_stage_logits).tolist() if all_stage_logits else [],
            },
            'subtype': {
                'labels': all_sub_labels,
                'probs': np.vstack(all_sub_probs).tolist() if all_sub_probs else [],
                'logits': np.vstack(all_sub_logits).tolist() if all_sub_logits else [],
            },
        },
        'primary_metric': primary_map.get(primary_task, pd_metrics['balanced_acc']),
    }
