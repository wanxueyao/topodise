"""
main.py - 多任务学习单次实验入口

执行一组配置下的完整 K-fold 交叉验证实验：
  - Branch A: CORN-1 分割/MAE/伪标签任务（三选一或联合）
  - Branch B: PD 分类（pd / stage / subtype）
  - 架构: dual_stream + ResNet18
  - K-fold: patient 模式，5 折

用法（单次运行）::

    cd codes
    python experiments/mtl_experiments/main.py          # 使用默认配置
    python experiments/mtl_experiments/main.py \\
        --branch_a_task segmentation \\
        --epochs 40 \\
        --lr 2e-4 \\
        --wd 3e-3 \\
        --output_dir results/mtl_seg_only

批量实验调用接口::

    from experiments.mtl_experiments.main import MTLExperimentConfig, run_single_mtl_experiment
    config = MTLExperimentConfig()
    config.BRANCH_A_TASK = 'mae'
    results = run_single_mtl_experiment(config, output_dir='results/mtl_mae')
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Literal, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from sklearn.metrics import f1_score, roc_auc_score

# ── 添加项目根目录 ──────────────────────────────────────────────
_THIS_FILE = Path(__file__).resolve()


def _discover_project_root() -> Path:
    """自动发现仓库根目录，兼容不同服务器挂载层级。"""
    for p in [_THIS_FILE.parent, *_THIS_FILE.parents]:
        if (p / 'codes' / 'multitasklearning').is_dir():
            return p
        if p.name == 'codes' and (p / 'multitasklearning').is_dir():
            return p.parent
    return _THIS_FILE.parents[3]


PROJECT_ROOT = _discover_project_root()
CODES_ROOT = PROJECT_ROOT / 'codes'

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(CODES_ROOT))

from multitasklearning.unet_multitask import UNetMultiTask
from multitasklearning.multitask_data_loader import (
    build_corn1_loaders,
    build_pd_patient_kfold_loaders,
    MultiTaskDataLoader,
)
from utils.patient_outputs import save_fold_patient_outputs
from utils.misc import set_seed


# ──────────────────────────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────────────────────────

@dataclass
class MTLExperimentConfig:
    """多任务学习实验配置"""

    # ── 架构 ──────────────────────────────────────────
    BACKBONE: str = 'resnet18'
    PRETRAINED: Literal[True, False, 'Half'] = 'Half'
    DROPOUT: float = 0.0

    # ── Branch A 任务选择 ───────────────────────────────
    # 'segmentation': 监督分割（使用 GT 掩模）
    # 'mae'         : MAE 自监督重构（无需掩模标签）
    # 'pseudo'      : Frangi 伪标签分割（无需 GT 掩模）
    BRANCH_A_TASK: Literal['segmentation', 'mae', 'pseudo'] = 'segmentation'

    # Branch A 损失权重（其余任务权重自动设为 0）
    SEG_WEIGHT:    float = 0.40
    MAE_WEIGHT:    float = 0.40
    PSEUDO_WEIGHT: float = 0.40

    # Branch B 损失权重
    PD_WEIGHT:      float = 0.40
    STAGE_WEIGHT:   float = 0.10
    SUBTYPE_WEIGHT: float = 0.10

    # ── 训练超参数 ─────────────────────────────────────
    NUM_EPOCHS: int = 40
    LEARNING_RATE: float = 2e-4
    WEIGHT_DECAY: float = 3e-3
    BACKBONE_LR_RATIO: float = 0.05
    BATCH_SIZE: int = 16
    LR_SCHEDULER: str = 'cosine'   # 可选 cosine 或 step
    EARLY_STOPPING: bool = True
    PATIENCE: int = 6
    MIXED_PRECISION: bool = True
    GRAD_CLIP: float = 1.0

    # ── 数据增强 ───────────────────────────────────────
    DATA_AUGMENTATION: bool = True

    # ── K-fold ────────────────────────────────────────
    NUM_FOLDS: int = 5
    KFOLD_MODE: str = 'patient'
    SEED: int = 42
    RUN_TASKS: Optional[List[str]] = None

    # ── 图像尺寸 ───────────────────────────────────────
    IMAGE_SIZE: int = 256

    # ── 数据路径 ───────────────────────────────────────
    CORN1_ROOT: str = 'datasets/CORN-1'
    PD_DATASET_JSON: str = 'datasets/pd_dataset_en/patients_dataset_en.json'
    DATASETS_BASE: str = 'datasets'
    DATASET_MODE: str = 'pd_all'

    # ── 系统 ──────────────────────────────────────────
    NUM_WORKERS: int = 4
    DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    OUTPUT_DIR: str = 'results/mtl'


# ──────────────────────────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────────────────────────

def _make_optimizer(model: nn.Module, config: MTLExperimentConfig) -> optim.Optimizer:
    """创建带差异化学习率的 AdamW 优化器。"""
    encoder_params, head_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'encoder' in name:
            encoder_params.append(param)
        else:
            head_params.append(param)

    return optim.AdamW([
        {'params': encoder_params, 'lr': config.LEARNING_RATE * config.BACKBONE_LR_RATIO},
        {'params': head_params,    'lr': config.LEARNING_RATE},
    ], weight_decay=config.WEIGHT_DECAY)


def _freeze_backbone_half(model: nn.Module):
    """冻结 encoder 前一半层的参数（Half 预训练策略）。"""
    enc = model.encoder
    layers = [enc.layer0, enc.maxpool, enc.layer1, enc.layer2,
              enc.layer3, enc.layer4]
    freeze_until = len(layers) // 2
    for layer in layers[:freeze_until]:
        for p in layer.parameters():
            p.requires_grad = False


def _compute_pd_metrics(logits: torch.Tensor, labels: torch.Tensor
                         ) -> Dict[str, float]:
    """计算 PD 分类指标（忽略 label=-1 的样本）。"""
    mask = labels >= 0
    if mask.sum() == 0:
        return {'acc': 0.0, 'f1': 0.0, 'auc': 0.0}

    logits_m = logits[mask]
    labels_m = labels[mask]
    preds = logits_m.argmax(dim=1).cpu().numpy()
    labs  = labels_m.cpu().numpy()
    probs = torch.softmax(logits_m, dim=1)[:, 1].detach().cpu().numpy()

    acc = float((preds == labs).mean())
    f1  = float(f1_score(labs, preds, average='macro', zero_division=0))
    try:
        auc = float(roc_auc_score(labs, probs))
    except ValueError:
        auc = 0.0
    return {'acc': acc, 'f1': f1, 'auc': auc}


def _compute_mc_metrics(logits: torch.Tensor, labels: torch.Tensor,
                         task_name: str) -> Dict[str, float]:
    """计算多类分类指标（stage / subtype），自动屏蔽 label<0 样本。"""
    mask = labels >= 0
    if mask.sum() == 0:
        return {f'{task_name}_acc': 0.0, f'{task_name}_f1': 0.0}

    preds = logits[mask].argmax(dim=1).cpu().numpy()
    labs  = labels[mask].cpu().numpy()
    acc = float((preds == labs).mean())
    f1  = float(f1_score(labs, preds, average='macro', zero_division=0))
    return {f'{task_name}_acc': acc, f'{task_name}_f1': f1}


def _dice(pred: torch.Tensor, target: torch.Tensor, thr: float = 0.5) -> float:
    p = (pred > thr).float()
    inter = (p * target).sum()
    return (2.0 * inter / (p.sum() + target.sum() + 1e-8)).item()


# ──────────────────────────────────────────────────────────────────
# 损失函数
# ──────────────────────────────────────────────────────────────────

class FocalCELoss(nn.Module):
    """Focal + CrossEntropy 混合损失，对类别不平衡更鲁棒。"""
    def __init__(self, gamma: float = 2.0, ignore_index: int = -1):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(logits, targets)
        valid = (targets != self.ignore_index)
        if valid.sum() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        pt = torch.exp(-ce_loss)
        focal = ((1 - pt) ** self.gamma * ce_loss)[valid]
        return focal.mean()


class DiceBCELoss(nn.Module):
    """Dice + BCE 分割损失。"""
    def __init__(self, bce_weight: float = 0.5, eps: float = 1e-6):
        super().__init__()
        self.bce_weight = bce_weight
        self.eps = eps
        self.bce_logits = nn.BCEWithLogitsLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # segmentation 分支输出已过 sigmoid，这里先安全反变换为 logits，
        # 再使用 autocast 安全的 BCEWithLogits。
        pred_clamped = pred.clamp(min=self.eps, max=1.0 - self.eps)
        pred_logits = torch.logit(pred_clamped)
        bce = self.bce_logits(pred_logits, target)
        inter = (pred * target).sum(dim=(2, 3))
        dice = 1.0 - (2.0 * inter + 1.0) / (pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + 1.0)
        dice = dice.mean()
        return self.bce_weight * bce + (1 - self.bce_weight) * dice


# ──────────────────────────────────────────────────────────────────
# 单个 fold 的训练和验证
# ──────────────────────────────────────────────────────────────────

class FoldTrainer:
    """负责单个 fold 的训练全流程。"""

    def __init__(self, model: UNetMultiTask, config: MTLExperimentConfig,
                 mtl_loader: MultiTaskDataLoader,
                 val_corn1_loader, val_pd_loader,
                 fold_dir: str):
        self.model   = model
        self.config  = config
        self.mtl_loader = mtl_loader
        self.val_corn1_loader = val_corn1_loader
        self.val_pd_loader    = val_pd_loader
        self.fold_dir = fold_dir
        os.makedirs(fold_dir, exist_ok=True)

        self.device = torch.device(config.DEVICE)
        self.model.to(self.device)

        self.optimizer = _make_optimizer(model, config)
        if config.LR_SCHEDULER == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=config.NUM_EPOCHS, eta_min=1e-7
            )
        else:
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=15, gamma=0.5
            )

        self.scaler = GradScaler() if (config.MIXED_PRECISION and
                                        'cuda' in config.DEVICE) else None

        # 损失
        self.seg_criterion      = DiceBCELoss()
        self.mae_criterion      = nn.MSELoss()
        self.cls_criterion_pd   = FocalCELoss(ignore_index=-1)
        self.cls_criterion_mc   = FocalCELoss(ignore_index=-1)

        # 早停
        self.best_val_score = -1.0
        self.patience_counter = 0
        self.best_ckpt = os.path.join(fold_dir, 'best_model.pth')

        # 历史
        self.history: Dict[str, List] = {
            'train_loss': [], 'val_score': [],
            'train_pd_acc': [], 'train_stage_acc': [], 'train_subtype_acc': [],
            'train_seg_dice': [], 'train_mae_loss': [],
            'val_pd_acc': [], 'val_pd_f1': [], 'val_pd_auc': [],
            'val_stage_acc': [], 'val_stage_f1': [],
            'val_subtype_acc': [], 'val_subtype_f1': [],
            'val_seg_dice': [], 'val_mae_loss': [], 'lr': [],
        }

    # ------------------------------------------------------------------ #
    # 计算 Branch A 损失
    # ------------------------------------------------------------------ #
    def _branch_a_loss(self, outputs: Dict, corn1_batch: Dict) -> torch.Tensor:
        task = self.config.BRANCH_A_TASK
        if task == 'segmentation':
            masks = corn1_batch['mask'].to(self.device)
            seg = outputs['segmentation']
            if seg.shape[2:] != masks.shape[2:]:
                seg = torch.nn.functional.interpolate(
                    seg, size=masks.shape[2:], mode='bilinear', align_corners=False
                )
            return self.seg_criterion(seg, masks) * self.config.SEG_WEIGHT

        elif task == 'mae':
            if 'mae_reconstructed' in outputs:
                return outputs['losses'].get('seg_mae',
                       torch.tensor(0.0, device=self.device)) * self.config.MAE_WEIGHT
            return torch.tensor(0.0, device=self.device)

        elif task == 'pseudo':
            masks = corn1_batch['mask'].to(self.device)   # 伪标签在 CORNDataset 中已生成
            seg = outputs['segmentation']
            if seg.shape[2:] != masks.shape[2:]:
                seg = torch.nn.functional.interpolate(
                    seg, size=masks.shape[2:], mode='bilinear', align_corners=False
                )
            return self.seg_criterion(seg, masks) * self.config.PSEUDO_WEIGHT

        return torch.tensor(0.0, device=self.device)

    # ------------------------------------------------------------------ #
    # 计算 Branch B 损失
    # ------------------------------------------------------------------ #
    def _branch_b_loss(self, outputs: Dict, pd_batch: Dict) -> torch.Tensor:
        pd_l  = pd_batch['pd_label'].to(self.device)
        stg_l = pd_batch['stage_label'].to(self.device)
        sub_l = pd_batch['subtype_label'].to(self.device)

        pd_loss  = self.cls_criterion_pd(outputs['pd_logits'], pd_l)      * self.config.PD_WEIGHT
        stg_loss = self.cls_criterion_mc(outputs['stage_logits'],  stg_l) * self.config.STAGE_WEIGHT
        sub_loss = self.cls_criterion_mc(outputs['subtype_logits'],sub_l) * self.config.SUBTYPE_WEIGHT
        return pd_loss + stg_loss + sub_loss

    # ------------------------------------------------------------------ #
    # 前向推断（统一接口，适配不同 Branch A 任务）
    # ------------------------------------------------------------------ #
    def _forward_corn1(self, corn1_batch: Dict) -> Dict:
        images = corn1_batch['image'].to(self.device)
        use_mae = (self.config.BRANCH_A_TASK == 'mae')
        return self.model(images, use_mae=use_mae, training=True)

    def _forward_pd(self, pd_batch: Dict) -> Dict:
        images    = pd_batch['image'].to(self.device)
        pd_l      = pd_batch['pd_label'].to(self.device)
        stage_l   = pd_batch['stage_label'].to(self.device)
        subtype_l = pd_batch['subtype_label'].to(self.device)
        return self.model(
            images,
            pd_labels=pd_l, stage_labels=stage_l, subtype_labels=subtype_l,
            use_mae=False, training=True
        )

    # ------------------------------------------------------------------ #
    # 训练单个 epoch（交替处理 CORN-1 和 PD 批次）
    # ------------------------------------------------------------------ #
    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        total_loss, pd_accs, stg_accs, sub_accs, seg_dices = 0.0, [], [], [], []
        mae_losses: List[float] = []
        n_batches = 0

        for corn1_batch, pd_batch in self.mtl_loader:
            self.optimizer.zero_grad(set_to_none=True)

            amp_ctx = autocast(device_type='cuda') if self.scaler else (
                torch.no_grad.__class__.__new__(torch.no_grad)  # 伪上下文，实际不用
            )
            # 分支 A
            with (autocast(device_type='cuda') if self.scaler else
                  torch.autocast('cpu', enabled=False)):
                out_a = self._forward_corn1(corn1_batch)
                loss_a = self._branch_a_loss(out_a, corn1_batch)

                # 分支 B
                out_b = self._forward_pd(pd_batch)
                loss_b = self._branch_b_loss(out_b, pd_batch)

                loss = loss_a + loss_b

            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.GRAD_CLIP)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.GRAD_CLIP)
                self.optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

            # 记录 Branch A 分割质量（如果有 GT 掩模）
            if self.config.BRANCH_A_TASK in ('segmentation', 'pseudo'):
                masks = corn1_batch['mask'].to(self.device)
                seg   = out_a['segmentation']
                if seg.shape[2:] != masks.shape[2:]:
                    seg = torch.nn.functional.interpolate(
                        seg, size=masks.shape[2:], mode='bilinear', align_corners=False)
                seg_dices.append(_dice(seg.detach(), masks))

            # 记录 Branch A MAE 重建损失（未加权，便于纯粹监控重建质量）
            if self.config.BRANCH_A_TASK == 'mae':
                mae_raw = out_a.get('losses', {}).get('seg_mae')
                if mae_raw is not None and torch.is_tensor(mae_raw):
                    mae_losses.append(float(mae_raw.detach().item()))

            # 记录 Branch B 各分类任务训练准确率
            pd_l  = pd_batch['pd_label'].to(self.device)
            valid = pd_l >= 0
            if valid.sum() > 0:
                preds = out_b['pd_logits'][valid].argmax(dim=1)
                acc   = (preds == pd_l[valid]).float().mean().item()
                pd_accs.append(acc)

            stg_l = pd_batch['stage_label'].to(self.device)
            vstg  = stg_l >= 0
            if vstg.sum() > 0:
                p_s = out_b['stage_logits'][vstg].argmax(dim=1)
                stg_accs.append((p_s == stg_l[vstg]).float().mean().item())

            sub_l = pd_batch['subtype_label'].to(self.device)
            vsub  = sub_l >= 0
            if vsub.sum() > 0:
                p_u = out_b['subtype_logits'][vsub].argmax(dim=1)
                sub_accs.append((p_u == sub_l[vsub]).float().mean().item())

        result = {
            'loss':        total_loss / max(n_batches, 1),
            'pd_acc':      float(np.mean(pd_accs))   if pd_accs   else 0.0,
            'stage_acc':   float(np.mean(stg_accs))  if stg_accs  else 0.0,
            'subtype_acc': float(np.mean(sub_accs))  if sub_accs  else 0.0,
            'seg_dice':    float(np.mean(seg_dices))  if seg_dices else -1.0,
            'mae_loss':    float(np.mean(mae_losses)) if mae_losses else -1.0,
        }
        return result

    # ------------------------------------------------------------------ #
    # 验证（单独验证 CORN-1 和 PD，以获得各任务准确指标）
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        self.model.eval()

        # --- 验证 PD 分类 ---
        all_pd_logits, all_pd_labels     = [], []
        all_stg_logits, all_stg_labels   = [], []
        all_sub_logits, all_sub_labels   = [], []
        all_patient_ids                  = []

        for pd_batch in self.val_pd_loader:
            imgs   = pd_batch['image'].to(self.device)
            out    = self.model(imgs, use_mae=False, training=False)
            all_pd_logits.append(out['pd_logits'].cpu())
            all_pd_labels.append(pd_batch['pd_label'])
            all_stg_logits.append(out['stage_logits'].cpu())
            all_stg_labels.append(pd_batch['stage_label'])
            all_sub_logits.append(out['subtype_logits'].cpu())
            all_sub_labels.append(pd_batch['subtype_label'])
            all_patient_ids.extend(self._batch_patient_ids(pd_batch['patient_idx']))

        pd_logits  = torch.cat(all_pd_logits)
        pd_labels  = torch.cat(all_pd_labels)
        stg_logits = torch.cat(all_stg_logits)
        stg_labels = torch.cat(all_stg_labels)
        sub_logits = torch.cat(all_sub_logits)
        sub_labels = torch.cat(all_sub_labels)

        pd_metrics  = _compute_pd_metrics(pd_logits, pd_labels)
        stg_metrics = _compute_mc_metrics(stg_logits, stg_labels, 'stage')
        sub_metrics = _compute_mc_metrics(sub_logits, sub_labels, 'subtype')
        patient_metrics = self._save_patient_logits(
            all_patient_ids, pd_logits, pd_labels,
            stg_logits, stg_labels, sub_logits, sub_labels
        )

        # --- 验证 CORN-1 分割 ---
        dices = []
        if self.config.BRANCH_A_TASK in ('segmentation', 'pseudo'):
            for corn1_batch in self.val_corn1_loader:
                imgs  = corn1_batch['image'].to(self.device)
                masks = corn1_batch['mask'].to(self.device)
                out   = self.model(imgs, use_mae=False, training=False)
                seg   = out['segmentation']
                if seg.shape[2:] != masks.shape[2:]:
                    seg = torch.nn.functional.interpolate(
                        seg, size=masks.shape[2:], mode='bilinear', align_corners=False)
                dices.append(_dice(seg, masks))

        # --- 验证 CORN-1 MAE 重建损失 ---
        # 注：model.forward 中 use_mae 需要 training=True 才生效（用作分支门控），
        # 但 self.model 已处于 eval() 模式且包裹 @torch.no_grad，行为安全。
        mae_val_losses: List[float] = []
        if self.config.BRANCH_A_TASK == 'mae':
            for corn1_batch in self.val_corn1_loader:
                imgs = corn1_batch['image'].to(self.device)
                out  = self.model(imgs, use_mae=True, training=True)
                mae_raw = out.get('losses', {}).get('seg_mae')
                if mae_raw is not None and torch.is_tensor(mae_raw):
                    mae_val_losses.append(float(mae_raw.item()))

        # 综合得分（用于早停）= pd_acc * 0.5 + (seg_dice 或 pd_f1) * 0.5
        seg_dice = float(np.mean(dices)) if dices else -1.0
        val_score = (pd_metrics['acc'] * 0.6 +
                     (seg_dice if seg_dice >= 0 else pd_metrics['f1']) * 0.4)

        return {
            'val_score':   val_score,
            'pd_acc':      pd_metrics['acc'],
            'pd_patient_acc': patient_metrics['pd']['patient_acc'],
            'pd_patient_logits_json': patient_metrics['pd']['patient_logits_json'],
            'pd_patient_logits_csv': patient_metrics['pd']['patient_logits_csv'],
            'pd_f1':       pd_metrics['f1'],
            'pd_auc':      pd_metrics['auc'],
            'stage_acc':   stg_metrics.get('stage_acc', 0.0),
            'stage_patient_acc': patient_metrics['stage']['patient_acc'],
            'stage_patient_logits_json': patient_metrics['stage']['patient_logits_json'],
            'stage_patient_logits_csv': patient_metrics['stage']['patient_logits_csv'],
            'stage_f1':    stg_metrics.get('stage_f1',  0.0),
            'subtype_acc': sub_metrics.get('subtype_acc', 0.0),
            'subtype_patient_acc': patient_metrics['subtype']['patient_acc'],
            'subtype_patient_logits_json': patient_metrics['subtype']['patient_logits_json'],
            'subtype_patient_logits_csv': patient_metrics['subtype']['patient_logits_csv'],
            'subtype_f1':  sub_metrics.get('subtype_f1',  0.0),
            'seg_dice':    seg_dice,
            'mae_loss':    float(np.mean(mae_val_losses)) if mae_val_losses else -1.0,
        }

    def _batch_patient_ids(self, patient_idx) -> List[str]:
        dataset = getattr(self.val_pd_loader, 'dataset', None)
        patients = getattr(dataset, 'patients', None)
        out: List[str] = []
        for idx in patient_idx.detach().cpu().tolist():
            if patients is not None and 0 <= int(idx) < len(patients):
                out.append(str(patients[int(idx)].get('patient_id', idx)))
            else:
                out.append(str(idx))
        return out

    def _save_patient_logits(self, patient_ids, pd_logits, pd_labels,
                             stg_logits, stg_labels, sub_logits, sub_labels) -> Dict[str, Dict[str, Any]]:
        def save_one(task: str, logits: torch.Tensor, labels: torch.Tensor):
            valid = labels >= 0
            valid_ids = [pid for pid, keep in zip(patient_ids, valid.cpu().tolist()) if keep]
            return save_fold_patient_outputs(
                self.fold_dir, task, '', 1,
                valid_ids,
                labels[valid].cpu().numpy().tolist(),
                logits=logits[valid].cpu().numpy(),
            )

        return {
            'pd': save_one('pd', pd_logits, pd_labels),
            'stage': save_one('stage', stg_logits, stg_labels),
            'subtype': save_one('subtype', sub_logits, sub_labels),
        }

    # ------------------------------------------------------------------ #
    # 完整 fold 训练
    # ------------------------------------------------------------------ #
    def run(self) -> Dict[str, Any]:
        config = self.config
        best_val_metrics: Dict = {}

        for epoch in range(1, config.NUM_EPOCHS + 1):
            train_metrics = self.train_epoch()
            self.scheduler.step()

            val_metrics = self.validate()
            val_score   = val_metrics['val_score']

            current_lr = self.optimizer.param_groups[-1]['lr']

            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_score'].append(val_score)
            self.history['train_pd_acc'].append(train_metrics['pd_acc'])
            self.history['train_stage_acc'].append(train_metrics.get('stage_acc', 0.0))
            self.history['train_subtype_acc'].append(train_metrics.get('subtype_acc', 0.0))
            self.history['train_seg_dice'].append(train_metrics['seg_dice'])
            self.history['val_pd_acc'].append(val_metrics['pd_acc'])
            self.history['val_pd_f1'].append(val_metrics['pd_f1'])
            self.history['val_pd_auc'].append(val_metrics['pd_auc'])
            self.history['val_stage_acc'].append(val_metrics['stage_acc'])
            self.history['val_stage_f1'].append(val_metrics['stage_f1'])
            self.history['val_subtype_acc'].append(val_metrics['subtype_acc'])
            self.history['val_subtype_f1'].append(val_metrics['subtype_f1'])
            self.history['val_seg_dice'].append(val_metrics['seg_dice'])
            self.history['train_mae_loss'].append(train_metrics.get('mae_loss', -1.0))
            self.history['val_mae_loss'].append(val_metrics.get('mae_loss', -1.0))
            self.history['lr'].append(current_lr)

            improved = val_score > self.best_val_score
            if improved:
                self.best_val_score = val_score
                self.patience_counter = 0
                best_val_metrics = val_metrics.copy()
                torch.save(self.model.state_dict(), self.best_ckpt)

            now = datetime.now().strftime('%H:%M:%S')
            mark = ' ↑' if improved else ''
            task = config.BRANCH_A_TASK
            if task in ('segmentation', 'pseudo'):
                br_a = (
                    f"    BrA({task[:3]}): tr_dice={train_metrics['seg_dice']:.3f}"
                    f"  val_dice={val_metrics['seg_dice']:.3f}"
                )
            else:
                br_a = f"    BrA(mae): (no pixel-level dice metric)"
            br_b = (
                f"    BrB(cls): "
                f"tr_pd={train_metrics['pd_acc']:.3f} "
                f"tr_stg={train_metrics.get('stage_acc', 0):.3f} "
                f"tr_sub={train_metrics.get('subtype_acc', 0):.3f}  |  "
                f"vl_pd={val_metrics['pd_acc']:.3f}/f1={val_metrics['pd_f1']:.3f}/auc={val_metrics['pd_auc']:.3f}  "
                f"vl_stg={val_metrics['stage_acc']:.3f}/f1={val_metrics['stage_f1']:.3f}  "
                f"vl_sub={val_metrics['subtype_acc']:.3f}/f1={val_metrics['subtype_f1']:.3f}"
            )
            print(
                f"  [{now}] E{epoch:03d}/{config.NUM_EPOCHS:03d}"
                f" | loss={train_metrics['loss']:.4f}"
                f" | LR={current_lr:.2e}"
                f" | val_score={val_score:.4f}{mark}",
                flush=True,
            )
            print(br_a, flush=True)
            print(br_b, flush=True)

            if config.EARLY_STOPPING:
                self.patience_counter += (0 if improved else 1)
                if self.patience_counter >= config.PATIENCE:
                    print(f"  早停触发 (patience={config.PATIENCE})")
                    break

        # 保存历史
        with open(os.path.join(self.fold_dir, 'history.json'), 'w') as f:
            json.dump(self.history, f, indent=2)

        return best_val_metrics


# ──────────────────────────────────────────────────────────────────
# K-fold 运行入口
# ──────────────────────────────────────────────────────────────────

def run_single_mtl_experiment(config: MTLExperimentConfig,
                               output_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    执行完整的 K-fold MTL 实验，返回汇总结果。

    Returns
    -------
    dict 包含：
        fold_results  : 各 fold 的验证指标
        mean_metrics  : 均值
        std_metrics   : 标准差
        config        : 配置字典
    """
    set_seed(config.SEED)

    # 绝对路径处理：同时兼容以仓库根或 codes 目录为基准的部署
    def _resolve_path(p: str, expect_dir: Optional[bool] = None) -> tuple[str, list[str]]:
        p_obj = Path(p)
        if p_obj.is_absolute():
            return str(p_obj), [str(p_obj)]

        candidates: list[Path] = [
            PROJECT_ROOT / p_obj,
            CODES_ROOT / p_obj,
        ]

        # 常见 DGX 目录：/home/.../PD
        if PROJECT_ROOT.name != 'PD' and (PROJECT_ROOT / 'PD').is_dir():
            candidates.extend([
                PROJECT_ROOT / 'PD' / p_obj,
                PROJECT_ROOT / 'PD' / 'codes' / p_obj,
            ])

        p_norm = p.replace('\\', '/').lstrip('./')
        if p_norm.startswith('codes/'):
            candidates.append(PROJECT_ROOT / p_norm[len('codes/'):])

        # 去重（保留顺序）
        unique_candidates: list[Path] = []
        seen = set()
        for c in candidates:
            s = str(c)
            if s in seen:
                continue
            seen.add(s)
            unique_candidates.append(c)

        for c in unique_candidates:
            if expect_dir is True and c.is_dir():
                return str(c), [str(x) for x in unique_candidates]
            if expect_dir is False and c.is_file():
                return str(c), [str(x) for x in unique_candidates]
            if expect_dir is None and c.exists():
                return str(c), [str(x) for x in unique_candidates]

        # 未命中时返回首选候选，随后统一由显式校验抛错
        return str(unique_candidates[0]), [str(x) for x in unique_candidates]

    def _assert_exists(path_str: str, label: str, is_dir: bool, tried: list[str]):
        p = Path(path_str)
        ok = p.is_dir() if is_dir else p.is_file()
        if ok:
            return
        expected = '目录' if is_dir else '文件'
        tried_msg = '\n'.join(f'  - {x}' for x in tried)
        raise FileNotFoundError(
            f"{label} 未找到有效{expected}: {path_str}\n已尝试路径:\n{tried_msg}"
        )

    corn1_root, corn1_tried = _resolve_path(config.CORN1_ROOT, expect_dir=True)
    pd_json, pd_json_tried = _resolve_path(config.PD_DATASET_JSON, expect_dir=False)
    datasets_base, datasets_tried = _resolve_path(config.DATASETS_BASE, expect_dir=True)

    _assert_exists(corn1_root, 'CORN-1 数据根目录', is_dir=True, tried=corn1_tried)
    _assert_exists(pd_json, 'PD 数据集 JSON', is_dir=False, tried=pd_json_tried)
    _assert_exists(datasets_base, 'PD 图像数据根目录', is_dir=True, tried=datasets_tried)

    exp_dir = output_dir or os.path.join(
        str(CODES_ROOT),
        config.OUTPUT_DIR,
        datetime.now().strftime('%Y%m%d_%H%M%S')
    )
    os.makedirs(exp_dir, exist_ok=True)

    # 保存配置
    cfg_dict = asdict(config) if hasattr(config, '__dataclass_fields__') else vars(config)
    with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
        json.dump(cfg_dict, f, indent=2, ensure_ascii=False)

    fold_results: List[Dict] = []

    for fold_idx in range(config.NUM_FOLDS):
        _fold_now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"\n{'='*60}")
        print(f"  Fold {fold_idx + 1}/{config.NUM_FOLDS}  |  BranchA={config.BRANCH_A_TASK.upper()}  |  {_fold_now}")
        print(f"  Backbone={config.BACKBONE}  Pretrained={config.PRETRAINED}  "
              f"Epochs={config.NUM_EPOCHS}  LR={config.LEARNING_RATE:.0e}  WD={config.WEIGHT_DECAY:.0e}")
        print('='*60)

        fold_dir = os.path.join(exp_dir, f'fold_{fold_idx}')

        # ── CORN-1 加载器 ──────────────────────────────────────────
        use_pseudo = (config.BRANCH_A_TASK == 'pseudo')
        corn1_train, corn1_val = build_corn1_loaders(
            corn1_root,
            batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS,
            augment_train=config.DATA_AUGMENTATION,
            image_size=config.IMAGE_SIZE,
            generate_pseudo=use_pseudo,
        )

        # ── PD 加载器 ──────────────────────────────────────────────
        pd_train, pd_val = build_pd_patient_kfold_loaders(
            dataset_json=pd_json,
            fold_idx=fold_idx,
            n_folds=config.NUM_FOLDS,
            datasets_base=datasets_base,
            dataset_mode=config.DATASET_MODE,
            batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS,
            augment_train=config.DATA_AUGMENTATION,
            image_size=config.IMAGE_SIZE,
            random_state=config.SEED,
        )

        # ── 多任务加载器 ───────────────────────────────────────────
        mtl_loader = MultiTaskDataLoader(corn1_train, pd_train)

        # ── 构建模型 ───────────────────────────────────────────────
        pretrained_bool = (config.PRETRAINED is True or config.PRETRAINED == 'Half')
        model = UNetMultiTask(
            encoder_type=config.BACKBONE,
            encoder_pretrained=pretrained_bool,
            classification_mode='dual_stream',
            num_pd_classes=2,
            num_stage_classes=5,
            num_subtype_classes=3,
            mae_enabled=(config.BRANCH_A_TASK == 'mae'),
            dropout_rate=config.DROPOUT,
        )

        if config.PRETRAINED == 'Half':
            _freeze_backbone_half(model)

        # ── 运行 fold ─────────────────────────────────────────────
        trainer = FoldTrainer(
            model=model,
            config=config,
            mtl_loader=mtl_loader,
            val_corn1_loader=corn1_val,
            val_pd_loader=pd_val,
            fold_dir=fold_dir,
        )
        fold_result = trainer.run()
        fold_result['fold'] = fold_idx
        fold_results.append(fold_result)

        print(f"\n  ── Fold {fold_idx+1} 最佳结果 ──")
        print(f"  BranchA  seg_dice: {fold_result.get('seg_dice', -1):.4f}")
        print(f"  BranchB  pd      : acc={fold_result.get('pd_acc',      0):.4f}  "
              f"pat={fold_result.get('pd_patient_acc', 0):.4f}  "
              f"f1={fold_result.get('pd_f1',      0):.4f}  "
              f"auc={fold_result.get('pd_auc',    0):.4f}")
        print(f"  BranchB  stage   : acc={fold_result.get('stage_acc',   0):.4f}  "
              f"pat={fold_result.get('stage_patient_acc', 0):.4f}  "
              f"f1={fold_result.get('stage_f1',   0):.4f}")
        print(f"  BranchB  subtype : acc={fold_result.get('subtype_acc', 0):.4f}  "
              f"pat={fold_result.get('subtype_patient_acc', 0):.4f}  "
              f"f1={fold_result.get('subtype_f1', 0):.4f}")

        # 释放显存
        del model, trainer, mtl_loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── 汇总统计 ───────────────────────────────────────────────────
    active_tasks = set(config.RUN_TASKS or ['pd', 'stage', 'subtype'])
    keys_to_agg = ['seg_dice', 'mae_loss']
    if 'pd' in active_tasks:
        keys_to_agg.extend(['pd_acc', 'pd_f1', 'pd_auc', 'pd_patient_acc'])
    if 'stage' in active_tasks:
        keys_to_agg.extend(['stage_acc', 'stage_f1', 'stage_patient_acc'])
    if 'subtype' in active_tasks:
        keys_to_agg.extend(['subtype_acc', 'subtype_f1', 'subtype_patient_acc'])
    mean_metrics, std_metrics = {}, {}
    for k in keys_to_agg:
        vals = [r[k] for r in fold_results if k in r and r[k] >= 0]
        if vals:
            mean_metrics[k] = float(np.mean(vals))
            std_metrics[k]  = float(np.std(vals))

    print(f"\n{'='*60}")
    print(f"K-fold 汇总 (BranchA={config.BRANCH_A_TASK})")
    print('='*60)
    for k, v in mean_metrics.items():
        print(f"  {k:20s}: {v:.4f} ± {std_metrics.get(k, 0):.4f}")

    summary = {
        'fold_results':   fold_results,
        'mean_metrics':   mean_metrics,
        'std_metrics':    std_metrics,
        'config':         cfg_dict,
        'output_dir':     exp_dir,
    }

    with open(os.path.join(exp_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n结果保存至: {exp_dir}")
    return summary


# ──────────────────────────────────────────────────────────────────
# 命令行入口
# ──────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description='MTL 多任务学习实验')
    p.add_argument('--branch_a_task', default='segmentation',
                   choices=['segmentation', 'mae', 'pseudo'],
                   help='Branch A 任务类型')
    p.add_argument('--epochs',    type=int,   default=40,    help='训练轮数')
    p.add_argument('--lr',        type=float, default=2e-4,  help='学习率')
    p.add_argument('--wd',        type=float, default=3e-3,  help='权重衰减')
    p.add_argument('--batch',     type=int,   default=16,    help='批大小')
    p.add_argument('--dropout',   type=float, default=0.0,   help='Dropout 率')
    p.add_argument('--pretrained', default='Half',
                   choices=['True', 'False', 'Half'], help='预训练策略')
    p.add_argument('--folds',     type=int,   default=5,     help='K-fold 数')
    p.add_argument('--seed',      type=int,   default=42,    help='随机种子')
    p.add_argument('--workers',   type=int,   default=4,     help='DataLoader worker 数')
    p.add_argument('--no_aug',    action='store_true',        help='禁用数据增强')
    p.add_argument('--output_dir', default=None,              help='输出目录')
    return p.parse_args()


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )

    args = _parse_args()

    config = MTLExperimentConfig()
    config.BRANCH_A_TASK    = args.branch_a_task
    config.NUM_EPOCHS       = args.epochs
    config.LEARNING_RATE    = args.lr
    config.WEIGHT_DECAY     = args.wd
    config.BATCH_SIZE       = args.batch
    config.DROPOUT          = args.dropout
    config.PRETRAINED       = (True if args.pretrained == 'True'
                               else False if args.pretrained == 'False'
                               else 'Half')
    config.NUM_FOLDS        = args.folds
    config.SEED             = args.seed
    config.NUM_WORKERS      = args.workers
    config.DATA_AUGMENTATION = not args.no_aug

    run_single_mtl_experiment(config, output_dir=args.output_dir)
