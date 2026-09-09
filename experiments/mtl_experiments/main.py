"""mtl 实验实现：训练、评估及单组执行；参数与批量控制见 batch_experiments.py。"""
from __future__ import annotations

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from experiments._execution import single_main
    single_main("experiments.mtl_experiments.main")
    raise SystemExit(0)

from utils.td_rest import resolve_baseline_tasks, td_rest_rows, td_rest_fold, save_td_rest_results
import json
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Literal, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from sklearn.metrics import f1_score, roc_auc_score
from multitasklearning.unet_multitask import UNetMultiTask
from multitasklearning.multitask_data_loader import build_corn1_loaders, build_pd_patient_kfold_loaders, MultiTaskDataLoader
from utils.patient_outputs import save_fold_patient_outputs
from utils.misc import set_seed
import hashlib
import time
from torch.utils.data import DataLoader
from multitasklearning.encoder import UNetEncoder
from multitasklearning.multitask_data_loader import PDImageDataset, _patient_kfold_split
from utils.patient_outputs import aggregate_patient_outputs
from collections import defaultdict
from typing import Tuple
from utils.subtype_hierarchy import SUBTYPE_TO_3CLS, build_patient_5x3_subtype_folds, subtype_pigd_mixed_label, subtype_tdrest_label




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
                br_a = "    BrA(mae): (no pixel-level dice metric)"
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



def build_paper_config(exp, device="cpu", num_workers=0):
    """Translate a selected paper recipe without changing its hyperparameters."""
    config = MTLExperimentConfig()
    config.BRANCH_A_TASK = exp.get('branch_a_task', 'segmentation')
    config.NUM_EPOCHS = exp.get('epochs', 40)
    config.LEARNING_RATE = exp.get('lr', 0.0002)
    config.WEIGHT_DECAY = exp.get('wd', 0.003)
    config.BATCH_SIZE = exp.get('batch', 16)
    config.DROPOUT = exp.get('dropout', 0.0)
    config.PRETRAINED = exp.get('pretrained', 'Half')
    config.DATA_AUGMENTATION = exp.get('data_augmentation', True)
    config.NUM_FOLDS = exp.get('folds', 5)
    config.SEED = exp.get('seed', 42)
    config.RUN_TASKS = exp.get('_run_tasks')
    if 'backbone_lr_ratio' in exp:
        config.BACKBONE_LR_RATIO = float(exp['backbone_lr_ratio'])
    _weight_overrides = {'pd_weight': 'PD_WEIGHT', 'stage_weight': 'STAGE_WEIGHT', 'subtype_weight': 'SUBTYPE_WEIGHT', 'seg_weight': 'SEG_WEIGHT', 'mae_weight': 'MAE_WEIGHT', 'pseudo_weight': 'PSEUDO_WEIGHT'}
    for (_k, _attr) in _weight_overrides.items():
        if _k in exp:
            setattr(config, _attr, float(exp[_k]))
    config.NUM_WORKERS = num_workers
    config.DEVICE = device
    return config


def run_paper_experiment(exp, output_dir, device="cpu", num_workers=0):
    exp = resolve_baseline_tasks(exp)
    cfg = build_paper_config(exp, device, num_workers)
    if exp.get('_subtype_td_rest'):
        return run_mtl_subtype_td_rest_experiment(cfg, output_dir=output_dir, seed=cfg.SEED,
            n_splits=exp.get('num_folds', 5), repeats=exp.get('kfold_repeats', 3))
    if exp.get('_transfer') == 'dm':
        return run_single_dm_transfer_experiment(cfg, tasks=exp['_run_tasks'], output_dir=output_dir)
    return run_single_mtl_experiment(cfg, output_dir=output_dir)

# ==================== DM workflow ====================



# ── 项目根目录 ─────────────────────────────────────────────────────────
_dm_THIS_FILE = Path(__file__).resolve()
_dm_CODES_ROOT = str(_dm_THIS_FILE.parents[2])
REPO_ROOT = str(_dm_THIS_FILE.parents[3])
if _dm_CODES_ROOT not in sys.path:
    sys.path.insert(0, _dm_CODES_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)




# ==================== 路径与任务常量 ====================

DM_JSON = 'datasets/dm_dataset_en/dm_patients_dataset_en.json'
DM_TASKS: List[str] = ['dm_binary', 'dm_subtype']

# DM 原始 group → 二分类标签（dm_subtype 排除 control）
_dm_DM_SUBTYPE_MAP = {'NDR': 0, 'NPDR': 1}

# pd 多任务预训练使用固定 seed，使所有 DM 实验共享同一份 pd encoder
PRETRAIN_SEED = 42

# 每条 MTL 线（branch_a_task）的最佳预训练超参（来自 pd_dataset_en 基线结论）
#   mae          : B03 (ep50, lr2e-4) —— 全实验验证集分数最高
#   segmentation : A05 (ep40, lr3e-4) —— Group A 最高
_dm_PRETRAIN_HP: Dict[str, Dict[str, Any]] = {
    'mae':          dict(epochs=50, lr=2e-4, wd=3e-3, dropout=0.0, pretrained='Half'),
    'segmentation': dict(epochs=40, lr=3e-4, wd=3e-3, dropout=0.0, pretrained='Half'),
    'pseudo':       dict(epochs=50, lr=2e-4, wd=3e-3, dropout=0.0, pretrained='Half'),
}

# encoder 缓存目录（相对 CODES_ROOT）
PRETRAIN_CACHE_DIR = 'results/mtl_dm_transfer/pretrain_cache'

# 预训练时用于切分 pd 训练/验证的 fold（取 fold 0：~80% 训练）
_dm_PRETRAIN_FOLD_IDX = 0
_dm_PRETRAIN_N_FOLDS = 5

# 进程内 encoder 缓存：sig -> encoder_state_dict
_dm_INPROC_ENCODER_CACHE: Dict[str, Dict[str, torch.Tensor]] = {}


# ==================== 路径辅助 ====================

def _dm_resolve_project_path(p: str) -> str:
    """把相对路径解析为绝对路径（优先 CODES_ROOT，再 REPO_ROOT）。"""
    p_obj = Path(p)
    if p_obj.is_absolute():
        return str(p_obj)
    for root in (_dm_CODES_ROOT, REPO_ROOT):
        cand = Path(root) / p_obj
        if cand.exists():
            return str(cand)
    return str(Path(_dm_CODES_ROOT) / p_obj)


# ==================== DM 数据构造 ====================

def build_dm_patients(task: str) -> List[Dict[str, Any]]:
    """把 DM 数据集映射为 ``PDImageDataset`` 可用的患者字典列表。

    - dm_binary ：control→0(group='control')，NDR/NPDR→1(group='pd')，80 患者。
    - dm_subtype：排除 control；NDR→0(group='control')，NPDR→1(group='pd')，60 患者。

    所有 DM 图像放入 image_paths['central']，whorl 留空（PDImageDataset 仅按
    group 推断二分类标签，stage/subtype 取 -1 被忽略）。
    """
    if task not in DM_TASKS:
        raise ValueError(f"未知 DM 任务: {task}，可选: {DM_TASKS}")

    dm_json_abs = _dm_resolve_project_path(DM_JSON)
    with open(dm_json_abs, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    patients: List[Dict[str, Any]] = []
    for rec in raw:
        grp = rec['group']
        imgs = rec.get('image_paths', []) or []
        if not imgs:
            continue

        if task == 'dm_binary':
            label = 0 if grp == 'control' else 1
        else:  # DM 亚型任务
            if grp == 'control':
                continue
            label = _dm_DM_SUBTYPE_MAP[grp]

        patients.append({
            'patient_id': rec.get('patient_id', rec.get('name', '')),
            'group': 'pd' if label == 1 else 'control',
            'stage': 0,
            'subtype': '',
            'image_paths': {'central': list(imgs), 'whorl': []},
        })
    return patients


# ==================== 预训练签名与缓存 ====================

def _dm_pretrain_signature(backbone: str, branch_a_task: str) -> str:
    hp = _dm_PRETRAIN_HP[branch_a_task]
    payload = {
        'backbone': backbone,
        'branch_a_task': branch_a_task,
        'epochs': hp['epochs'],
        'lr': hp['lr'],
        'wd': hp['wd'],
        'dropout': hp['dropout'],
        'pretrained': hp['pretrained'],
        'pretrain_seed': PRETRAIN_SEED,
        'fold_idx': _dm_PRETRAIN_FOLD_IDX,
        'n_folds': _dm_PRETRAIN_N_FOLDS,
    }
    raw = json.dumps(payload, sort_keys=True).encode('utf-8')
    return hashlib.sha1(raw).hexdigest()[:12]


def _dm_pretrain_mtl_encoder(backbone: str, branch_a_task: str,
                          device: str, log) -> Dict[str, torch.Tensor]:
    """在 pd_dataset_en (+ CORN-1) 上做一次多任务训练，返回 encoder 权重。"""
    hp = _dm_PRETRAIN_HP[branch_a_task]

    cfg = MTLExperimentConfig()
    cfg.BACKBONE = backbone
    cfg.BRANCH_A_TASK = branch_a_task
    cfg.NUM_EPOCHS = int(hp['epochs'])
    cfg.LEARNING_RATE = float(hp['lr'])
    cfg.WEIGHT_DECAY = float(hp['wd'])
    cfg.DROPOUT = float(hp['dropout'])
    cfg.PRETRAINED = hp['pretrained']
    cfg.NUM_FOLDS = _dm_PRETRAIN_N_FOLDS
    cfg.SEED = PRETRAIN_SEED
    cfg.DEVICE = device
    cfg.NUM_WORKERS = 0

    corn1_root = _dm_resolve_project_path(cfg.CORN1_ROOT)
    pd_json = _dm_resolve_project_path(cfg.PD_DATASET_JSON)
    datasets_base = _dm_resolve_project_path(cfg.DATASETS_BASE)

    set_seed(PRETRAIN_SEED)

    use_pseudo = (branch_a_task == 'pseudo')
    corn1_train, corn1_val = build_corn1_loaders(
        corn1_root,
        batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        augment_train=cfg.DATA_AUGMENTATION,
        image_size=cfg.IMAGE_SIZE,
        generate_pseudo=use_pseudo,
    )
    pd_train, pd_val = build_pd_patient_kfold_loaders(
        dataset_json=pd_json,
        fold_idx=_dm_PRETRAIN_FOLD_IDX,
        n_folds=_dm_PRETRAIN_N_FOLDS,
        datasets_base=datasets_base,
        dataset_mode=cfg.DATASET_MODE,
        batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        augment_train=cfg.DATA_AUGMENTATION,
        image_size=cfg.IMAGE_SIZE,
        random_state=PRETRAIN_SEED,
    )
    mtl_loader = MultiTaskDataLoader(corn1_train, pd_train)

    pretrained_bool = (cfg.PRETRAINED is True or cfg.PRETRAINED == 'Half')
    model = UNetMultiTask(
        encoder_type=cfg.BACKBONE,
        encoder_pretrained=pretrained_bool,
        classification_mode='dual_stream',
        num_pd_classes=2,
        num_stage_classes=5,
        num_subtype_classes=3,
        mae_enabled=(branch_a_task == 'mae'),
        dropout_rate=cfg.DROPOUT,
    )
    if cfg.PRETRAINED == 'Half':
        _freeze_backbone_half(model)

    cache_dir = _dm_resolve_project_path(PRETRAIN_CACHE_DIR)
    os.makedirs(cache_dir, exist_ok=True)
    fold_dir = os.path.join(cache_dir, f'_pretrain_{backbone}_{branch_a_task}')

    log(f"[dm-transfer] 在 pd_dataset_en (+CORN-1) 上做一次 MTL 预训练 "
        f"(backbone={backbone}, branch_a={branch_a_task}, epochs={cfg.NUM_EPOCHS}) ...")
    trainer = FoldTrainer(
        model=model,
        config=cfg,
        mtl_loader=mtl_loader,
        val_corn1_loader=corn1_val,
        val_pd_loader=pd_val,
        fold_dir=fold_dir,
    )
    trainer.run()

    # 载入早停期间保存的最优权重（best_model.pth 为完整模型 state_dict）
    best_ckpt = os.path.join(fold_dir, 'best_model.pth')
    if os.path.exists(best_ckpt) and os.path.getsize(best_ckpt) > 0:
        full_state = torch.load(best_ckpt, map_location='cpu')
        model.load_state_dict(full_state)
        log(f"[dm-transfer] 已载入预训练最优权重: {best_ckpt}")

    encoder_state = {k: v.detach().cpu().clone()
                     for k, v in model.encoder.state_dict().items()}
    return encoder_state


def get_or_build_pd_encoder(backbone: str, branch_a_task: str,
                            device: str, log) -> Dict[str, torch.Tensor]:
    """获取在 pd_dataset_en 上预训练好的 encoder 权重（复用优先，必要时训练一次）。

    复用优先级：
    1. 环境变量 ``RUN_MTL_PRETRAIN_CKPT`` 指定的 encoder ckpt；
    2. 进程内缓存；
    3. 磁盘缓存（按签名）；
    4. 都没有则在 pd_dataset_en (+CORN-1) 上预训练一次并原子化写盘。
    """
    sig = _dm_pretrain_signature(backbone, branch_a_task)

    # 1) 显式 ckpt 覆盖
    env_ckpt = os.environ.get('RUN_MTL_PRETRAIN_CKPT', '').strip()
    if env_ckpt:
        ckpt_abs = _dm_resolve_project_path(env_ckpt)
        if os.path.exists(ckpt_abs):
            log(f"[dm-transfer] 复用显式 encoder ckpt: {ckpt_abs}")
            state = torch.load(ckpt_abs, map_location='cpu')
            enc_state = state.get('encoder_state', state) if isinstance(state, dict) else state
            return enc_state
        log(f"[dm-transfer][warn] RUN_MTL_PRETRAIN_CKPT 指定文件不存在: {ckpt_abs}")

    # 2) 进程内缓存
    if sig in _dm_INPROC_ENCODER_CACHE:
        return _dm_INPROC_ENCODER_CACHE[sig]

    # 3) 磁盘缓存
    cache_dir = _dm_resolve_project_path(PRETRAIN_CACHE_DIR)
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{backbone}_{branch_a_task}_{sig}.pt")
    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
        log(f"[dm-transfer] 复用 pd_dataset_en MTL encoder 缓存（跳过预训练）:\n"
            f"             {cache_path}")
        payload = torch.load(cache_path, map_location='cpu')
        enc_state = payload['encoder_state']
        _dm_INPROC_ENCODER_CACHE[sig] = enc_state
        return enc_state

    # 4) 预训练一次
    enc_state = _dm_pretrain_mtl_encoder(backbone, branch_a_task, device, log)

    payload = {
        'encoder_state': enc_state,
        'meta': {
            'backbone': backbone,
            'branch_a_task': branch_a_task,
            'signature': sig,
            'pretrain_hp': _dm_PRETRAIN_HP[branch_a_task],
            'pretrain_seed': PRETRAIN_SEED,
            'created': datetime.now().isoformat(timespec='seconds'),
        },
    }
    tmp_path = f"{cache_path}.tmp.{os.getpid()}"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, cache_path)
    log(f"[dm-transfer] 已缓存 pd MTL encoder: {cache_path}")

    _dm_INPROC_ENCODER_CACHE[sig] = enc_state
    return enc_state


# ==================== 迁移模型（encoder + 分类头）====================

class MTLTransferModel(nn.Module):
    """复用 MTL encoder 表征 + 一个全局池化分类头，用于 DM 二分类迁移评估。"""

    def __init__(self, backbone: str, num_classes: int = 2,
                 dropout: float = 0.3, freeze_encoder: bool = True):
        super().__init__()
        self.encoder = UNetEncoder(backbone, pretrained=False)
        feat_dim = self.encoder.bottleneck_channels
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, num_classes),
        )
        self._freeze_encoder = freeze_encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def load_encoder(self, encoder_state: Dict[str, torch.Tensor]):
        self.encoder.load_state_dict(encoder_state)

    def train(self, mode: bool = True):
        super().train(mode)
        # 冻结 encoder 时让其 BN/Dropout 始终处于 eval，稳定迁移表征
        if self._freeze_encoder:
            self.encoder.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._freeze_encoder:
            with torch.no_grad():
                feats = self.encoder(x)
        else:
            feats = self.encoder(x)
        pooled = self.pool(feats[-1])
        return self.head(pooled)


# ==================== DM 下游训练 ====================

def _dm_compute_metrics(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    preds = logits.argmax(dim=1).cpu().numpy()
    labs = labels.cpu().numpy()
    probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
    acc = float((preds == labs).mean())
    f1 = float(f1_score(labs, preds, average='macro', zero_division=0))
    try:
        auc = float(roc_auc_score(labs, probs))
    except ValueError:
        auc = 0.0
    return {'acc': acc, 'f1': f1, 'auc': auc}


def _dm_make_loader(patients: List[Dict[str, Any]], datasets_base: str,
                 augment: bool, image_size: int, batch_size: int,
                 num_workers: int, shuffle: bool, drop_last: bool) -> DataLoader:
    ds = PDImageDataset(patients, datasets_base=datasets_base,
                        augment=augment, image_size=image_size)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True, drop_last=drop_last)


def _dm_class_weights(patients: List[Dict[str, Any]], device: str) -> torch.Tensor:
    counts = [0, 0]
    for p in patients:
        counts[1 if p['group'] == 'pd' else 0] += 1
    total = sum(counts)
    w = [total / (2.0 * c) if c > 0 else 0.0 for c in counts]
    return torch.tensor(w, dtype=torch.float32, device=device)


def _dm_train_one_fold(config: MTLExperimentConfig,
                    encoder_state: Dict[str, torch.Tensor],
                    train_patients: List[Dict[str, Any]],
                    val_patients: List[Dict[str, Any]],
                    datasets_base: str, log,
                    output_dir: Optional[str] = None,
                    task: str = 'dm',
                    fold_idx: int = 1) -> Dict[str, float]:
    device = config.DEVICE
    freeze = (config.BACKBONE_LR_RATIO <= 0.0)

    model = MTLTransferModel(
        backbone=config.BACKBONE,
        num_classes=2,
        dropout=config.DROPOUT,
        freeze_encoder=freeze,
    ).to(device)
    model.load_encoder(encoder_state)

    train_loader = _dm_make_loader(
        train_patients, datasets_base, augment=config.DATA_AUGMENTATION,
        image_size=config.IMAGE_SIZE, batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS, shuffle=True, drop_last=True,
    )
    val_loader = _dm_make_loader(
        val_patients, datasets_base, augment=False,
        image_size=config.IMAGE_SIZE, batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS, shuffle=False, drop_last=False,
    )

    # 优化器：冻结时只训练分类头；否则 encoder 用更小学习率
    if freeze:
        params = [{'params': model.head.parameters(), 'lr': config.LEARNING_RATE}]
    else:
        params = [
            {'params': model.encoder.parameters(),
             'lr': config.LEARNING_RATE * config.BACKBONE_LR_RATIO},
            {'params': model.head.parameters(), 'lr': config.LEARNING_RATE},
        ]
    optimizer = torch.optim.AdamW(params, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.NUM_EPOCHS, eta_min=1e-7)

    criterion = nn.CrossEntropyLoss(weight=_dm_class_weights(train_patients, device))

    best_score = -1.0
    best_metrics: Dict[str, float] = {'acc': 0.0, 'f1': 0.0, 'auc': 0.0}
    best_train_acc = 0.0
    patience_counter = 0

    for epoch in range(1, config.NUM_EPOCHS + 1):
        model.train()
        tr_correct, tr_total, run_loss = 0, 0, 0.0
        for batch in train_loader:
            imgs = batch['image'].to(device)
            labels = batch['pd_label'].to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            if config.GRAD_CLIP and config.GRAD_CLIP > 0:
                nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    config.GRAD_CLIP)
            optimizer.step()
            run_loss += float(loss.item())
            tr_correct += int((logits.argmax(1) == labels).sum().item())
            tr_total += labels.numel()
        scheduler.step()
        train_acc = tr_correct / max(tr_total, 1)

        # 验证
        model.eval()
        all_logits, all_labels, all_patient_ids = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch['image'].to(device)
                logits = model(imgs)
                all_logits.append(logits.cpu())
                all_labels.append(batch['pd_label'])
                dataset = getattr(val_loader, 'dataset', None)
                patients = getattr(dataset, 'patients', None)
                for idx in batch['patient_idx'].detach().cpu().tolist():
                    if patients is not None and 0 <= int(idx) < len(patients):
                        all_patient_ids.append(str(patients[int(idx)].get('patient_id', idx)))
                    else:
                        all_patient_ids.append(str(idx))
        v_logits = torch.cat(all_logits)
        v_labels = torch.cat(all_labels)
        metrics = _dm_compute_metrics(v_logits, v_labels)
        patient_rows = aggregate_patient_outputs(
            all_patient_ids,
            v_labels.cpu().numpy().tolist(),
            logits=v_logits.cpu().numpy(),
        )
        metrics['patient_acc'] = float(np.mean([r['correct'] for r in patient_rows])) if patient_rows else 0.0
        score = metrics['patient_acc']

        improved = score > best_score
        if improved:
            if output_dir:
                patient_result = save_fold_patient_outputs(
                    output_dir, task, '', fold_idx,
                    all_patient_ids,
                    v_labels.cpu().numpy().tolist(),
                    logits=v_logits.cpu().numpy(),
                )
                metrics['patient_acc'] = patient_result['patient_acc']
                metrics['patient_logits_json'] = patient_result['patient_logits_json']
                metrics['patient_logits_csv'] = patient_result['patient_logits_csv']
            best_score = score
            best_metrics = metrics
            best_train_acc = train_acc
            patience_counter = 0
        else:
            patience_counter += 1

        log(f"    E{epoch:03d}/{config.NUM_EPOCHS:03d} loss={run_loss/max(len(train_loader),1):.4f} "
            f"tr_acc={train_acc:.3f} | val acc={metrics['acc']:.3f} "
            f"f1={metrics['f1']:.3f} auc={metrics['auc']:.3f} "
            f"pat={metrics['patient_acc']:.3f}{' ↑' if improved else ''}")

        if config.EARLY_STOPPING and patience_counter >= config.PATIENCE:
            log(f"    早停触发 (patience={config.PATIENCE})")
            break

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        'acc': best_metrics['acc'],
        'patient_acc': best_metrics.get('patient_acc', 0.0),
        'patient_logits_json': best_metrics.get('patient_logits_json', ''),
        'patient_logits_csv': best_metrics.get('patient_logits_csv', ''),
        'f1': best_metrics['f1'],
        'auc': best_metrics['auc'],
        'train_acc': best_train_acc,
    }


def _dm_run_dm_task_kfold(config: MTLExperimentConfig,
                       encoder_state: Dict[str, torch.Tensor],
                       task: str, log,
                       output_dir: Optional[str] = None) -> Dict[str, Any]:
    """对单个 DM 任务执行 patient-level K-fold（复用 pd encoder）。"""
    datasets_base = _dm_resolve_project_path(config.DATASETS_BASE)
    patients = build_dm_patients(task)

    fold_metrics: List[Dict[str, float]] = []
    for fold_idx in range(config.NUM_FOLDS):
        train_patients, val_patients = _patient_kfold_split(
            patients, fold_idx, config.NUM_FOLDS, random_state=config.SEED)
        log(f"\n  ── {task} | Fold {fold_idx + 1}/{config.NUM_FOLDS} "
            f"(train_pat={len(train_patients)}, val_pat={len(val_patients)}) ──")
        m = _dm_train_one_fold(config, encoder_state, train_patients,
                            val_patients, datasets_base, log,
                            output_dir=output_dir, task=task, fold_idx=fold_idx + 1)
        m['fold'] = fold_idx
        fold_metrics.append(m)
        log(f"  Fold {fold_idx + 1} best: acc={m['acc']:.4f} f1={m['f1']:.4f} "
            f"auc={m['auc']:.4f} pat={m.get('patient_acc', 0.0):.4f}")

    accs = [m['acc'] for m in fold_metrics]
    f1s = [m['f1'] for m in fold_metrics]
    aucs = [m['auc'] for m in fold_metrics]
    patient_accs = [m.get('patient_acc', 0.0) for m in fold_metrics]
    train_accs = [m['train_acc'] for m in fold_metrics]
    return {
        'task': task,
        'fold_metrics': fold_metrics,
        'avg_train_acc': float(np.mean(train_accs)),
        'avg_acc': float(np.mean(accs)),
        'std_acc': float(np.std(accs)),
        'avg_patient_acc': float(np.mean(patient_accs)),
        'std_patient_acc': float(np.std(patient_accs)),
        'avg_f1': float(np.mean(f1s)),
        'std_f1': float(np.std(f1s)),
        'avg_auc': float(np.mean(aucs)),
        'num_folds': len(fold_metrics),
    }


# ==================== 单实验运行 ====================

class _dm_Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def _dm_write_dm_report(report_path: str, config: MTLExperimentConfig,
                     task_results: List[Dict[str, Any]]):
    lines = [
        '# MTL → DM 迁移实验报告',
        '',
        f"- backbone: `{config.BACKBONE}`",
        f"- 预训练线 (branch_a_task): `{config.BRANCH_A_TASK}`",
        f"- 迁移策略: {'冻结 encoder（线性探针）' if config.BACKBONE_LR_RATIO <= 0 else f'微调 encoder (lr×{config.BACKBONE_LR_RATIO})'}",
        f"- DM 微调: epochs={config.NUM_EPOCHS}, lr={config.LEARNING_RATE:.0e}, "
        f"wd={config.WEIGHT_DECAY:.0e}, dropout={config.DROPOUT}, "
        f"folds={config.NUM_FOLDS}, seed={config.SEED}",
        '',
        '| 任务 | 验证 Acc | 验证 F1 | 验证 AUC | 训练 Acc | 折数 |',
        '|------|----------|---------|----------|----------|------|',
    ]
    for r in task_results:
        lines.append(
            f"| {r['task']} | {r['avg_acc']:.4f} ± {r['std_acc']:.4f} "
            f"| {r['avg_f1']:.4f} ± {r['std_f1']:.4f} "
            f"| {r['avg_auc']:.4f} | {r['avg_train_acc']:.4f} | {r['num_folds']} |"
        )
    lines.append('')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def run_single_dm_transfer_experiment(config: MTLExperimentConfig,
                                      output_dir: Optional[str] = None,
                                      tasks: Optional[List[str]] = None
                                      ) -> Dict[str, Any]:
    """运行单个 MTL→DM 迁移实验（复用 pd encoder，重训 DM 分类头）。"""
    tasks = tasks or DM_TASKS

    exp_dir = output_dir or os.path.join(
        _dm_CODES_ROOT, 'results/mtl_dm_transfer',
        datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(exp_dir, exist_ok=True)

    log_f = open(os.path.join(exp_dir, 'output.txt'), 'a', encoding='utf-8', buffering=1)
    _orig_stdout = sys.stdout
    sys.stdout = _dm_Tee(_orig_stdout, log_f)

    def log(msg: str = ''):
        print(msg, flush=True)

    try:
        log('=' * 70)
        log(f"MTL→DM 迁移实验 | backbone={config.BACKBONE} "
            f"branch_a={config.BRANCH_A_TASK} tasks={tasks}")
        log(f"DM 微调: ep={config.NUM_EPOCHS} lr={config.LEARNING_RATE:.0e} "
            f"wd={config.WEIGHT_DECAY:.0e} drop={config.DROPOUT} "
            f"bblr_ratio={config.BACKBONE_LR_RATIO} folds={config.NUM_FOLDS} "
            f"seed={config.SEED}")
        log('=' * 70)

        # 1) 取/建 pd encoder（按线缓存复用）
        encoder_state = get_or_build_pd_encoder(
            config.BACKBONE, config.BRANCH_A_TASK, config.DEVICE, log)

        # 2) 逐任务迁移微调
        set_seed(config.SEED)
        task_results: List[Dict[str, Any]] = []
        for task in tasks:
            t0 = time.time()
            res = _dm_run_dm_task_kfold(config, encoder_state, task, log, output_dir=exp_dir)
            res['duration_min'] = (time.time() - t0) / 60.0
            task_results.append(res)
            log(f"\n[{task}] 平均 acc={res['avg_acc']:.4f}±{res['std_acc']:.4f} "
                f"pat={res.get('avg_patient_acc', 0.0):.4f}±{res.get('std_patient_acc', 0.0):.4f} "
                f"f1={res['avg_f1']:.4f}±{res['std_f1']:.4f} "
                f"auc={res['avg_auc']:.4f}  ({res['duration_min']:.1f}min)")

        # 3) 保存结果
        summary = {
            'backbone': config.BACKBONE,
            'branch_a_task': config.BRANCH_A_TASK,
            'transfer': 'dm',
            'config': {
                'num_epochs': config.NUM_EPOCHS,
                'lr': config.LEARNING_RATE,
                'weight_decay': config.WEIGHT_DECAY,
                'dropout': config.DROPOUT,
                'backbone_lr_ratio': config.BACKBONE_LR_RATIO,
                'num_folds': config.NUM_FOLDS,
                'seed': config.SEED,
            },
            'task_results': task_results,
            'output_dir': exp_dir,
        }
        with open(os.path.join(exp_dir, 'results.json'), 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        _dm_write_dm_report(os.path.join(exp_dir, 'experiment_report.md'),
                         config, task_results)
        log(f"\n结果保存至: {exp_dir}")
        return summary
    finally:
        sys.stdout = _orig_stdout
        log_f.close()


# ==================== SUBTYPE workflow ====================



_subtype_THIS_FILE = Path(__file__).resolve()


def _subtype_discover_project_root() -> Path:
    """自动发现仓库根目录，兼容服务器上多一层 PD 的挂载结构。"""
    candidates: List[Path] = []
    for p in [_subtype_THIS_FILE.parent, *_subtype_THIS_FILE.parents]:
        candidates.append(p)
        candidates.append(p / 'PD')

    seen = set()
    for p in candidates:
        if p in seen:
            continue
        seen.add(p)
        if (p / 'codes' / 'multitasklearning').is_dir():
            return p
        if p.name == 'codes' and (p / 'multitasklearning').is_dir():
            return p.parent
    return _subtype_THIS_FILE.parents[3]


_subtype_PROJECT_ROOT = _subtype_discover_project_root()
_subtype_CODES_ROOT = _subtype_PROJECT_ROOT / 'codes'
sys.path.insert(0, str(_subtype_PROJECT_ROOT))
sys.path.insert(0, str(_subtype_CODES_ROOT))



def _subtype_resolve_dir(rel: str, is_dir: bool = True) -> str:
    """兼容以仓库根、codes 或服务器 PD 子目录为基准的部署，解析数据路径。"""
    rel_path = Path(rel)
    if rel_path.is_absolute():
        candidates = [rel_path]
    else:
        candidates = [
            _subtype_PROJECT_ROOT / rel_path,
            _subtype_CODES_ROOT / rel_path,
        ]
        if _subtype_PROJECT_ROOT.name != 'PD' and (_subtype_PROJECT_ROOT / 'PD').is_dir():
            candidates.extend([
                _subtype_PROJECT_ROOT / 'PD' / rel_path,
                _subtype_PROJECT_ROOT / 'PD' / 'codes' / rel_path,
            ])
        rel_norm = rel.replace('\\', '/').lstrip('./')
        if rel_norm.startswith('codes/'):
            candidates.append(_subtype_PROJECT_ROOT / rel_norm[len('codes/'):])

    unique_candidates: List[Path] = []
    seen = set()
    for cand in candidates:
        cand = cand.resolve() if cand.exists() else cand
        key = str(cand)
        if key in seen:
            continue
        seen.add(key)
        unique_candidates.append(cand)

    for cand in unique_candidates:
        if cand.is_dir() if is_dir else cand.is_file():
            return str(cand)

    expected = '目录' if is_dir else '文件'
    tried = '\n'.join(f'  - {cand}' for cand in unique_candidates)
    raise FileNotFoundError(f"MTL subtype hierarchy 路径未找到有效{expected}: {rel}\n已尝试路径:\n{tried}")


def _subtype_relabel_samples(dataset: PDImageDataset, patients: List[Dict], task: str) -> None:
    """覆盖 PDImageDataset.samples 中的 subtype 标签为二分类标签。"""
    new_samples = []
    for (path, pd_l, stage_l, _sub_l, p_idx) in dataset.samples:
        y3 = SUBTYPE_TO_3CLS[patients[p_idx]['subtype']]
        if task == 'pd_subtype_tdrest':
            bin_l = subtype_tdrest_label(y3)
        else:  # pd_subtype_pigd_mixed
            lab = subtype_pigd_mixed_label(y3)
            bin_l = -1 if lab is None else lab
        new_samples.append((path, pd_l, stage_l, bin_l, p_idx))
    dataset.samples = new_samples


def _subtype_build_model(config: MTLExperimentConfig) -> UNetMultiTask:
    pretrained_bool = (config.PRETRAINED is True or config.PRETRAINED == 'Half')
    model = UNetMultiTask(
        encoder_type=config.BACKBONE,
        encoder_pretrained=pretrained_bool,
        classification_mode='dual_stream',
        num_pd_classes=2,
        num_stage_classes=5,
        num_subtype_classes=2,   # 二分类子任务
        mae_enabled=(config.BRANCH_A_TASK == 'mae'),
        dropout_rate=config.DROPOUT,
    )
    if config.PRETRAINED == 'Half':
        _freeze_backbone_half(model)
    return model


def _subtype_jsonable_config(config: MTLExperimentConfig) -> Dict:
    """返回可序列化为 JSON 的 MTL 配置快照。"""
    out = {}
    for key, value in vars(config).items():
        if key.startswith('_'):
            continue
        try:
            json.dumps(value)
            out[key] = value
        except TypeError:
            out[key] = str(value)
    return out


def _subtype_branch_a_loss(config: MTLExperimentConfig, outputs: Dict,
                   corn1_batch: Dict, seg_criterion, device) -> torch.Tensor:
    task = config.BRANCH_A_TASK
    if task in ('segmentation', 'pseudo'):
        masks = corn1_batch['mask'].to(device)
        seg = outputs['segmentation']
        if seg.shape[2:] != masks.shape[2:]:
            seg = torch.nn.functional.interpolate(
                seg, size=masks.shape[2:], mode='bilinear', align_corners=False)
        weight = config.SEG_WEIGHT if task == 'segmentation' else config.PSEUDO_WEIGHT
        return seg_criterion(seg, masks) * weight
    if task == 'mae':
        mae_raw = outputs.get('losses', {}).get('seg_mae')
        if mae_raw is not None and torch.is_tensor(mae_raw):
            return mae_raw * config.MAE_WEIGHT
    return torch.tensor(0.0, device=device)


def _subtype_train_one_subtask(config: MTLExperimentConfig, corn1_root: str,
                       train_patients: List[Dict], val_patients: List[Dict],
                       task: str, datasets_base: str,
                       artifact_dir: Optional[str] = None,
                       reuse_checkpoint: bool = True) -> UNetMultiTask:
    """训练单个二分类子任务，返回 best 模型。"""
    device = torch.device(config.DEVICE)
    model = _subtype_build_model(config).to(device)
    checkpoint_path = os.path.join(artifact_dir, 'best_model.pth') if artifact_dir else None
    history_path = os.path.join(artifact_dir, 'history.json') if artifact_dir else None

    if checkpoint_path and reuse_checkpoint and os.path.exists(checkpoint_path):
        payload = torch.load(checkpoint_path, map_location=device)
        state = payload.get('model_state', payload) if isinstance(payload, dict) else payload
        model.load_state_dict(state)
        print(f"    [{task}] 复用已有模型，跳过训练: {checkpoint_path}", flush=True)
        return model

    if artifact_dir:
        os.makedirs(artifact_dir, exist_ok=True)

    # Branch A: CORN-1 加载器
    corn1_train, _ = build_corn1_loaders(
        corn1_root, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS,
        augment_train=config.DATA_AUGMENTATION, image_size=config.IMAGE_SIZE,
        generate_pseudo=(config.BRANCH_A_TASK == 'pseudo'),
    )

    # Branch B: subtype 二分类 PD 加载器
    train_ds = PDImageDataset(train_patients, datasets_base=datasets_base,
                              augment=config.DATA_AUGMENTATION, image_size=config.IMAGE_SIZE)
    val_ds = PDImageDataset(val_patients, datasets_base=datasets_base,
                            augment=False, image_size=config.IMAGE_SIZE)
    _subtype_relabel_samples(train_ds, train_patients, task)
    _subtype_relabel_samples(val_ds, val_patients, task)

    pd_train = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
                          num_workers=config.NUM_WORKERS, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False,
                            num_workers=config.NUM_WORKERS)
    mtl_loader = MultiTaskDataLoader(corn1_train, pd_train)

    optimizer = _make_optimizer(model, config)
    if config.LR_SCHEDULER == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.NUM_EPOCHS, eta_min=1e-7)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    use_amp = config.MIXED_PRECISION and 'cuda' in str(config.DEVICE)
    scaler = GradScaler() if use_amp else None
    seg_criterion = DiceBCELoss()
    sub_criterion = FocalCELoss(ignore_index=-1)

    best_acc, best_state = -1.0, None
    patience_counter = 0
    history = []
    for epoch in range(1, config.NUM_EPOCHS + 1):
        model.train()
        epoch_losses = []
        for corn1_batch, pd_batch in mtl_loader:
            optimizer.zero_grad(set_to_none=True)
            with (autocast(device_type='cuda') if scaler
                  else torch.autocast('cpu', enabled=False)):
                out_a = model(corn1_batch['image'].to(device),
                              use_mae=(config.BRANCH_A_TASK == 'mae'), training=True)
                loss_a = _subtype_branch_a_loss(config, out_a, corn1_batch, seg_criterion, device)

                sub_l = pd_batch['subtype_label'].to(device)
                out_b = model(pd_batch['image'].to(device),
                              subtype_labels=sub_l, use_mae=False, training=True)
                loss_b = sub_criterion(out_b['subtype_logits'], sub_l) * config.SUBTYPE_WEIGHT
                loss = loss_a + loss_b
                epoch_losses.append(float(loss.detach().cpu().item()))

            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
                optimizer.step()
        scheduler.step()

        # 验证：image-level subtype 二分类准确率（用于早停/选最佳）
        val_acc = _subtype_eval_subtype_acc(model, val_loader, device)
        improved = val_acc > best_acc
        if improved:
            best_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            if checkpoint_path:
                torch.save({
                    'model_state': best_state,
                    'task': task,
                    'best_val_acc': best_acc,
                    'epoch': epoch,
                    'config': _subtype_jsonable_config(config),
                }, checkpoint_path)
        else:
            patience_counter += 1
        history.append({
            'epoch': epoch,
            'train_loss': float(np.mean(epoch_losses)) if epoch_losses else 0.0,
            'val_bin_acc': float(val_acc),
            'best_val_acc': float(best_acc),
            'improved': bool(improved),
        })
        if history_path:
            with open(history_path, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
        print(f"    [{task}] E{epoch:03d}/{config.NUM_EPOCHS} "
              f"val_bin_acc={val_acc:.4f}{' ↑' if improved else ''}", flush=True)
        if config.EARLY_STOPPING and patience_counter >= config.PATIENCE:
            print(f"    [{task}] 早停 (patience={config.PATIENCE})", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def _subtype_eval_subtype_acc(model, loader, device) -> float:
    model.eval()
    correct, total = 0, 0
    for batch in loader:
        labels = batch['subtype_label'].to(device)
        mask = labels >= 0
        if mask.sum() == 0:
            continue
        out = model(batch['image'].to(device), use_mae=False, training=False)
        preds = out['subtype_logits'].argmax(dim=1)
        correct += int((preds[mask] == labels[mask]).sum().item())
        total += int(mask.sum().item())
    return correct / total if total else 0.0


@torch.no_grad()
def _subtype_infer_patient_level(model, patients: List[Dict], config: MTLExperimentConfig,
                         datasets_base: str) -> Tuple[Dict[str, int], Dict[str, float]]:
    """对一组患者做 patient-level 推理（图像 softmax 概率均值聚合）。"""
    device = torch.device(config.DEVICE)
    ds = PDImageDataset(patients, datasets_base=datasets_base,
                        augment=False, image_size=config.IMAGE_SIZE)
    loader = DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=False,
                        num_workers=config.NUM_WORKERS)

    model.eval()
    sum_probs: Dict[int, np.ndarray] = defaultdict(lambda: np.zeros(2))
    counts: Dict[int, int] = defaultdict(int)
    for batch in loader:
        out = model(batch['image'].to(device), use_mae=False, training=False)
        probs = torch.softmax(out['subtype_logits'], dim=1).cpu().numpy()
        pidxs = batch['patient_idx']
        pidxs = pidxs.tolist() if torch.is_tensor(pidxs) else list(pidxs)
        for i, pidx in enumerate(pidxs):
            sum_probs[int(pidx)] += probs[i]
            counts[int(pidx)] += 1

    pred_by_pid: Dict[str, int] = {}
    prob_class1_by_pid: Dict[str, float] = {}
    for pidx, total in sum_probs.items():
        mean_prob = total / max(1, counts[pidx])
        patient_id = patients[pidx]['patient_id']
        pred_by_pid[patient_id] = int(np.argmax(mean_prob))
        prob_class1_by_pid[patient_id] = float(mean_prob[1])
    return pred_by_pid, prob_class1_by_pid


def run_mtl_subtype_td_rest_experiment(config: MTLExperimentConfig, output_dir: str,
                                         seed: int = 42, n_splits: int = 5,
                                         repeats: int = 3) -> Dict:
    """
    运行 MTL subtype TD-rest 二分类 5-fold x 3 重算实验（只跑 subtype 任务）。

    Returns:
        summary 字典（同时写入 summary.json）。
    """
    os.makedirs(output_dir, exist_ok=True)
    set_seed(seed)

    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump({
            'config': _subtype_jsonable_config(config),
            'seed': seed,
            'n_splits': n_splits,
            'repeats': repeats,
        }, f, indent=2, ensure_ascii=False)

    # 只跑 subtype：关闭 PD / Stage 分类损失
    config.PD_WEIGHT = 0.0
    config.STAGE_WEIGHT = 0.0
    if config.SUBTYPE_WEIGHT <= 0:
        config.SUBTYPE_WEIGHT = 1.0

    corn1_root = _subtype_resolve_dir(config.CORN1_ROOT, is_dir=True)
    datasets_base = _subtype_resolve_dir(config.DATASETS_BASE, is_dir=True)
    pd_json = _subtype_resolve_dir(config.PD_DATASET_JSON, is_dir=False)

    with open(pd_json, 'r', encoding='utf-8') as f:
        all_patients = json.load(f)
    subtype_patients = [p for p in all_patients
                        if p['group'] == 'pd' and p.get('subtype') in SUBTYPE_TO_3CLS]
    pid_to_patient = {p['patient_id']: p for p in subtype_patients}
    records = [{'patient_id': p['patient_id'], 'subtype': p['subtype']}
               for p in subtype_patients]
    folds = build_patient_5x3_subtype_folds(records, seed=seed,
                                            n_splits=n_splits, repeats=repeats)

    print("=" * 60)
    print("MTL Subtype TD-rest 二分类公平重算 (5-fold x 3, 仅 subtype 任务)")
    print(f"subtype 患者数: {len(subtype_patients)} | "
          f"backbone: {config.BACKBONE} | BranchA: {config.BRANCH_A_TASK} | "
          f"fold 总数: {len(folds)}")
    print("=" * 60)

    fold_results: List[Dict] = []
    pred_rows: List[Dict] = []

    for global_idx, fold in enumerate(folds, start=1):
        repeat, fold_no = fold['repeat'], fold['fold']
        train_patients = [pid_to_patient[pid] for pid in fold['train_patients_full']]
        val_patients = [pid_to_patient[pid] for pid in fold['val_patients_full']]

        print(f"\n--- repeat {repeat} fold {fold_no} ({global_idx}/{len(folds)}) | "
              f"train={len(train_patients)} val={len(val_patients)} ---", flush=True)
        fold_artifact_dir = os.path.join(output_dir, f'repeat_{repeat}_fold_{fold_no}')

        # 任务: TD vs rest（全部 subtype 患者）
        tdrest_model = _subtype_train_one_subtask(
            config, corn1_root, train_patients, val_patients,
            'pd_subtype_tdrest', datasets_base,
            artifact_dir=os.path.join(fold_artifact_dir, 'pd_subtype_tdrest'))
        tdrest_pred, tdrest_prob = _subtype_infer_patient_level(
            tdrest_model, val_patients, config, datasets_base)
        del tdrest_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        rows = td_rest_rows(pid_to_patient, fold['val_patients_full'],
                            tdrest_pred, tdrest_prob, repeat, fold_no)
        fold_results.append(td_rest_fold(rows))
        pred_rows.extend(rows)

    return save_td_rest_results(output_dir, fold_results, pred_rows,
                                n_splits, repeats, seed)
