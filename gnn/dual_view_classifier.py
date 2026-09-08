"""
dual_view_classifier.py - 双视角 CNN 分类器（高层封装）

对标 core.basic_classifier.PDClassifier，提供统一的 train_one_fold 接口。

功能
----
- 多任务训练（pd / stage / subtype）
- 差异化学习率（backbone vs 融合层 / 分类头）
- Linear Warmup + Cosine 调度
- 梯度裁剪 + Early Stopping
"""

import os
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from torchvision import transforms

from core.basic_models import get_image_size, BACKBONE_CONFIG
from core.light_models import LIGHT_BACKBONE_CONFIG
from .dual_view_dataset import DualViewDataset
from .dual_view_model import DualViewModel, DUAL_VIEW_TASK_CONFIG


def _get_image_size(backbone: str, pretrained):
    """兼容 light backbone（均使用 224 固定尺寸，不做 use_original 处理）。"""
    if backbone in LIGHT_BACKBONE_CONFIG:
        if pretrained is False:
            return (None, None, True)
        size = LIGHT_BACKBONE_CONFIG[backbone]['input_size']
        return (size + 32, size, False)
    return get_image_size(backbone, pretrained)


# ─── 配置 ─────────────────────────────────────────────────────────────────────

@dataclass
class DualViewConfig:
    """双视角 CNN 分类器全参数配置。"""

    # ── 任务 ─────────────────────────────────────────────────────────────────────
    task: str = 'pd'                     # 'pd' | 'stage' | 'subtype'

    # ── 模型 ─────────────────────────────────────────────────────────────────────
    backbone: str = 'resnet18'
    pretrained: object = 'Half'          # True | False | 'Half'
    shared_backbone: bool = True         # en 和 unet 是否共享 backbone 权重

    # ── 融合 ─────────────────────────────────────────────────────────────────────
    fusion_dim: int = 256
    dropout: float = 0.3
    classifier_dropout: float = 0.3

    # ── 训练 ─────────────────────────────────────────────────────────────────────
    lr: float = 1e-4
    backbone_lr_ratio: float = 0.1      # backbone 实际 lr = lr * ratio
    weight_decay: float = 1e-3
    optimizer: str = 'adamw'            # 'adam' | 'adamw'
    lr_scheduler: str = 'cosine'        # 'cosine' | 'step' | 'none'
    warmup_epochs: int = 5
    epochs: int = 60
    patience: int = 12
    batch_size: int = 8

    # ── 数据 ─────────────────────────────────────────────────────────────────────
    region: str = 'both'               # 'both' | 'central' | 'whorl'
    data_augmentation: bool = True
    aug_horizontal_flip: float = 0.5
    aug_rotation: float = 6.0
    aug_brightness: float = 0.12
    aug_contrast: float = 0.12

    # ── 损失 ─────────────────────────────────────────────────────────────────────
    label_smoothing: float = 0.0

    # ── 其他 ─────────────────────────────────────────────────────────────────────
    num_workers: int = 0
    seed: int = 42


# ─── 变换 ─────────────────────────────────────────────────────────────────────

def _build_transforms(cfg: DualViewConfig, for_unet: bool = False):
    """
    构建图像变换。

    unet 图像与 en 图像使用相同的几何变换配置，
    但均采用 ImageNet 归一化（两种图像都是 RGB 三通道）。
    """
    resize, crop, use_orig = _get_image_size(cfg.backbone, cfg.pretrained)
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    base = [] if use_orig else [transforms.Resize((resize, resize)), transforms.CenterCrop(crop)]

    aug = []
    if cfg.data_augmentation:
        if cfg.aug_horizontal_flip > 0:
            aug.append(transforms.RandomHorizontalFlip(cfg.aug_horizontal_flip))
        if cfg.aug_rotation > 0:
            aug.append(transforms.RandomRotation(cfg.aug_rotation))
        if cfg.aug_brightness > 0 or cfg.aug_contrast > 0:
            aug.append(transforms.ColorJitter(
                brightness=cfg.aug_brightness, contrast=cfg.aug_contrast
            ))

    train_tf = transforms.Compose(base + aug + [transforms.ToTensor(), normalize])
    val_tf = transforms.Compose(base + [transforms.ToTensor(), normalize])
    return train_tf, val_tf


# ─── 评估 ─────────────────────────────────────────────────────────────────────

def _evaluate(model: DualViewModel, loader: DataLoader, device: str, task: str) -> Dict[str, Any]:
    """单任务评估，返回与 basic_trainer.evaluate 对齐的结果字典。"""
    model.eval()
    all_preds, all_labels, all_probs, all_logits, all_patient_ids = [], [], [], [], []
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            en_img, unet_img, pd_l, stage_l, sub_l = batch[:5]
            patient_ids = batch[5] if len(batch) > 5 else [str(i) for i in range(len(pd_l))]
            label_b = {'pd': pd_l, 'stage': stage_l, 'subtype': sub_l}[task]
            mask = label_b >= 0
            if not mask.any():
                continue

            en_m = en_img[mask].to(device)
            unet_m = unet_img[mask].to(device)
            labels_m = label_b[mask].to(device)

            logits = model(en_m, unet_m)
            loss = F.cross_entropy(logits, labels_m)
            total_loss += loss.item()
            n_batches += 1

            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels_m.cpu().numpy())
            all_probs.extend(torch.softmax(logits, 1).cpu().numpy())
            all_logits.extend(logits.cpu().numpy())
            keep = mask.cpu().numpy().tolist()
            all_patient_ids.extend([str(pid) for pid, ok in zip(patient_ids, keep) if ok])

    if not all_labels:
        return {'loss': 0.0, 'acc': 0.0, 'balanced_acc': 0.5, 'f1': 0.0, 'auc': 0.5,
                'preds': [], 'labels': [], 'probs': np.array([]),
                'logits': np.array([]), 'patient_ids': []}

    probs = np.array(all_probs)
    acc = accuracy_score(all_labels, all_preds)
    bacc = balanced_accuracy_score(all_labels, all_preds) if len(set(all_labels)) >= 2 else 0.5
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    try:
        if len(set(all_labels)) >= 2:
            auc = roc_auc_score(all_labels, probs[:, 1]) if probs.shape[1] == 2 else \
                  roc_auc_score(all_labels, probs, multi_class='ovr', average='weighted')
        else:
            auc = 0.5
    except Exception:
        auc = 0.5

    return {
        'loss': total_loss / max(n_batches, 1),
        'acc': acc,
        'balanced_acc': bacc,
        'f1': f1,
        'auc': auc,
        'preds': all_preds,
        'labels': all_labels,
        'probs': probs,
        'logits': np.array(all_logits),
        'patient_ids': all_patient_ids,
    }


# ─── 分类器 ───────────────────────────────────────────────────────────────────

class DualViewClassifier:
    """双视角 CNN 多任务分类器。"""

    def __init__(self, config: DualViewConfig, device: Optional[str] = None):
        self.cfg = config
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

    def train_one_fold(
        self,
        train_en: List[Dict],
        val_en: List[Dict],
        train_unet: List[Dict],
        val_unet: List[Dict],
        base_path: str,
        task: str = 'pd',
        fold_idx: int = 1,
        checkpoint_path: Optional[str] = None,
        reuse_checkpoint: bool = True,
    ) -> Dict[str, Any]:
        """
        训练单个 fold（单任务）。

        Args:
            train_en / val_en:     pd_dataset_en 的训练/验证患者列表
            train_unet / val_unet: pd_dataset_unet_en 对应的训练/验证患者列表
            base_path:             图像路径基准目录（例如 codes/datasets）
            task:                  训练任务 ('pd' | 'stage' | 'subtype')
            fold_idx:              fold 编号（用于日志）

        Returns:
            包含评价指标、train_history、model_state 的字典
        """
        cfg = self.cfg
        torch.manual_seed(cfg.seed + fold_idx)
        np.random.seed(cfg.seed + fold_idx)

        train_tf, val_tf = _build_transforms(cfg)

        train_ds = DualViewDataset(
            train_en, train_unet, base_path,
            en_transform=train_tf, unet_transform=train_tf,
            region=cfg.region,
        )
        val_ds = DualViewDataset(
            val_en, val_unet, base_path,
            en_transform=val_tf, unet_transform=val_tf,
            region=cfg.region,
        )

        train_loader = DataLoader(
            train_ds, batch_size=cfg.batch_size,
            shuffle=True, num_workers=cfg.num_workers, drop_last=False,
        )
        val_loader = DataLoader(
            val_ds, batch_size=cfg.batch_size,
            shuffle=False, num_workers=cfg.num_workers,
        )

        print(f"  Fold {fold_idx} | 训练样本: {len(train_ds)}, 验证样本: {len(val_ds)}")

        # ── 构建模型（按任务实例化）────────────────────────────────────────────────
        model = DualViewModel(
            task=task,
            backbone=cfg.backbone,
            pretrained=cfg.pretrained,
            shared_backbone=cfg.shared_backbone,
            fusion_dim=cfg.fusion_dim,
            dropout=cfg.dropout,
            classifier_dropout=cfg.classifier_dropout,
        ).to(self.device)

        # ── 差异化学习率 ──────────────────────────────────────────────────────────
        backbone_params, other_params = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if 'backbone_' in name:
                backbone_params.append(p)
            else:
                other_params.append(p)

        param_groups = [
            {'params': backbone_params, 'lr': cfg.lr * cfg.backbone_lr_ratio},
            {'params': other_params, 'lr': cfg.lr},
        ]

        if cfg.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)
        else:
            optimizer = torch.optim.Adam(param_groups, weight_decay=cfg.weight_decay)

        # ── 调度器 ────────────────────────────────────────────────────────────────
        def _lr_lambda(epoch):
            if epoch < cfg.warmup_epochs:
                return (epoch + 1) / max(cfg.warmup_epochs, 1)
            if cfg.lr_scheduler == 'cosine':
                import math
                prog = (epoch - cfg.warmup_epochs) / max(cfg.epochs - cfg.warmup_epochs, 1)
                return 0.5 * (1.0 + math.cos(math.pi * prog))
            return 1.0

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

        # ── 训练循环 ──────────────────────────────────────────────────────────────
        best_metric = -1.0
        best_state = None
        patience_counter = 0
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [], 'val_f1': []
        }
        reused_checkpoint = None
        if checkpoint_path and reuse_checkpoint and os.path.exists(checkpoint_path):
            payload = torch.load(checkpoint_path, map_location='cpu')
            state = payload.get('model_state', payload) if isinstance(payload, dict) else payload
            model.load_state_dict(state)
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            reused_checkpoint = checkpoint_path
            history['reused_checkpoint'] = checkpoint_path
            print(f"  Fold {fold_idx} | 复用已有模型，跳过训练: {checkpoint_path}", flush=True)

        if reused_checkpoint is None:
            for epoch in range(cfg.epochs):
                model.train()
                total_loss = 0.0
                n_batches = 0
                train_preds, train_labels_all = [], []

                for batch in train_loader:
                    en_img, unet_img, pd_l, stage_l, sub_l = batch[:5]
                    # 按任务选取对应标签，应用 mask 过滤无效样本（stage/subtype 中 control=-1）
                    labels_b = {'pd': pd_l, 'stage': stage_l, 'subtype': sub_l}[task]
                    mask_b = labels_b >= 0
                    if not mask_b.any():
                        continue

                    en_m = en_img[mask_b].to(self.device)
                    unet_m = unet_img[mask_b].to(self.device)
                    labels_m = labels_b[mask_b].to(self.device)

                    logits = model(en_m, unet_m)
                    loss = F.cross_entropy(logits, labels_m, label_smoothing=cfg.label_smoothing)
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    total_loss += loss.item()
                    n_batches += 1
                    train_preds.extend(logits.argmax(1).cpu().numpy())
                    train_labels_all.extend(labels_m.cpu().numpy())

                scheduler.step()

                train_loss = total_loss / max(n_batches, 1)
                train_acc = accuracy_score(train_labels_all, train_preds) if train_labels_all else 0.0

                val_metrics = _evaluate(model, val_loader, self.device, task)
                primary_metric = val_metrics['balanced_acc']
                lr_now = optimizer.param_groups[1]['lr']

                history['train_loss'].append(train_loss)
                history['train_acc'].append(train_acc)
                history['val_loss'].append(val_metrics['loss'])
                history['val_acc'].append(val_metrics['balanced_acc'])
                history['val_f1'].append(val_metrics['f1'])

                print(
                    f"  Epoch {epoch + 1:3d}/{cfg.epochs} | "
                    f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                    f"val_loss={val_metrics['loss']:.4f} "
                    f"val_bacc={val_metrics['balanced_acc']:.4f} "
                    f"val_auc={val_metrics['auc']:.4f} | "
                    f"lr={lr_now:.2e}",
                    flush=True,
                )

                if primary_metric > best_metric:
                    best_metric = primary_metric
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= cfg.patience:
                        print(f"  Early stopping at epoch {epoch + 1}", flush=True)
                        break

        history['best_val_acc'] = best_metric

        # ── 最终评估（最佳模型） ──────────────────────────────────────────────────
        if best_state is None:
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        model.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})
        final_train_metrics = _evaluate(model, train_loader, self.device, task)
        final_metrics = _evaluate(model, val_loader, self.device, task)

        print(
            f"  Fold {fold_idx} 完成 | "
            f"train_bacc={final_train_metrics['balanced_acc']:.4f} | "
            f"val_bacc={final_metrics['balanced_acc']:.4f} | "
            f"val_auc={final_metrics['auc']:.4f} | "
            f"val_f1={final_metrics['f1']:.4f}",
            flush=True,
        )
        return {
            'acc': final_metrics['acc'],
            'balanced_acc': final_metrics['balanced_acc'],
            'f1': final_metrics['f1'],
            'auc': final_metrics['auc'],
            'preds': final_metrics['preds'],
            'labels': final_metrics['labels'],
            'probs': final_metrics['probs'],
            'logits': final_metrics['logits'],
            'patient_ids': final_metrics['patient_ids'],
            'train_acc': final_train_metrics['acc'],
            'train_balanced_acc': final_train_metrics['balanced_acc'],
            'model_state': best_state,
            'train_history': history,
        }
