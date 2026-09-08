"""
classifier.py - 高层 GNN 分类器

对标 core.basic_classifier.PDClassifier，提供:
  - GNNClassifierConfig: 统一超参数配置
  - GNNClassifier:       封装模型构建、优化器、调度器、单 fold 训练
"""

import logging
import os
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import torch
from torch.utils.data import DataLoader

from .models import GNNMultiTaskModel
from .trainer import (
    GNNTrainerConfig,
    train_epoch,
    evaluate,
    batch_to_device,
)
from .graph_dataset import NerveGraphDataset, collate_fn

logger = logging.getLogger(__name__)


# ─── 配置 ─────────────────────────────────────────────────────────────────────

@dataclass
class GNNClassifierConfig:
    """GNN 分类器的全部超参数。"""
    # 模型结构
    gnn_type: str = 'gat'           # 'gat' | 'graphsage'
    hidden_dim: int = 64
    num_layers: int = 2
    num_heads: int = 4              # GAT 专用
    shared_encoder: bool = False    # True = central/whorl 共享 GNN 权重

    # 训练
    lr: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = 'adam'         # 'adam' | 'adamw'
    lr_scheduler: str = 'cosine'    # 'cosine' | 'step' | 'none'
    warmup_epochs: int = 5
    epochs: int = 50
    patience: int = 10
    batch_size: int = 16

    # 正则化
    dropout: float = 0.2
    classifier_dropout: float = 0.3
    num_subtype_classes: int = 3

    # 多任务损失权重
    pd_loss_weight: float = 1.0
    stage_loss_weight: float = 0.5
    subtype_loss_weight: float = 0.5
    label_smoothing: float = 0.0

    # 模型选择标准
    primary_task: str = 'pd'        # 'pd' | 'stage' | 'subtype'

    # 数据
    graph_cache_dir: Optional[str] = None  # None = 不缓存图
    num_workers: int = 0


# ─── 分类器 ───────────────────────────────────────────────────────────────────

class GNNClassifier:
    """
    GNN 分类器封装。

    封装模型构建、优化器调度、单折训练循环，接口对标 PDClassifier。

    Args:
        config: GNNClassifierConfig
        device: 运算设备，None = 自动检测 (cuda > mps > cpu)
    """

    def __init__(
        self,
        config: GNNClassifierConfig,
        device: Optional[torch.device] = None,
    ):
        self.config = config
        self.device = device or self._auto_device()
        self.model: Optional[GNNMultiTaskModel] = None

    # ── 设备检测 ────────────────────────────────────────────────────────────────

    @staticmethod
    def _auto_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device('cuda')
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')

    # ── 模型构建 ────────────────────────────────────────────────────────────────

    def build_model(self) -> GNNMultiTaskModel:
        """构建并返回 GNNMultiTaskModel。"""
        cfg = self.config
        model = GNNMultiTaskModel(
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            gnn_type=cfg.gnn_type,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
            classifier_dropout=cfg.classifier_dropout,
            shared_encoder=cfg.shared_encoder,
            num_subtype_classes=cfg.num_subtype_classes,
        )
        return model.to(self.device)

    def _build_optimizer(self, model: GNNMultiTaskModel) -> torch.optim.Optimizer:
        cfg = self.config
        if cfg.optimizer == 'adamw':
            return torch.optim.AdamW(
                model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
            )
        return torch.optim.Adam(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )

    def _build_scheduler(self, optimizer, num_epochs: int):
        cfg = self.config
        if cfg.lr_scheduler == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(num_epochs - cfg.warmup_epochs, 1),
            )
        if cfg.lr_scheduler == 'step':
            return torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=max(num_epochs // 3, 1), gamma=0.5
            )
        return None

    # ── 单 Fold 训练 ────────────────────────────────────────────────────────────

    def train_one_fold(
        self,
        train_patients: List[Dict],
        val_patients: List[Dict],
        base_path: str,
        fold_idx: int = 0,
        checkpoint_path: Optional[str] = None,
        reuse_checkpoint: bool = True,
    ) -> Dict[str, Any]:
        """
        训练单个 fold，返回最优验证指标。

        Args:
            train_patients: 训练集患者元数据列表
            val_patients:   验证集患者元数据列表
            base_path:      图像路径根目录
            fold_idx:       当前 fold 编号（仅用于日志）

        Returns:
            dict 包含各任务最优验证指标
        """
        cfg = self.config

        trainer_cfg = GNNTrainerConfig(
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            optimizer=cfg.optimizer,
            lr_scheduler=cfg.lr_scheduler,
            warmup_epochs=cfg.warmup_epochs,
            epochs=cfg.epochs,
            patience=cfg.patience,
            pd_loss_weight=cfg.pd_loss_weight,
            stage_loss_weight=cfg.stage_loss_weight,
            subtype_loss_weight=cfg.subtype_loss_weight,
            label_smoothing=cfg.label_smoothing,
        )

        # ── 数据集 ──────────────────────────────────────────────────────────────
        train_ds = NerveGraphDataset(
            train_patients, base_path, cache_dir=cfg.graph_cache_dir
        )
        val_ds = NerveGraphDataset(
            val_patients, base_path, cache_dir=cfg.graph_cache_dir
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=cfg.num_workers,
            drop_last=(len(train_ds) > cfg.batch_size),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=cfg.num_workers,
        )

        # ── 模型 & 优化器 ────────────────────────────────────────────────────────
        model = self.build_model()
        optimizer = self._build_optimizer(model)
        scheduler = self._build_scheduler(optimizer, cfg.epochs)

        best_metric = -1.0
        best_result: Dict[str, Any] = {}
        patience_counter = 0
        reused_checkpoint = None

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(
            f"[Fold {fold_idx}] 开始训练 | "
            f"train={len(train_ds)} val={len(val_ds)} | "
            f"device={self.device} | params={n_params:,}"
        )

        if checkpoint_path and reuse_checkpoint and os.path.exists(checkpoint_path):
            payload = torch.load(checkpoint_path, map_location='cpu')
            state = payload.get('model_state', payload) if isinstance(payload, dict) else payload
            model.load_state_dict(state)
            best_result = evaluate(model, val_loader, self.device, trainer_cfg, cfg.primary_task)
            best_metric = best_result['primary_metric']
            reused_checkpoint = checkpoint_path
            logger.info(f"[Fold {fold_idx}] 复用已有模型，跳过训练: {checkpoint_path}")

        # ── 训练循环 ─────────────────────────────────────────────────────────────
        if reused_checkpoint is None:
            for epoch in range(1, cfg.epochs + 1):
                # Warmup: 线性增加 LR
                if 0 < cfg.warmup_epochs and epoch <= cfg.warmup_epochs:
                    for pg in optimizer.param_groups:
                        pg['lr'] = cfg.lr * epoch / cfg.warmup_epochs

                t0 = time.time()
                train_loss = train_epoch(model, train_loader, optimizer, self.device, trainer_cfg)
                val_result = evaluate(model, val_loader, self.device, trainer_cfg, cfg.primary_task)
                t1 = time.time()

                if scheduler is not None and epoch > cfg.warmup_epochs:
                    scheduler.step()

                primary = val_result['primary_metric']

                if epoch % 10 == 0 or epoch == 1 or epoch == cfg.epochs:
                    logger.info(
                        f"  ep {epoch:3d}/{cfg.epochs} | "
                        f"train_loss {train_loss:.4f} | val_loss {val_result['loss']:.4f} | "
                        f"pd_bacc {val_result['pd']['balanced_acc']:.3f} | "
                        f"stage_bacc {val_result['stage']['balanced_acc']:.3f} | "
                        f"sub_bacc {val_result['subtype']['balanced_acc']:.3f} | "
                        f"primary {primary:.3f} | {t1 - t0:.1f}s"
                    )

                if primary > best_metric:
                    best_metric = primary
                    best_result = val_result
                    patience_counter = 0
                    if checkpoint_path:
                        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                        torch.save({'model_state': model.state_dict(), 'fold': fold_idx}, checkpoint_path)
                else:
                    patience_counter += 1
                    if patience_counter >= cfg.patience:
                        logger.info(f"  [Fold {fold_idx}] Early stopping at epoch {epoch}")
                        break

        logger.info(
            f"[Fold {fold_idx}] 完成 | best_primary={best_metric:.4f} | "
            f"pd: bacc={best_result.get('pd', {}).get('balanced_acc', 0):.3f} "
            f"f1={best_result.get('pd', {}).get('f1', 0):.3f} "
            f"auc={best_result.get('pd', {}).get('auc', 0):.3f}"
        )

        self.model = model   # 保留最后 fold 的模型引用

        pd_m    = best_result.get('pd',      {})
        stage_m = best_result.get('stage',   {})
        sub_m   = best_result.get('subtype', {})

        return {
            'fold': fold_idx,
            'best_primary_metric': best_metric,
            'pd_acc':           pd_m.get('acc',           0.0),
            'pd_balanced_acc': pd_m.get('balanced_acc', 0.0),
            'pd_f1':           pd_m.get('f1',           0.0),
            'pd_auc':          pd_m.get('auc',          0.0),
            'stage_acc': stage_m.get('acc',           0.0),
            'stage_balanced_acc': stage_m.get('balanced_acc', 0.0),
            'stage_f1':           stage_m.get('f1',           0.0),
            'stage_auc':          stage_m.get('auc',          0.0),
            'subtype_acc': sub_m.get('acc',           0.0),
            'subtype_balanced_acc': sub_m.get('balanced_acc', 0.0),
            'subtype_f1':           sub_m.get('f1',           0.0),
            'subtype_auc':          sub_m.get('auc',          0.0),
            'predictions':          best_result.get('predictions', {}),
            'checkpoint_path':      checkpoint_path or '',
            'reused_checkpoint':    bool(reused_checkpoint),
        }
