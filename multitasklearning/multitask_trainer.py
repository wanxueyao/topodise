"""
多任务学习训练器
===============

支持:
- 多任务学习训练
- K-fold 交叉验证
- 早停
- 混合精度训练
- 模型检查点
- 详细日志记录
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
import logging
import json
import os
from datetime import datetime
from pathlib import Path
import numpy as np

from .config import MTLConfig
from .unet_multitask import UNetMultiTask
from .multitask_utils import (
    compute_dice_score, compute_iou_score, 
    compute_classification_metrics, MultiTaskLoss, FrangiFilter
)

logger = logging.getLogger(__name__)


class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience: int = 20, min_delta: float = 0.0, mode: str = 'min'):
        """
        Parameters
        ----------
        patience : int
            容忍无改进的轮数
        min_delta : float
            最小改进阈值
        mode : str
            'min' 或 'max'
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, current_score: float) -> bool:
        """
        Returns
        -------
        bool
            是否应该停止训练
        """
        if self.best_score is None:
            self.best_score = current_score
        elif self._is_improvement(current_score):
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def _is_improvement(self, current_score: float) -> bool:
        if self.mode == 'min':
            return current_score < self.best_score - self.min_delta
        else:
            return current_score > self.best_score + self.min_delta


class MultiTaskTrainer:
    """多任务学习训练器"""
    
    def __init__(self, config: MTLConfig, model: UNetMultiTask,
                 train_loader: DataLoader, val_loader: DataLoader = None,
                 test_loader: DataLoader = None):
        """
        Parameters
        ----------
        config : MTLConfig
            配置对象
        model : UNetMultiTask
            模型
        train_loader : DataLoader
            训练数据加载器
        val_loader : DataLoader, optional
            验证数据加载器
        test_loader : DataLoader, optional
            测试数据加载器
        """
        self.config = config
        self.model = model.to(config.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        # 优化器
        self.optimizer = self._build_optimizer()
        
        # 学习率调度
        self.scheduler = self._build_scheduler()
        
        # 混合精度训练
        self.scaler = GradScaler() if config.mixed_precision else None
        
        # 损失函数
        self.criterion = self._build_criterion()
        
        # 早停
        self.early_stopping = EarlyStopping(patience=config.early_stopping_patience)
        
        # 统计
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': [],
        }
        
        # 输出目录
        self.output_dir = self._setup_output_dir()
        
        logger.info("训练器初始化完成")
    
    def _build_optimizer(self) -> optim.Optimizer:
        """构建优化器"""
        if self.config.optimizer == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"未知的优化器: {self.config.optimizer}")
    
    def _build_scheduler(self) -> optim.lr_scheduler.LRScheduler:
        """构建学习率调度"""
        if self.config.lr_scheduler == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs,
                eta_min=1e-7
            )
        elif self.config.lr_scheduler == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        else:
            raise ValueError(f"未知的调度方式: {self.config.lr_scheduler}")
    
    def _build_criterion(self) -> MultiTaskLoss:
        """构建多任务损失函数"""
        weights = {
            'segmentation': self.config.loss_weight_segmentation,
            'mae': self.config.loss_weight_mae,
            'pseudo': self.config.loss_weight_pseudo,
            'pd_classification': self.config.loss_weight_pd_classification,
            'stage': self.config.loss_weight_stage,
            'subtype': self.config.loss_weight_subtype,
        }
        return MultiTaskLoss(weights)
    
    def _setup_output_dir(self) -> Path:
        """设置输出目录"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path(self.config.checkpoint_dir) / f"mtl_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存配置
        config_path = output_dir / 'config.json'
        self._save_config(config_path)
        
        return output_dir
    
    def _save_config(self, path: Path):
        """保存配置到JSON"""
        config_dict = {
            'encoder_type': self.config.encoder_type,
            'classification_mode': self.config.classification_mode,
            'batch_size': self.config.batch_size,
            'num_epochs': self.config.num_epochs,
            'learning_rate': self.config.learning_rate,
            'loss_weights': self.config.normalize_weights(),
        }
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        metrics = {
            'dice': [],
            'iou': [],
            'pd_acc': [],
        }
        
        for batch_idx, batch in enumerate(self.train_loader):
            # 获取数据
            images = batch['image'].to(self.config.device)
            
            # 根据任务类型获取标签
            if batch.get('task') == 'segmentation':
                masks = batch['mask'].to(self.config.device)
                pd_labels = None
                stage_labels = None
                subtype_labels = None
            else:  # classification
                masks = None
                pd_labels = batch['pd_label'].to(self.config.device)
                stage_labels = batch['stage_label'].to(self.config.device)
                subtype_labels = batch['subtype_label'].to(self.config.device)
            
            self.optimizer.zero_grad()
            
            # 前向传播
            with autocast(enabled=self.config.mixed_precision):
                outputs = self.model(
                    images,
                    segmentation_targets=masks,
                    pd_labels=pd_labels,
                    stage_labels=stage_labels,
                    subtype_labels=subtype_labels,
                    use_mae=True,
                    training=True
                )
                
                # 计算总损失
                loss = self._compute_total_loss(outputs['losses'])
            
            # 反向传播
            if self.config.mixed_precision:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # 计算指标
            if batch.get('task') == 'segmentation' and masks is not None:
                dice = compute_dice_score(outputs['segmentation'], masks)
                iou = compute_iou_score(outputs['segmentation'], masks)
                metrics['dice'].append(dice)
                metrics['iou'].append(iou)
            elif batch.get('task') == 'classification':
                cls_metrics = compute_classification_metrics(
                    outputs['pd_logits'], pd_labels, 2
                )
                metrics['pd_acc'].append(cls_metrics['accuracy'])
            
            # 日志
            if (batch_idx + 1) % self.config.log_interval == 0:
                logger.info(
                    f"批次 {batch_idx + 1}/{len(self.train_loader)}, "
                    f"损失: {loss.item():.4f}"
                )
        
        avg_loss = total_loss / num_batches
        
        result = {'loss': avg_loss}
        for key, vals in metrics.items():
            if vals:
                result[f'avg_{key}'] = np.mean(vals)
        
        return result
    
    def val_epoch(self) -> Dict[str, float]:
        """验证一个epoch"""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        metrics = {
            'dice': [],
            'iou': [],
            'pd_acc': [],
        }
        
        with torch.no_grad():
            for batch in self.val_loader:
                images = batch['image'].to(self.config.device)
                
                if batch.get('task') == 'segmentation':
                    masks = batch['mask'].to(self.config.device)
                    pd_labels = None
                    stage_labels = None
                    subtype_labels = None
                else:
                    masks = None
                    pd_labels = batch['pd_label'].to(self.config.device)
                    stage_labels = batch['stage_label'].to(self.config.device)
                    subtype_labels = batch['subtype_label'].to(self.config.device)
                
                outputs = self.model(
                    images,
                    segmentation_targets=masks,
                    pd_labels=pd_labels,
                    stage_labels=stage_labels,
                    subtype_labels=subtype_labels,
                    use_mae=False,
                    training=False
                )
                
                loss = self._compute_total_loss(outputs['losses'])
                total_loss += loss.item()
                num_batches += 1
                
                # 计算指标
                if batch.get('task') == 'segmentation' and masks is not None:
                    dice = compute_dice_score(outputs['segmentation'], masks)
                    iou = compute_iou_score(outputs['segmentation'], masks)
                    metrics['dice'].append(dice)
                    metrics['iou'].append(iou)
                elif batch.get('task') == 'classification':
                    cls_metrics = compute_classification_metrics(
                        outputs['pd_logits'], pd_labels, 2
                    )
                    metrics['pd_acc'].append(cls_metrics['accuracy'])
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        result = {'loss': avg_loss}
        for key, vals in metrics.items():
            if vals:
                result[f'avg_{key}'] = np.mean(vals)
        
        return result
    
    def _compute_total_loss(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算加权总损失"""
        total = torch.tensor(0.0, device=next(self.model.parameters()).device)
        
        for loss_type, weight in self.config.normalize_weights().items():
            for key, value in losses.items():
                if loss_type in key and isinstance(value, torch.Tensor):
                    total = total + weight * value
                    break
        
        return total
    
    def train(self):
        """训练模型"""
        logger.info("开始训练...")
        
        for epoch in range(self.config.num_epochs):
            logger.info(f"\n=== Epoch {epoch + 1}/{self.config.num_epochs} ===")
            
            # 训练
            train_metrics = self.train_epoch()
            self.history['train_loss'].append(train_metrics['loss'])
            
            logger.info(f"训练损失: {train_metrics['loss']:.4f}")
            for key, val in train_metrics.items():
                if key != 'loss':
                    logger.info(f"  {key}: {val:.4f}")
            
            # 验证
            if self.val_loader is not None and (epoch + 1) % self.config.validation_interval == 0:
                val_metrics = self.val_epoch()
                self.history['val_loss'].append(val_metrics['loss'])
                
                logger.info(f"验证损失: {val_metrics['loss']:.4f}")
                for key, val in val_metrics.items():
                    if key != 'loss':
                        logger.info(f"  {key}: {val:.4f}")
                
                # 早停
                if self.early_stopping(val_metrics['loss']):
                    logger.info("触发早停")
                    break
            
            # 更新学习率
            self.scheduler.step()
            
            # 保存检查点
            if (epoch + 1) % self.config.save_interval == 0:
                self.save_checkpoint(epoch + 1)
        
        logger.info("训练完成")
        self.save_history()
    
    def save_checkpoint(self, epoch: int):
        """保存模型检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'config': self.config.__dict__,
        }
        
        path = self.output_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, path)
        logger.info(f"检查点保存到: {path}")
    
    def save_history(self):
        """保存训练历史"""
        path = self.output_dir / 'history.json'
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"历史保存到: {path}")
    
    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.config.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state'])
        logger.info(f"检查点加载从: {path}")


if __name__ == '__main__':
    # 测试训练器初始化
    from .unet_multitask import UNetMultiTask
    
    config = MTLConfig()
    model = UNetMultiTask()
    
    # 创建虚拟数据加载器
    from torch.utils.data import TensorDataset
    
    images = torch.randn(10, 3, 256, 256)
    masks = torch.randint(0, 2, (10, 1, 256, 256)).float()
    dataset = TensorDataset(images, masks)
    loader = DataLoader(dataset, batch_size=2)
    
    trainer = MultiTaskTrainer(config, model, loader, loader)
    print("训练器初始化成功")
