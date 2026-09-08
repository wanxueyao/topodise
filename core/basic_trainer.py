"""
basic_trainer.py - 模型训练模块

模块职责
--------
提供模型训练和评估的核心功能:
- 单 epoch 训练/评估
- 完整 fold 训练流程
- 学习率调度器和早停机制
- 差异化学习率优化器

主要函数
--------
- train_epoch: 训练单个 epoch
- evaluate: 评估模型
- train_fold_single_task: 训练完整的一个 fold
- create_optimizer: 创建支持差异化学习率的优化器
- create_scheduler: 创建学习率调度器

主要类
------
- TrainerConfig: 训练器配置
- EarlyStopping: 早停机制
"""
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from typing import Dict, Any, Optional


# ==================== 配置类 ====================

class TrainerConfig:
    """
    训练器配置类
    
    属性
    ----
    学习率调度:
        LR_SCHEDULER: 调度器类型 ('step'/'cosine'/'none')
        LR_STEP_SIZE: StepLR 步长
        LR_GAMMA: StepLR 衰减因子
    
    早停:
        EARLY_STOPPING: 是否启用
        PATIENCE: 耐心值 (连续无改善的 epoch 数)
    
    差异化学习率:
        BACKBONE_LR_RATIO: backbone 学习率倍率
    """
    LR_SCHEDULER: str = 'step'
    LR_STEP_SIZE: int = 10
    LR_GAMMA: float = 0.5
    EARLY_STOPPING: bool = True
    PATIENCE: int = 10
    BACKBONE_LR_RATIO: float = 0.1


# ==================== 早停类 ====================

class EarlyStopping:
    """
    早停机制
    
    当验证指标连续多个 epoch 无改善时，提前终止训练。
    
    Args:
        patience: 耐心值，连续多少个 epoch 无改善后停止
        min_delta: 最小改善阈值，小于此值视为无改善
        
    使用示例
    --------
    >>> early_stop = EarlyStopping(patience=10)
    >>> for epoch in range(100):
    ...     val_acc = train_and_eval()
    ...     if early_stop(val_acc):
    ...         print("Early stopping!")
    ...         break
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.should_stop = False
    
    def __call__(self, score: float) -> bool:
        """
        检查是否应该早停
        
        Args:
            score: 当前 epoch 的验证指标 (越高越好)
            
        Returns:
            True 表示应该停止，False 表示继续训练
        """
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_score = score
            self.counter = 0
        return self.should_stop


# ==================== 核心训练函数 ====================

def train_epoch(model: nn.Module, dataloader, criterion, optimizer, 
                device: str, label_key: str) -> tuple:
    """
    训练单个 epoch
    
    遍历整个数据集一次，更新模型参数。
    
    Args:
        model: 神经网络模型
        dataloader: 数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 计算设备 ('cuda'/'cpu')
        label_key: 标签键名 ('pd'/'stage'/'subtype')
        
    Returns:
        (average_loss, accuracy): 平均损失和准确率
    """
    model.train()
    total_loss, all_preds, all_labels = 0.0, [], []
    valid_batches = 0
    
    for batch in dataloader:
        images, labels, mask = _parse_batch(batch, label_key, device)
        if mask.sum() == 0:
            continue
        
        valid_batches += 1
        outputs = _forward(model, images, mask)
        loss = criterion(outputs, labels[mask])
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        all_preds.extend(outputs.argmax(1).cpu().numpy())
        all_labels.extend(labels[mask].cpu().numpy())
    
    if valid_batches == 0:
        return 0.0, 0.0
    
    return total_loss / valid_batches, accuracy_score(all_labels, all_preds)


def evaluate(model: nn.Module, dataloader, criterion, 
             device: str, label_key: str) -> tuple:
    """
    评估模型
    
    在验证集/测试集上评估模型性能，不更新参数。
    
    Args:
        model: 神经网络模型
        dataloader: 数据加载器
        criterion: 损失函数
        device: 计算设备
        label_key: 标签键名
        
    Returns:
        (average_loss, result_dict): 平均损失和详细结果字典
        
    result_dict 包含:
        - loss, acc, precision, recall, f1: 评估指标
        - preds, labels, probs: 预测结果和真实标签
    """
    model.eval()
    total_loss, all_preds, all_labels, all_probs, all_logits = 0.0, [], [], [], []
    valid_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            images, labels, mask = _parse_batch(batch, label_key, device)
            if mask.sum() == 0:
                continue
            
            valid_batches += 1
            outputs = _forward(model, images, mask)
            total_loss += criterion(outputs, labels[mask]).item()
            
            probs = torch.softmax(outputs, dim=1)
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(labels[mask].cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_logits.extend(outputs.cpu().numpy())
    
    if valid_batches == 0:
        return 0.0, _empty_result()
    
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    
    return total_loss / valid_batches, {
        'loss': total_loss / valid_batches,
        'acc': acc, 'precision': precision, 'recall': recall, 'f1': f1,
        'preds': all_preds, 'labels': all_labels,
        'probs': np.array(all_probs), 'logits': np.array(all_logits)
    }


def train_fold_single_task(
    train_loader, val_loader, model: nn.Module, criterion, optimizer,
    num_epochs: int, device: str, task_name: str, fold_idx: int, label_key: str,
    scheduler=None, early_stopping: Optional[EarlyStopping] = None
) -> Dict[str, Any]:
    """
    训练一个完整的 fold
    
    包含完整的训练循环：多个 epoch 的训练、验证、早停检查、最佳模型保存。
    
    Args:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        model: 神经网络模型
        criterion: 损失函数
        optimizer: 优化器
        num_epochs: 最大训练轮数
        device: 计算设备
        task_name: 任务名称 (用于日志)
        fold_idx: fold 索引 (用于日志)
        label_key: 标签键名
        scheduler: 学习率调度器 (可选)
        early_stopping: 早停器 (可选)
        
    Returns:
        训练历史字典:
        - train_loss, train_acc: 训练曲线
        - val_loss, val_acc, val_f1: 验证曲线
        - best_val_acc: 最佳验证准确率
    """
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_f1': []
    }
    best_acc, best_state = 0.0, None
    
    for epoch in range(num_epochs):
        # 训练一个 epoch
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, label_key
        )
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        # 验证
        val_loss, val_result = evaluate(model, val_loader, criterion, device, label_key)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_result['acc'])
        history['val_f1'].append(val_result['f1'])
        
        # 保存最佳模型
        if val_result['acc'] > best_acc:
            best_acc = val_result['acc']
            best_state = model.state_dict().copy()
        
        # 学习率调度
        if scheduler:
            scheduler.step()
        
        # 早停检查
        if early_stopping and early_stopping(val_result['acc']):
            print(f"  Early stopping at epoch {epoch + 1}")
            break
        
        # 日志输出 (每个 epoch 均打印详细进度)
        lr = optimizer.param_groups[0]['lr']
        print(f"  Epoch {epoch+1:3d}/{num_epochs} | "
              f"Train: {train_loss:.4f}/{train_acc:.4f} | "
              f"Val: {val_loss:.4f}/{val_result['acc']:.4f} | LR: {lr:.6f}")
    
    # 恢复最佳模型
    if best_state:
        model.load_state_dict(best_state)
    
    history['best_val_acc'] = best_acc
    return history


# ==================== 辅助函数 ====================

def _parse_batch(batch, label_key: str, device: str) -> tuple:
    """
    解析数据 batch (内部函数)
    
    处理单流和双流两种数据格式，提取图像、标签和有效样本掩码。
    
    Args:
        batch: DataLoader 返回的 batch
        label_key: 标签键名
        device: 计算设备
        
    Returns:
        (images, labels, valid_mask): 元组
        - images: 单流为 Tensor，双流为 (central, whorl) 元组
        - valid_mask: 有效样本的布尔掩码
    """
    is_dual = isinstance(batch[0], (tuple, list))
    
    if is_dual:
        (central, whorl), pd, stage, subtype = batch
        images = (central.to(device), whorl.to(device))
    else:
        img, pd, stage, subtype = batch
        images = img.to(device)
    
    if label_key == 'pd_subtype_tdrest':
        labels = torch.where(subtype == 0, 0, torch.where(subtype > 0, 1, -1))
    elif label_key == 'pd_subtype_pigd_mixed':
        labels = torch.where(subtype == 1, 0, torch.where(subtype == 2, 1, -1))
    else:
        label_map = {'pd': pd, 'stage': stage, 'subtype': subtype}
        labels = label_map[label_key]
    labels = labels.to(device)
    
    # stage/subtype 可能有 -1 表示无效样本
    if label_key == 'pd':
        mask = torch.ones(len(labels), dtype=torch.bool, device=device)
    else:
        mask = (labels != -1)
    
    return images, labels, mask


def _forward(model: nn.Module, images, mask: torch.Tensor) -> torch.Tensor:
    """
    前向传播 (内部函数)
    
    根据单流/双流模式调用模型，仅对有效样本进行推理。
    """
    if isinstance(images, tuple):
        return model(images[0][mask], images[1][mask])
    return model(images[mask])


def _empty_result() -> Dict[str, Any]:
    """返回空评估结果 (内部函数)"""
    return {
        'loss': 0.0, 'acc': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
        'preds': [], 'labels': [], 'probs': [], 'logits': []
    }


# ==================== 优化器工厂 ====================

def create_optimizer(model: nn.Module, base_lr: float, weight_decay: float,
                     backbone_lr_ratio: float = 0.1) -> torch.optim.Optimizer:
    """
    创建优化器 (支持差异化学习率)
    
    迁移学习最佳实践: backbone 使用较小学习率，classifier 使用正常学习率。
    
    Args:
        model: 神经网络模型
        base_lr: 基础学习率 (classifier 使用)
        weight_decay: L2 正则化系数
        backbone_lr_ratio: backbone 学习率倍率
            - backbone_lr = base_lr * backbone_lr_ratio
            - 例如 ratio=0.1，base_lr=0.001 → backbone_lr=0.0001
            
    Returns:
        Adam 优化器
    """
    backbone_params = []
    classifier_params = []
    
    for name, param in model.named_parameters():
        if 'backbone' in name:
            backbone_params.append(param)
        else:
            classifier_params.append(param)
    
    param_groups = [
        {'params': backbone_params, 'lr': base_lr * backbone_lr_ratio},
        {'params': classifier_params, 'lr': base_lr}
    ]
    
    return torch.optim.Adam(param_groups, weight_decay=weight_decay)


def create_scheduler(optimizer: torch.optim.Optimizer, scheduler_type: str,
                     num_epochs: int, step_size: int = 10, gamma: float = 0.5):
    """
    创建学习率调度器
    
    Args:
        optimizer: 优化器
        scheduler_type: 调度器类型
            - 'step': StepLR，每 step_size 个 epoch 衰减
            - 'cosine': CosineAnnealingLR，余弦退火
            - 'none' 或其他: 不使用调度器
        num_epochs: 总训练轮数 (cosine 调度器需要)
        step_size: StepLR 步长
        gamma: StepLR 衰减因子
        
    Returns:
        调度器实例，或 None (不使用调度器时)
    """
    if scheduler_type == 'step':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    return None
