"""Shared classification losses, optimizers, scheduling and training loops."""
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from typing import Dict, Any, Optional

from core.classification_losses import OrdinalRegressionLoss


# ==================== 配置类 ====================



# ==================== 早停 ====================

class EarlyStopping:
    """
    早停机制

    当验证准确率连续 patience 个 epoch 无改善时，停止训练。

    Args:
        patience: 连续多少个 epoch 无改善后停止
        min_delta: 最小改善阈值
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, score: float) -> bool:
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


# ==================== 损失函数工厂 ====================

def create_classification_criterion(task: str, num_classes: int,
                           loss_type: str = 'ce',
                           label_smooth_eps: float = 0.1,
                           device: str = 'cpu') -> nn.Module:
    """
    根据配置创建损失函数

    Args:
        task: 任务名称 ('pd' / 'stage' / 'subtype')
        num_classes: 类别数
        loss_type: 损失类型
            - 'ce': 标准 CrossEntropyLoss
            - 'label_smooth': Label Smoothing CrossEntropy
            - 'ordinal': OrdinalRegressionLoss（仅 stage 任务，强制转为 ordinal）
        label_smooth_eps: label smooth 的 epsilon
        device: 计算设备

    Returns:
        损失函数实例

    注意:
        - ordinal 模式要求模型输出维度为 num_classes-1
        - pd / subtype 任务的 ordinal 会自动降级为 'label_smooth'
    """
    # ordinal 仅对 stage 任务有意义（有序关系）
    if loss_type == 'ordinal' and task != 'stage':
        loss_type = 'label_smooth'

    if loss_type == 'ordinal':
        return OrdinalRegressionLoss(num_classes=num_classes).to(device)
    elif loss_type == 'label_smooth':
        return nn.CrossEntropyLoss(label_smoothing=label_smooth_eps).to(device)
    else:
        return nn.CrossEntropyLoss().to(device)


# ==================== 优化器 ====================

def create_classification_optimizer(model: nn.Module, base_lr: float,
                           weight_decay: float = 1e-4,
                           backbone_lr_ratio: float = 0.1,
                           optimizer_type: str = 'adamw') -> torch.optim.Optimizer:
    """
    创建支持差异化学习率的优化器

    backbone 使用较小学习率，分类头使用正常学习率。
    AdamW 相比 Adam 有更好的权重衰减效果，推荐用于 ViT 类模型。

    Args:
        model: 神经网络模型
        base_lr: 分类头学习率
        weight_decay: L2 正则化系数
        backbone_lr_ratio: backbone 学习率倍率
        optimizer_type: 优化器类型 ('adam' / 'adamw')

    Returns:
        优化器实例
    """
    backbone_params = []
    classifier_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # timm 模型的 backbone 参数包含 'backbone' 关键字
        if 'backbone' in name:
            backbone_params.append(param)
        else:
            classifier_params.append(param)

    param_groups = []
    if backbone_params:
        param_groups.append({'params': backbone_params, 'lr': base_lr * backbone_lr_ratio})
    if classifier_params:
        param_groups.append({'params': classifier_params, 'lr': base_lr})

    if not param_groups:
        # fallback
        param_groups = [{'params': model.parameters(), 'lr': base_lr}]

    if optimizer_type.lower() == 'adamw':
        return torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    else:
        return torch.optim.Adam(param_groups, weight_decay=weight_decay)


# ==================== 学习率调度器 ====================

def create_classification_scheduler(optimizer: torch.optim.Optimizer,
                           scheduler_type: str,
                           num_epochs: int,
                           step_size: int = 10,
                           gamma: float = 0.5):
    """
    创建学习率调度器

    Args:
        optimizer: 优化器
        scheduler_type: 调度器类型
            - 'step': StepLR
            - 'cosine': CosineAnnealingLR（推荐轻量模型）
            - 'warmup_cosine': 含 warmup 的余弦退火（推荐 ViT）
            - 'none': 不使用调度器
        num_epochs: 总训练轮数
        step_size: StepLR 步长
        gamma: StepLR 衰减因子

    Returns:
        调度器实例或 None
    """
    if scheduler_type == 'step':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    elif scheduler_type == 'warmup_cosine':
        # 线性 warmup (前 5 epochs) + 余弦退火
        warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=5)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, num_epochs - 5), eta_min=1e-6)
        return torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[5])
    else:
        return None





# ==================== 核心训练函数 ====================

def _parse_batch(batch, label_key: str, device: str) -> tuple:
    """
    解析 batch 数据 (内部函数)

    处理单流和双流两种格式。

    Returns:
        (images, labels, valid_mask)
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

    # stage/subtype 存在 -1 无效标签
    if label_key == 'pd':
        mask = torch.ones(len(labels), dtype=torch.bool, device=device)
    else:
        mask = (labels != -1)

    return images, labels, mask


def _forward(model: nn.Module, images, mask: torch.Tensor) -> torch.Tensor:
    """
    前向传播 (内部函数)，仅对有效样本推理。
    """
    if isinstance(images, tuple):
        return model(images[0][mask], images[1][mask])
    return model(images[mask])


def _is_ordinal_criterion(criterion) -> bool:
    """判断是否使用序数回归损失"""
    return isinstance(criterion, OrdinalRegressionLoss)


def _get_preds_and_probs(model_output: torch.Tensor,
                         use_ordinal: bool) -> tuple:
    """
    从模型输出获取预测类别和概率 (内部函数)

    Args:
        model_output: 模型输出 logits
        use_ordinal: 是否为序数回归输出

    Returns:
        (preds, probs): numpy 数组
    """
    if use_ordinal:
        probs = OrdinalRegressionLoss.logits_to_probs(model_output)
        preds = probs.argmax(dim=1)
    else:
        probs = torch.softmax(model_output, dim=1)
        preds = model_output.argmax(dim=1)

    return preds.cpu().numpy(), probs.cpu().numpy()


def classification_train_epoch(model: nn.Module, dataloader, criterion,
                      optimizer, device: str, label_key: str) -> tuple:
    """
    训练单个 epoch (支持序数回归和标准 CE)

    Args:
        model: 神经网络模型
        dataloader: 训练数据加载器
        criterion: 损失函数 (CE / LabelSmooth / Ordinal)
        optimizer: 优化器
        device: 计算设备
        label_key: 标签键名 ('pd'/'stage'/'subtype')

    Returns:
        (avg_loss, accuracy): 平均损失和准确率
    """
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []
    valid_batches = 0
    use_ordinal = _is_ordinal_criterion(criterion)

    for batch in dataloader:
        images, labels, mask = _parse_batch(batch, label_key, device)
        valid_count = int(mask.sum().item())
        if valid_count == 0:
            continue
        # 训练态 BatchNorm 需要每个通道至少 2 个样本，
        # 对 mask 过滤后仅剩 1 个有效样本的 batch 直接跳过。
        if valid_count < 2:
            continue

        valid_batches += 1
        outputs = _forward(model, images, mask)
        loss = criterion(outputs, labels[mask])

        optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪，防止 ViT 模型训练不稳定
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        preds, _ = _get_preds_and_probs(outputs.detach(), use_ordinal)
        all_preds.extend(preds)
        all_labels.extend(labels[mask].cpu().numpy())

    if valid_batches == 0:
        return 0.0, 0.0

    return total_loss / valid_batches, accuracy_score(all_labels, all_preds)


def classification_evaluate(model: nn.Module, dataloader, criterion,
                   device: str, label_key: str) -> tuple:
    """
    评估模型 (支持序数回归和标准 CE)

    Args:
        model: 神经网络模型
        dataloader: 验证数据加载器
        criterion: 损失函数
        device: 计算设备
        label_key: 标签键名

    Returns:
        (avg_loss, result_dict)

    result_dict 包含:
        loss, acc, precision, recall, f1, preds, labels, probs
    """
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_probs, all_logits = [], [], [], []
    valid_batches = 0
    use_ordinal = _is_ordinal_criterion(criterion)

    with torch.no_grad():
        for batch in dataloader:
            images, labels, mask = _parse_batch(batch, label_key, device)
            if mask.sum() == 0:
                continue

            valid_batches += 1
            outputs = _forward(model, images, mask)
            total_loss += criterion(outputs, labels[mask]).item()

            preds, probs = _get_preds_and_probs(outputs, use_ordinal)
            all_preds.extend(preds)
            all_labels.extend(labels[mask].cpu().numpy())
            all_probs.extend(probs)
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


def _empty_result() -> Dict[str, Any]:
    """返回空评估结果"""
    return {
        'loss': 0.0, 'acc': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
        'preds': [], 'labels': [], 'probs': [], 'logits': []
    }


def classification_train_fold(
    train_loader, val_loader, model: nn.Module, criterion, optimizer,
    num_epochs: int, device: str, task_name: str, fold_idx: int, label_key: str,
    scheduler=None, early_stopping: Optional[EarlyStopping] = None,
) -> Dict[str, Any]:
    """
    训练一个完整的 fold (轻量化版本)

    支持序数回归损失，集成梯度裁剪，与 basic_trainer.train_fold_single_task 同接口。

    Args:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        model: 神经网络模型
        criterion: 损失函数
        optimizer: 优化器
        num_epochs: 最大训练轮数
        device: 计算设备
        task_name: 任务名称 (日志显示)
        fold_idx: fold 索引 (日志显示)
        label_key: 标签键名
        scheduler: 学习率调度器 (可选)
        early_stopping: 早停器 (可选)

    Returns:
        训练历史字典:
            train_loss, train_acc, val_loss, val_acc, val_f1, best_val_acc
    """
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_f1': []
    }
    best_acc, best_state = 0.0, None

    for epoch in range(num_epochs):

        train_loss, train_acc = classification_train_epoch(
            model, train_loader, criterion, optimizer, device, label_key
        )
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        val_loss, val_result = classification_evaluate(
            model, val_loader, criterion, device, label_key
        )
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_result['acc'])
        history['val_f1'].append(val_result['f1'])

        # 保存最佳模型
        if val_result['acc'] > best_acc:
            best_acc = val_result['acc']
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        # 学习率调度
        if scheduler:
            scheduler.step()

        # 早停检查
        if early_stopping and early_stopping(val_result['acc']):
            print(f"  Early stopping at epoch {epoch + 1}")
            break

        lr = optimizer.param_groups[-1]['lr']
        print(f"  Epoch {epoch+1:3d}/{num_epochs} | "
              f"Train: {train_loss:.4f}/{train_acc:.4f} | "
              f"Val: {val_loss:.4f}/{val_result['acc']:.4f} | LR: {lr:.6f}")

    # 恢复最佳模型
    if best_state:
        model.load_state_dict(best_state)

    history['best_val_acc'] = best_acc
    return history
