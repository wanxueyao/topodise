"""ResNet18 SimCLR augmentations, contrastive loss and pretraining loop."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from PIL import Image
from typing import Callable, Optional, Tuple
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from contextlib import nullcontext

from core.basic_models import (
    get_backbone as _get_basic_backbone,
    get_backbone_feature_dim as _get_basic_feature_dim,
)
from core.progress import ETAEstimator, format_duration


# Backbone 路由：使用 basic_models 的名称集合


def _get_backbone(backbone: str, pretrained) -> 'nn.Module':
    return _get_basic_backbone(backbone, pretrained)


def _get_feature_dim(backbone: str) -> int:
    return _get_basic_feature_dim(backbone)


def _get_input_size(backbone: str) -> int:
    """获取 backbone 输入尺寸（统一入口）"""
    from core.basic_models import BACKBONE_CONFIG
    return BACKBONE_CONFIG[backbone]['input_size']


# ==================== SimCLR 数据增强 ====================

class SimCLRTransform:
    """
    SimCLR 对比学习专用图像增强

    针对眼底视网膜图像设计：
    - 色调扰动关闭（hue=0），保留视网膜的诊断色彩信息
    - 饱和度扰动降低，避免破坏眼底血管与病灶颜色
    - 随机裁剪模拟视角/缩放变化

    增强流程
    --------
    1. RandomResizedCrop: 随机裁剪并缩放至目标尺寸
    2. RandomHorizontalFlip: 左右翻转 (p=0.5)
    3. ColorJitter: 亮度/对比度/饱和度扰动 (p=0.8)
    4. RandomGrayscale: 灰度化 (p=0.1)
    5. ToTensor + Normalize (ImageNet 均值/方差)

    Args:
        input_size: 目标图像尺寸
        scale: RandomResizedCrop 面积比例范围
        jitter_strength: 色彩扰动强度系数 (0~1，默认 0.5)
    """

    def __init__(self, input_size: int = 224,
                 scale: Tuple[float, float] = (0.5, 1.0),
                 jitter_strength: float = 0.5):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=scale),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(
                    brightness=0.4 * jitter_strength,
                    contrast=0.4 * jitter_strength,
                    saturation=0.2 * jitter_strength,
                    hue=0.0,  # 眼底图像不扰动色调
                )
            ], p=0.8),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __call__(self, img) -> torch.Tensor:
        return self.transform(img)


# ==================== SimCLR 数据集包装器 ====================

class SimCLRDatasetWrapper(Dataset):
    """
    SimCLR 预训练数据集包装器

    将 PDDataset（或任意包含 .samples / .mode 的 Dataset）包装为双视图格式。
    每次 __getitem__ 对同一图像独立应用两次随机增强，返回 (view1, view2)。

    预训练阶段不使用标签，因此只返回图像视图。

    Args:
        base_dataset: 基础数据集（PDDataset 实例）
        simclr_transform: SimCLRTransform 或等价的可调用对象
    """

    def __init__(self, base_dataset, simclr_transform: SimCLRTransform):
        self.samples, self.mode = self._resolve_samples_and_mode(base_dataset)
        self.simclr_transform = simclr_transform

    @staticmethod
    def _resolve_samples_and_mode(base_dataset):
        """兼容 Dataset 与 Subset，解析出样本路径列表与模式。"""
        dataset = base_dataset
        indices = None

        # 支持 Subset(Subset(...)) 的索引映射
        while isinstance(dataset, Subset):
            current_indices = list(dataset.indices)
            if indices is None:
                indices = current_indices
            else:
                indices = [current_indices[i] for i in indices]
            dataset = dataset.dataset

        if not hasattr(dataset, 'samples'):
            raise AttributeError(
                "SimCLRDatasetWrapper 需要基础数据集包含 samples 属性，"
                f"当前类型: {type(dataset).__name__}"
            )

        mode = getattr(dataset, 'mode', 'single_stream')
        if indices is None:
            samples = dataset.samples
        else:
            samples = [dataset.samples[i] for i in indices]

        return samples, mode

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        # dual_stream 模式下取 central 图像用于预训练
        path = sample['central'] if isinstance(sample, dict) else sample
        img = Image.open(path).convert('RGB')
        view1 = self.simclr_transform(img)
        view2 = self.simclr_transform(img)
        return view1, view2


# ==================== NT-Xent 损失 ====================

class NTXentLoss(nn.Module):
    """
    归一化温度交叉熵损失 (NT-Xent Loss)

    SimCLR 核心损失函数。对于 batch 中 N 个样本，每个样本生成两个视图，
    共 2N 个样本。每个视图的正样本对是同一图像的另一视图，
    其余 2N-2 个视图均为负样本。

    数学形式（对第 i 个视图）:
        ℓ(i, j) = -log( exp(sim(z_i, z_j)/τ) / Σ_{k≠i} exp(sim(z_i, z_k)/τ) )

    Args:
        temperature: 温度系数 τ，控制对比强度（推荐 0.1~0.5）
    """

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        计算 NT-Xent 损失

        Args:
            z1: L2 归一化后的投影向量 (B, D)
            z2: L2 归一化后的投影向量 (B, D)

        Returns:
            标量损失
        """
        N = z1.shape[0]
        z = torch.cat([z1, z2], dim=0)  # (2N, D)

        # 余弦相似度矩阵 (已归一化，点积 = 余弦相似度)
        sim = torch.mm(z, z.T) / self.temperature  # (2N, 2N)

        # 屏蔽对角线（不与自身对比）
        mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
        sim = sim.masked_fill(mask, -float('inf'))

        # 正样本对标签: z1[i] 对应 z2[i]（即位置 i+N），z2[i] 对应 z1[i]（位置 i）
        labels = torch.cat([
            torch.arange(N, 2 * N, device=z.device),
            torch.arange(0, N, device=z.device),
        ])

        return F.cross_entropy(sim, labels)


# ==================== 投影头 ====================

class SimCLRProjectionHead(nn.Module):
    """
    SimCLR MLP 投影头

    将 backbone 特征映射到对比学习空间。
    预训练结束后此模块被丢弃，backbone 特征直接用于下游任务。

    结构: Linear(in_dim → hidden_dim) → BN → ReLU → Linear(hidden_dim → out_dim)

    Args:
        in_dim: backbone 输出特征维度
        hidden_dim: 隐层维度（默认 256）
        out_dim: 对比空间输出维度（默认 128）
    """

    def __init__(self, in_dim: int, hidden_dim: int = 256, out_dim: int = 128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# ==================== SimCLR 模型 ====================

class SimCLRModel(nn.Module):
    """
    SimCLR 预训练模型

    组合轻量化 backbone 与 MLP 投影头，用于自监督对比预训练。
    预训练完成后，通过 get_backbone() 取出 backbone 迁移到分类任务。

    Args:
        backbone_name: 轻量化 backbone 名称（见 BACKBONE_CONFIG）
        pretrained: ImageNet 预训练权重加载模式
        proj_hidden_dim: 投影头隐层维度
        proj_out_dim: 投影头输出维度
    """

    def __init__(self, backbone_name: str, pretrained=True,
                 proj_hidden_dim: int = 256, proj_out_dim: int = 128):
        super().__init__()
        self.backbone_name = backbone_name
        self.backbone = _get_backbone(backbone_name, pretrained)
        feat_dim = _get_feature_dim(backbone_name)
        self.projection_head = SimCLRProjectionHead(feat_dim, proj_hidden_dim, proj_out_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Returns:
            (features, projections): backbone 特征 (B, feat_dim) 和
                                     L2 归一化投影向量 (B, proj_out_dim)
        """
        feat = self.backbone(x)
        proj = self.projection_head(feat)
        proj = F.normalize(proj, dim=1)
        return feat, proj

    def get_backbone(self) -> nn.Module:
        """返回预训练 backbone（用于初始化下游任务模型）"""
        return self.backbone



# ==================== 预训练单 epoch ====================

def simclr_pretrain_epoch(
    model: SimCLRModel,
    dataloader: DataLoader,
    criterion: NTXentLoss,
    optimizer: torch.optim.Optimizer,
    device: str,
    amp: bool = False,
    channels_last: bool = False,
    non_blocking: bool = False,
) -> float:
    """
    SimCLR 预训练单个 epoch

    Args:
        model: SimCLRModel
        dataloader: 双视图 DataLoader（SimCLRDatasetWrapper）
        criterion: NTXentLoss
        optimizer: 优化器
        device: 计算设备

    Returns:
        本 epoch 的平均 NT-Xent 损失
    """
    model.train()

    total_loss = 0.0
    n_batches = 0

    for view1, view2 in dataloader:
        view1 = view1.to(device, non_blocking=non_blocking)
        view2 = view2.to(device, non_blocking=non_blocking)
        if channels_last:
            view1 = view1.contiguous(memory_format=torch.channels_last)
            view2 = view2.contiguous(memory_format=torch.channels_last)

        # BatchNorm 需要至少 2 个样本
        if view1.shape[0] < 2:
            continue

        optimizer.zero_grad(set_to_none=True)
        amp_context = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if amp and torch.device(device).type == "cuda"
            else nullcontext()
        )
        with amp_context:
            _, z1 = model(view1)
            _, z2 = model(view2)
        # Keep the O(2B^2) similarity matrix and softmax in FP32.
        loss = criterion(z1.float(), z2.float())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


# ==================== SimCLR 预训练主循环 ====================

def simclr_pretrain(
    backbone_name: str,
    pretrained,
    train_dataset,
    num_epochs: int,
    batch_size: int,
    lr: float,
    temperature: float,
    device: str,
    proj_hidden_dim: int = 256,
    proj_out_dim: int = 128,
    weight_decay: float = 1e-4,
    num_workers: int = 4,
    input_size: Optional[int] = None,
    verbose: bool = True,
    amp: bool = False,
    channels_last: bool = False,
    persistent_workers: bool = False,
    pin_memory: Optional[bool] = None,
    fused_adamw: bool = False,
    drop_last: bool = True,
    initial_model: Optional[SimCLRModel] = None,
    progress_callback: Optional[Callable[[dict], None]] = None,
    progress_stage: str = "simclr",
) -> SimCLRModel:
    """
    执行 SimCLR 自监督预训练

    对训练集中的所有图像（不使用标签）进行对比学习预训练。
    预训练完成后返回 SimCLRModel，其 backbone 已适应数据分布。

    Args:
        backbone_name: 轻量化 backbone 名称
        pretrained: ImageNet 预训练权重加载模式（True / False / 'Half'）
        train_dataset: 训练数据集（PDDataset 实例，需含 .samples / .mode）
        num_epochs: 预训练轮数（推荐 50~200，小数据集用 100）
        batch_size: 预训练 batch 大小（SimCLR 大 batch 效果更好，建议 ≥32）
        lr: 预训练学习率（推荐 1e-3，使用 AdamW）
        temperature: NT-Xent 温度系数（推荐 0.3~0.7）
        device: 计算设备
        proj_hidden_dim: 投影头隐层维度
        proj_out_dim: 投影头输出维度
        weight_decay: 权重衰减（预训练用较小值）
        num_workers: DataLoader 工作进程数
        input_size: backbone 输入尺寸（None 时自动从 BACKBONE_CONFIG 获取）
        verbose: 是否打印训练进度

    Returns:
        预训练完成的 SimCLRModel
    """
    if input_size is None:
        input_size = _get_input_size(backbone_name)

    # 构建 SimCLR 双视图数据集
    simclr_transform = SimCLRTransform(input_size=input_size)
    simclr_dataset = SimCLRDatasetWrapper(train_dataset, simclr_transform)
    device_obj = torch.device(device)
    if pin_memory is None:
        pin_memory = device_obj.type == 'cuda'
    loader_kwargs = {
        'batch_size': batch_size,
        'shuffle': True,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        'drop_last': drop_last,
        'persistent_workers': persistent_workers and num_workers > 0,
    }
    if num_workers > 0:
        loader_kwargs['prefetch_factor'] = 2
    simclr_loader = DataLoader(simclr_dataset, **loader_kwargs)

    # 构建模型、损失、优化器、调度器
    if initial_model is None:
        model = SimCLRModel(
            backbone_name, pretrained,
            proj_hidden_dim=proj_hidden_dim,
            proj_out_dim=proj_out_dim,
        ).to(device)
    else:
        if initial_model.backbone_name != backbone_name:
            raise ValueError(
                f"initial SimCLR model backbone mismatch: "
                f"{initial_model.backbone_name} != {backbone_name}"
            )
        model = initial_model.to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)

    criterion = NTXentLoss(temperature=temperature)
    optimizer_kwargs = {'lr': lr, 'weight_decay': weight_decay}
    if fused_adamw and device_obj.type == 'cuda':
        try:
            optimizer = torch.optim.AdamW(model.parameters(), fused=True, **optimizer_kwargs)
        except (TypeError, RuntimeError):
            optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=lr * 1e-3
    )

    if verbose:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  [SimCLR] backbone={backbone_name}, trainable_params={n_params:,}, "
              f"pretrain_epochs={num_epochs}, T={temperature}, "
              f"batch={batch_size}, lr={lr}, amp_bf16={amp and device_obj.type == 'cuda'}, "
              f"channels_last={channels_last}, workers={num_workers}")

    epoch_progress = ETAEstimator(num_epochs, window=5)
    for epoch in range(num_epochs):
        epoch_started = time.perf_counter()
        loss = simclr_pretrain_epoch(
            model, simclr_loader, criterion, optimizer, device,
            amp=amp, channels_last=channels_last, non_blocking=bool(pin_memory),
        )
        scheduler.step()
        epoch_seconds = time.perf_counter() - epoch_started
        progress = epoch_progress.update(epoch + 1, interval_seconds=epoch_seconds)
        lr_cur = optimizer.param_groups[0]['lr']
        event = {
            "kind": "epoch_progress",
            "stage": progress_stage,
            "epoch": epoch + 1,
            "epochs": num_epochs,
            "epoch_seconds": round(epoch_seconds, 3),
            "loss": loss,
            "lr": lr_cur,
            **progress,
        }
        if progress_callback is not None:
            progress_callback(event)
        if verbose:
            print(
                f"  [SimCLR][{progress_stage}] epoch={epoch+1:3d}/{num_epochs} "
                f"({progress['progress_pct']:.1f}%) loss={loss:.4f} lr={lr_cur:.6f} "
                f"epoch_s={epoch_seconds:.1f} "
                f"elapsed={format_duration(progress['elapsed_seconds'])} "
                f"ETA={format_duration(progress['eta_seconds'])} "
                f"finish={progress['estimated_finish_at']} "
                f"quality={progress['eta_quality']}",
                flush=True,
            )

    return model
