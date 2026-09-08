"""
多任务学习全局配置
=================

设置:
- 模型超参数
- 训练配置
- 数据配置
- 任务权重
"""

from typing import Literal, Dict, List
from dataclasses import dataclass, field


@dataclass
class MTLConfig:
    """多任务学习配置"""
    
    # ==================== 模型架构 ====================
    # U-Net 编码器配置
    encoder_type: str = 'resnet50'  # resnet50, resnet101, efficientnet_b4
    encoder_pretrained: bool = True
    input_channels: int = 3
    
    # U-Net 通道配置
    initial_channels: int = 64
    max_channels: int = 512
    num_encoder_levels: int = 5  # 编码器深度
    
    # 分类分支架构
    classification_mode: Literal['single_stream', 'dual_stream', 'mixed_single_stream'] = 'dual_stream'
    num_pd_subtypes: int = 3  # TD, PIGD, mixed
    num_classification_heads: int = 3  # PD vs Control, Stage (HY分级), Subtype
    
    # 分割分支配置
    segmentation_output_channels: int = 1  # 神经骨架二值化
    mae_patch_size: int = 16
    mae_mask_ratio: float = 0.75  # MAE 遮挡比例
    mae_embedding_dim: int = 768
    
    # ==================== 训练配置 ====================
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_epochs: int = 5
    
    # 优化器
    optimizer: str = 'adamw'  # adamw, sgd
    gradient_clip: float = 1.0
    
    # 学习率调度
    lr_scheduler: str = 'cosine'  # cosine, step, exponential
    scheduler_params: Dict = field(default_factory=lambda: {
        'T_max': 100,  # cosine
        'eta_min': 1e-7
    })
    
    # ==================== 任务权重 ====================
    # 分支A (分割/重建)
    loss_weight_segmentation: float = 0.3
    loss_weight_mae: float = 0.2
    loss_weight_pseudo: float = 0.1
    
    # 分支B (分类)
    loss_weight_pd_classification: float = 0.2
    loss_weight_stage: float = 0.1
    loss_weight_subtype: float = 0.1
    
    # ==================== 数据配置 ====================
    # CORN-1 数据集
    corn1_root: str = 'datasets/CORN-1'
    corn1_split_ratio: float = 0.8  # 训练:测试比例
    
    # PD 数据集
    pd_dataset_root: str = 'datasets/pd_dataset_en'
    pd_kfold_splits: int = 5
    pd_kfold_mode: Literal['patient', 'image'] = 'patient'
    
    # 数据增强
    image_size: int = 256
    crop_size: int = 224
    augmentation_prob: float = 0.5
    use_data_augmentation: bool = True
    
    # ==================== 伪标签配置 ====================
    frangi_sigma_range: tuple = (1, 10)  # Frangi filter sigma范围
    frangi_scales: int = 50
    hessian_sigma: float = 2.0
    pseudo_label_threshold: float = 0.5
    
    # ==================== 设备和日志 ====================
    device: str = 'cuda'
    num_workers: int = 4
    pin_memory: bool = True
    seed: int = 42
    
    # 检查点和日志
    checkpoint_dir: str = 'checkpoints'
    log_dir: str = 'logs'
    save_interval: int = 10  # 每N个epoch保存一次
    log_interval: int = 10  # 每N步记录一次
    
    # ==================== 评估配置 ====================
    use_ema: bool = False  # Exponential Moving Average
    ema_decay: float = 0.999
    validation_interval: int = 1  # 每N个epoch验证一次
    
    # ==================== 高级配置 ====================
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    early_stopping_patience: int = 20
    resume_checkpoint: str = None  # 恢复训练的检查点路径
    
    def __post_init__(self):
        """验证配置"""
        assert 0 < self.mae_mask_ratio < 1, "MAE mask ratio 必须在 0-1 之间"
        assert self.num_pd_subtypes == 3, "PD子类型数必须为3 (TD, PIGD, mixed)"
        assert self.loss_weight_segmentation + self.loss_weight_mae + self.loss_weight_pseudo + \
               self.loss_weight_pd_classification + self.loss_weight_stage + self.loss_weight_subtype > 0
        
    def get_total_weight(self) -> float:
        """获取总权重"""
        return (self.loss_weight_segmentation + self.loss_weight_mae + 
                self.loss_weight_pseudo + self.loss_weight_pd_classification + 
                self.loss_weight_stage + self.loss_weight_subtype)
    
    def normalize_weights(self) -> Dict[str, float]:
        """归一化权重 (总和为1)"""
        total = self.get_total_weight()
        return {
            'segmentation': self.loss_weight_segmentation / total,
            'mae': self.loss_weight_mae / total,
            'pseudo': self.loss_weight_pseudo / total,
            'pd_classification': self.loss_weight_pd_classification / total,
            'stage': self.loss_weight_stage / total,
            'subtype': self.loss_weight_subtype / total,
        }


# ==================== 预定义配置 ====================

def get_config_lightweight() -> MTLConfig:
    """轻量级配置 (快速原型)"""
    return MTLConfig(
        encoder_type='resnet18',
        batch_size=16,
        num_epochs=50,
        learning_rate=5e-4,
        image_size=224,
        crop_size=192,
    )


def get_config_standard() -> MTLConfig:
    """标准配置 (推荐)"""
    return MTLConfig(
        encoder_type='resnet50',
        batch_size=32,
        num_epochs=100,
        learning_rate=1e-4,
        image_size=256,
        crop_size=224,
    )


def get_config_large() -> MTLConfig:
    """大规模配置 (更好的性能)"""
    return MTLConfig(
        encoder_type='efficientnet_b4',
        batch_size=64,
        num_epochs=150,
        learning_rate=5e-5,
        image_size=380,
        crop_size=320,
        mixed_precision=True,
        gradient_accumulation_steps=2,
    )
