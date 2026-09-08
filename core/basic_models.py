"""
basic_models.py - 神经网络模型定义

模块职责
--------
定义帕金森病分类任务的神经网络模型，包括:
- Backbone 管理: 创建和配置特征提取网络
- 模型类: 单流和双流分类模型
- 预训练支持: 完整预训练、随机初始化、半冻结

支持的 Backbone
---------------
- resnet18/resnet34: 512 维特征, 推荐输入 224×224
- densenet121: 1024 维特征, 推荐输入 224×224
- efficientnet (B4): 1792 维特征, 推荐输入 380×380

预训练模式
----------
- True: 加载 ImageNet 预训练权重，所有层可训练
- False: 随机初始化，从零训练
- Half: 加载预训练权重，冻结低层特征提取器

尺寸策略
--------
- 迁移学习 (True/Half): 使用官方推荐尺寸
- 从零训练 (False): 保留原始尺寸，利用 AdaptiveAvgPool
"""
import torch
import torch.nn as nn
from torchvision.models import (
    resnet18, resnet34, densenet121, efficientnet_b4,
    ResNet18_Weights, ResNet34_Weights, DenseNet121_Weights, EfficientNet_B4_Weights
)
from typing import Literal

PretrainedMode = Literal[True, False, 'Half']


# ==================== Backbone 配置表 ====================

BACKBONE_CONFIG = {
    'resnet18': {
        'weights': ResNet18_Weights.IMAGENET1K_V1,
        'feature_dim': 512,
        'input_size': 224,
        'freeze_layers': ['conv1', 'bn1', 'layer1', 'layer2'],
    },
    'resnet34': {
        'weights': ResNet34_Weights.IMAGENET1K_V1,
        'feature_dim': 512,
        'input_size': 224,
        'freeze_layers': ['conv1', 'bn1', 'layer1', 'layer2'],
    },
    'densenet121': {
        'weights': DenseNet121_Weights.IMAGENET1K_V1,
        'feature_dim': 1024,
        'input_size': 224,
        'freeze_count': 6,
    },
    'efficientnet': {
        'weights': EfficientNet_B4_Weights.IMAGENET1K_V1,
        'feature_dim': 1792,
        'input_size': 380,
        'freeze_count': 4,
    },
}
"""
Backbone 配置表

每个 backbone 包含:
- weights: PyTorch 预训练权重枚举
- feature_dim: 特征提取器输出维度
- input_size: 官方推荐输入尺寸
- freeze_layers/freeze_count: 半冻结时要冻结的层
"""


# ==================== 辅助函数 ====================

def get_image_size(backbone: str, pretrained: PretrainedMode = True) -> tuple:
    """
    获取图像预处理尺寸配置
    
    根据 backbone 类型和预训练模式，决定是否需要 resize/crop。
    迁移学习时使用官方推荐尺寸以获得最佳效果。
    
    Args:
        backbone: backbone 名称
        pretrained: 预训练模式
        
    Returns:
        (resize_size, crop_size, use_original): 元组
        - use_original=True 时不做 resize，使用原始尺寸
        - use_original=False 时先 resize 再 center crop
    """
    if pretrained is False:
        return (None, None, True)
    
    size = BACKBONE_CONFIG[backbone]['input_size']
    resize = size + 32
    return (resize, size, False)


def get_backbone_feature_dim(backbone: str) -> int:
    """
    获取 backbone 输出特征维度
    
    Args:
        backbone: backbone 名称
        
    Returns:
        特征向量维度 (如 resnet18 返回 512)
    """
    return BACKBONE_CONFIG[backbone]['feature_dim']


def _freeze_half(model: nn.Module, backbone: str):
    """
    冻结模型前半部分层 (内部函数)
    
    用于半冻结预训练模式，保留低层通用特征，只训练高层任务特征。
    
    Args:
        model: 已加载权重的模型
        backbone: backbone 名称
    """
    config = BACKBONE_CONFIG[backbone]
    
    if backbone in ['resnet18', 'resnet34']:
        # ResNet: 冻结 conv1, bn1, layer1, layer2
        for name in config['freeze_layers']:
            for param in getattr(model, name).parameters():
                param.requires_grad = False
    else:
        # DenseNet/EfficientNet: 冻结 features 的前 N 个模块
        for i, module in enumerate(model.features):
            if i < config['freeze_count']:
                for param in module.parameters():
                    param.requires_grad = False


def get_backbone(backbone: str, pretrained: PretrainedMode = True) -> nn.Module:
    """
    创建 backbone 特征提取器
    
    加载指定的 CNN backbone，移除分类头，可选加载预训练权重。
    
    Args:
        backbone: backbone 名称 ('resnet18'/'resnet34'/'densenet121'/'efficientnet')
        pretrained: 预训练模式
            - True: 加载 ImageNet 权重
            - False: 随机初始化
            - 'Half': 加载权重并冻结前半部分
            
    Returns:
        特征提取器模型 (无分类头，输出为特征向量)
        
    Raises:
        ValueError: backbone 名称无效时抛出
    """
    config = BACKBONE_CONFIG.get(backbone)
    if not config:
        raise ValueError(f"未知 backbone: {backbone}，可选: {list(BACKBONE_CONFIG.keys())}")
    
    # 确定权重
    weights = config['weights'] if pretrained else None
    
    # 创建模型并移除分类头
    if backbone == 'resnet18':
        model = resnet18(weights=weights)
        model.fc = nn.Identity()
    elif backbone == 'resnet34':
        model = resnet34(weights=weights)
        model.fc = nn.Identity()
    elif backbone == 'densenet121':
        model = densenet121(weights=weights)
        model.classifier = nn.Identity()
    elif backbone == 'efficientnet':
        model = efficientnet_b4(weights=weights)
        model.classifier = nn.Identity()
    
    # 半冻结处理
    if pretrained == 'Half':
        _freeze_half(model, backbone)
    
    return model


# ==================== 模型类 ====================

class SingleTaskModel(nn.Module):
    """
    单流分类模型
    
    结构: Backbone -> Dropout -> FC(256) -> ReLU -> Dropout -> FC(num_classes)
    
    适用于:
    - single_stream 架构: 单张图像输入
    - mixed_single_stream 架构: 混合区域单流训练
    
    Args:
        backbone: backbone 名称
        num_classes: 输出类别数
        pretrained: 预训练模式
    """
    
    def __init__(self, backbone: str, num_classes: int, pretrained: PretrainedMode = True):
        super().__init__()
        self.backbone = get_backbone(backbone, pretrained)
        dim = get_backbone_feature_dim(backbone)
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入图像 (B, C, H, W)
            
        Returns:
            分类 logits (B, num_classes)
        """
        return self.classifier(self.backbone(x))


class DualStreamSingleTaskModel(nn.Module):
    """
    双流分类模型 (central + whorl)
    
    结构: 
    - Central Backbone -> features_c
    - Whorl Backbone -> features_w
    - Concat(features_c, features_w) -> 分类头
    
    适用于 dual_stream 架构，同时输入 central 和 whorl 两张图像。
    
    Args:
        backbone: backbone 名称
        num_classes: 输出类别数
        pretrained: 预训练模式
    """
    
    def __init__(self, backbone: str, num_classes: int, pretrained: PretrainedMode = True):
        super().__init__()
        self.backbone_central = get_backbone(backbone, pretrained)
        self.backbone_whorl = get_backbone(backbone, pretrained)
        
        dim = get_backbone_feature_dim(backbone) * 2  # 拼接后维度翻倍
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, central: torch.Tensor, whorl: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            central: central 区域图像 (B, C, H, W)
            whorl: whorl 区域图像 (B, C, H, W)
            
        Returns:
            分类 logits (B, num_classes)
        """
        f_c = self.backbone_central(central)
        f_w = self.backbone_whorl(whorl)
        return self.classifier(torch.cat([f_c, f_w], dim=1))
