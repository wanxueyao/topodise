"""
U-Net 编码器
===========

支持多种backbone:
- ResNet (18, 34, 50, 101)
- EfficientNet (B0-B7)

输出: 多尺度特征图用于解码器
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import List, Tuple, Literal
import logging

logger = logging.getLogger(__name__)


class UNetEncoder(nn.Module):
    """
    U-Net 编码器，提取多尺度特征
    
    Parameters
    ----------
    encoder_type : str
        Backbone 类型: 'resnet18', 'resnet50', 'efficientnet_b4'
    pretrained : bool
        是否使用预训练权重
    input_channels : int
        输入通道数 (通常为3)
    """
    
    def __init__(self, encoder_type: str = 'resnet50', pretrained: bool = True,
                 input_channels: int = 3):
        super().__init__()
        self.encoder_type = encoder_type
        self.pretrained = pretrained
        
        if encoder_type.startswith('resnet'):
            self._build_resnet(encoder_type, pretrained, input_channels)
        elif encoder_type.startswith('efficientnet'):
            self._build_efficientnet(encoder_type, pretrained, input_channels)
        else:
            raise ValueError(f"不支持的编码器类型: {encoder_type}")
    
    def _build_resnet(self, model_name: str, pretrained: bool, input_channels: int):
        """构建ResNet编码器"""
        # 获取ResNet模型
        model_fn = getattr(models, model_name)
        backbone = model_fn(weights='DEFAULT' if pretrained else None)
        
        # 处理输入通道
        if input_channels != 3:
            original_conv = backbone.conv1
            backbone.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, 
                                       stride=2, padding=3, bias=False)
        
        # 提取特征提取模块
        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1  # /4, 256
        self.layer2 = backbone.layer2  # /8, 512
        self.layer3 = backbone.layer3  # /16, 1024
        self.layer4 = backbone.layer4  # /32, 2048
        
        # 特征维度
        self._feature_dims = {
            'resnet18': (64, 64, 128, 256, 512),
            'resnet34': (64, 64, 128, 256, 512),
            'resnet50': (64, 256, 512, 1024, 2048),
            'resnet101': (64, 256, 512, 1024, 2048),
        }
        self.feature_dims = self._feature_dims.get(model_name, (64, 256, 512, 1024, 2048))
    
    def _build_efficientnet(self, model_name: str, pretrained: bool, input_channels: int):
        """构建EfficientNet编码器"""
        model_fn = getattr(models, model_name)
        backbone = model_fn(weights='DEFAULT' if pretrained else None)
        
        # 处理输入通道
        if input_channels != 3:
            original_conv = backbone.features[0][0]
            backbone.features[0][0] = nn.Conv2d(input_channels, 32, kernel_size=3,
                                                stride=2, padding=1, bias=False)
        
        # EfficientNet 没有直接的layer访问，需要通过features
        # 分成若干stage
        self.layer0 = backbone.features[0]  # stem
        self.layer1 = backbone.features[1]  # block 0
        self.layer2 = backbone.features[2:4]  # blocks 1-3
        self.layer3 = backbone.features[4:6]  # blocks 4-6
        self.layer4 = backbone.features[6:]  # blocks 7+
        self.maxpool = nn.Identity()  # EfficientNet 不需要单独的maxpool
        
        # EfficientNet 特征维度 (B4为例)
        self._efficientnet_dims = {
            'efficientnet_b0': (32, 16, 24, 40, 112, 320),
            'efficientnet_b4': (48, 24, 56, 112, 320, 1408),
        }
        dims = self._efficientnet_dims.get(model_name, (48, 24, 56, 112, 320, 1408))
        self.feature_dims = dims
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        获取多尺度特征
        
        Parameters
        ----------
        x : torch.Tensor
            输入张量 [B, C, H, W]
        
        Returns
        -------
        List[torch.Tensor]
            多尺度特征列表 [低分辨率...高分辨率]
        """
        features = []
        
        # Layer 0 (stem)
        x = self.layer0(x)  # /2
        features.append(x)
        
        # Layer 1
        x = self.maxpool(x)  # /4
        x = self.layer1(x)
        features.append(x)
        
        # Layer 2
        x = self.layer2(x)  # /8
        features.append(x)
        
        # Layer 3
        x = self.layer3(x)  # /16
        features.append(x)
        
        # Layer 4
        x = self.layer4(x)  # /32
        features.append(x)
        
        return features
    
    @property
    def num_levels(self) -> int:
        """编码器深度 (特征图数量)"""
        return 5
    
    @property
    def output_stride(self) -> int:
        """最后一层特征图步长"""
        return 32
    
    @property
    def bottleneck_channels(self) -> int:
        """瓶颈层通道数"""
        return self.feature_dims[-1]


class ResNetEncoder(UNetEncoder):
    """ResNet 编码器的便捷包装"""
    
    def __init__(self, depth: int = 50, pretrained: bool = True):
        model_name = f'resnet{depth}'
        super().__init__(encoder_type=model_name, pretrained=pretrained)


class EfficientNetEncoder(UNetEncoder):
    """EfficientNet 编码器的便捷包装"""
    
    def __init__(self, level: str = 'b4', pretrained: bool = True):
        model_name = f'efficientnet_{level}'
        super().__init__(encoder_type=model_name, pretrained=pretrained)


# 测试代码
if __name__ == '__main__':
    import torch
    
    # 测试 ResNet50
    encoder = ResNetEncoder(depth=50, pretrained=False)
    x = torch.randn(2, 3, 256, 256)
    features = encoder(x)
    
    print("ResNet50 编码器")
    print(f"输入: {x.shape}")
    for i, feat in enumerate(features):
        print(f"特征{i}: {feat.shape}")
    
    # 测试 EfficientNet B4
    encoder = EfficientNetEncoder(level='b4', pretrained=False)
    features = encoder(x)
    
    print("\nEfficientNet B4 编码器")
    print(f"输入: {x.shape}")
    for i, feat in enumerate(features):
        print(f"特征{i}: {feat.shape}")
