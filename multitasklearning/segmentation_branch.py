"""
分割/重建分支 (Branch A)
=======================

支持三个任务:
1. 神经骨架分割 (CORN-1 数据集)
2. MAE 自监督重构
3. 伪标签生成 (Frangi Filter / Hessian)

输出: 重建的神经骨架图
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
import math
import logging

logger = logging.getLogger(__name__)


class ConvBlock(nn.Module):
    """标准的卷积块: Conv -> BN -> ReLU"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1, use_bn: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=not use_bn)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class DecoderBlock(nn.Module):
    """解码器块: 上采样 + 拼接 + 卷积"""
    
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int,
                 scale_factor: int = 2):
        super().__init__()
        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)
        self.conv_skip_reduce = ConvBlock(in_channels + skip_channels, out_channels, 3, 1, 1)
        self.conv = ConvBlock(out_channels, out_channels, 3, 1, 1)
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # 确保空间维度匹配
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv_skip_reduce(x)
        x = self.conv(x)
        return x


class SegmentationDecoder(nn.Module):
    """
    分割解码器
    
    从编码器的瓶颈特征恢复到原始分辨率，
    生成神经骨架分割掩模
    """
    
    def __init__(self, encoder_dims: Tuple[int], output_channels: int = 1):
        """
        Parameters
        ----------
        encoder_dims : Tuple[int]
            编码器各层的通道数 (从低到高分辨率)
        output_channels : int
            输出通道数，通常为1 (二值分割)
        """
        super().__init__()
        self.encoder_dims = encoder_dims
        
        # 解码器块
        self.decoder4 = DecoderBlock(encoder_dims[-1], encoder_dims[-2], encoder_dims[-2])
        self.decoder3 = DecoderBlock(encoder_dims[-2], encoder_dims[-3], encoder_dims[-3])
        self.decoder2 = DecoderBlock(encoder_dims[-3], encoder_dims[-4], encoder_dims[-4])
        self.decoder1 = DecoderBlock(encoder_dims[-4], encoder_dims[-5], encoder_dims[-5])
        
        # 最后的上采样和输出层
        self.final_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.final_conv = ConvBlock(encoder_dims[-5], encoder_dims[-5], 3, 1, 1)
        self.output_layer = nn.Conv2d(encoder_dims[-5], output_channels, 1)
        self.sigmoid = nn.Sigmoid()  # 二值分割概率
    
    def forward(self, features: List[torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Parameters
        ----------
        features : List[torch.Tensor]
            编码器输出的多尺度特征
        
        Returns
        -------
        Tuple[torch.Tensor, List[torch.Tensor]]
            (分割输出, 中间特征列表)
        """
        # 反向遍历特征进行解码
        x = features[-1]  # 瓶颈特征
        decoder_features = [x]
        
        x = self.decoder4(x, features[-2])
        decoder_features.append(x)
        
        x = self.decoder3(x, features[-3])
        decoder_features.append(x)
        
        x = self.decoder2(x, features[-4])
        decoder_features.append(x)
        
        x = self.decoder1(x, features[-5])
        decoder_features.append(x)
        
        x = self.final_up(x)
        x = self.final_conv(x)
        
        # 输出分割概率图
        segmentation = self.sigmoid(self.output_layer(x))
        
        return segmentation, decoder_features


class MAEReconstructor(nn.Module):
    """
    遮挡自编码 (Masked Autoencoder) 重构器
    
    用于自监督学习，通过预测被遮挡的图像patch
    来增强特征表示
    """
    
    def __init__(self, patch_size: int = 16, mask_ratio: float = 0.75,
                 embedding_dim: int = 768, input_channels: int = 3):
        super().__init__()
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.embedding_dim = embedding_dim
        self.input_channels = input_channels
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(input_channels, embedding_dim, 
                                    kernel_size=patch_size, stride=patch_size)
        
        # 编码器 (简化版本，实际使用会更复杂)
        self.encoder = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.GELU(),
            nn.Linear(embedding_dim * 2, embedding_dim),
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.GELU(),
            nn.Linear(embedding_dim * 2, patch_size * patch_size * input_channels),
        )
        
        # 位置编码
        self._register_buffers()
    
    def _register_buffers(self):
        """注册位置编码等"""
        # 简化实现：不显式添加位置编码，依赖卷积的局部性
        pass
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : torch.Tensor
            输入图像 [B, C, H, W]
        
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            (重构图像, 损失掩模)
        """
        B, C, H, W = x.shape
        
        # Patch embedding
        patches = self.patch_embed(x)  # [B, embedding_dim, H/patch_size, W/patch_size]
        num_patches = (H // self.patch_size) * (W // self.patch_size)
        
        # 随机遮挡patches
        num_mask = int(num_patches * self.mask_ratio)
        mask_indices = torch.randperm(num_patches)[:num_mask]
        mask = torch.zeros(B, num_patches, device=x.device)
        mask[:, mask_indices] = 1.0
        
        # Flatten patches
        patches_flat = patches.flatten(2).transpose(1, 2)  # [B, num_patches, embedding_dim]
        
        # 编码
        encoded = self.encoder(patches_flat)
        
        # 解码
        decoded = self.decoder(encoded)  # [B, num_patches, patch_size^2 * C]
        
        # 重塑回图像
        decoded = decoded.view(B, H // self.patch_size, W // self.patch_size,
                             self.patch_size, self.patch_size, C)
        decoded = decoded.permute(0, 5, 1, 3, 2, 4).contiguous()
        decoded = decoded.view(B, C, H, W)
        
        return decoded, mask.view(B, 1, H // self.patch_size, W // self.patch_size)


class SegmentationBranch(nn.Module):
    """
    分割分支 (Branch A) - 完整实现
    
    整合:
    1. 分割解码器 (恢复神经骨架)
    2. MAE 重构器 (自监督学习)
    3. 伪标签支持 (Frangi/Hessian)
    """
    
    def __init__(self, encoder_dims: Tuple[int], output_channels: int = 1,
                 patch_size: int = 16, mae_mask_ratio: float = 0.75,
                 mae_embedding_dim: int = 768):
        super().__init__()
        
        self.output_channels = output_channels
        
        # 分割解码器
        self.segmentation_decoder = SegmentationDecoder(encoder_dims, output_channels)
        
        # MAE 重构器
        self.mae_reconstructor = MAEReconstructor(
            patch_size=patch_size,
            mask_ratio=mae_mask_ratio,
            embedding_dim=mae_embedding_dim,
            input_channels=3
        )
        
        # 损失函数
        self.segmentation_loss = nn.BCELoss()  # 二值交叉熵
        self.mae_loss = nn.MSELoss()  # 均方误差重构损失
    
    def forward(self, features: List[torch.Tensor], images: torch.Tensor = None,
                segmentation_targets: torch.Tensor = None,
                use_mae: bool = False, use_segmentation: bool = True):
        """
        Parameters
        ----------
        features : List[torch.Tensor]
            编码器输出的多尺度特征
        images : torch.Tensor, optional
            原始输入图像 (用于MAE)
        segmentation_targets : torch.Tensor, optional
            分割真值标签 [B, 1, H, W]
        use_mae : bool
            是否使用MAE重构
        use_segmentation : bool
            是否使用分割解码器
        
        Returns
        -------
        dict
            包含 segmentation, mae_reconstruction, losses 的字典
        """
        outputs = {}
        losses = {}
        
        # 分割分支
        if use_segmentation:
            segmentation, decoder_features = self.segmentation_decoder(features)
            outputs['segmentation'] = segmentation
            
            if segmentation_targets is not None:
                seg_loss = self.segmentation_loss(segmentation, segmentation_targets)
                losses['segmentation'] = seg_loss
        
        # MAE 重构分支
        if use_mae and images is not None:
            mae_reconstructed, mask = self.mae_reconstructor(images)
            outputs['mae_reconstructed'] = mae_reconstructed
            outputs['mae_mask'] = mask

            # 将 patch 级 mask [B,1,H/p,W/p] 上采样到图像分辨率再计算损失
            mask_upsampled = torch.nn.functional.interpolate(
                mask, size=images.shape[2:], mode='nearest'
            )
            mae_loss = self.mae_loss(mae_reconstructed * mask_upsampled,
                                     images * mask_upsampled)
            losses['mae'] = mae_loss
        
        outputs['losses'] = losses
        return outputs
    
    def get_pseudolabels(self, segmentation: torch.Tensor, method: str = 'frangi') -> torch.Tensor:
        """
        获取伪标签 (用于无监督学习)
        
        Parameters
        ----------
        segmentation : torch.Tensor
            分割输出 [B, 1, H, W]
        method : str
            方法: 'frangi' 或 'hessian'
        
        Returns
        -------
        torch.Tensor
            伪标签
        """
        # 这部分由外部的伪标签生成器处理
        # 这里仅作占位符
        return segmentation


if __name__ == '__main__':
    # 测试
    encoder_dims = (64, 64, 128, 256, 512)
    
    seg_branch = SegmentationBranch(
        encoder_dims=encoder_dims,
        output_channels=1,
        patch_size=16,
        mae_mask_ratio=0.75
    )
    
    # 模拟编码器特征
    features = [
        torch.randn(2, 64, 128, 128),
        torch.randn(2, 64, 64, 64),
        torch.randn(2, 128, 32, 32),
        torch.randn(2, 256, 16, 16),
        torch.randn(2, 512, 8, 8),
    ]
    
    images = torch.randn(2, 3, 256, 256)
    targets = torch.randint(0, 2, (2, 1, 256, 256)).float()
    
    outputs = seg_branch(features, images, targets, use_mae=True, use_segmentation=True)
    print("分割输出:", outputs['segmentation'].shape)
    print("MAE重构:", outputs['mae_reconstructed'].shape)
    print("损失:", outputs['losses'])
