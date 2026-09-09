"""
分类分支 (Branch B)
===================

支持三种架构:
1. Single Stream: 单一流，基于解码器特征的分类
2. Dual Stream: 双流，并行处理分割和分类信息
3. Mixed Single Stream: 混合单流，先融合再分类

输出: PD vs Control, HY分级, 亚型分类 (TD/PIGD/mixed)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Literal, Dict
import logging

logger = logging.getLogger(__name__)


class AdaptiveAvgMaxPool(nn.Module):
    """自适应平均池化 + 最大池化的组合"""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_pool = F.adaptive_avg_pool2d(x, 1)
        max_pool = F.adaptive_max_pool2d(x, 1)
        return torch.cat([avg_pool, max_pool], dim=1)


class ClassificationHead(nn.Module):
    """分类头: 池化 -> MLP -> 分类"""
    
    def __init__(self, in_channels: int, num_classes: int, hidden_dim: int = 512,
                 dropout_rate: float = 0.5):
        super().__init__()
        self.pool = AdaptiveAvgMaxPool()
        
        # 池化后的通道数翻倍
        pooled_channels = in_channels * 2
        
        self.mlp = nn.Sequential(
            nn.Linear(pooled_channels, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            特征图 [B, C, H, W]
        
        Returns
        -------
        torch.Tensor
            分类逻辑 [B, num_classes]
        """
        x = self.pool(x)  # [B, 2*C, 1, 1]
        x = x.view(x.size(0), -1)  # [B, 2*C]
        x = self.mlp(x)  # [B, num_classes]
        return x


class SingleStreamClassifier(nn.Module):
    """
    单流分类器
    
    从解码器特征直接进行分类
    """
    
    def __init__(self, decoder_features_channels: List[int],
                 num_pd_classes: int = 2,
                 num_stage_classes: int = 5,
                 num_subtype_classes: int = 3,
                 hidden_dim: int = 512,
                 dropout_rate: float = 0.5):
        super().__init__()
        
        # 使用最高分辨率的解码器特征
        feature_channel = decoder_features_channels[0]
        
        # 三个分类头
        self.pd_classifier = ClassificationHead(
            feature_channel, num_pd_classes, hidden_dim, dropout_rate
        )
        self.stage_classifier = ClassificationHead(
            feature_channel, num_stage_classes, hidden_dim, dropout_rate
        )
        self.subtype_classifier = ClassificationHead(
            feature_channel, num_subtype_classes, hidden_dim, dropout_rate
        )
    
    def forward(self, decoder_features: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        decoder_features : List[torch.Tensor]
            解码器输出的特征列表
        
        Returns
        -------
        Dict[str, torch.Tensor]
            分类输出字典
        """
        # 使用最高分辨率特征
        feat = decoder_features[0]
        
        pd_logits = self.pd_classifier(feat)
        stage_logits = self.stage_classifier(feat)
        subtype_logits = self.subtype_classifier(feat)
        
        return {
            'pd_logits': pd_logits,
            'stage_logits': stage_logits,
            'subtype_logits': subtype_logits,
        }


class DualStreamClassifier(nn.Module):
    """
    双流分类器
    
    Stream 1: 分割特征 (Branch A输出)
    Stream 2: 解码器特征 (Branch B的原始特征)
    
    两个流各自产生分类，然后融合
    """
    
    def __init__(self, segmentation_channel: int,
                 decoder_features_channels: List[int],
                 num_pd_classes: int = 2,
                 num_stage_classes: int = 5,
                 num_subtype_classes: int = 3,
                 hidden_dim: int = 512,
                 dropout_rate: float = 0.5,
                 fusion_method: str = 'concat'):
        super().__init__()
        
        self.fusion_method = fusion_method
        
        # Stream A: 分割特征
        self.seg_pd_classifier = ClassificationHead(
            segmentation_channel, num_pd_classes, hidden_dim, dropout_rate
        )
        self.seg_stage_classifier = ClassificationHead(
            segmentation_channel, num_stage_classes, hidden_dim, dropout_rate
        )
        self.seg_subtype_classifier = ClassificationHead(
            segmentation_channel, num_subtype_classes, hidden_dim, dropout_rate
        )
        
        # Stream B: 解码器特征
        feature_channel = decoder_features_channels[0]
        self.dec_pd_classifier = ClassificationHead(
            feature_channel, num_pd_classes, hidden_dim, dropout_rate
        )
        self.dec_stage_classifier = ClassificationHead(
            feature_channel, num_stage_classes, hidden_dim, dropout_rate
        )
        self.dec_subtype_classifier = ClassificationHead(
            feature_channel, num_subtype_classes, hidden_dim, dropout_rate
        )
        
        # 融合层 (如果需要)
        if fusion_method == 'concat':
            fusion_input_dim = num_pd_classes * 2
            self.pd_fusion = nn.Linear(fusion_input_dim, num_pd_classes)
            
            fusion_input_dim = num_stage_classes * 2
            self.stage_fusion = nn.Linear(fusion_input_dim, num_stage_classes)
            
            fusion_input_dim = num_subtype_classes * 2
            self.subtype_fusion = nn.Linear(fusion_input_dim, num_subtype_classes)
    
    def forward(self, segmentation: torch.Tensor,
                decoder_features: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        segmentation : torch.Tensor
            分割输出 [B, 1, H, W]
        decoder_features : List[torch.Tensor]
            解码器特征列表
        
        Returns
        -------
        Dict[str, torch.Tensor]
            双流分类输出字典
        """
        # Stream A 分类
        seg_pd_logits = self.seg_pd_classifier(segmentation)
        seg_stage_logits = self.seg_stage_classifier(segmentation)
        seg_subtype_logits = self.seg_subtype_classifier(segmentation)
        
        # Stream B 分类
        dec_feat = decoder_features[0]
        dec_pd_logits = self.dec_pd_classifier(dec_feat)
        dec_stage_logits = self.dec_stage_classifier(dec_feat)
        dec_subtype_logits = self.dec_subtype_classifier(dec_feat)
        
        # 融合
        if self.fusion_method == 'concat':
            pd_logits = self.pd_fusion(
                torch.cat([seg_pd_logits, dec_pd_logits], dim=1)
            )
            stage_logits = self.stage_fusion(
                torch.cat([seg_stage_logits, dec_stage_logits], dim=1)
            )
            subtype_logits = self.subtype_fusion(
                torch.cat([seg_subtype_logits, dec_subtype_logits], dim=1)
            )
        elif self.fusion_method == 'mean':
            pd_logits = (seg_pd_logits + dec_pd_logits) / 2
            stage_logits = (seg_stage_logits + dec_stage_logits) / 2
            subtype_logits = (seg_subtype_logits + dec_subtype_logits) / 2
        else:
            raise ValueError(f"未知的融合方法: {self.fusion_method}")
        
        return {
            'pd_logits': pd_logits,
            'stage_logits': stage_logits,
            'subtype_logits': subtype_logits,
            'seg_pd_logits': seg_pd_logits,
            'seg_stage_logits': seg_stage_logits,
            'seg_subtype_logits': seg_subtype_logits,
            'dec_pd_logits': dec_pd_logits,
            'dec_stage_logits': dec_stage_logits,
            'dec_subtype_logits': dec_subtype_logits,
        }


class MixedSingleStreamClassifier(nn.Module):
    """
    混合单流分类器
    
    融合分割特征和解码器特征，然后进行分类
    """
    
    def __init__(self, segmentation_channel: int,
                 decoder_features_channels: List[int],
                 num_pd_classes: int = 2,
                 num_stage_classes: int = 5,
                 num_subtype_classes: int = 3,
                 hidden_dim: int = 512,
                 dropout_rate: float = 0.5,
                 fusion_method: str = 'concat'):
        super().__init__()
        
        self.fusion_method = fusion_method
        dec_channel = decoder_features_channels[0]
        
        if fusion_method == 'concat':
            # 拼接分割和解码器特征
            self.fusion_conv = nn.Sequential(
                nn.Conv2d(segmentation_channel + dec_channel,
                         dec_channel, kernel_size=1),
                nn.BatchNorm2d(dec_channel),
                nn.ReLU(inplace=True),
            )
            fused_channel = dec_channel
        elif fusion_method == 'add':
            # 相加融合 (需要通道数相同)
            if segmentation_channel != dec_channel:
                self.seg_align = nn.Conv2d(segmentation_channel, dec_channel, 1)
            else:
                self.seg_align = nn.Identity()
            fused_channel = dec_channel
        else:
            raise ValueError(f"未知的融合方法: {fusion_method}")
        
        # 三个分类头
        self.pd_classifier = ClassificationHead(
            fused_channel, num_pd_classes, hidden_dim, dropout_rate
        )
        self.stage_classifier = ClassificationHead(
            fused_channel, num_stage_classes, hidden_dim, dropout_rate
        )
        self.subtype_classifier = ClassificationHead(
            fused_channel, num_subtype_classes, hidden_dim, dropout_rate
        )
    
    def forward(self, segmentation: torch.Tensor,
                decoder_features: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        segmentation : torch.Tensor
            分割输出 [B, 1, H, W]
        decoder_features : List[torch.Tensor]
            解码器特征列表
        
        Returns
        -------
        Dict[str, torch.Tensor]
            混合分类输出字典
        """
        dec_feat = decoder_features[0]
        
        if self.fusion_method == 'concat':
            # 确保空间维度一致
            if segmentation.shape[2:] != dec_feat.shape[2:]:
                seg = F.interpolate(segmentation, size=dec_feat.shape[2:],
                                   mode='bilinear', align_corners=False)
            else:
                seg = segmentation
            
            fused = torch.cat([seg, dec_feat], dim=1)
            fused = self.fusion_conv(fused)
        elif self.fusion_method == 'add':
            seg = F.interpolate(segmentation, size=dec_feat.shape[2:],
                               mode='bilinear', align_corners=False)
            seg = self.seg_align(seg)
            fused = seg + dec_feat
        
        pd_logits = self.pd_classifier(fused)
        stage_logits = self.stage_classifier(fused)
        subtype_logits = self.subtype_classifier(fused)
        
        return {
            'pd_logits': pd_logits,
            'stage_logits': stage_logits,
            'subtype_logits': subtype_logits,
        }


class ClassificationBranch(nn.Module):
    """
    分类分支 (Branch B) - 完整实现
    
    根据指定的架构模式选择对应的分类器
    """
    
    def __init__(self, mode: Literal['single_stream', 'dual_stream', 'mixed_single_stream'],
                 segmentation_channel: int = 1,
                 decoder_features_channels: List[int] = None,
                 num_pd_classes: int = 2,
                 num_stage_classes: int = 5,
                 num_subtype_classes: int = 3,
                 hidden_dim: int = 512,
                 dropout_rate: float = 0.5):
        super().__init__()
        
        self.mode = mode
        
        if decoder_features_channels is None:
            decoder_features_channels = [256, 512]
        
        if mode == 'single_stream':
            self.classifier = SingleStreamClassifier(
                decoder_features_channels, num_pd_classes,
                num_stage_classes, num_subtype_classes,
                hidden_dim, dropout_rate
            )
        elif mode == 'dual_stream':
            self.classifier = DualStreamClassifier(
                segmentation_channel, decoder_features_channels,
                num_pd_classes, num_stage_classes, num_subtype_classes,
                hidden_dim, dropout_rate, fusion_method='concat'
            )
        elif mode == 'mixed_single_stream':
            self.classifier = MixedSingleStreamClassifier(
                segmentation_channel, decoder_features_channels,
                num_pd_classes, num_stage_classes, num_subtype_classes,
                hidden_dim, dropout_rate, fusion_method='concat'
            )
        else:
            raise ValueError(f"未知的分类模式: {mode}")
        
        # 损失函数
        self.pd_loss = nn.CrossEntropyLoss()
        self.stage_loss = nn.CrossEntropyLoss(ignore_index=-1)  # 忽略无效类
        self.subtype_loss = nn.CrossEntropyLoss(ignore_index=-1)
    
    def forward(self, segmentation: torch.Tensor = None,
                decoder_features: List[torch.Tensor] = None,
                pd_labels: torch.Tensor = None,
                stage_labels: torch.Tensor = None,
                subtype_labels: torch.Tensor = None) -> Dict:
        """
        Parameters
        ----------
        segmentation : torch.Tensor, optional
            分割输出 [B, 1, H, W]
        decoder_features : List[torch.Tensor], optional
            解码器特征列表
        pd_labels : torch.Tensor, optional
            PD标签 [B]
        stage_labels : torch.Tensor, optional
            HY分级标签 [B]
        subtype_labels : torch.Tensor, optional
            亚型标签 [B]
        
        Returns
        -------
        Dict
            分类输出和损失
        """
        if self.mode == 'single_stream':
            outputs = self.classifier(decoder_features)
        else:  # dual_stream 或 mixed_single_stream
            outputs = self.classifier(segmentation, decoder_features)
        
        # 计算损失
        losses = {}
        if pd_labels is not None:
            losses['pd'] = self.pd_loss(outputs['pd_logits'], pd_labels)
        if stage_labels is not None:
            losses['stage'] = self.stage_loss(outputs['stage_logits'], stage_labels)
        if subtype_labels is not None:
            losses['subtype'] = self.subtype_loss(outputs['subtype_logits'], subtype_labels)
        
        outputs['losses'] = losses
        return outputs
