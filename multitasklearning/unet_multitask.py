"""
U-Net 多任务学习主架构
====================

完整的多任务学习模型:
- 左侧: Encoder (特征提取)
- 右侧: 两个分支
  - Branch A: 分割/MAE重构 (神经骨架)
  - Branch B: 分类 (PD分类)

架构流程:
Input -> Encoder -> Decoder -> [Branch A (分割), Branch B (分类)]
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Literal
import logging

from .encoder import UNetEncoder
from .segmentation_branch import SegmentationBranch, SegmentationDecoder
from .classification_branch import ClassificationBranch

logger = logging.getLogger(__name__)


class UNetMultiTask(nn.Module):
    """
    U-Net 双分支多任务学习模型
    
    Parameters
    ----------
    encoder_type : str
        Backbone编码器类型 (resnet50, resnet101, efficientnet_b4)
    encoder_pretrained : bool
        是否使用预训练权重
    classification_mode : str
        分类架构模式 (single_stream, dual_stream, mixed_single_stream)
    num_pd_classes : int
        PD分类类别数
    num_stage_classes : int
        HY分级类别数
    num_subtype_classes : int
        PD亚型类别数
    mae_enabled : bool
        是否启用MAE自监督学习
    dropout_rate : float
        Dropout率
    """
    
    def __init__(self,
                 encoder_type: str = 'resnet50',
                 encoder_pretrained: bool = True,
                 classification_mode: Literal['single_stream', 'dual_stream', 'mixed_single_stream'] = 'dual_stream',
                 num_pd_classes: int = 2,
                 num_stage_classes: int = 5,
                 num_subtype_classes: int = 3,
                 mae_enabled: bool = True,
                 dropout_rate: float = 0.5,
                 input_channels: int = 3):
        super().__init__()
        
        self.encoder_type = encoder_type
        self.classification_mode = classification_mode
        self.mae_enabled = mae_enabled
        
        # ==================== 编码器 ====================
        self.encoder = UNetEncoder(encoder_type, encoder_pretrained, input_channels)
        
        # 获取编码器特征维度
        encoder_dims = self.encoder.feature_dims
        logger.info(f"编码器特征维度: {encoder_dims}")
        
        # ==================== 分支A: 分割/重建 ====================
        self.segmentation_branch = SegmentationBranch(
            encoder_dims=encoder_dims,
            output_channels=1,
            patch_size=16,
            mae_mask_ratio=0.75,
            mae_embedding_dim=768
        )
        
        # ==================== 分支B: 分类 ====================
        # _get_decoder_features 返回 encoder_features[-3:]，
        # 对应通道数为 encoder_dims[-3:]（而非 encoder_dims[:-1]）。
        self.classification_branch = ClassificationBranch(
            mode=classification_mode,
            segmentation_channel=1,
            decoder_features_channels=list(encoder_dims[-3:]),
            num_pd_classes=num_pd_classes,
            num_stage_classes=num_stage_classes,
            num_subtype_classes=num_subtype_classes,
            hidden_dim=512,
            dropout_rate=dropout_rate
        )
        
        logger.info(f"多任务模型构建完成")
        logger.info(f"  编码器: {encoder_type}")
        logger.info(f"  分类模式: {classification_mode}")
        logger.info(f"  MAE: {'启用' if mae_enabled else '禁用'}")
    
    def forward(self,
                images: torch.Tensor,
                segmentation_targets: torch.Tensor = None,
                pd_labels: torch.Tensor = None,
                stage_labels: torch.Tensor = None,
                subtype_labels: torch.Tensor = None,
                use_mae: bool = True,
                training: bool = True) -> Dict:
        """
        前向传播
        
        Parameters
        ----------
        images : torch.Tensor
            输入图像 [B, 3, H, W]
        segmentation_targets : torch.Tensor, optional
            分割真值 [B, 1, H, W]
        pd_labels : torch.Tensor, optional
            PD标签 [B]
        stage_labels : torch.Tensor, optional
            HY分级标签 [B]
        subtype_labels : torch.Tensor, optional
            亚型标签 [B]
        use_mae : bool
            是否使用MAE
        training : bool
            是否为训练模式
        
        Returns
        -------
        Dict
            包含所有输出和损失的字典
        """
        outputs = {}
        
        # ==================== 编码器 ====================
        encoder_features = self.encoder(images)
        
        # ==================== 分支A: 分割/重建 ====================
        seg_outputs = self.segmentation_branch(
            features=encoder_features,
            images=images,
            segmentation_targets=segmentation_targets,
            use_mae=use_mae and self.mae_enabled and training,
            use_segmentation=True
        )
        
        outputs['segmentation'] = seg_outputs['segmentation']
        if 'mae_reconstructed' in seg_outputs:
            outputs['mae_reconstructed'] = seg_outputs['mae_reconstructed']
        
        # ==================== 分支B: 分类 ====================
        # 获取解码器特征 (从分割分支的解码器)
        # 这里我们需要额外获取分类所需的特征
        decoder_features = self._get_decoder_features(encoder_features)
        
        cls_outputs = self.classification_branch(
            segmentation=seg_outputs['segmentation'],
            decoder_features=decoder_features,
            pd_labels=pd_labels,
            stage_labels=stage_labels,
            subtype_labels=subtype_labels
        )
        
        outputs['pd_logits'] = cls_outputs['pd_logits']
        outputs['stage_logits'] = cls_outputs['stage_logits']
        outputs['subtype_logits'] = cls_outputs['subtype_logits']
        
        # 双流特有的输出
        if self.classification_mode == 'dual_stream':
            outputs.update({
                'seg_pd_logits': cls_outputs.get('seg_pd_logits'),
                'seg_stage_logits': cls_outputs.get('seg_stage_logits'),
                'seg_subtype_logits': cls_outputs.get('seg_subtype_logits'),
                'dec_pd_logits': cls_outputs.get('dec_pd_logits'),
                'dec_stage_logits': cls_outputs.get('dec_stage_logits'),
                'dec_subtype_logits': cls_outputs.get('dec_subtype_logits'),
            })
        
        # ==================== 合并损失 ====================
        all_losses = {}
        
        # 分割损失
        if 'losses' in seg_outputs:
            for key, val in seg_outputs['losses'].items():
                all_losses[f'seg_{key}'] = val
        
        # 分类损失
        if 'losses' in cls_outputs:
            for key, val in cls_outputs['losses'].items():
                all_losses[f'cls_{key}'] = val
        
        outputs['losses'] = all_losses
        
        return outputs
    
    def _get_decoder_features(self, encoder_features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        从编码器特征生成解码器特征
        
        这是简化实现，实际可能需要额外的解码模块
        """
        # 返回编码器的高层特征，用于分类
        # 通常我们会使用最后几层的特征
        return encoder_features[-3:]  # 返回最后3层
    
    def get_segmentation(self, images: torch.Tensor) -> torch.Tensor:
        """获取分割输出 (推理)"""
        with torch.no_grad():
            outputs = self.forward(images, training=False, use_mae=False)
        return outputs['segmentation']
    
    def get_classification(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """获取分类输出 (推理)"""
        with torch.no_grad():
            outputs = self.forward(images, training=False, use_mae=False)
        return {
            'pd_logits': outputs['pd_logits'],
            'stage_logits': outputs['stage_logits'],
            'subtype_logits': outputs['subtype_logits'],
        }
    
    def get_pruned_model(self, remove_mae: bool = False) -> 'UNetMultiTask':
        """
        获取精简版本的模型 (用于推理)
        
        Parameters
        ----------
        remove_mae : bool
            是否移除MAE模块
        
        Returns
        -------
        UNetMultiTask
            精简后的模型
        """
        if remove_mae:
            self.mae_enabled = False
        return self
    
    def count_parameters(self) -> Dict[str, int]:
        """计算参数数量"""
        def count_params(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)
        
        return {
            'encoder': count_params(self.encoder),
            'segmentation_branch': count_params(self.segmentation_branch),
            'classification_branch': count_params(self.classification_branch),
            'total': count_params(self),
        }


class MTLBuilder:
    """多任务学习模型构建工厂"""
    
    @staticmethod
    def build_lightweight() -> UNetMultiTask:
        """构建轻量级模型"""
        return UNetMultiTask(
            encoder_type='resnet18',
            encoder_pretrained=True,
            classification_mode='single_stream',
            dropout_rate=0.3,
        )
    
    @staticmethod
    def build_standard() -> UNetMultiTask:
        """构建标准模型"""
        return UNetMultiTask(
            encoder_type='resnet50',
            encoder_pretrained=True,
            classification_mode='dual_stream',
            dropout_rate=0.5,
        )
    
    @staticmethod
    def build_large() -> UNetMultiTask:
        """构建大规模模型"""
        return UNetMultiTask(
            encoder_type='efficientnet_b4',
            encoder_pretrained=True,
            classification_mode='dual_stream',
            dropout_rate=0.6,
        )
    
    @staticmethod
    def build_custom(config) -> UNetMultiTask:
        """根据配置构建模型"""
        return UNetMultiTask(
            encoder_type=config.encoder_type,
            encoder_pretrained=config.encoder_pretrained,
            classification_mode=config.classification_mode,
            num_pd_classes=2,
            num_stage_classes=5,
            num_subtype_classes=config.num_pd_subtypes,
            mae_enabled=True,
            dropout_rate=0.5,
        )


if __name__ == '__main__':
    # 测试模型
    model = UNetMultiTask(
        encoder_type='resnet50',
        encoder_pretrained=False,  # 测试用
        classification_mode='dual_stream'
    )
    
    # 打印模型参数
    params = model.count_parameters()
    print("模型参数数量:")
    for key, val in params.items():
        print(f"  {key}: {val:,}")
    
    # 前向传播测试
    B, C, H, W = 2, 3, 256, 256
    images = torch.randn(B, C, H, W)
    
    # 训练模式
    outputs = model(images, training=True, use_mae=True)
    print("\n训练模式输出:")
    print(f"  分割: {outputs['segmentation'].shape}")
    print(f"  PD分类: {outputs['pd_logits'].shape}")
    print(f"  HY分级: {outputs['stage_logits'].shape}")
    print(f"  亚型分类: {outputs['subtype_logits'].shape}")
    
    # 推理模式
    seg = model.get_segmentation(images)
    print(f"\n推理模式分割: {seg.shape}")
    
    cls = model.get_classification(images)
    print(f"推理模式分类: PD={cls['pd_logits'].shape}")
