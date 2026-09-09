"""
dual_view_model.py - 双视角单任务 CNN 模型

架构（每个任务独立实例化一个模型，与 basic_experiments 对齐）
----
    en_img   → backbone_en  → [B, cnn_dim]  ─┬
    unet_img → backbone_unet → [B, cnn_dim] ─┤→ concat [B, 2*cnn_dim]
                                               → Fusion MLP (Linear + LN + ReLU + Dropout)
                                               └→ head (num_classes)

参数
----
- task: 'pd' | 'stage' | 'subtype'，决定分类头维度
- shared_backbone=True:  backbone_en 与 backbone_unet 共享权重（减少参数）
- shared_backbone=False: 两个独立的 backbone（允许各自适配不同图像风格）
"""

import torch
import torch.nn as nn

from core.basic_models import get_backbone, get_backbone_feature_dim


DUAL_VIEW_TASK_CONFIG = {
    'pd':      {'num_classes': 2, 'name': 'PD_Classification',      'class_names': ['Non-PD', 'PD']},
    'stage':   {'num_classes': 3, 'name': 'Stage_Classification',   'class_names': ['Stage 1', 'Stage 2', 'Stage 3']},
    'subtype': {'num_classes': 3, 'name': 'Subtype_Classification', 'class_names': ['TD', 'PIGD', 'Mixed']},
    'pd_subtype_tdrest': {'num_classes': 2, 'name': 'Subtype_TDvsRest', 'class_names': ['TD', 'rest']},
    'pd_subtype_pigd_mixed': {'num_classes': 2, 'name': 'Subtype_PIGDvsMixed', 'class_names': ['PIGD', 'mixed']},
}


def _get_backbone(backbone: str, pretrained):
    """使用论文 ResNet18 骨干。"""
    return get_backbone(backbone, pretrained)


def _get_feature_dim(backbone: str) -> int:
    return get_backbone_feature_dim(backbone)


class DualViewModel(nn.Module):
    """
    双视角单任务分类模型（en 原始图像 + unet 神经纤维图像）。
    每个任务独立训练一个模型实例，与 basic_experiments 对齐。

    Args:
        task:              任务名 ('pd' | 'stage' | 'subtype')
        backbone:          backbone 名称 ('resnet18')
        pretrained:        预训练模式 (True / False / 'Half')
        shared_backbone:   True → en 和 unet 共享同一组 backbone 权重
        fusion_dim:        融合 MLP 隐层维度
        dropout:           融合层 dropout
        classifier_dropout: 分类头 dropout
    """

    def __init__(
        self,
        task: str = 'pd',
        backbone: str = 'resnet18',
        pretrained=True,
        shared_backbone: bool = True,
        fusion_dim: int = 256,
        dropout: float = 0.3,
        classifier_dropout: float = 0.3,
    ):
        super().__init__()
        self.task = task
        self.shared_backbone = shared_backbone
        num_classes = DUAL_VIEW_TASK_CONFIG[task]['num_classes']
        cnn_dim = _get_feature_dim(backbone)

        self.backbone_en = _get_backbone(backbone, pretrained)
        if shared_backbone:
            self.backbone_unet = self.backbone_en   # 共享权重
        else:
            self.backbone_unet = _get_backbone(backbone, pretrained)

        fused_dim = cnn_dim * 2

        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.head = nn.Sequential(
            nn.Dropout(classifier_dropout),
            nn.Linear(fusion_dim, num_classes),
        )

    def forward(
        self,
        en_img: torch.Tensor,
        unet_img: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            en_img:   [B, C, H, W]  原始眼底图像
            unet_img: [B, C, H, W]  UNet 神经纤维图像
        Returns:
            logits [B, num_classes]
        """
        f_en = self.backbone_en(en_img)        # [B, cnn_dim]
        f_unet = self.backbone_unet(unet_img)  # [B, cnn_dim]

        fused = self.fusion(torch.cat([f_en, f_unet], dim=1))  # [B, fusion_dim]

        return self.head(fused)  # [B, num_classes]
