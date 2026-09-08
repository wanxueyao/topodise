"""
light_models.py - 轻量化神经网络模型定义

模块职责
--------
定义轻量化骨干网络用于帕金森病分类，专为小样本高过拟合场景设计:
- Backbone 管理: MobileViT-XXS/XS, VanillaNet-5, EfficientNet-B0
- 模型类: 单流和双流分类模型，包含增强正则化
- 序数回归损失: 针对 Stage 任务的有序分类

核心设计原则
----------
- 参数量远小于 ResNet，降低过拟合风险
- 内置 Dropout (0.3~0.5) 正则化
- Stage 任务支持 Label Smoothing 和 Ordinal Regression Loss

支持的 Backbone
---------------
- mobilevit_xxs: ~1.3M 参数，输入 256×256，轻量 ViT
- mobilevit_xs: ~2.3M 参数，输入 256×256，轻量 ViT
- vanillanet_5: ~22M 参数（act_learn 激活，原生实现），输入 224×224，特征维 4096
- efficientnet_b0: ~5.3M 参数，输入 224×224，推荐小数据首选

输入尺寸
--------
- MobileViT: 256×256 (官方推荐)
- VanillaNet-5: 224×224
- EfficientNet-B0: 224×224
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, Optional

try:
    import timm
    _TIMM_AVAILABLE = True
except ImportError:
    _TIMM_AVAILABLE = False

from core.vanillanet import vanillanet_5 as _vanillanet_5_factory, VanillaNet5

PretrainedMode = Literal[True, False, 'Half']


# ==================== Backbone 配置表 ====================

LIGHT_BACKBONE_CONFIG = {
    'mobilevit_xxs': {
        'timm_name': 'mobilevit_xxs',
        'feature_dim': 320,
        'input_size': 256,
        'freeze_stages': 2,
    },
    'mobilevit_xs': {
        'timm_name': 'mobilevit_xs',
        'feature_dim': 384,
        'input_size': 256,
        'freeze_stages': 2,
    },
    'vanillanet_5': {
        # 使用本地原生实现 (core/vanillanet.py)，不依赖 timm
        'timm_name': None,
        'feature_dim': VanillaNet5.FEATURE_DIM,   # 4096
        'input_size': 224,
        'freeze_stages': 1,
    },
    'efficientnet_b0': {
        'timm_name': 'efficientnet_b0',
        'feature_dim': 1280,
        'input_size': 224,
        'freeze_stages': 3,
    },
}


# ==================== 序数回归损失 ====================

class OrdinalRegressionLoss(nn.Module):
    """
    序数回归损失 (适用于有序多分类任务)

    将 K 类有序问题建模为 K-1 个二分类问题:
    - p(y > k) 对 k = 0, ..., K-2
    - 输出层维度为 K-1 (而非 K)

    优势: 保留类别间有序关系，比纯 CrossEntropy 更适合 Stage 1<2<3 场景。

    Args:
        num_classes: 类别总数 K
        reduction: 损失聚合方式 ('mean' / 'sum')

    前向传播 (forward):
        logits: (B, K-1) 每个阈值的 sigmoid logit
        targets: (B,) 有序类别标签 [0, K-1]
    """

    def __init__(self, num_classes: int = 3, reduction: str = 'mean'):
        super().__init__()
        self.num_classes = num_classes
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        K = self.num_classes
        device = logits.device
        target_bin = torch.zeros(targets.size(0), K - 1, device=device)
        for k in range(K - 1):
            target_bin[:, k] = (targets > k).float()

        loss = F.binary_cross_entropy_with_logits(
            logits, target_bin, reduction=self.reduction
        )
        return loss

    @staticmethod
    def logits_to_probs(logits: torch.Tensor) -> torch.Tensor:
        """将 OrdinalRegression logits 转为各类别概率"""
        cum_probs = torch.sigmoid(logits)  # (B, K-1)
        B, Km1 = cum_probs.shape
        K = Km1 + 1

        probs = torch.zeros(B, K, device=logits.device)
        probs[:, 0] = 1.0 - cum_probs[:, 0]
        for k in range(1, K - 1):
            probs[:, k] = cum_probs[:, k - 1] - cum_probs[:, k]
        probs[:, K - 1] = cum_probs[:, K - 2]

        probs = probs.clamp(min=1e-7)
        probs = probs / probs.sum(dim=1, keepdim=True)
        return probs

    @staticmethod
    def logits_to_preds(logits: torch.Tensor) -> torch.Tensor:
        """将 logits 转为类别预测"""
        return OrdinalRegressionLoss.logits_to_probs(logits).argmax(dim=1)


# ==================== 辅助函数 ====================

def _check_timm():
    """检查 timm 库可用性"""
    if not _TIMM_AVAILABLE:
        raise ImportError(
            "light_models 需要 timm 库，请安装: pip install timm"
        )


def get_image_size_light(backbone: str, pretrained: PretrainedMode = True) -> tuple:
    """
    获取轻量化 backbone 的图像预处理尺寸

    Returns:
        (resize_size, crop_size, use_original)
    """
    if pretrained is False:
        return (None, None, True)

    cfg = LIGHT_BACKBONE_CONFIG.get(backbone)
    if cfg is None:
        raise ValueError(
            f"未知轻量化 backbone: {backbone}，可选: {list(LIGHT_BACKBONE_CONFIG.keys())}"
        )

    size = cfg['input_size']
    resize = int(size * 1.14)
    return (resize, size, False)


def get_light_backbone_feature_dim(backbone: str) -> int:
    """获取轻量化 backbone 输出特征维度"""
    cfg = LIGHT_BACKBONE_CONFIG.get(backbone)
    if cfg is None:
        raise ValueError(f"未知轻量化 backbone: {backbone}")
    return cfg['feature_dim']


def _freeze_half_light(model: nn.Module, backbone: str):
    """冻结轻量化模型的前 N 个 stage"""
    cfg = LIGHT_BACKBONE_CONFIG[backbone]
    n_freeze = cfg.get('freeze_stages', 2)

    children = list(model.named_children())
    for i, (_, module) in enumerate(children):
        if i < n_freeze:
            for param in module.parameters():
                param.requires_grad = False


def get_light_backbone(backbone: str, pretrained: PretrainedMode = True) -> nn.Module:
    """
    创建轻量化 backbone 特征提取器

    - MobileViT-XXS/XS, EfficientNet-B0: 通过 timm 加载
    - VanillaNet-5: 使用本地原生实现 (core/vanillanet.py)，不依赖 timm

    Args:
        backbone: backbone 名称
        pretrained: True / False / 'Half'

    Returns:
        特征提取器，输出为 1D 特征向量
    """
    cfg = LIGHT_BACKBONE_CONFIG.get(backbone)
    if cfg is None:
        raise ValueError(
            f"未知轻量化 backbone: {backbone}，可选: {list(LIGHT_BACKBONE_CONFIG.keys())}"
        )

    # vanillanet_5: 走本地原生实现
    if backbone == 'vanillanet_5':
        use_pretrained = pretrained is not False
        model = _vanillanet_5_factory(pretrained=use_pretrained)
        if pretrained == 'Half':
            _freeze_half_light(model, backbone)
        return model

    # 其余: 走 timm
    _check_timm()
    use_pretrained = pretrained is not False
    model = timm.create_model(
        cfg['timm_name'],
        pretrained=use_pretrained,
        num_classes=0,
        global_pool='avg',
    )
    if pretrained == 'Half':
        _freeze_half_light(model, backbone)
    return model


# ==================== 分类头 ====================

class LightClassifierHead(nn.Module):
    """
    强正则化分类头

    结构: Linear → BN → GELU → Dropout → Linear

    Args:
        feature_dim: 输入特征维度
        num_classes: 输出类别数
        dropout1: 第一个 Dropout 概率
        dropout2: 第二个 Dropout 概率 (当前未使用，保留接口)
        use_ordinal: 是否使用序数回归输出 (输出维度为 num_classes-1)
    """

    def __init__(self, feature_dim: int, num_classes: int,
                 dropout1: float = 0.5, dropout2: float = 0.3,
                 use_ordinal: bool = False):
        super().__init__()
        self.use_ordinal = use_ordinal
        out_dim = (num_classes - 1) if use_ordinal else num_classes

        self.head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout1),
            nn.Linear(256, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


# ==================== 单流模型 ====================

class LightSingleTaskModel(nn.Module):
    """
    轻量化单流分类模型

    结构: LightBackbone → Dropout → LightClassifierHead

    Args:
        backbone: 轻量化 backbone 名称
        num_classes: 输出类别数
        pretrained: 预训练模式 (True / False / 'Half')
        dropout: 特征向量后的 Dropout 概率 (建议 0.3~0.5)
        use_ordinal: 是否使用序数回归输出 (适用于 Stage 任务)
    """

    def __init__(self, backbone: str, num_classes: int,
                 pretrained: PretrainedMode = True,
                 dropout: float = 0.4,
                 use_ordinal: bool = False):
        super().__init__()
        self.use_ordinal = use_ordinal
        self.backbone = get_light_backbone(backbone, pretrained)
        dim = get_light_backbone_feature_dim(backbone)

        self.feat_drop = nn.Dropout(dropout)
        self.classifier = LightClassifierHead(
            dim, num_classes,
            dropout1=dropout,
            dropout2=max(0.0, dropout - 0.1),
            use_ordinal=use_ordinal,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        feat = self.feat_drop(feat)
        return self.classifier(feat)

    def set_act_num(self, n: int):
        """将 act_num 调度透传到 VanillaNet backbone（其他 backbone 调用无效）"""
        if hasattr(self.backbone, 'set_act_num'):
            self.backbone.set_act_num(n)


# ==================== 双流模型 ====================

class LightDualStreamModel(nn.Module):
    """
    轻量化双流分类模型 (central + whorl)

    结构:
    - Central Backbone → f_c
    - Whorl Backbone   → f_w
    - Concat(f_c, f_w) → Dropout → LightClassifierHead

    两个 backbone 结构相同但参数独立（不共享权重）。

    Args:
        backbone: 轻量化 backbone 名称
        num_classes: 输出类别数
        pretrained: 预训练模式
        dropout: 融合层 Dropout 概率
        use_ordinal: 是否使用序数回归输出
    """

    def __init__(self, backbone: str, num_classes: int,
                 pretrained: PretrainedMode = True,
                 dropout: float = 0.5,
                 use_ordinal: bool = False):
        super().__init__()
        self.use_ordinal = use_ordinal
        self.backbone_central = get_light_backbone(backbone, pretrained)
        self.backbone_whorl = get_light_backbone(backbone, pretrained)

        dim = get_light_backbone_feature_dim(backbone) * 2  # 拼接后翻倍

        self.feat_drop = nn.Dropout(dropout)
        self.classifier = LightClassifierHead(
            dim, num_classes,
            dropout1=dropout,
            dropout2=max(0.0, dropout - 0.1),
            use_ordinal=use_ordinal,
        )

    def forward(self, central: torch.Tensor, whorl: torch.Tensor) -> torch.Tensor:
        f_c = self.backbone_central(central)
        f_w = self.backbone_whorl(whorl)
        feat = torch.cat([f_c, f_w], dim=1)
        feat = self.feat_drop(feat)
        return self.classifier(feat)

    def set_act_num(self, n: int):
        """将 act_num 调度同时透传到两路 VanillaNet backbone"""
        if hasattr(self.backbone_central, 'set_act_num'):
            self.backbone_central.set_act_num(n)
        if hasattr(self.backbone_whorl, 'set_act_num'):
            self.backbone_whorl.set_act_num(n)
