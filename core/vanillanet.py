"""
vanillanet.py - VanillaNet-5 原生 PyTorch 实现

基于论文: "VanillaNet: the Power of Minimalism in Deep Learning" (arXiv 2305.12972)
官方代码: https://github.com/huawei-noah/VanillaNet

设计原则
--------
VanillaNet 的极简性体现在:
- 无 skip connection（纯直连 conv stack）
- 无 attention / multi-head
- 使用 "act_learn": 可学习缩放的 ReLU 系列（act_num+1 个并联 BN 分支）
- 训练时双分支结构（减少梯度消失），推理时可合并为单分支（structural re-param）

VanillaNet-5 结构
-----------------
  stem  (Conv 4×4, s4):  3 → 512 channels
  block1 (Conv 1×1 + Pool s2): 512  → 1024
  block2 (Conv 1×1 + Pool s2): 1024 → 2048
  block3 (Conv 1×1 + Pool s2): 2048 → 4096
  AdaptiveAvgPool → flatten → (B, 4096)

参数量: ~22M  |  推荐输入: 224×224  |  特征维度: 4096

预训练权重
----------
官方发布于 GitHub Releases，可用 load_pretrained_vanillanet5() 下载加载。
若网络不可达，则静默跳过（返回随机初始化模型）。
"""
import math
import os
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# 官方预训练权重 URL（vanillanet_5.pth，ImageNet-1k）
_PRETRAINED_URL = (
    "https://github.com/huawei-noah/VanillaNet/releases/download/ckpt/vanillanet_5.pth"
)
_PRETRAINED_CACHE = os.path.join(
    os.path.expanduser("~"), ".cache", "vanillanet", "vanillanet_5.pth"
)


# ==================== 可学习激活函数 ====================

class _ActLearn(nn.Module):
    """
    VanillaNet 可学习激活函数 (act_learn)

    核心思想: 对输入分别经过 act_num+1 个独立 BN 后加权求和、再过 ReLU。
    这等价于在不同"阈值"下的 ReLU 线性组合，增强了模型表达力。

    训练时: 多 BN 并联 → 丰富梯度流
    推理时: 可将多 BN 合并为单 BN (re-param)，此处保留训练结构以保证权重可加载。

    Args:
        dim: 通道数
        act_num: 额外 BN 数量（总 BN 数 = act_num + 1）
        deploy: 是否为推理简化模式（合并后单 BN）
    """

    def __init__(self, dim: int, act_num: int = 3, deploy: bool = False):
        super().__init__()
        self.act_num = act_num
        self._current_act_num = act_num   # 训练时动态递减
        self.deploy = deploy

        # 可学习权重: (act_num+1, dim, 1, 1)，通道维度独立缩放
        self.weight = nn.Parameter(torch.randn(act_num + 1, dim, 1, 1) * 0.01)

        if deploy:
            # 推理时只保留一个合并后的 BN
            self.bn = nn.BatchNorm2d(dim, eps=1e-6)
        else:
            # 训练时每个分支独立 BN
            self.bns = nn.ModuleList([
                nn.BatchNorm2d(dim, eps=1e-6) for _ in range(act_num + 1)
            ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.deploy:
            # 推理模式：等效为单 BN + 线性缩放（无 ReLU，对应 act_num=0 的极限形态）
            return self.bn(x) * self.weight.sum(0)
        else:
            # 训练模式：线性基础分支 + 动态数量的非线性分支
            # - branch 0: 只做 BN，不经 ReLU（线性基础，保证 act_num=0 时近似恒等变换）
            # - branch 1.._current_act_num: BN + ReLU（非线性增强，根据训练进度逐步关闭）
            n = self._current_act_num + 1   # 当前激活的分支数
            res = self.bns[0](x) * self.weight[0]   # 线性基础分支
            for i in range(1, n):
                res = res + F.relu(self.bns[i](x)) * self.weight[i]
            return res

    def set_act_num(self, n: int):
        """动态更新激活分支数（训练调度专用，不改变已分配的 BN 数量）"""
        self._current_act_num = max(0, min(n, self.act_num))


# ==================== Block ====================

class _Block(nn.Module):
    """
    VanillaNet Block

    训练时双分支 (conv1 + conv2)，推理时合并为单 conv（structural re-param）。
    Pool 在 conv 之后、act 之前。

    Args:
        in_channels: 输入通道
        out_channels: 输出通道
        act_num: act_learn 的 BN 数量
        stride: 下采样步幅（使用 MaxPool）
        deploy: 是否推理模式
    """

    def __init__(self, in_channels: int, out_channels: int,
                 act_num: int = 3, stride: int = 2, deploy: bool = False):
        super().__init__()
        self.deploy = deploy

        if deploy:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
            self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.pool = nn.MaxPool2d(stride, stride) if stride > 1 else nn.Identity()
        self.act = _ActLearn(out_channels, act_num, deploy=deploy)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.deploy:
            x = self.conv(x)
        else:
            x = self.conv1(x) + self.conv2(x)
        x = self.pool(x)
        x = self.act(x)
        return x

    def set_act_num(self, n: int):
        """透传 act_num 更新到内部 _ActLearn 模块"""
        self.act.set_act_num(n)


# ==================== VanillaNet-5 ====================

class VanillaNet5(nn.Module):
    """
    VanillaNet-5: 极简卷积网络特征提取器

    输出全局平均池化后的 4096 维特征向量。
    移除了原始分类头，适合迁移学习场景。

    Args:
        act_num: act_learn 内 BN 分支数 (默认 3，与官方一致)
        deploy: 是否使用推理简化模式
        in_channels: 输入通道数 (默认 3)

    输入: (B, 3, 224, 224)
    输出: (B, 4096)
    """

    FEATURE_DIM = 4096
    DIMS = [512, 1024, 2048, 4096]     # 128*4, 256*4, 512*4, 1024*4
    STRIDES = [2, 2, 2]

    def __init__(self, act_num: int = 3, deploy: bool = False, in_channels: int = 3):
        super().__init__()
        dims = self.DIMS
        self.deploy = deploy

        # Stem: Conv 4×4 stride 4
        if deploy:
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4),
                _ActLearn(dims[0], act_num, deploy=True),
            )
        else:
            # 双分支 stem，输出同维度叠加
            self.stem_conv1 = nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4)
            self.stem_conv2 = nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4)
            self.stem_act = _ActLearn(dims[0], act_num, deploy=False)

        # Blocks
        self.stages = nn.ModuleList([
            _Block(dims[i], dims[i + 1], act_num=act_num,
                   stride=self.STRIDES[i], deploy=deploy)
            for i in range(len(self.STRIDES))
        ])

        self.gap = nn.AdaptiveAvgPool2d(1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Parameter):
                nn.init.trunc_normal_(m, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stem
        if self.deploy:
            x = self.stem(x)
        else:
            x = self.stem_conv1(x) + self.stem_conv2(x)
            x = self.stem_act(x)

        # Blocks
        for stage in self.stages:
            x = stage(x)

        # Global average pool + flatten
        x = self.gap(x)
        x = x.flatten(1)
        return x

    def set_act_num(self, n: int):
        """
        更新所有 act_learn 模块的激活分支数（训练调度专用）

        随着训练进行，从 act_num 线性递减到 0：
        - act_num=3（初始）：4 个 BN 分支（1 线性 + 3 非线性），激活富有表达力，梯度充足
        - act_num=0（末尾）：仅 1 个 BN 分支（纯线性），模型接近极简形态
        """
        if hasattr(self, 'stem_act'):
            self.stem_act.set_act_num(n)
        for stage in self.stages:
            stage.set_act_num(n)


# ==================== 工厂函数 ====================

def vanillanet_5(pretrained: bool = False, deploy: bool = False) -> VanillaNet5:
    """
    创建 VanillaNet-5 特征提取器

    Args:
        pretrained: 是否尝试加载 ImageNet 预训练权重
            - True: 从官方 GitHub Releases 下载（~88MB）并缓存到 ~/.cache/vanillanet/
            - False: 随机初始化
        deploy: 是否使用推理简化模式（合并双分支）

    Returns:
        VanillaNet5 实例（无分类头，输出 4096 维特征）

    注意:
        预训练权重为原始带分类头的权重，加载时自动忽略分类头参数。
        若下载失败，会打印警告并回退到随机初始化。
    """
    model = VanillaNet5(deploy=deploy)

    if pretrained:
        _load_pretrained(model)

    return model


def _load_pretrained(model: VanillaNet5):
    """尝试加载官方预训练权重 (内部函数)"""
    import urllib.request

    os.makedirs(os.path.dirname(_PRETRAINED_CACHE), exist_ok=True)

    # 下载（如未缓存）
    if not os.path.exists(_PRETRAINED_CACHE):
        logger.info(f"下载 VanillaNet-5 预训练权重 → {_PRETRAINED_CACHE}")
        try:
            urllib.request.urlretrieve(_PRETRAINED_URL, _PRETRAINED_CACHE)
        except Exception as e:
            logger.warning(f"⚠ VanillaNet-5 预训练权重下载失败: {e}，使用随机初始化")
            return

    # 加载权重
    try:
        ckpt = torch.load(_PRETRAINED_CACHE, map_location='cpu', weights_only=False)

        # 官方权重可能包含 'model' 键
        state_dict = ckpt.get('model', ckpt)

        # 过滤: 忽略分类头 (head.*)，只加载特征提取部分
        model_keys = set(model.state_dict().keys())
        filtered = {
            k: v for k, v in state_dict.items()
            if k in model_keys and model.state_dict()[k].shape == v.shape
        }

        missing, unexpected = model.load_state_dict(filtered, strict=False)
        loaded = len(filtered)
        total = len(model_keys)
        logger.info(f"✓ 加载 VanillaNet-5 预训练权重: {loaded}/{total} 层"
                    f"（缺失 {len(missing)} 层，忽略 {len(unexpected)} 层）")
    except Exception as e:
        logger.warning(f"⚠ VanillaNet-5 权重加载失败: {e}，使用随机初始化")
