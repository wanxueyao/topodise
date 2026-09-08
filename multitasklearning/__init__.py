"""
多任务学习模块 - U-Net双分支架构
========================================

架构设计:
  Left (Encoder): 共享特征提取
  Right (Decoder):
    - Branch A: 神经骨架重建 (分割/MAE/伪标签)
    - Branch B: PD分类 (支持双流/单流/混合)

模块组织:
  - config.py: 全局配置
  - unet_multitask.py: U-Net主架构
  - encoder.py: 编码器
  - decoder_branches.py: 解码器两个分支
  - segmentation_branch.py: 分割和重建任务
  - classification_branch.py: 分类任务
  - multitask_data_loader.py: 数据加载 (CORN-1 + PD)
  - multitask_trainer.py: 多任务训练器
  - multitask_utils.py: 工具函数 (Frangi Filter等)
"""

from .config import MTLConfig
from .unet_multitask import UNetMultiTask
from .segmentation_branch import SegmentationBranch, MAEReconstructor
from .classification_branch import ClassificationBranch
from .multitask_data_loader import load_corn1_dataset, load_pd_dataset
from .multitask_trainer import MultiTaskTrainer
from .multitask_utils import FrangiFilter, HessianPseudoLabeler

__all__ = [
    'MTLConfig',
    'UNetMultiTask',
    'SegmentationBranch',
    'MAEReconstructor',
    'ClassificationBranch',
    'load_corn1_dataset',
    'load_pd_dataset',
    'MultiTaskTrainer',
    'FrangiFilter',
    'HessianPseudoLabeler',
]
