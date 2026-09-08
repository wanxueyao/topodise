"""
codes/gnn - 神经纤维图神经网络 (GNN) 模块

将 UNet 分割后的二值神经纤维图像转化为图结构，
使用 GAT 或 GraphSAGE 完成三个分类任务：
  - PD 诊断 (二分类)
  - PD 分期 (三分类: 1/2/3)
  - PD 亚型 (三分类: TD/PIGD/mixed)

主要子模块:
  skeletonize   - 图像骨架化
  graph_builder - 骨架 → 图 (节点/边检测 + 特征提取)
  graph_dataset - PyTorch 数据集 (含磁盘缓存)
  models        - GAT / GraphSAGE 多任务模型
  trainer       - 训练和评估循环
  classifier    - 高层分类器接口 (对标 core.basic_classifier.PDClassifier)
"""

from .graph_builder import skeleton_to_graph, NerveGraphData
from .graph_dataset import NerveGraphDataset, collate_fn
from .models import GNNMultiTaskModel
from .classifier import GNNClassifier, GNNClassifierConfig

__all__ = [
    'skeleton_to_graph',
    'NerveGraphData',
    'NerveGraphDataset',
    'collate_fn',
    'GNNMultiTaskModel',
    'GNNClassifier',
    'GNNClassifierConfig',
]
