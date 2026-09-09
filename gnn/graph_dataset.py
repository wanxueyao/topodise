"""
graph_dataset.py - 图数据集

将患者元数据和图像路径转化为可供 DataLoader 使用的 PyTorch 数据集。

每个样本: (central_graph, whorl_graph, pd_label, stage_label, subtype_label)

支持磁盘缓存以避免重复的骨架化和图构建计算。
图以 dict-of-tensors 格式缓存，兼容 torch.load(weights_only=True)。

主要组件:
  GraphTensor        - 单图 Tensor 数据结构
  GraphBatch         - 多图合并 Batch（块对角拼接）
  collate_fn         - DataLoader 专用 collate 函数
  NerveGraphDataset  - 双流图数据集
"""

import os
import hashlib
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

import torch
from torch.utils.data import Dataset

from .skeletonize import image_to_skeleton
from .graph_builder import skeleton_to_graph, NerveGraphData, EDGE_FEAT_DIM


# ─── 数据结构 ──────────────────────────────────────────────────────────────────

@dataclass
class GraphTensor:
    """单图 Tensor 形式。"""
    x: torch.Tensor           # [N, NODE_FEAT_DIM]  节点特征
    edge_index: torch.Tensor  # [2, E]              边索引 (无向双向)
    edge_attr: torch.Tensor   # [E, EDGE_FEAT_DIM]  边特征
    num_nodes: int


@dataclass
class GraphBatch:
    """
    多图合并 Batch（块对角拼接）。
    兼容纯 PyTorch 实现的 GNN 消息传递（无需 PyG）。
    """
    x: torch.Tensor           # [total_N, NODE_FEAT_DIM]
    edge_index: torch.Tensor  # [2, total_E]  (已按图偏移)
    edge_attr: torch.Tensor   # [total_E, EDGE_FEAT_DIM]
    batch: torch.Tensor       # [total_N]  各节点所属图 index
    num_graphs: int
    ptr: torch.Tensor         # [num_graphs+1]  节点计数前缀和


# ─── GraphData 与 Tensor 互转 ─────────────────────────────────────────────────

def graph_data_to_tensor(g: NerveGraphData) -> GraphTensor:
    """NerveGraphData (numpy) → GraphTensor (torch)。"""
    return GraphTensor(
        x=torch.from_numpy(g.node_features),
        edge_index=torch.from_numpy(g.edge_index),
        edge_attr=torch.from_numpy(g.edge_features),
        num_nodes=g.num_nodes,
    )


# ─── Batch 合并 ───────────────────────────────────────────────────────────────

def collate_graph_tensors(graph_list: List[GraphTensor]) -> GraphBatch:
    """
    将多个 GraphTensor 合并为一个 GraphBatch（块对角拼接）。

    Args:
        graph_list: 待合并的图列表

    Returns:
        GraphBatch
    """
    total_nodes = 0
    xs, edge_indices, edge_attrs, batches = [], [], [], []
    ptr = [0]

    for i, g in enumerate(graph_list):
        N = g.num_nodes
        xs.append(g.x)
        if g.edge_index.size(1) > 0:
            edge_indices.append(g.edge_index + total_nodes)
        else:
            edge_indices.append(torch.zeros((2, 0), dtype=torch.long))
        edge_attrs.append(g.edge_attr)
        batches.append(torch.full((N,), i, dtype=torch.long))
        total_nodes += N
        ptr.append(total_nodes)

    # 拼接所有图
    x_cat = torch.cat(xs, dim=0)
    ea_cat = torch.cat(edge_attrs, dim=0) if edge_attrs else torch.zeros((0, EDGE_FEAT_DIM))
    has_edges = any(ei.size(1) > 0 for ei in edge_indices)
    ei_cat = (torch.cat(edge_indices, dim=1)
               if has_edges
               else torch.zeros((2, 0), dtype=torch.long))

    return GraphBatch(
        x=x_cat,
        edge_index=ei_cat,
        edge_attr=ea_cat,
        batch=torch.cat(batches, dim=0),
        num_graphs=len(graph_list),
        ptr=torch.tensor(ptr, dtype=torch.long),
    )


def collate_fn(batch) -> Tuple:
    """
    DataLoader collate 函数。

    每个 batch item:
        (central_graph: GraphTensor,
         whorl_graph:   GraphTensor,
         pd_label:      int,
         stage_label:   int,
         subtype_label: int)

    Returns:
        (central_batch, whorl_batch, pd_labels, stage_labels, subtype_labels)
    """
    central_gs, whorl_gs, pd_ls, stage_ls, sub_ls, patient_ids = zip(*batch)
    return (
        collate_graph_tensors(list(central_gs)),
        collate_graph_tensors(list(whorl_gs)),
        torch.tensor(pd_ls,    dtype=torch.long),
        torch.tensor(stage_ls, dtype=torch.long),
        torch.tensor(sub_ls,   dtype=torch.long),
        list(patient_ids),
    )


# ─── 磁盘缓存 ──────────────────────────────────────────────────────────────────

def _cache_path(image_path: str, cache_dir: str) -> str:
    """为图像路径计算缓存文件路径（含 MD5 校验）。"""
    stem = os.path.splitext(os.path.basename(image_path))[0]
    h = hashlib.md5(image_path.encode()).hexdigest()[:8]
    return os.path.join(cache_dir, f"{stem}_{h}.pt")


def _save_graph_tensor(gt: GraphTensor, path: str) -> None:
    """将 GraphTensor 序列化为 dict，兼容 weights_only=True。"""
    torch.save({
        'x':          gt.x,
        'edge_index': gt.edge_index,
        'edge_attr':  gt.edge_attr,
        'num_nodes':  gt.num_nodes,
    }, path)


def _load_graph_tensor(path: str) -> GraphTensor:
    """从缓存文件加载 GraphTensor。"""
    d = torch.load(path, weights_only=True)
    return GraphTensor(
        x=d['x'],
        edge_index=d['edge_index'],
        edge_attr=d['edge_attr'],
        num_nodes=int(d['num_nodes']),
    )


def load_or_build_graph(
    image_path: str,
    cache_dir: Optional[str],
    threshold: int = 128,
) -> GraphTensor:
    """
    从缓存加载图，若缓存不存在则从图像构建并写入缓存。

    Args:
        image_path: 神经纤维图像路径
        cache_dir:  缓存目录，为 None 时不使用缓存
        threshold:  二值化阈值

    Returns:
        GraphTensor
    """
    if cache_dir:
        cp = _cache_path(image_path, cache_dir)
        if os.path.exists(cp):
            return _load_graph_tensor(cp)

    binary, skeleton = image_to_skeleton(image_path, threshold)
    g = skeleton_to_graph(skeleton, binary=binary)
    gt = graph_data_to_tensor(g)

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        _save_graph_tensor(gt, cp)

    return gt


# ─── 数据集类 ──────────────────────────────────────────────────────────────────

class NerveGraphDataset(Dataset):
    """
    神经纤维图数据集（双流，中央 + 旋涡）。

    每个样本 = (central_graph, whorl_graph, pd_label, stage_label, subtype_label)

    标签含义:
        pd_label:      0=control, 1=PD
        stage_label:   0/1/2 对应 stage 1/2/3，非 PD 患者为 -1
        subtype_label: 0=TD, 1=PIGD, 2=mixed，非 PD 或无亚型为 -1

    Args:
        patients:  患者元数据列表（来自 core.data_loader.load_and_filter_dataset）
        base_path: 图像路径前缀（JSON 中路径相对于此目录）
        cache_dir: 图缓存目录，None 表示不缓存
        threshold: 图像二值化阈值
    """

    SUBTYPE_MAP = {'TD': 0, 'PIGD': 1, 'mixed': 2}

    def __init__(
        self,
        patients: List[Dict],
        base_path: str,
        cache_dir: Optional[str] = None,
        threshold: int = 128,
    ):
        super().__init__()
        self.base_path = base_path
        self.cache_dir = cache_dir
        self.threshold = threshold

        # 每个 sample = (c_path, w_path, pd_label, stage_label, subtype_label, patient_id)
        self.samples: List[Tuple] = []
        self._build_samples(patients)

    # ── 构建样本列表 ────────────────────────────────────────────────────────────

    def _build_samples(self, patients: List[Dict]) -> None:
        for patient in patients:
            central = patient['image_paths'].get('central', [])
            whorl   = patient['image_paths'].get('whorl',   [])
            pd_l, stage_l, sub_l = self._encode_labels(patient)

            for c_rel, w_rel in zip(central, whorl):
                c_path = os.path.join(self.base_path, c_rel)
                w_path = os.path.join(self.base_path, w_rel)
                if os.path.exists(c_path) and os.path.exists(w_path):
                    self.samples.append((
                        c_path, w_path, pd_l, stage_l, sub_l,
                        patient.get('patient_id', len(self.samples)),
                    ))

    def _encode_labels(self, patient: Dict) -> Tuple[int, int, int]:
        """将患者元数据编码为 (pd_label, stage_label, subtype_label)。"""
        group = patient.get('group', '')
        is_pd = (group == 'pd')

        pd_label = 1 if is_pd else 0

        stage = patient.get('stage', 0)
        stage_label = (int(stage) - 1) if (is_pd and stage and int(stage) > 0) else -1

        subtype = patient.get('subtype')
        subtype_label = self.SUBTYPE_MAP.get(subtype, -1) if (is_pd and subtype) else -1

        return pd_label, stage_label, subtype_label

    # ── Dataset 接口 ────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        c_path, w_path, pd_l, stage_l, sub_l, patient_id = self.samples[idx]
        c_graph = load_or_build_graph(c_path, self.cache_dir, self.threshold)
        w_graph = load_or_build_graph(w_path, self.cache_dir, self.threshold)
        return c_graph, w_graph, pd_l, stage_l, sub_l, patient_id
