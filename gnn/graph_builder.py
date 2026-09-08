"""
graph_builder.py - 从骨架构建图 (节点/边检测 + 特征提取)

算法:
  1. 用 3×3 卷积高效计算骨架每个像素的 8-连通度 (degree)
  2. 分类像素:
       degree == 0: 孤立点  → 端点
       degree == 1: 端点
       degree == 2: 普通路径像素
       degree >= 3: 分支点
  3. 节点 = 端点 + 分支点 (degree != 2)
  4. 移除节点像素 → 剩余像素用 scipy 连通分量标记，每个分量 = 一段路径
  5. 每段路径 + 其相邻节点 → 一条边
  6. 计算节点和边的特征向量

节点特征 (NODE_FEAT_DIM = 5):
  [x_norm, y_norm, node_type, degree_norm, local_density]

边特征 (EDGE_FEAT_DIM = 3):
  [path_len_norm, eucl_norm, tortuosity]
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List
from scipy.ndimage import label as scipy_label, convolve, binary_dilation

# 特征维度常量
NODE_FEAT_DIM = 5
EDGE_FEAT_DIM = 3

# 8-连通结构元素
_STRUCT8 = np.ones((3, 3), dtype=bool)


@dataclass
class NerveGraphData:
    """单张图像的图数据结构（numpy 格式，供后续转 tensor 使用）。"""
    node_features: np.ndarray   # [N, NODE_FEAT_DIM] float32
    edge_index: np.ndarray      # [2, E_directed] int64  (无向图双向存储)
    edge_features: np.ndarray   # [E_directed, EDGE_FEAT_DIM] float32
    num_nodes: int
    image_shape: Tuple[int, int]


# ─── 内部辅助函数 ─────────────────────────────────────────────────────────────

def _compute_degrees(skel: np.ndarray) -> np.ndarray:
    """用 3×3 卷积高效计算骨架各像素的 8-连通度。"""
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)
    skel_u8 = skel.astype(np.uint8)
    deg = convolve(skel_u8, kernel, mode='constant', cval=0)
    return deg * skel_u8   # 只保留骨架像素的度数，背景为 0


def _local_density(arr: np.ndarray, y: int, x: int, radius: int = 5) -> float:
    """计算 (y, x) 周边 (2*radius+1)×(2*radius+1) 区域的白像素比例。"""
    H, W = arr.shape
    y1, y2 = max(0, y - radius), min(H, y + radius + 1)
    x1, x2 = max(0, x - radius), min(W, x + radius + 1)
    patch = arr[y1:y2, x1:x2]
    if patch.size == 0:
        return 0.0
    return float(np.mean(patch.astype(np.float32)))


def _make_empty_graph(H: int, W: int) -> NerveGraphData:
    """返回只含 1 个虚节点 (全零) 的空图占位符。"""
    return NerveGraphData(
        node_features=np.zeros((1, NODE_FEAT_DIM), dtype=np.float32),
        edge_index=np.zeros((2, 0), dtype=np.int64),
        edge_features=np.zeros((0, EDGE_FEAT_DIM), dtype=np.float32),
        num_nodes=1,
        image_shape=(H, W),
    )


# ─── 主要函数 ─────────────────────────────────────────────────────────────────

def skeleton_to_graph(
    skel: np.ndarray,
    binary: Optional[np.ndarray] = None,
    min_component_size: int = 1,
) -> NerveGraphData:
    """
    将二值骨架转化为带特征的图数据结构。

    Args:
        skel: 二值骨架 [H, W], dtype bool 或 uint8
        binary: 可选，原始二值图像 [H, W]（用于计算局部密度特征）
                如为 None，则以骨架自身估计密度
        min_component_size: 忽略像素数 < 该值的微小连通分量

    Returns:
        NerveGraphData
    """
    H, W = skel.shape
    img_diag = max(float(np.sqrt(H * H + W * W)), 1.0)
    skel = skel.astype(bool)

    if not skel.any():
        return _make_empty_graph(H, W)

    # 用骨架自身作密度参考（若无原始二值图像）
    density_src = binary.astype(bool) if binary is not None else skel

    # ── Step 1: 计算像素度数 ──────────────────────────────────────────────────
    degree_map = _compute_degrees(skel)   # [H, W] uint8

    # ── Step 2: 识别节点像素 ──────────────────────────────────────────────────
    # 节点 = degree != 2 （端点 deg≤1，分支点 deg≥3，孤立点 deg=0）
    node_mask = skel & (degree_map != 2)

    # 若所有像素均为 degree=2（纯环），强制选一个像素为节点以断环
    if not node_mask.any():
        ys0, xs0 = np.where(skel)
        node_mask[ys0[0], xs0[0]] = True

    # ── Step 3: 建立节点列表 ──────────────────────────────────────────────────
    node_ys, node_xs = np.where(node_mask)
    num_nodes = len(node_ys)

    # (y, x) → 节点索引
    node_map: dict = {}
    for idx in range(num_nodes):
        node_map[(int(node_ys[idx]), int(node_xs[idx]))] = idx

    # ── Step 4: 找边 (非节点像素连通分量) ────────────────────────────────────
    non_node_skel = skel & ~node_mask
    labeled, num_comps = scipy_label(non_node_skel, structure=_STRUCT8)

    # ── Step 5: 建立边列表 ────────────────────────────────────────────────────
    edge_list: List[Tuple[int, int, int]] = []  # (i, j, path_pixel_count)
    seen_pairs: set = set()

    for comp_id in range(1, num_comps + 1):
        comp_mask = labeled == comp_id
        comp_size = int(comp_mask.sum())
        if comp_size < min_component_size:
            continue

        # 膨胀连通分量一像素，找相邻节点
        dilated = binary_dilation(comp_mask, structure=_STRUCT8)
        adj_mask = dilated & node_mask
        adj_ys, adj_xs = np.where(adj_mask)

        # 收集唯一节点索引
        adj_idx: List[int] = list({
            node_map[(int(y), int(x))]
            for y, x in zip(adj_ys.tolist(), adj_xs.tolist())
            if (int(y), int(x)) in node_map
        })

        # 对所有相邻节点对建边
        for k1 in range(len(adj_idx)):
            for k2 in range(k1 + 1, len(adj_idx)):
                i, j = adj_idx[k1], adj_idx[k2]
                pair = (min(i, j), max(i, j))
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    edge_list.append((i, j, comp_size))

    # 直连相邻节点 (两个节点像素 8-连通相邻，中间无普通像素)
    for (y, x), i in node_map.items():
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if dy == 0 and dx == 0:
                    continue
                nb = (y + dy, x + dx)
                if nb in node_map:
                    j = node_map[nb]
                    if i < j:
                        pair = (i, j)
                        if pair not in seen_pairs:
                            seen_pairs.add(pair)
                            edge_list.append((i, j, 0))

    # ── Step 6: 计算边特征 (无向图 → 双向存储) ────────────────────────────────
    src_list: List[int] = []
    dst_list: List[int] = []
    ef_list: List[np.ndarray] = []

    for (i, j, path_size) in edge_list:
        yi, xi = int(node_ys[i]), int(node_xs[i])
        yj, xj = int(node_ys[j]), int(node_xs[j])

        eucl = float(np.sqrt((yi - yj) ** 2 + (xi - xj) ** 2))
        path_len = float(path_size + 1)   # +1 保证路径长度 ≥ 1
        tort = min(path_len / max(eucl, 1.0), 5.0) / 5.0

        ef = np.array([
            path_len / img_diag,
            eucl / img_diag,
            tort,
        ], dtype=np.float32)

        # 双向
        src_list += [i, j]
        dst_list += [j, i]
        ef_list += [ef, ef]

    if src_list:
        edge_index = np.array([src_list, dst_list], dtype=np.int64)   # [2, E]
        edge_features = np.stack(ef_list, axis=0)                      # [E, EDGE_FEAT_DIM]
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_features = np.zeros((0, EDGE_FEAT_DIM), dtype=np.float32)

    # ── Step 7: 计算节点特征 ──────────────────────────────────────────────────
    # 先统计每个节点的最终度数（在 edge_list 中）
    node_degree = np.zeros(num_nodes, dtype=np.float32)
    for (i, j, _) in edge_list:
        node_degree[i] += 1.0
        node_degree[j] += 1.0
    max_deg = float(max(node_degree.max(), 1.0))

    node_features = np.zeros((num_nodes, NODE_FEAT_DIM), dtype=np.float32)
    for idx in range(num_nodes):
        y, x = int(node_ys[idx]), int(node_xs[idx])
        deg = int(degree_map[y, x])
        node_features[idx, 0] = x / max(W - 1, 1)                     # 归一化 x
        node_features[idx, 1] = y / max(H - 1, 1)                     # 归一化 y
        node_features[idx, 2] = 1.0 if deg >= 3 else 0.0              # 节点类型: 0=端点, 1=分支
        node_features[idx, 3] = node_degree[idx] / max_deg            # 归一化度数
        node_features[idx, 4] = _local_density(density_src, y, x, radius=5)  # 局部密度

    return NerveGraphData(
        node_features=node_features,
        edge_index=edge_index,
        edge_features=edge_features,
        num_nodes=num_nodes,
        image_shape=(H, W),
    )
