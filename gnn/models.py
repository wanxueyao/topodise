"""Pure PyTorch GAT layers and graph classification for the paper GNN recipes."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .graph_builder import NODE_FEAT_DIM, EDGE_FEAT_DIM
from .graph_dataset import GraphBatch


# ─── 聚合工具函数 ─────────────────────────────────────────────────────────────

def scatter_mean(
    src: torch.Tensor,
    index: torch.Tensor,
    dim_size: int,
) -> torch.Tensor:
    """
    按 index 分组，对 src 求均值。

    Args:
        src:      [E, D]
        index:    [E]  目标节点索引
        dim_size: 目标维度大小 (节点数)

    Returns:
        [dim_size, D]
    """
    D = src.size(1)
    out = torch.zeros(dim_size, D, device=src.device, dtype=src.dtype)
    cnt = torch.zeros(dim_size, 1, device=src.device, dtype=src.dtype)
    idx_exp = index.unsqueeze(1).expand_as(src)
    out.scatter_add_(0, idx_exp, src)
    cnt.scatter_add_(0, index.unsqueeze(1),
                     torch.ones(index.size(0), 1, device=src.device, dtype=src.dtype))
    return out / cnt.clamp(min=1.0)


def global_mean_pool(
    x: torch.Tensor,
    batch: torch.Tensor,
    num_graphs: int,
) -> torch.Tensor:
    """全局均值池化: [total_N, D] → [num_graphs, D]"""
    return scatter_mean(x, batch, num_graphs)


# ─── GAT 层 ───────────────────────────────────────────────────────────────────

class GATLayer(nn.Module):
    """
    单层图注意力网络 (Graph Attention Network)。

    各注意力头独立计算，最终对头输出取均值，保持维度为 out_dim。
    支持边特征参与注意力计算。

    Args:
        in_dim:   输入节点特征维度
        out_dim:  输出节点特征维度
        num_heads:注意力头数
        dropout:  dropout 比率
        edge_dim: 边特征维度 (0 = 不使用边特征)
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        edge_dim: int = EDGE_FEAT_DIM,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.out_dim = out_dim

        # 节点变换: in_dim → num_heads * out_dim
        self.W = nn.Linear(in_dim, num_heads * out_dim, bias=False)

        # 注意力参数 a: a^T * [h_i || h_j]  per head
        self.a = nn.Parameter(torch.zeros(num_heads, 2 * out_dim))

        # 边特征变换
        if edge_dim > 0:
            self.W_edge = nn.Linear(edge_dim, num_heads * out_dim, bias=False)
            self.a_edge = nn.Parameter(torch.zeros(num_heads, out_dim))
        else:
            self.W_edge = None
            self.a_edge = None

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_normal_(self.a.unsqueeze(0))
        if self.W_edge is not None:
            nn.init.xavier_uniform_(self.W_edge.weight)
            nn.init.xavier_normal_(self.a_edge.unsqueeze(0))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:          [N, in_dim]
            edge_index: [2, E]  (src → dst)
            edge_attr:  [E, edge_dim] 或 None

        Returns:
            [N, out_dim]
        """
        N = x.size(0)
        H = self.num_heads
        D = self.out_dim

        # [N, H, D]
        h = self.W(x).view(N, H, D)

        # 无边时直接返回头均值
        if edge_index.size(1) == 0:
            return h.mean(dim=1)   # [N, D]

        src_idx, dst_idx = edge_index[0], edge_index[1]
        E = src_idx.size(0)

        h_src = h[src_idx]   # [E, H, D]
        h_dst = h[dst_idx]   # [E, H, D]

        # 注意力 logit: [E, H]
        cat_hd = torch.cat([h_src, h_dst], dim=-1)        # [E, H, 2D]
        # self.a: [H, 2D]  →  unsqueeze: [1, H, 2D]
        e = (cat_hd * self.a.unsqueeze(0)).sum(dim=-1)     # [E, H]

        # 加入边特征
        if edge_attr is not None and self.W_edge is not None:
            he = self.W_edge(edge_attr).view(E, H, D)      # [E, H, D]
            e = e + (he * self.a_edge.unsqueeze(0)).sum(dim=-1)

        e = self.leaky_relu(e).clamp(-10.0, 10.0)          # [E, H]

        # Softmax over 每个目标节点的入射边
        alpha = torch.exp(e)                                # [E, H]
        alpha_sum = torch.zeros(N, H, device=x.device, dtype=x.dtype)
        alpha_sum.scatter_add_(0, dst_idx.unsqueeze(1).expand(-1, H), alpha)
        alpha = alpha / (alpha_sum[dst_idx] + 1e-8)        # [E, H]
        alpha = self.dropout(alpha)

        # 聚合邻居: sum_j alpha_ij * Wh_j  →  [N, H, D]
        msg = h_src * alpha.unsqueeze(-1)                   # [E, H, D]
        out = torch.zeros(N, H, D, device=x.device, dtype=x.dtype)
        idx_exp = dst_idx.unsqueeze(1).unsqueeze(2).expand(-1, H, D)
        out.scatter_add_(0, idx_exp, msg)

        return out.mean(dim=1)   # [N, D]  取头均值





# ─── GNN 编码器 ───────────────────────────────────────────────────────────────

class GNNEncoder(nn.Module):
    """
    多层 GNN 编码器：节点特征 → 图级嵌入。

    流程:
      输入节点特征 → 嵌入层 → L 层 GNN → 全局均值池化 → 图级向量

    Args:
        node_feat_dim: 输入节点特征维度 (默认 NODE_FEAT_DIM)
        edge_feat_dim: 边特征维度 (默认 EDGE_FEAT_DIM)
        hidden_dim:    隐藏层维度
        num_layers:    GNN 层数
        gnn_type:      'gat'
        num_heads:     注意力头数 (GAT 专用)
        dropout:       dropout 比率
    """

    def __init__(
        self,
        node_feat_dim: int = NODE_FEAT_DIM,
        edge_feat_dim: int = EDGE_FEAT_DIM,
        hidden_dim: int = 64,
        num_layers: int = 2,
        gnn_type: str = 'gat',
        num_heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.out_dim = hidden_dim

        # 节点特征嵌入
        self.node_embed = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # GNN 层
        gnn_layers = []
        for _ in range(num_layers):
            if gnn_type != "gat":
                raise ValueError("Paper GNN recipes require gnn_type=gat")
            gnn_layers.append(GATLayer(
                in_dim=hidden_dim,
                out_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                edge_dim=edge_feat_dim,
            ))
        self.gnn_layers = nn.ModuleList(gnn_layers)

        # 输出归一化
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, graph_batch: GraphBatch) -> torch.Tensor:
        """
        Args:
            graph_batch: GraphBatch（来自 collate_graph_tensors）

        Returns:
            [num_graphs, hidden_dim]  图级嵌入
        """
        x  = graph_batch.x           # [total_N, node_feat_dim]
        ei = graph_batch.edge_index  # [2, total_E]
        ea = graph_batch.edge_attr   # [total_E, edge_feat_dim]
        b  = graph_batch.batch       # [total_N]

        # 节点嵌入
        h = self.node_embed(x)       # [total_N, hidden_dim]

        # GNN 层（带残差连接）
        for layer in self.gnn_layers:
            h_new = layer(h, ei, ea)
            h = F.relu(h_new + h)   # 残差 + 激活

        h = self.out_norm(h)

        # 全局均值池化 → 图级嵌入
        return global_mean_pool(h, b, graph_batch.num_graphs)  # [num_graphs, hidden_dim]


# ─── 多任务 GNN 分类模型 ──────────────────────────────────────────────────────

class GNNMultiTaskModel(nn.Module):
    """
    双流 GNN 多任务分类模型。

    两路输入 (central + whorl) 分别经 GNN 编码，拼接后通过三个独立分类头:
      - PD 诊断:  二分类 (control=0, PD=1)
      - PD 分期:  三分类 (stage 1→0, 2→1, 3→2)
      - PD 亚型:  三分类 (TD=0, PIGD=1, mixed=2)

    标签 = -1 的样本在损失计算中自动被忽略（见 trainer.py）。

    Args:
        node_feat_dim:      输入节点特征维度
        edge_feat_dim:      边特征维度
        hidden_dim:         隐藏层维度
        num_layers:         GNN 层数
        gnn_type:           'gat'
        num_heads:          注意力头数 (GAT 专用)
        dropout:            GNN dropout
        classifier_dropout: 分类头 dropout
        shared_encoder:     True = central/whorl 共享 GNN 权重
        num_subtype_classes: subtype 输出类别数，默认 3；层级二分类实验设为 2
    """

    def __init__(
        self,
        node_feat_dim: int = NODE_FEAT_DIM,
        edge_feat_dim: int = EDGE_FEAT_DIM,
        hidden_dim: int = 64,
        num_layers: int = 2,
        gnn_type: str = 'gat',
        num_heads: int = 4,
        dropout: float = 0.2,
        classifier_dropout: float = 0.3,
        shared_encoder: bool = False,
        num_subtype_classes: int = 3,
    ):
        super().__init__()

        encoder_kwargs = dict(
            node_feat_dim=node_feat_dim,
            edge_feat_dim=edge_feat_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            gnn_type=gnn_type,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.central_encoder = GNNEncoder(**encoder_kwargs)
        self.whorl_encoder = (
            self.central_encoder if shared_encoder
            else GNNEncoder(**encoder_kwargs)
        )

        fusion_dim = hidden_dim * 2

        def _make_head(out_classes: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(fusion_dim, fusion_dim // 2),
                nn.ReLU(),
                nn.Dropout(classifier_dropout),
                nn.Linear(fusion_dim // 2, out_classes),
            )

        self.pd_head      = _make_head(2)                    # PD binary
        self.stage_head   = _make_head(3)                    # stage 3-class
        self.subtype_head = _make_head(num_subtype_classes)  # subtype / hierarchy binary

    def forward(
        self,
        central_batch: GraphBatch,
        whorl_batch: GraphBatch,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            central_batch: GraphBatch，central 图
            whorl_batch:   GraphBatch，whorl 图

        Returns:
            (pd_logits, stage_logits, subtype_logits)
            shapes: ([B, 2], [B, 3], [B, num_subtype_classes])
        """
        c_emb = self.central_encoder(central_batch)   # [B, hidden_dim]
        w_emb = self.whorl_encoder(whorl_batch)        # [B, hidden_dim]

        patient_emb = torch.cat([c_emb, w_emb], dim=-1)  # [B, 2*hidden_dim]

        return (
            self.pd_head(patient_emb),
            self.stage_head(patient_emb),
            self.subtype_head(patient_emb),
        )
