from __future__ import annotations

import torch
from torch import nn


class ResidualAdapter(nn.Module):
    def __init__(self, dim: int = 512, rank: int = 32):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(torch.nn.functional.gelu(self.down(self.norm(x))))


class GatedAttentionMIL(nn.Module):
    def __init__(self, dim: int = 512, hidden: int = 128, dropout: float = 0.25):
        super().__init__()
        self.v = nn.Linear(dim, hidden)
        self.u = nn.Linear(dim, hidden)
        self.w = nn.Linear(hidden, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, instances: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        single_bag = instances.ndim == 2
        if single_bag:
            instances = instances.unsqueeze(0)
        if instances.ndim != 3:
            raise ValueError("MIL instances must have shape [K,D] or [B,K,D]")
        if mask is None:
            mask = torch.ones(instances.shape[:2], dtype=torch.bool, device=instances.device)
        elif mask.ndim == 1:
            mask = mask.unsqueeze(0)
        if mask.shape != instances.shape[:2] or not bool(mask.any(dim=1).all()):
            raise ValueError("each bag mask must select at least one valid instance")
        scores = self.w(torch.tanh(self.v(instances)) * torch.sigmoid(self.u(instances))).squeeze(-1)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=1)
        attention = attention.masked_fill(~mask, 0.0)
        pooled = (attention.unsqueeze(-1) * instances).sum(dim=1)
        pooled = self.dropout(pooled)
        if single_bag:
            return pooled, attention.squeeze(0)
        return pooled, attention


class CommonStage2(nn.Module):
    """所有冻结编码器共用的唯一可训练下游路径。"""

    def __init__(
        self,
        raw_dim: int,
        num_classes: int,
        common_dim: int = 512,
        adapter_rank: int = 32,
        attention_hidden: int = 128,
        attention_dropout: float = 0.25,
        morphology_dim: int = 32,
        input_mode: str = "region",
    ):
        super().__init__()
        if input_mode not in {"region", "native_pair"}:
            raise ValueError(f"unsupported Stage2 input_mode: {input_mode}")
        self.input_mode = input_mode
        self.input_projector = nn.Sequential(nn.LayerNorm(raw_dim), nn.Linear(raw_dim, common_dim))
        self.pair_composer = (
            nn.Sequential(
                nn.LayerNorm(common_dim * 3),
                nn.Linear(common_dim * 3, common_dim),
                nn.GELU(),
            )
            if input_mode == "region" else None
        )
        self.adapter = ResidualAdapter(common_dim, adapter_rank)
        self.mil = GatedAttentionMIL(common_dim, attention_hidden, attention_dropout)
        self.morphology = nn.Sequential(
            nn.LayerNorm(21), nn.Linear(21, morphology_dim), nn.GELU()
        )
        self.classifier = nn.Linear(common_dim + morphology_dim, num_classes)

    def forward(
        self,
        central_features: torch.Tensor,
        whorl_features: torch.Tensor | None,
        morphology: torch.Tensor,
        pair_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        single_bag = central_features.ndim == 2
        if single_bag:
            central_features = central_features.unsqueeze(0)
            if whorl_features is not None:
                whorl_features = whorl_features.unsqueeze(0)
            morphology = morphology.reshape(1, 21)
            if pair_mask is not None and pair_mask.ndim == 1:
                pair_mask = pair_mask.unsqueeze(0)
        if central_features.ndim != 3:
            raise ValueError("features must have shape [K,D] or [B,K,D]")
        if self.input_mode == "region":
            if whorl_features is None or whorl_features.ndim != 3:
                raise ValueError("region mode requires whorl features")
            if central_features.shape != whorl_features.shape:
                raise ValueError("central/whorl feature shapes differ")
        elif whorl_features is not None:
            raise ValueError("native_pair mode must not receive separate whorl features")
        if morphology.shape != (central_features.shape[0], 21):
            raise ValueError("morphology must have shape [21] or [B,21]")
        central = self.input_projector(central_features)
        if self.input_mode == "native_pair":
            pair = central
        else:
            whorl = self.input_projector(whorl_features)
            pair = self.pair_composer(
                torch.cat([central, whorl, (central - whorl).abs()], dim=-1)
            )
        pair = self.adapter(pair)
        bag, attention = self.mil(pair, pair_mask)
        morph = self.morphology(morphology)
        logits = self.classifier(torch.cat([bag, morph], dim=-1))
        if single_bag and attention.ndim == 2:
            attention = attention.squeeze(0)
        return logits, attention
