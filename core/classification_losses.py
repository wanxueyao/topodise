"""Losses used by the shared classification trainer."""
import torch
import torch.nn as nn
import torch.nn.functional as F

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
