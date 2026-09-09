"""Loss functions: SupCon, focal-BCE, region InfoNCE."""
import torch
import torch.nn.functional as F


def supcon_loss(features: torch.Tensor, labels: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    """Supervised Contrastive Loss (Khosla et al. 2020).

    features: [B, D] L2-normalized.
    labels:   [B] int.
    """
    B = features.size(0)
    if B < 2:
        return features.sum() * 0.0
    sim = features @ features.t() / tau
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()
    mask_pos = (labels[:, None] == labels[None, :]).float()
    eye = torch.eye(B, device=features.device)
    mask_pos = mask_pos * (1 - eye)
    mask_self = 1 - eye
    exp_sim = sim.exp() * mask_self
    log_prob = sim - exp_sim.sum(dim=1, keepdim=True).clamp(min=1e-12).log()
    pos_count = mask_pos.sum(dim=1).clamp(min=1)
    mean_log = (mask_pos * log_prob).sum(dim=1) / pos_count
    # 仅在存在正对的样本上求平均，避免 NaN
    has_pos = (mask_pos.sum(dim=1) > 0).float()
    if has_pos.sum() == 0:
        return features.sum() * 0.0
    return -(mean_log * has_pos).sum() / has_pos.sum()


def focal_bce_loss(logits: torch.Tensor, targets: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = torch.where(targets > 0.5, p, 1 - p)
    return (((1 - pt) ** gamma) * ce).mean()


def nt_xent_loss(za: torch.Tensor, zb: torch.Tensor, tau: float = 0.2) -> torch.Tensor:
    """SimCLR NT-Xent instance-discrimination loss.

    za, zb: [B, D] embeddings of two augmented views; (za_i, zb_i) is the only
    positive pair for row i, every other sample is a negative.
    """
    B = za.size(0)
    if B < 2:
        return za.sum() * 0.0
    a = F.normalize(za, dim=-1)
    b = F.normalize(zb, dim=-1)
    reps = torch.cat([a, b], dim=0)              # [2B, D]
    sim = reps @ reps.t() / tau                  # [2B, 2B]
    eye = torch.eye(2 * B, device=za.device, dtype=torch.bool)
    sim = sim.masked_fill(eye, float("-inf"))
    targets = torch.cat([
        torch.arange(B, 2 * B, device=za.device),
        torch.arange(0, B, device=za.device),
    ])
    return F.cross_entropy(sim, targets)


def focal_ce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    weight: torch.Tensor = None,
    label_smoothing: float = 0.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Multi-class focal loss.

    Focal modulation ``(1 - p_t)^gamma`` is applied on top of (optionally
    class-weighted, label-smoothed) cross-entropy. Reduces to standard CE when
    ``gamma == 0``.

    logits:  [B, C]
    targets: [B] int (>=0 entries only; caller should mask invalid labels)
    weight:  [C] optional class weights (inverse-frequency, etc.)
    """
    ce = F.cross_entropy(
        logits, targets,
        weight=weight,
        label_smoothing=label_smoothing,
        reduction="none",
    )
    # p_t = prob of the true class (from un-smoothed softmax)
    logp = F.log_softmax(logits, dim=-1)
    pt = logp.gather(1, targets.clamp(min=0).unsqueeze(1)).squeeze(1).exp()
    loss = ((1.0 - pt) ** gamma) * ce
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    return loss.mean()


def region_infonce_loss(
    cls_c: torch.Tensor, cls_w: torch.Tensor, tau: float = 0.2
) -> torch.Tensor:
    """Region-discrimination InfoNCE.

    cls_c, cls_w: [B, D] CLS embeddings from valid central / whorl pairs.
    Duplicated single-region placeholders should be masked out before calling.

    正对：(c_i, c_j) i≠j -- 同区域跨 subject
    负对：(c_i, w_k)    -- 任意 whorl
    """
    B = cls_c.size(0)
    if B < 2:
        return cls_c.sum() * 0.0
    cN = F.normalize(cls_c, dim=-1)
    wN = F.normalize(cls_w, dim=-1)
    sim_cc = cN @ cN.t() / tau         # [B, B]
    sim_cw = cN @ wN.t() / tau         # [B, B] 全负
    eye = torch.eye(B, device=cls_c.device, dtype=torch.bool)
    # 数值稳定
    M = torch.max(sim_cc.max(), sim_cw.max()).detach()
    pos = sim_cc.masked_fill(eye, float("-inf")) - M
    neg = sim_cw - M
    # log sum exp over 正对（除自身）+ 全部 whorl 负对
    denom_terms = torch.cat([pos, neg], dim=1)
    log_denom = torch.logsumexp(denom_terms, dim=1)
    # 每行的正对数 = B-1
    log_num = torch.logsumexp(pos, dim=1)
    loss_c = -(log_num - log_denom).mean()

    # 对称：whorl 侧
    sim_ww = wN @ wN.t() / tau
    sim_wc = wN @ cN.t() / tau
    M = torch.max(sim_ww.max(), sim_wc.max()).detach()
    pos = sim_ww.masked_fill(eye, float("-inf")) - M
    neg = sim_wc - M
    denom_terms = torch.cat([pos, neg], dim=1)
    log_denom = torch.logsumexp(denom_terms, dim=1)
    log_num = torch.logsumexp(pos, dim=1)
    loss_w = -(log_num - log_denom).mean()

    return 0.5 * (loss_c + loss_w)


def coral_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    n_classes: int,
    weight: torch.Tensor = None,
) -> torch.Tensor:
    """CORAL ordinal-regression loss (Cao et al., 2020).

    logits:  [B, n_classes-1] cumulative logits from ``CoralHead``.
    targets: [B] int in [0, n_classes-1].
    weight:  optional [n_classes] per-(true-)class importance weights.

    Each level k is a binary task "is y > k?"; loss is the mean binary
    cross-entropy over the (B x (K-1)) extended-binary label matrix.
    """
    B = logits.size(0)
    levels = torch.zeros(B, n_classes - 1, device=logits.device, dtype=logits.dtype)
    for k in range(n_classes - 1):
        levels[:, k] = (targets > k).float()
    loss = F.binary_cross_entropy_with_logits(logits, levels, reduction="none")
    if weight is not None:
        w = weight[targets.clamp(min=0)].unsqueeze(1)        # [B, 1]
        loss = loss * w
    return loss.mean()


def coral_predict(logits: torch.Tensor) -> torch.Tensor:
    """Map CORAL cumulative logits [B, K-1] to ordinal class ranks [B] in [0, K-1]."""
    return (torch.sigmoid(logits) > 0.5).sum(dim=1)
