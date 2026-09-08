"""Classifier and projection heads for RACF stage."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassifierHead(nn.Module):
    def __init__(self, z_dim: int = 512, n_classes: int = 2, hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, z):
        return self.net(z)


class CoralHead(nn.Module):
    """CORAL ordinal-regression head (Cao et al., 2020).

    Produces ``n_classes - 1`` cumulative logits that share a single scalar
    projection and only differ by ``n_classes - 1`` ordered bias thresholds, so
    the predicted cumulative probabilities P(y > k) are rank-monotonic by
    construction. Suited to ordinal targets such as H&Y stage (s1 < s2 < s3).
    """

    def __init__(self, z_dim: int = 512, n_classes: int = 3, hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        self.n_classes = n_classes
        self.backbone = nn.Sequential(
            nn.Linear(z_dim, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.fc = nn.Linear(hidden, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(n_classes - 1))

    def forward(self, z):
        h = self.backbone(z)
        return self.fc(h) + self.bias            # [B, n_classes-1]


class ProjectionHead(nn.Module):
    """L2-normalized projection for SupCon."""

    def __init__(self, z_dim: int = 512, proj_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, proj_dim),
            nn.SiLU(),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward(self, z):
        u = self.net(z)
        return F.normalize(u, dim=-1)


class PrototypeHead(nn.Module):
    """Prototypical metric head (Snell et al., 2017, "Prototypical Networks").

    Borrowed from the few-shot prototype classifier: project a pooled embedding
    into an L2-normalized metric space (with strong feature dropout for
    regularization on tiny datasets), then classify by negative distance to
    per-class prototypes instead of a parametric softmax. This is far more
    sample-efficient than a linear head when each class has only a handful of
    patients, which is exactly the regime of the PD stage / subtype tasks.
    """

    def __init__(self, z_dim: int = 512, proj_dim: int = 128,
                 dropout: float = 0.65, metric: str = "euclidean"):
        super().__init__()
        assert metric in ("euclidean", "cosine")
        self.metric = metric
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(z_dim, proj_dim)

    def forward(self, z):
        u = self.proj(self.dropout(z))
        return F.normalize(u, p=2, dim=-1)

    @staticmethod
    def compute_prototypes(emb: torch.Tensor, labels: torch.Tensor, n_classes: int):
        """Mean L2-normalized embedding per class.

        Returns ``(prototypes [C, D], present_mask [C])`` where ``present_mask``
        is 1.0 for classes that had at least one support sample.
        """
        protos, present = [], []
        for c in range(n_classes):
            sel = emb[labels == c]
            if sel.numel() == 0:
                protos.append(torch.zeros(emb.size(1), device=emb.device, dtype=emb.dtype))
                present.append(0.0)
            else:
                protos.append(F.normalize(sel.mean(dim=0), p=2, dim=-1))
                present.append(1.0)
        return (torch.stack(protos, dim=0),
                torch.tensor(present, device=emb.device, dtype=emb.dtype))

    def distance_logits(self, query: torch.Tensor, prototypes: torch.Tensor,
                        temp: float = 1.0) -> torch.Tensor:
        """Classification logits = (cosine sim | negative sq-euclidean) / temp."""
        if self.metric == "cosine":
            return (query @ prototypes.t()) / max(temp, 1e-6)
        d = self._squared_euclidean(query, prototypes)
        return -d / max(temp, 1e-6)

    @staticmethod
    def _squared_euclidean(query: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        if query.device.type != "mps":
            return torch.cdist(query, support, p=2).pow(2)
        q2 = query.pow(2).sum(dim=1, keepdim=True)
        s2 = support.pow(2).sum(dim=1).unsqueeze(0)
        return (q2 + s2 - 2.0 * (query @ support.t())).clamp_min(0.0)

    def support_logits(self, query: torch.Tensor, support_emb: torch.Tensor,
                       support_labels: torch.Tensor, n_classes: int,
                       temp: float = 1.0, mode: str = "support_lse",
                       top_k: int = 3):
        """Classify against the full support embedding bank.

        ``compute_prototypes`` collapses every class into one mean vector. That
        is brittle for fine-grained PD subtype/stage, where a class can be
        multi-modal. This method keeps the full training-fold support bank and
        aggregates class-wise distances with either log-mean-exp (soft nearest
        prototypes) or top-k averaging (kNN-style prototype classification).

        Returns ``(logits [Q, C], present_mask [C])``.
        """
        class_logits, present = [], []
        for c in range(n_classes):
            sel = support_emb[support_labels == c]
            if sel.numel() == 0:
                class_logits.append(torch.full((query.size(0),), -1e4,
                                               device=query.device, dtype=query.dtype))
                present.append(0.0)
                continue
            if self.metric == "cosine":
                scores = (query @ sel.t()) / max(temp, 1e-6)
            else:
                scores = -self._squared_euclidean(query, sel) / max(temp, 1e-6)

            if top_k and top_k > 0 and scores.size(1) > top_k:
                scores = scores.topk(top_k, dim=1).values
            if mode == "support_topk":
                logit = scores.mean(dim=1)
            else:
                logit = torch.logsumexp(scores, dim=1) - math.log(max(scores.size(1), 1))
            class_logits.append(logit)
            present.append(1.0)
        return (torch.stack(class_logits, dim=1),
                torch.tensor(present, device=query.device, dtype=query.dtype))


def episodic_proto_loss(emb: torch.Tensor, labels: torch.Tensor, n_classes: int,
                        proto_head: "PrototypeHead", temp: float = 1.0,
                        min_per_class: int = 2,
                        generator: torch.Generator = None) -> torch.Tensor:
    """In-batch episodic prototypical loss (ProtoNet style).

    Within the current mini-batch of (already projected, L2-normalized) metric
    embeddings ``emb``, classes that have at least ``min_per_class`` samples are
    split into a support half (used to build prototypes) and a query half
    (classified by distance). Cross-entropy over the query set encourages a
    metric space where same-class embeddings cluster, mirroring the prototype
    method that beats the parametric head on small PD data.

    Returns a scalar loss (0 if no class has enough samples to form an episode).
    """
    device = emb.device
    classes = []
    for c in range(n_classes):
        idx = (labels == c).nonzero(as_tuple=True)[0]
        if idx.numel() >= min_per_class:
            classes.append((c, idx))
    if len(classes) < 2:
        return torch.zeros((), device=device)

    support_emb, support_lab = [], []
    query_emb, query_lab = [], []
    for epi_label, (_c, idx) in enumerate(classes):
        perm = idx[torch.randperm(idx.numel(), generator=generator, device=device)]
        n_support = max(1, perm.numel() // 2)
        s_idx, q_idx = perm[:n_support], perm[n_support:]
        if q_idx.numel() == 0:
            q_idx = s_idx[-1:]
        support_emb.append(emb[s_idx])
        support_lab.append(torch.full((s_idx.numel(),), epi_label, device=device, dtype=torch.long))
        query_emb.append(emb[q_idx])
        query_lab.append(torch.full((q_idx.numel(),), epi_label, device=device, dtype=torch.long))

    support_emb = torch.cat(support_emb, dim=0)
    support_lab = torch.cat(support_lab, dim=0)
    query_emb = torch.cat(query_emb, dim=0)
    query_lab = torch.cat(query_lab, dim=0)

    n_way = len(classes)
    protos, _ = PrototypeHead.compute_prototypes(support_emb, support_lab, n_way)
    logits = proto_head.distance_logits(query_emb, protos, temp=temp)
    return F.cross_entropy(logits, query_lab)
